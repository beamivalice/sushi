#!/usr/bin/env python3
"""Oracle for the deepseek_v41 port: DeepSeek's own inference/model.py (the V4.1-Flash release),
run in fp32 on the CPU over a TINY random model.

The checkpoint is written in the ORIGINAL release layout (tensor names, fp8 e4m3 trunk with 32x32
ue8m0 block scales, int8-packed fp4 experts with one e8m0 per 32, fp8 Engram rows), at toy geometry
that keeps every mechanism: sliding window, compressed KV at ratio 2 and 1 with KV/index source
layers, candidate blocks, hyper-connections, Engram, sqrtsoftplus routing. The reference then runs
over the values those bytes decode to (torch's own fp8/fp4/e8m0 casts), so the fixture measures the
engine, not a quantizer.

The reference's TileLang kernels are replaced by the CPU stub below (the same math, in torch). Its
declared adaptations: the input of a GEMM is not fake-quantized (`act_quant` without `inplace`
returns x), the Engram rows stay fp32, and the head returns every position. The QAT round trips that
the attention and compressor run with `inplace=True` stay on, and `fixture_noqat.safetensors` is the
same run with them off (plain math, the tight bar).

  python -I tests/dump_dsv41_fixtures.py --reference <DeepSeek-V4.1-Flash>/inference --out <dir> [--geometry stq]
  DSV41_TINY=<dir> zig build test -Doptimize=ReleaseFast -Dtest-filter="dsv41"

Writes under <dir>: the pack (config.json, model-*.safetensors, model.safetensors.index.json),
fixture.safetensors, fixture_noqat.safetensors and qat.safetensors (the three round trips over inputs
with exact rounding midpoints).
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

S, D = 40, 6
VOCAB = 96
CFG = dict(
    vocab_size=VOCAB, hidden_size=128, moe_intermediate_size=128, num_hidden_layers=8,
    num_attention_heads=4, num_key_value_heads=1, hidden_act="silu", head_dim=64, qk_rope_head_dim=16,
    q_lora_rank=64, o_lora_rank=32, o_groups=2, swiglu_limit=1.0, rms_norm_eps=1e-20,
    max_position_embeddings=128, rope_theta=10000.0,
    rope_scaling=dict(rope_type="yarn", factor=4, beta_fast=32, beta_slow=1, original_max_position_embeddings=64),
    n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=3, scoring_func="sqrtsoftplus",
    topk_method="noaux_tc", norm_topk_prob=True, routed_scaling_factor=1.5, sliding_window=8,
    compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0], compress_rope_theta=40000.0,
    kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5, 7], index_n_heads=8, index_head_dim=32,
    index_topk=4, candidate_source_layer_id=5, candidate_topk_blocks=3, candidate_block_size=2,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, engram_layer_ids=[1, 2], engram_max_ngram_size=4,
    engram_vocab_size=50, engram_n_heads=2, engram_head_dim=64, engram_pad_token_id=2,
    engram_compressed_vocab_size=VOCAB, num_nextn_predict_layers=3, dspark_block_size=4,
    dspark_noise_token_id=VOCAB - 1, dspark_target_layer_ids=[5, 6, 7], dspark_markov_rank=32,
    dspark_n_routed_experts=4, dspark_num_experts_per_tok=2,
)
# `--geometry stq`: every trunk linear but the indexer's weights_proj on the 128 grid a trellis bank needs, for the
# STQ pack tests (`DSV41_STQ_TINY`); the mechanisms are the default geometry's.
STQ_GEOMETRY = dict(
    hidden_size=256, num_attention_heads=2, head_dim=128, q_lora_rank=128, o_lora_rank=128, o_groups=2,
    index_head_dim=128,
)
STQ_VOCAB = 256
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
E4M3_MAX = 448.0
# the kernels multiply by the f32 reciprocal, which differs from a division at exact powers of two
INV_E4M3_MAX = torch.tensor(1.0 / 448.0, dtype=torch.float32)
INV_E2M1_MAX = torch.tensor(1.0 / 6.0, dtype=torch.float32)


# ---- the CPU stub of inference/kernel.py -------------------------------------------------------

class Kernel(types.ModuleType):
    """`act_quant`, `fp4_act_quant`, `sparse_attn`, `hc_split_sinkhorn`; the GEMMs are never reached (dense reference)."""

    def __init__(self):
        super().__init__("kernel")
        self.DISABLE_FAKE_QUANT = False

    @staticmethod
    def _round_scale(v):
        # 2 ** ceil(log2(v)) for positive f32, exactly (fast_round_scale)
        m, e = torch.frexp(v)
        return torch.ldexp(torch.ones_like(v), torch.where(m == 0.5, e - 1, e))

    def act_quant(self, x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if not inplace:
            return x.float(), torch.ones(*x.shape[:-1], x.shape[-1] // block_size)
        if self.DISABLE_FAKE_QUANT:
            return x
        z = x.contiguous().float().unflatten(-1, (-1, block_size))
        amax = z.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        s = self._round_scale(amax * INV_E4M3_MAX) if scale_fmt is not None else amax * INV_E4M3_MAX
        y = (z / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * s
        x.copy_(y.flatten(-2).to(x.dtype))
        return x

    @staticmethod
    def _fp4_round(v):
        grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
        a = v.abs()
        idx = (a.unsqueeze(-1) - grid).abs().argmin(-1)
        # argmin takes the lower code at an exact midpoint; round-to-nearest-even moves it up from an odd one
        mid = (grid[:-1] + grid[1:]) / 2
        up = (a.unsqueeze(-1) == mid).any(-1) & (torch.searchsorted(mid, a, right=True) % 2 == 0)
        idx = torch.where(up, idx + 1, idx)
        return torch.copysign(grid[idx.clamp(max=7)], v)

    def fp4_act_quant(self, x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        assert inplace
        if self.DISABLE_FAKE_QUANT:
            return x
        z = x.contiguous().float().unflatten(-1, (-1, block_size))
        amax = z.abs().amax(-1, keepdim=True)
        if scale_dtype == torch.float8_e4m3fn:
            amax = amax.clamp_min(6 * 2.0 ** -9)
            s = (amax / 6).to(torch.float8_e4m3fn).float()
        else:
            amax = amax.clamp_min(6 * 2.0 ** -126)
            s = self._round_scale(amax * INV_E2M1_MAX)
        y = self._fp4_round((z / s).clamp(-6, 6)) * s
        x.copy_(y.flatten(-2).to(x.dtype))
        return x

    def fp8_gemm(self, *args, **kwargs):
        raise NotImplementedError("the reference runs dense over decoded weights")

    fp4_gemm = fp8_gemm

    def sparse_attn(self, q, kv, attn_sink, topk_idxs, softmax_scale):
        b, s, h, d = q.shape
        idx = topk_idxs.long()
        valid = idx >= 0
        gathered = torch.gather(kv.float().unsqueeze(1).expand(b, s, -1, d), 2,
                                idx.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, d))
        scores = torch.einsum("bshd,bskd->bshk", q.float(), gathered) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), -math.inf)
        sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
        p = torch.softmax(torch.cat([scores, sink], -1), -1)[..., :-1]
        return torch.einsum("bshk,bskd->bshd", p, gathered).to(q.dtype)

    def hc_split_sinkhorn(self, mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        hc = hc_mult
        pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
        comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
        comb = torch.softmax(comb, -1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb


def load_reference(ref_dir, kernel):
    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace()
    pil.ImageOps = types.SimpleNamespace()
    sys.modules.setdefault("PIL", pil)
    sys.modules["kernel"] = kernel
    sys.path.insert(0, ref_dir)
    import engram as ref_engram
    import model as ref
    ref_engram.build_compressed_token_map = lambda tok: (list(range(VOCAB)), VOCAB)

    def engram_f32(self, indices):
        v = F.embedding(indices, self.weight.float())
        sc = F.embedding(indices, self.scale.float())
        return (v.unflatten(-1, (-1, self.block_size)) * sc.unsqueeze(-1)).flatten(-2)

    ref.ParallelEngramEmbedding.forward = engram_f32
    head_fwd = ref.ParallelHead.forward
    ref.ParallelHead.forward = lambda self, x, full_logits=False: head_fwd(self, x, True)
    return ref, ref_engram


def ref_args(ref, ref_engram):
    c = CFG
    rs = c["rope_scaling"]
    args = ref.ModelArgs(
        max_batch_size=1, max_seq_len=c["max_position_embeddings"], temperature=0, dtype="bf16",
        expert_dtype=None, vocab_size=VOCAB, dim=c["hidden_size"], moe_inter_dim=c["moe_intermediate_size"],
        n_layers=c["num_hidden_layers"], n_mtp_layers=c["num_nextn_predict_layers"], n_heads=c["num_attention_heads"],
        n_routed_experts=c["n_routed_experts"], n_activated_experts=c["num_experts_per_tok"],
        route_scale=c["routed_scaling_factor"], swiglu_limit=c["swiglu_limit"], q_lora_rank=c["q_lora_rank"],
        head_dim=c["head_dim"], rope_head_dim=c["qk_rope_head_dim"], norm_eps=c["rms_norm_eps"],
        o_groups=c["o_groups"], o_lora_rank=c["o_lora_rank"], window_size=c["sliding_window"],
        compress_ratios=tuple(c["compress_ratios"]), kv_source_layers=tuple(c["kv_source_layer_ids"]),
        index_source_layers=tuple(c["index_source_layer_ids"]), compress_rope_theta=c["compress_rope_theta"],
        original_seq_len=rs["original_max_position_embeddings"], rope_theta=c["rope_theta"],
        rope_factor=rs["factor"], beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"],
        index_n_heads=c["index_n_heads"], index_head_dim=c["index_head_dim"], index_topk=c["index_topk"],
        candidate_source_layer=c["candidate_source_layer_id"], candidate_topk_blocks=c["candidate_topk_blocks"],
        candidate_block_size=c["candidate_block_size"], hc_mult=c["hc_mult"],
        hc_sinkhorn_iters=c["hc_sinkhorn_iters"], hc_eps=c["hc_eps"],
        engram_layer_ids=tuple(c["engram_layer_ids"]), engram_max_ngram_size=c["engram_max_ngram_size"],
        engram_vocab_size=c["engram_vocab_size"], engram_n_heads=c["engram_n_heads"],
        engram_head_dim=c["engram_head_dim"], engram_pad_id=c["engram_pad_token_id"],
        engram_compressed_vocab_size=VOCAB, vision_n_layers=0, dspark_block_size=c["dspark_block_size"],
        dspark_noise_token_id=c["dspark_noise_token_id"], dspark_target_layer_ids=tuple(c["dspark_target_layer_ids"]),
        dspark_markov_rank=c["dspark_markov_rank"], dspark_n_routed_experts=c["dspark_n_routed_experts"],
        dspark_n_activated_experts=c["dspark_num_experts_per_tok"],
    )
    layout = ref_engram.EngramLayout.from_args(types.SimpleNamespace(**{**c, "engram_num_embeddings": (0, 0)}))
    rows = tuple(sum(p for per in layer for p in per) for layer in layout.primes)
    args.engram_num_embeddings = rows
    return args, rows


# ---- the tiny checkpoint, in the release's layout ----------------------------------------------

class Pack:
    """`stored`: the tensors as the release names and types them; `deq`: the f32 values the reference
    sees for each (torch's own casts)."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.stored, self.deq = {}, {}

    def dense(self, name, a, dtype=torch.bfloat16):
        t = torch.from_numpy(np.asarray(a, np.float32)).to(dtype)
        self.stored[name] = t
        self.deq[name] = t.float()

    def fp8(self, base, out, inp):
        """e4m3 codes with one ue8m0 scale per 32x32 block, `<base>.weight` / `<base>.scale`."""
        w = self.rng.standard_normal((out, inp)) * inp ** -0.5
        bo, bi = -(-out // 32), -(-inp // 32)
        pad = np.zeros((bo * 32, bi * 32))
        pad[:out, :inp] = w
        blocks = pad.reshape(bo, 32, bi, 32)
        amax = np.abs(blocks).max(axis=(1, 3))
        e = np.clip(np.ceil(np.log2(np.maximum(amax, 1e-30) / E4M3_MAX)), -127, 127)
        scale = 2.0 ** e
        q = (blocks / scale[:, None, :, None]).reshape(bo * 32, bi * 32)[:out, :inp]
        codes = torch.from_numpy(q.astype(np.float32)).to(torch.float8_e4m3fn)
        sc = torch.from_numpy((e + 127).astype(np.uint8)).view(torch.float8_e8m0fnu)
        self.stored[base + ".weight"] = codes
        self.stored[base + ".scale"] = sc
        full = np.repeat(np.repeat(scale, 32, 0), 32, 1)[:out, :inp]
        self.deq[base + ".weight"] = codes.float() * torch.from_numpy(full.astype(np.float32))

    def fp8_rows(self, base, rows, inp):
        """Engram rows: e4m3 with one ue8m0 per 32 columns of each row."""
        w = self.rng.standard_normal((rows, inp)) * 0.5
        blocks = w.reshape(rows, inp // 32, 32)
        e = np.clip(np.ceil(np.log2(np.maximum(np.abs(blocks).max(-1), 1e-30) / E4M3_MAX)), -127, 127)
        q = blocks / (2.0 ** e)[..., None]
        codes = torch.from_numpy(q.reshape(rows, inp).astype(np.float32)).to(torch.float8_e4m3fn)
        self.stored[base + ".weight"] = codes
        self.stored[base + ".scale"] = torch.from_numpy((e + 127).astype(np.uint8)).view(torch.float8_e8m0fnu)
        self.deq[base + ".weight"] = codes.float()
        self.deq[base + ".scale"] = torch.from_numpy((2.0 ** e).astype(np.float32))

    def fp4(self, base, out, inp):
        """Experts: two e2m1 per int8 (the even element in the low nibble), one e8m0 per 32 of K."""
        codes = self.rng.integers(0, 16, size=(out, inp))
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
        e = np.round(np.log2(inp ** -0.5 / 2)) + self.rng.integers(-1, 2, size=(out, inp // 32))
        scale = 2.0 ** e
        self.stored[base + ".weight"] = torch.from_numpy(packed).view(torch.int8)
        self.stored[base + ".scale"] = torch.from_numpy((e + 127).astype(np.uint8)).view(torch.float8_e8m0fnu)
        vals = E2M1[codes].reshape(out, inp // 32, 32) * scale[..., None]
        self.deq[base + ".weight"] = torch.from_numpy(vals.reshape(out, inp).astype(np.float32))


def gauss(rng, *shape, s):
    return rng.standard_normal(shape) * s


def block_tensors(p, pfx, li, E):
    """One block's tensors under `pfx` (a backbone layer `li`, or a DSpark stage with `li` = None): attention,
    hyper-connection mixes, the router and `E` routed experts plus the shared one."""
    c = CFG
    rng = p.rng
    d, hd, ql = c["hidden_size"], c["head_dim"], c["q_lora_rank"]
    H, og, ol = c["num_attention_heads"], c["o_groups"], c["o_lora_rank"]
    inter = c["moe_intermediate_size"]
    hc, mix = c["hc_mult"], (2 + c["hc_mult"]) * c["hc_mult"]
    norm = lambda n: 1 + 0.1 * rng.standard_normal(n)
    for nm in ("attn", "ffn"):
        p.dense(f"{pfx}.hc_{nm}_fn", gauss(rng, mix, hc * d, s=0.05), torch.float32)
        p.dense(f"{pfx}.hc_{nm}_base", gauss(rng, mix, s=0.2), torch.float32)
        p.dense(f"{pfx}.hc_{nm}_scale", 0.5 + gauss(rng, 3, s=0.2), torch.float32)
    p.dense(f"{pfx}.attn_norm.weight", norm(d))
    p.dense(f"{pfx}.ffn_norm.weight", norm(d))
    a = f"{pfx}.attn"
    p.fp8(f"{a}.wq_a", ql, d)
    p.dense(f"{a}.q_norm.weight", norm(ql))
    p.fp8(f"{a}.wq_b", H * hd, ql)
    p.fp8(f"{a}.wkv", hd, d)
    p.dense(f"{a}.kv_norm.weight", norm(hd))
    p.fp8(f"{a}.wo_a", og * ol, H * hd // og)
    p.fp8(f"{a}.wo_b", d, og * ol)
    p.dense(f"{a}.attn_sink", gauss(rng, H, s=0.5), torch.float32)
    if li is not None and li in c["kv_source_layer_ids"]:
        p.dense(f"{a}.compressor.wkv.weight", gauss(rng, hd, d, s=d ** -0.5))
        if c["compress_ratios"][li] > 1:
            p.dense(f"{a}.compressor.wgate.weight", gauss(rng, hd, d, s=d ** -0.5))
        p.dense(f"{a}.compressor.norm.weight", norm(hd))
    if li is not None and li in c["index_source_layer_ids"]:
        ih, ihd = c["index_n_heads"], c["index_head_dim"]
        p.fp8(f"{a}.indexer.wq_b", ih * ihd, ql)
        p.dense(f"{a}.indexer.weights_proj.weight", gauss(rng, ih, d, s=d ** -0.5))
        if li in c["kv_source_layer_ids"]:
            p.dense(f"{a}.indexer.wk.weight", gauss(rng, ihd, hd, s=hd ** -0.5))
            p.dense(f"{a}.indexer.k_norm.weight", norm(ihd))
    f = f"{pfx}.ffn"
    p.dense(f"{f}.gate.weight", gauss(rng, E, d, s=d ** -0.5))
    p.dense(f"{f}.gate.bias", gauss(rng, E, s=0.3), torch.float32)
    p.dense(f"{f}.gate.bias_vl", gauss(rng, E, s=0.3), torch.float32)
    p.fp8(f"{f}.shared_experts.w1", inter, d)
    p.fp8(f"{f}.shared_experts.w3", inter, d)
    p.fp8(f"{f}.shared_experts.w2", d, inter)
    for e in range(E):
        p.fp4(f"{f}.experts.{e}.w1", inter, d)
        p.fp4(f"{f}.experts.{e}.w3", inter, d)
        p.fp4(f"{f}.experts.{e}.w2", d, inter)


def build_pack(rows):
    c = CFG
    p = Pack(7)
    rng = p.rng
    d, hd, ql = c["hidden_size"], c["head_dim"], c["q_lora_rank"]
    H, og, ol = c["num_attention_heads"], c["o_groups"], c["o_lora_rank"]
    inter, E = c["moe_intermediate_size"], c["n_routed_experts"]
    hc, mix = c["hc_mult"], (2 + c["hc_mult"]) * c["hc_mult"]
    norm = lambda n: 1 + 0.1 * rng.standard_normal(n)
    p.dense("embed.weight", gauss(rng, VOCAB, d, s=0.5))
    p.dense("head.weight", gauss(rng, VOCAB, d, s=0.3))
    p.dense("norm.weight", norm(d))
    for li in range(c["num_hidden_layers"]):
        block_tensors(p, f"layers.{li}", li, E)
    ehd = c["engram_head_dim"]
    cols = (c["engram_max_ngram_size"] - 1) * c["engram_n_heads"]
    for k, li in enumerate(c["engram_layer_ids"]):
        e = f"layers.{li}.engram"
        p.fp8(f"{e}.wkv", d * (hc + 1), cols * ehd)
        p.dense(f"{e}.q_weight", 1 + gauss(rng, hc, d, s=0.3))
        p.dense(f"{e}.k_weight", 1 + gauss(rng, hc, d, s=0.3))
        p.fp8_rows(f"{e}.embed", rows[k], ehd)
    # the DSpark stages come last: the draws above keep their values
    n_stage, rank = c["num_nextn_predict_layers"], c["dspark_markov_rank"]
    for si in range(n_stage):
        pfx = f"mtp.{si}"
        block_tensors(p, pfx, None, c["dspark_n_routed_experts"])
        if si == 0:
            p.fp8(f"{pfx}.main_proj", d, d * len(c["dspark_target_layer_ids"]))
            p.dense(f"{pfx}.main_norm.weight", norm(d))
        if si == n_stage - 1:
            p.dense(f"{pfx}.norm.weight", norm(d))
            p.dense(f"{pfx}.markov_head.embed.weight", gauss(rng, VOCAB, rank, s=0.5))
            p.dense(f"{pfx}.markov_head.head.weight", gauss(rng, VOCAB, rank, s=0.5))
            p.dense(f"{pfx}.confidence_head.proj.weight", gauss(rng, 1, d + rank, s=0.2))
    return p


def write_pack(out, p):
    os.makedirs(out, exist_ok=True)
    cfg = {
        "architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "dtype": "bfloat16",
        "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "image_token_id": 129264,
        "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
                                "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
        "text_config": {"model_type": "deepseek_v41_text", **CFG, "dtype": "bfloat16",
                        "engram_num_embeddings": list(ROWS)},
    }
    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    for name, t in p.stored.items():
        shards["model-00002-of-00002.safetensors" if ".engram." in name or name == "head.weight" else
               "model-00001-of-00002.safetensors"][name] = t
    weight_map = {}
    for fname, tensors in shards.items():
        save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(out, fname), metadata={"format": "pt"})
        weight_map.update({k: fname for k in tensors})
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=1)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(out, "engram_token_map.json"), "w") as f:
        json.dump(list(range(VOCAB)), f)


def load_into(rm, p):
    for name, param in rm.named_parameters(remove_duplicate=True):
        if name not in p.deq:
            raise SystemExit(f"no value for reference parameter {name}")
        with torch.no_grad():
            param.copy_(p.deq[name].reshape(param.shape))


# ---- the run ----------------------------------------------------------------------------------

def run(ref, ref_engram, out, qat):
    kernel = sys.modules["kernel"]
    args, rows = ref_args(ref, ref_engram)
    p = build_pack(rows)
    if qat:
        write_pack(out, p)
    kernel.DISABLE_FAKE_QUANT = not qat
    torch.manual_seed(0)
    with torch.no_grad():
        rm = ref.Transformer(args, tokenizer=None).float()
    load_into(rm, p)
    rng = np.random.default_rng(11)
    ids = rng.integers(3, VOCAB - 1, size=S + D)
    ids[0] = 2
    ids[9] = 2
    t = torch.from_numpy(ids[None]).long()
    res = {"ids": torch.from_numpy(ids.astype(np.int32))}
    hooks = []
    # layer 0's attention in pieces: its normed input, then the sparse-attention operands and heads
    calls = {"attn": 0, "split": 0}
    orig_attn, orig_split = ref.sparse_attn, ref.hc_split_sinkhorn

    def attn_spy(q, kv, sink, idxs, scale):
        o = orig_attn(q, kv, sink, idxs, scale)
        if calls["attn"] == 0:
            res.update(sa_q_0=q[0].float().clone(), sa_kv_0=kv[0].float().clone(), sa_o_0=o[0].float().clone())
        calls["attn"] += 1
        return o

    def split_spy(mixes, scale, base, *a):
        out = orig_split(mixes, scale, base, *a)
        if calls["split"] == 0:
            res.update(hc_pre_0=out[0][0].float().clone(), hc_post_0=out[1][0].float().clone(), hc_comb_0=out[2][0].float().clone())
        calls["split"] += 1
        return out

    ref.sparse_attn, ref.hc_split_sinkhorn = attn_spy, split_spy
    hooks.append(rm.layers[0].attn.register_forward_pre_hook(lambda mod, inp: res.__setitem__("attn_in_0", inp[0][0].float().clone())))
    for li, blk in enumerate(rm.layers):
        hooks.append(blk.register_forward_hook(lambda mod, inp, o, li=li: res.__setitem__(f"stream_{li}", o[0][0].float().clone())))
        hooks.append(blk.attn.register_forward_hook(lambda mod, inp, o, li=li: res.__setitem__(f"attn_{li}", o[0].float().clone())))
        hooks.append(blk.ffn.register_forward_hook(lambda mod, inp, o, li=li: res.__setitem__(f"ffn_{li}", o[0].float().clone())))
        if blk.engram is not None:
            hooks.append(blk.engram.register_forward_hook(lambda mod, inp, o, li=li: res.__setitem__(f"engram_{li}", o[0].float().clone())))
        if blk.attn.indexer is not None:
            # the selected compressed rows, window offset removed, in position order
            hooks.append(blk.attn.indexer.register_forward_hook(lambda mod, inp, o, li=li: res.__setitem__(
                f"topk_{li}", torch.where(o[0] >= 0, o[0] - inp[4], -1).sort(-1).values.int())))
    with torch.no_grad():
        _, full, _ = rm(t, 0)
        res["logits_full"] = full[0].float()
        for h in hooks:
            h.remove()
        ref.sparse_attn, ref.hc_split_sinkhorn = orig_attn, orig_split
        n = S + D
        for li, blk in enumerate(rm.layers):
            r = CFG["compress_ratios"][li]
            if blk.attn.is_kv_source:
                res[f"comp_{li}"] = blk.attn.compress_kv_cache[0, : n // r].float().clone()
            if blk.attn.indexer is not None and blk.attn.indexer.owns_k:
                res[f"idxk_{li}"] = blk.attn.indexer.k_cache[0, : n // r].float().clone()
        # DSpark: the prefill seeds each stage's window with the main rows, then every decode step drafts a block
        # from the token the trunk just picked and the main row of the token it just read
        drafter = rm.mtp[0]
        _, _, mh = rm(t[:, :S], 0)
        rm.forward_spec(t[:, S - 1], mh, 0)
        res["ds_mh_prefill"] = mh[0].float().clone()
        res["ds_mainx_prefill"] = drafter.main_norm(drafter.main_proj(mh))[0].float().clone()
        dec, ds_ids, ds_logits, ds_conf, ds_t1, ds_mh = [], [], [], [], [], []
        for i in range(S, S + D):
            ids_out, lg, mh = rm(t[:, i:i + 1], i)
            dec.append(lg[0, -1].float())
            t1 = ids_out.reshape(-1)[-1:].clone()
            out_ids, d_logits, conf = rm.forward_spec(t1, mh, i)
            ds_t1.append(t1[0].int())
            ds_mh.append(mh[0, 0].float().clone())
            ds_ids.append(out_ids[0].int())
            ds_logits.append(d_logits[0].float().clone())
            ds_conf.append(conf[0].float().clone())
    res["logits_decode"] = torch.stack(dec)
    res.update(ds_t1=torch.stack(ds_t1), ds_mh=torch.stack(ds_mh), ds_ids=torch.stack(ds_ids),
               ds_logits=torch.stack(ds_logits), ds_conf=torch.stack(ds_conf))
    drift = (res["logits_decode"] - res["logits_full"][S:]).abs().max().item()
    name = "fixture.safetensors" if qat else "fixture_noqat.safetensors"
    save_file({k: v.contiguous() for k, v in res.items()}, os.path.join(out, name))
    print(f"{out}/{name}: engram rows {rows}, prefill-vs-decode max |dlogit| {drift:.2e}")


def write_qat(out, kernel):
    """The three QAT round trips over inputs that include exact rounding midpoints and
    power-of-two amax boundaries."""
    rng = np.random.default_rng(3)
    mid = np.array([0.033203125, -0.033203125, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -5.0, 448.0, 6.0,
                    1e-9, 0.0, -0.0, 2.0] * 8, np.float32).reshape(4, 32) * (2.0 ** rng.integers(-8, 9, size=(4, 1))).astype(np.float32)
    x = np.concatenate([(rng.standard_normal((60, 32)) * np.exp(rng.standard_normal((60, 1)))).astype(np.float32), mid]).reshape(32, 64)
    kernel.DISABLE_FAKE_QUANT = False
    t = torch.from_numpy(x.copy())
    res = {
        "x": t.clone(),
        "fp8_ue8m0_32": kernel.act_quant(t.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True),
        "fp4_ue8m0_32": kernel.fp4_act_quant(t.clone(), 32, True),
        "fp4_e4m3_16": kernel.fp4_act_quant(t.clone(), 16, True, torch.float8_e4m3fn),
    }
    os.makedirs(out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in res.items()}, os.path.join(out, "qat.safetensors"))


ROWS = ()


def main():
    global ROWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="DeepSeek-V4.1-Flash inference/ directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--geometry", choices=("default", "stq"), default="default")
    a = ap.parse_args()
    if a.geometry == "stq":
        global VOCAB
        VOCAB = STQ_VOCAB
        CFG.update(STQ_GEOMETRY, vocab_size=VOCAB, engram_compressed_vocab_size=VOCAB, dspark_noise_token_id=VOCAB - 1)
    torch.set_default_dtype(torch.float32)
    kernel = Kernel()
    ref, ref_engram = load_reference(os.path.abspath(a.reference), kernel)
    out = os.path.abspath(os.path.expanduser(a.out))
    ROWS = ref_args(ref, ref_engram)[1]
    write_qat(out, kernel)
    for qat in (True, False):
        run(ref, ref_engram, out, qat)


if __name__ == "__main__":
    main()
