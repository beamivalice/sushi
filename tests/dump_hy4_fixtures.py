#!/usr/bin/env python3
# sushi-test: tier=tool
"""Oracle for the hy_v4_flash port: transformers' own modeling_hy_v4_flash (huggingface/transformers PR 50903, head
c2ec8505), run in f32 on the CPU over a TINY random model written in the release's own layout and names.

The checkpoint is the release's form at toy geometry that keeps every mechanism: sparse and sliding MLA with sinks and
the output gate, the compressed indexer (selection engages past 16 keys), the dense layer 0 and the latent MoE with its
sigmoid router, iHC, one Engram module with a non-identity token map, and the release's vocabulary and special ids.
Weights are BF16 on disk (transformers' f32 keep list in f32), and the reference runs over the values those bytes
decode to, so the fixture measures the engine, not a rounding.

llama.cpp's choices, which Sushi follows where the references differ (docs/hy4-flash-arch.md#where-the-references-
differ), are patched in; each patch is checked to change the output it should:
  - Engram rows are read as stored (no FP8 round trip);
  - Engram norms take the f32 machine epsilon;
  - Engram ids at or past the token map's size hash as themselves.
With --vision the pack already written in --out is read back and one image prompt is dumped beside it as
oracle_vision.safetensors (the pack's bytes untouched), with llama.cpp's three image choices patched in, each checked
to move the output: the tower's GELUs are the tanh form, image rows take consecutive positions on one plane (not the
interleaved 3-D M-RoPE), and an image position reads as the pad id in its neighbours' Engram n-grams.
With --processor one 70 x 45 image goes through transformers' PIL image processor at the release's
processor_config.json and is written as oracle_processor.safetensors: its RGB bytes, patches and grid.

  <venv: torch, transformers at PR 50903 head c2ec8505>/bin/python -I tests/dump_hy4_fixtures.py \\
      --release <Hy4-Flash checkout> --out <dir> [--vision | --processor]
  HY4_BF16_TINY=<dir> zig build test -Doptimize=ReleaseFast -Dtest-filter="hy4"

Writes under <dir>: the pack (config.json, model.safetensors, the release's tokenizer and template files) and
oracle.safetensors (the prompt, logits, every layer's four streams in and out, and the captures), with its prefill
and greedy decode steps.

`--geometry stq` writes the source of the tiny STQ packs instead (`HY4_BF16_TINY_STQ`): the same model with every routed
expert and every trunk linear the release puts on the trellis 128 grid on it there too (8 heads of both kinds, q and
KV LoRA 128, experts 128 x 128), plus random DFly2 tensors in the release's names and shape pattern (transformers
holds no drafter). It writes no oracle: the engine's BF16 run of this checkpoint is its packs' teacher.
"""
import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import transformers.models.hy_v4_flash.modeling_hy_v4_flash as hy
from transformers import AutoConfig, AutoModelForImageTextToText

PROMPT_LEN = 40  # past 16 keys the indexer selects; past 8 the sliding window wraps
DECODE_STEPS = 8
SEED = 0
MAP_RANGE = 100_000  # the token map's compressed ids: C = max + 1 sets the hash multipliers' bound


def tiny_config(release: dict, geometry: str = "oracle") -> dict:
    c = copy.deepcopy(release)
    t = c["text_config"]
    kinds = ["deepseek_sparse_attention", "deepseek_sparse_attention", "sliding_attention", "sliding_attention",
             "sliding_attention", "deepseek_sparse_attention"]
    n = len(kinds)
    t.update(
        num_hidden_layers=n, layer_types=kinds, swa_pattern=[int(k == "sliding_attention") for k in kinds],
        indexer_types=["full"] * n, mlp_layer_types=["dense"] + ["sparse"] * (n - 1),
        hidden_size=128, intermediate_size=256, num_attention_heads=2, num_key_value_heads=2,
        swa_num_attention_heads=4, q_lora_rank=64, kv_lora_rank=64,
        qk_nope_head_dim=32, qk_rope_head_dim=16, qk_head_dim=48, head_dim=16, v_head_dim=32,
        swa_qk_nope_head_dim=32, swa_qk_rope_head_dim=16, swa_qk_head_dim=48, swa_v_head_dim=32,
        sliding_window=8, window_size=[7, 0],
        index_n_heads=4, index_head_dim=32, compressed_indexer_num_attention_heads=4, compressed_indexer_head_dim=32,
        compressed_indexer_group_topk=4, compressed_indexer_candidate_topk=16, index_topk=16,
        n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, routed_expert_hidden_size=32,
        engram_layer_ids=[2], engram_n_head_per_ngram=2, engram_n_embed_per_ngram=32, engram_vocab_size=[100, 100],
        # The primes after engram_vocab_size - 1, continuing across heads and orders, as the release's list is.
        engram_layer_head_vocab_sizes={"2": [[101, 103], [107, 109]]},
    )
    for r in t["rope_parameters"].values():
        r["mrope_section"] = [4, 2, 2]
    c["dfly_target_kv_layers"] = [5]
    c["dfly_target_index_layer"] = [5]
    c["dfly_num_hidden_layers"] = 1
    c["vision_config"].update(num_hidden_layers=2, hidden_size=32, intermediate_size=64, num_attention_heads=2,
                              num_key_value_heads=2, out_hidden_size=128, text_hidden_size=128)
    if geometry == "stq":
        t.update(num_attention_heads=8, num_key_value_heads=8, swa_num_attention_heads=8, q_lora_rank=128, kv_lora_rank=128,
                 moe_intermediate_size=128, routed_expert_hidden_size=128)
    return c


def dfly_tensors(cfg: dict, g: torch.Generator) -> dict:
    """The drafter's tensors in the release's names, at this geometry: each layer's MLA attention in the sparse layers'
    shape and a dense SwiGLU of 3x the width; the fusion, hidden correction, confidence head and final norm."""
    t = cfg["text_config"]
    h, ql, kl = t["hidden_size"], t["q_lora_rank"], t["kv_lora_rank"]
    heads, nope, rope, v = t["num_attention_heads"], t["qk_nope_head_dim"], t["qk_rope_head_dim"], t["v_head_dim"]
    shapes = {}
    for i in range(cfg["dfly_num_hidden_layers"]):
        p = f"model.dfly_layers.{i}."
        shapes.update({
            p + "input_layernorm.weight": [h], p + "post_attention_layernorm.weight": [h],
            p + "self_attn.q_a_proj.weight": [ql, h], p + "self_attn.q_a_layernorm.weight": [ql],
            p + "self_attn.q_b_proj.weight": [heads * (nope + rope), ql],
            p + "self_attn.kv_a_proj_with_mqa.weight": [kl + rope, h], p + "self_attn.kv_a_layernorm.weight": [kl],
            p + "self_attn.kv_b_proj.weight": [heads * (nope + v), kl], p + "self_attn.o_proj.weight": [h, heads * v],
            p + "self_attn.linear_gate.weight": [heads * v, h], p + "self_attn.post_proj_norm.weight": [h],
            p + "self_attn.learnable_sink_param": [heads],
            p + "mlp.gate_proj.weight": [3 * h, h], p + "mlp.up_proj.weight": [3 * h, h], p + "mlp.down_proj.weight": [h, 3 * h],
        })
    p = "model.dfly_layers."
    shapes.update({
        p + "final_norm.weight": [h], p + "confidence_head.proj.weight": [1, h], p + "confidence_head.proj.bias": [1],
        p + "hidden_correction.embed_norm.weight": [h], p + "hidden_correction.hidden_norm.weight": [h],
        p + "hidden_correction.gate_proj.weight": [h, 2 * h], p + "hidden_correction.up_proj.weight": [h, 2 * h],
        p + "hidden_correction.down_proj.weight": [h, h], p + "mtp_fusion.eh_proj.weight": [h, 2 * h],
        p + "mtp_fusion.enorm.weight": [h], p + "mtp_fusion.hnorm.weight": [h], p + "mtp_fusion.gate": [h],
    })
    out = {}
    for name, shape in shapes.items():
        if len(shape) == 2 and shape[0] > 1:
            x = torch.randn(shape, generator=g) / shape[-1] ** 0.5
        elif name.endswith(("norm.weight", "sink_param")):
            x = 1.0 + torch.randn(shape, generator=g) * 0.1
        else:
            x = torch.randn(shape, generator=g) * 0.1
        out[name] = x.to(torch.bfloat16)
    return out


def write_stq_source(rel: Path, out: Path, tiny: dict):
    """The tiny STQ packs' source: the checkpoint and the drafter's tensors, no oracle."""
    json.dump(tiny, open(out / "config.json", "w"), indent=1)
    cfg = AutoConfig.from_pretrained(out)
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    model = AutoModelForImageTextToText.from_config(cfg).float().eval()
    randomize(model, g)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not keep_f32(name):
                p.data = p.data.to(torch.bfloat16)
    model.save_pretrained(out, safe_serialization=True)
    tensors = load_file(out / "model.safetensors")
    tensors.update(dfly_tensors(tiny, g))
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    as_release(out / "model.safetensors")
    json.dump(tiny, open(out / "config.json", "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        shutil.copy(rel / f, out / f)
    print(f"wrote {out}: {len(tensors)} tensors, the STQ geometry")


def keep_f32(name: str) -> bool:
    # The model's own list: the generation class adds lm_head, which the release stores BF16 (`as_release` holds every
    # tensor to the release's dtype, so a rounding missed here fails the dump).
    keep = hy.HYV4FlashPreTrainedModel._keep_in_fp32_modules_strict
    return any(k in name.split(".") or name.endswith(k) or f".{k}." in f".{name}." for k in keep)


def randomize(model, g: torch.Generator):
    """Weights at scales whose activations neither vanish nor saturate: a projection at 1/sqrt(fan_in), a norm near 1,
    the iHC mixes, sinks, router bias and Engram scales spread enough that each one moves the output."""
    for name, p in model.named_parameters():
        with torch.no_grad():
            if name.startswith("model.vision_tower"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
            elif p.ndim >= 2 and "experts." in name:
                p.copy_(torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5)
            elif p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5)
            elif "sinks" in name:
                p.copy_(torch.randn(p.shape, generator=g))
            elif name.endswith((".scale", "hc_scale")):
                p.copy_(0.5 + torch.rand(p.shape, generator=g))
            elif name.endswith((".base", "hc_base")) or "intra_group_ape" in name or "unified_per_hc_scales" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif name.endswith(".bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
            else:  # norm weights, unified_pre_gamma
                p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.1)
    for mod in model.modules():
        if isinstance(mod, hy.HYV4FlashTopkRouter) or hasattr(mod, "e_score_correction_bias"):
            mod.e_score_correction_bias.copy_(torch.randn(mod.e_score_correction_bias.shape, generator=g) * 0.1)
        if isinstance(mod, hy.HYV4FlashEngramLayer):
            table = torch.randint(0, MAP_RANGE, mod.lookup_table.shape, generator=g, dtype=torch.int32)
            mod.lookup_table.copy_(table)
            mod.hash_multipliers.copy_(mod.draw_hash_multipliers(int(table.max()) + 1))


def round_to_disk(model):
    """Each weight as the BF16 (or kept f32) bytes the checkpoint holds, then back to f32 for the reference run."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not keep_f32(name):
                p.copy_(p.to(torch.bfloat16).to(torch.float32))


def release_dtypes() -> dict:
    """The release's dtype for each tensor-name pattern (src/hy4flash/fixtures/hy4_tensors.txt, read from its shard
    headers), digits folded so a layer or head index matches any."""
    out = {}
    lst = Path(__file__).resolve().parents[1] / "src/hy4flash/fixtures/hy4_tensors.txt"
    for line in lst.read_text().splitlines():
        name, dtype, _ = line.split()
        out[pattern(name)] = dtype
    return out


def pattern(name: str) -> str:
    return re.sub(r"\.\d+(?=\.|$)", ".N", name)


def as_release(path: Path):
    """Rewrites the pack's tensors in the dtypes the release stores them in (its head BF16, key_norm F32, ...); every
    value is already one its dtype holds, so nothing rounds. A name the release lacks fails the dump."""
    want = release_dtypes()
    tensors = load_file(path)
    dt = {"BF16": torch.bfloat16, "F32": torch.float32, "I32": torch.int32}
    out = {}
    for name, t in tensors.items():
        d = dt[want[pattern(name)]] if pattern(name) in want else sys.exit(f"the release ships no tensor like {name}")
        conv = t.to(d)
        if not torch.equal(conv.to(torch.float64), t.to(torch.float64)):
            sys.exit(f"{name} does not hold in the release's {want[pattern(name)]}")
        out[name] = conv.contiguous()
    save_file(out, path, metadata={"format": "pt"})
    # The release's index, as the expert stream opens its banks by it.
    total = sum(t.numel() * t.element_size() for t in out.values())
    index = {"metadata": {"total_size": total}, "weight_map": {k: path.name for k in sorted(out)}}
    json.dump(index, open(path.parent / "model.safetensors.index.json", "w"), indent=1)


# ---- llama.cpp's choices -------------------------------------------------------------------------------------------

def no_fp8(x):
    return x


def unsigned_mod(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """`x mod m` with x's 64 bits read unsigned, as a C++ uint64_t hash computes it. A compressed id below the map's
    size keeps every product under 2^63, where this equals torch's own %; an id hashed as itself (past the map) can
    pass 2^63, where torch's floor-mod of the negative value would differ."""
    neg = x < 0
    high = torch.remainder(torch.tensor(2**62, dtype=torch.int64), m) * 2 % m  # 2^63 mod m
    low = torch.remainder(torch.where(neg, x + torch.iinfo(torch.int64).max + 1, x), m)  # x - 2^63 when negative
    return torch.where(neg, (low + high) % m, low)


def hash_ngrams_identity_past_map(self, ngram_tokens):
    table_size = self.lookup_table.shape[0]
    compressed = self.lookup_table[ngram_tokens.clamp_max(table_size - 1)].long()
    compressed = torch.where(ngram_tokens < table_size, compressed, ngram_tokens.long())
    mixed = compressed * self.hash_multipliers
    hash_ids = []
    for ngram_size in range(2, self.max_ngram_size + 1):
        mix = mixed[..., 0]
        for offset in range(1, ngram_size):
            mix = torch.bitwise_xor(mix, mixed[..., offset])
        order_start = (ngram_size - 2) * self.n_head_per_ngram
        moduli = self.head_vocab_sizes[order_start: order_start + self.n_head_per_ngram]
        hash_ids.append(unsigned_mod(mix.unsqueeze(-1), moduli))
    return torch.cat(hash_ids, dim=-1)


ORIGINAL = {"fp8": hy._fp8_round_trip, "hash": hy.HYV4FlashEngramLayer.hash_ngrams, "rope": hy.apply_rotary_pos_emb}


def rope_keys_bf16(q, k, cos, sin, unsqueeze_dim=1):
    """The attention's rotated rope keys held at BF16, as the cache keeps them. The indexer's partial rotary passes one
    tensor as both (`q is k`) and keeps full precision."""
    q2, k2 = ORIGINAL["rope"](q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    return (q2, k2) if q is k else (q2, k2.to(torch.bfloat16).to(k2.dtype))


def latent_bf16(module, args, out):
    return out.to(torch.bfloat16).to(out.dtype)


def patch(model, on: dict):
    hy._fp8_round_trip = no_fp8 if on.get("fp8") else ORIGINAL["fp8"]
    hy.HYV4FlashEngramLayer.hash_ngrams = hash_ngrams_identity_past_map if on.get("hash") else ORIGINAL["hash"]
    # transformers leaves the epsilon unset, which torch resolves from the input dtype: f32's on this CPU run, BF16's
    # (7.8e-3) on the release's BF16 run. Off here is that BF16 resolution, so the check sees what the patch fixes.
    eps = torch.finfo(torch.float32).eps if on.get("eps") else torch.finfo(torch.bfloat16).eps
    for mod in model.modules():
        if isinstance(mod, hy.HYV4FlashEngramLayer):
            for n in mod.lookup_per_head_norms:
                n.eps = eps
    # The latent and the rope keys as the engine's cache holds them at its BF16 width (the 8-bit width is held to this
    # oracle by its own bar): a served width, so the oracle and the engine round at the same points.
    hy.apply_rotary_pos_emb = rope_keys_bf16 if on.get("kv16") else ORIGINAL["rope"]
    for mod in model.modules():
        if hasattr(mod, "kv_a_layernorm"):
            for h in getattr(mod.kv_a_layernorm, "_kv16_hooks", []):
                h.remove()
            mod.kv_a_layernorm._kv16_hooks = [mod.kv_a_layernorm.register_forward_hook(latent_bf16)] if on.get("kv16") else []


LLAMA = {"fp8": True, "hash": True, "eps": True, "kv16": True}


# ---- captures ------------------------------------------------------------------------------------------------------

class Capture:
    def __init__(self, model):
        self.out = {}
        self.handles = []
        lm = model.model.language_model
        for i, layer in enumerate(lm.layers):
            self.hook(layer, f"l{i}", self.layer_io)
            self.hook(layer.attn_hc, f"l{i}.attn_hc", self.hc)
            self.hook(layer.ffn_hc, f"l{i}.ffn_hc", self.hc)
            self.hook(layer.self_attn, f"l{i}.attn", self.first)
            if getattr(layer.self_attn, "compressed_indexer", None) is not None:
                self.hook(layer.self_attn.compressed_indexer, f"l{i}.recall", self.plain)
            if hasattr(layer.mlp, "gate"):
                self.hook(layer.mlp.gate, f"l{i}.router", self.router)
            self.hook(layer.mlp, f"l{i}.ffn", self.plain)
            if layer.engram_layer is not None:
                self.hook(layer.engram_layer, f"l{i}.engram", self.plain)
                self.wrap(layer.engram_layer, "hash_ngrams", f"l{i}.engram.ids")
            if getattr(layer.self_attn, "compressed_indexer", None) is not None:
                self.wrap(layer.self_attn.compressed_indexer, "pool_groups", f"l{i}.pooled")
        self.hook(lm.hc_head, "hc_head", self.plain)

    def wrap(self, mod, method, key):
        """An instance method's result, each call's batch row; the last call's result also as `<key>.last` (a pooled
        key set is the whole cache's at each forward, so only its newest is kept)."""
        fn = getattr(mod, method)

        def call(*a, **k):
            r = fn(*a, **k)
            if key.endswith(".pooled"):
                self.out[key] = [r[0].detach().float().clone()]
            else:
                self.put(key, r[0])
            return r
        setattr(mod, method, call)
        self.handles.append(type("U", (), {"remove": lambda _s, m=mod, n=method: delattr(m, n)})())

    def hook(self, mod, key, fn):
        self.handles.append(mod.register_forward_hook(lambda m, a, o, k=key, f=fn: f(k, m, a, o)))

    def put(self, key, t):
        self.out.setdefault(key, []).append(t.detach().float().clone())

    def layer_io(self, k, m, a, o):
        self.put(k + ".in", a[0][0])
        self.put(k + ".out", (o[0] if isinstance(o, tuple) else o)[0])

    def hc(self, k, m, a, o):
        self.put(k + ".post", o[0][0])
        self.put(k + ".x", o[1][0])

    def first(self, k, m, a, o):
        self.put(k + ".out", o[0][0])

    def plain(self, k, m, a, o):
        self.put(k, (o[0] if isinstance(o, tuple) else o)[0])

    def router(self, k, m, a, o):
        _, w, idx = o
        self.put(k + ".weights", w)
        self.put(k + ".picks", idx)

    def tensors(self) -> dict:
        res = {}
        for k, v in self.out.items():
            try:
                res[k] = torch.cat(v, dim=0)  # positions concatenated across prefill and decode steps
            except RuntimeError:
                for j, t in enumerate(v):
                    res[f"{k}.{j}"] = t
        return res


def boundary_ties(model, prompt: torch.Tensor, steps: int, runner=None) -> list:
    """Rows where the indexer's last recalled group and the first one left out score the same. ReLU leaves exact-zero
    scores, so with few heads a tie can decide a pick, and either pick is right: the fixture must not hold one."""
    ties = []
    for layer in model.model.language_model.layers:
        ix = getattr(layer.self_attn, "compressed_indexer", None)
        if ix is None:
            continue
        orig = ix.forward

        def scored(hidden_states, q_resid, group_keys, group_position_ids, position_ids, query_group_index, ix=ix,
                   orig=orig, li=layer.self_attn.layer_idx):
            b, t, _ = hidden_states.shape
            g = group_keys.shape[1]
            q = ix.query_projection(q_resid).view(b, t, ix.num_heads, ix.head_dim)
            qc, qs = ix.rotary_emb(q, position_ids, layer_type="indexed_attention")
            q = hy._apply_partial_rotary_pos_emb(q, qc, qs, ix.rope_dim, unsqueeze_dim=2)
            gc, gs = ix.rotary_emb(group_keys, group_position_ids, layer_type="indexed_attention")
            k = hy._apply_partial_rotary_pos_emb(group_keys.unsqueeze(2), gc, gs, ix.rope_dim, unsqueeze_dim=2).squeeze(2)
            hw = ix.head_weight_projection(hidden_states).float() * (ix.num_heads ** -0.5 * ix.head_dim ** -0.5)
            sc = torch.nn.functional.relu(torch.matmul(q.float(), k.float().unsqueeze(1).transpose(-1, -2)))
            logits = torch.matmul(hw.unsqueeze(-2), sc).squeeze(-2)[0]
            slots = ix.group_topk - 1
            for row in range(t):
                own = int(query_group_index[0, row])
                valid = torch.sort(logits[row, :own], descending=True).values
                if valid.numel() > slots and valid[slots - 1] == valid[slots]:
                    ties.append((li, int(position_ids[0, 0, row]), float(valid[slots])))
            return orig(hidden_states, q_resid, group_keys, group_position_ids, position_ids, query_group_index)
        ix.forward = scored
    try:
        runner() if runner else run(model, prompt, steps)
    finally:
        for layer in model.model.language_model.layers:
            ix = getattr(layer.self_attn, "compressed_indexer", None)
            if ix is not None and "forward" in ix.__dict__:
                del ix.forward
    return ties


def run(model, prompt: torch.Tensor, steps: int):
    """Prefill the prompt in one forward, then decode greedily one token a forward with the cache."""
    cap = Capture(model)
    with torch.no_grad():
        out = model(input_ids=prompt[None], use_cache=True)
        logits = [out.logits[0]]
        cache = out.past_key_values if hasattr(out, "past_key_values") else out.cache
        ids = prompt.tolist()
        for _ in range(steps):
            nxt = int(logits[-1][-1].argmax())
            ids.append(nxt)
            # As generate's prepare_inputs_for_generation: Engram's n-grams are built over the whole sequence and
            # the new position's taken, or a lone decode token would hash with a padded history.
            ngram, live = model.model.build_engram_inputs(torch.tensor([ids]), None)
            o = model(input_ids=torch.tensor([[nxt]]), cache=cache, use_cache=True,
                      ngram_tokens=ngram[:, -1:], engram_mask=live[:, -1:])
            cache = o.past_key_values if hasattr(o, "past_key_values") else o.cache
            logits.append(o.logits[0])
    for h in cap.handles:
        h.remove()
    res = cap.tensors()
    res["logits"] = torch.cat(logits, dim=0).float()
    res["ids"] = torch.tensor(ids, dtype=torch.int32)
    return res


def prompt_ids(g: torch.Generator, cfg) -> torch.Tensor:
    """Ordinary ids, plus the ids that take each Engram arm: the pad id (masked out), a special id past the token
    map (hashed as itself under llama.cpp's choice), and a repeat (its n-grams hash alike)."""
    tc = cfg.text_config
    ids = torch.randint(0, tc.engram_lookup_table_size, (PROMPT_LEN,), generator=g)
    ids[0] = tc.bos_token_id
    ids[5] = tc.engram_pad_id
    ids[9] = 120_825  # past the 120,818-entry token map, inside the vocabulary
    ids[20:23] = ids[10:13]
    return ids


# ---- the image case -------------------------------------------------------------------------------------------------

IMAGE_GRID = (4, 6)  # patches: merged 2 x 3, so 2 rows of 3 + a newline
IMAGE_DECODE_STEPS = 4


def image_patches(g: torch.Generator, vc) -> torch.Tensor:
    """A random image's patches as the processor flattens them: rescaled by 1/255, normalized with mean and std 0.5,
    each patch's (C, P, P) values a row, the patches in row-major order."""
    h, w = IMAGE_GRID
    p = vc.patch_size
    img = torch.randint(0, 256, (3, h * p, w * p), generator=g).float()
    x = (img / 255.0 - 0.5) / 0.5
    return x.reshape(3, h, p, w, p).permute(1, 3, 0, 2, 4).reshape(h * w, 3 * p * p)


def redraw_tower(path: Path, g: torch.Generator):
    """The pack's tower and merger drawn at scales where their GELUs leave the near-linear range, so erf and tanh tell
    apart at the tower's own bar: norms near 1, biases 0.1, the position table and newline 0.5, a projection or
    convolution at 1/sqrt(fan_in). Every other tensor of the pack is kept as stored (the text oracle reads none of
    the tower), and the dump fails if one changes."""
    before = load_file(path)
    out = dict(before)
    for name in sorted(before):
        if not name.startswith("vit."):
            continue
        t = before[name]
        if name.endswith("norm.weight"):
            v = 1.0 + torch.randn(t.shape, generator=g) * 0.1
        elif name.endswith(".bias"):
            v = torch.randn(t.shape, generator=g) * 0.1
        elif name.endswith(("position_embedding.weight", "image_newline")):
            v = torch.randn(t.shape, generator=g) * 0.5
        else:
            fan_in = 1
            for d in t.shape[1:]:
                fan_in *= d
            v = torch.randn(t.shape, generator=g) / fan_in ** 0.5
        out[name] = v.to(t.dtype).contiguous()
    save_file(out, path, metadata={"format": "pt"})
    after = load_file(path)
    for name in before:
        if not name.startswith("vit.") and not torch.equal(before[name], after[name]):
            sys.exit(f"redrawing the tower changed {name}")


def tanh_gelu(model, on: bool):
    act = torch.nn.GELU(approximate="tanh") if on else hy.ACT2FN["gelu"]
    vt = model.model.vision_tower
    for layer in vt.layers:
        layer.mlp.activation_fn = act
    vt.patch_merger.proj_act = act


def image_prompt(cfg, rows: int) -> torch.Tensor:
    head = [cfg.text_config.bos_token_id, 1000, 1001]
    tail = [2000, 2001, 2002, 2003]
    return torch.tensor(head + [cfg.image_start_token_id] + [cfg.image_token_id] * rows + [cfg.image_end_token_id] + tail)


def image_run(model, cfg, ids: torch.Tensor, pixels: torch.Tensor, grid: torch.Tensor, steps: int, llama: dict):
    """Prefill the image prompt in one forward, then decode greedily. Positions are one plane of consecutive ids when
    `llama["positions"]`, else transformers' own 3-D M-RoPE; image ids read as the pad id in the n-gram history when
    `llama["engram"]`."""
    tc = cfg.text_config
    out = {}
    hooks = [model.model.vision_tower.embeddings.register_forward_hook(lambda m, a, o: out.__setitem__("vis.embed", o[0].float().clone()))]
    for i, layer in enumerate(model.model.vision_tower.layers):
        hooks.append(layer.register_forward_hook(lambda m, a, o, i=i: out.__setitem__(f"vis.block{i}", (o[0] if isinstance(o, tuple) else o)[0].float().clone())))
    hooks.append(model.model.vision_tower.patch_merger.register_forward_hook(lambda m, a, o: out.__setitem__("vis.merged", o[0].float().clone())))

    def engram_inputs(seq):
        t = torch.tensor([seq])
        hashed = torch.where(t == cfg.image_token_id, tc.engram_pad_id, t) if llama["engram"] else t
        ngram, _ = model.model.build_engram_inputs(hashed, None)
        _, live = model.model.build_engram_inputs(t, None)
        return ngram, live

    seq = ids.tolist()
    with torch.no_grad():
        ngram, live = engram_inputs(seq)
        kw = {}
        if llama["positions"]:
            kw["position_ids"] = torch.arange(len(seq)).view(1, 1, -1).expand(3, 1, -1)
        else:
            kw["mm_token_type_ids"] = (ids == cfg.image_token_id).int()[None]
        o = model(input_ids=ids[None], pixel_values=pixels, image_grid_thw=grid, use_cache=True,
                  ngram_tokens=ngram, engram_mask=live, **kw)
        logits = [o.logits[0]]
        cache = o.past_key_values if hasattr(o, "past_key_values") else o.cache
        for _ in range(steps):
            nxt = int(logits[-1][-1].argmax())
            seq.append(nxt)
            ngram, live = engram_inputs(seq)
            kw = {"position_ids": torch.tensor([[[len(seq) - 1]]]).expand(3, 1, 1)} if llama["positions"] else {}
            o = model(input_ids=torch.tensor([[nxt]]), cache=cache, use_cache=True,
                      ngram_tokens=ngram[:, -1:], engram_mask=live[:, -1:], **kw)
            cache = o.past_key_values if hasattr(o, "past_key_values") else o.cache
            logits.append(o.logits[0])
    for h in hooks:
        h.remove()
    out["vis.logits"] = torch.cat(logits, dim=0).float()
    out["vis.ids"] = torch.tensor(seq, dtype=torch.int32)
    return out


def dump_vision(out: Path):
    redraw_tower(out / "model.safetensors", torch.Generator().manual_seed(SEED + 2000))
    cfg = AutoConfig.from_pretrained(out)
    model = AutoModelForImageTextToText.from_pretrained(out, dtype=torch.float32).eval()
    for mod in model.modules():
        if isinstance(mod, hy.HYV4FlashEngramLayer):  # non-persistent: drawn from the loaded map
            mod.hash_multipliers.copy_(mod.draw_hash_multipliers(int(mod.lookup_table.max()) + 1))
    patch(model, LLAMA)
    grid = torch.tensor([[1, *IMAGE_GRID]])
    rows = (IMAGE_GRID[0] // 2) * (IMAGE_GRID[1] // 2 + 1)
    ids = image_prompt(cfg, rows)
    llama = {"gelu": True, "positions": True, "engram": True}
    tanh_gelu(model, True)
    # The first image whose prompt holds no indexer tie at a pick's edge, as the text prompt's seed.
    for draw in range(50):
        g = torch.Generator().manual_seed(SEED + 1000 + draw)
        pixels = image_patches(g, cfg.vision_config)
        ties = boundary_ties(model, ids, 0, lambda: image_run(model, cfg, ids, pixels, grid, IMAGE_DECODE_STEPS, llama))
        if not ties:
            print(f"image draw {draw}: no indexer tie decides a pick")
            break
        print(f"image draw {draw}: indexer ties at {ties[:3]}, next draw")
    else:
        sys.exit("no image in 50 draws is free of indexer ties")
    ref = image_run(model, cfg, ids, pixels, grid, 0, llama)
    if ref["vis.merged"].shape[0] != rows:
        sys.exit(f"the merger gave {ref['vis.merged'].shape[0]} rows, the prompt holds {rows}")
    # Each llama.cpp image choice changes what it should: the GELU the tower's rows by five times their bar (the
    # engine's tower stages are held to 2e-5 of the peak in f32; erf and tanh differ by at most ~5e-4 of a unit
    # input, so no draw moves them much further), the positions and the Engram history the logits.
    for name in llama:
        alt_on = {k: v for k, v in llama.items() if k != name} | {name: False}
        tanh_gelu(model, alt_on["gelu"])
        alt = image_run(model, cfg, ids, pixels, grid, 0, alt_on)
        key = "vis.merged" if name == "gelu" else "vis.logits"
        delta = ((alt[key] - ref[key]).abs().max() / ref[key].abs().max()).item()
        if delta <= (1e-4 if name == "gelu" else 0.0):
            sys.exit(f"the llama.cpp image choice '{name}' moves {key} by only {delta:.3e} of its peak")
        print(f"image choice {name}: {key} moves by {delta:.3e} of its peak")
    tanh_gelu(model, True)
    res = image_run(model, cfg, ids, pixels, grid, IMAGE_DECODE_STEPS, llama)
    res["vis.pixels"] = pixels.contiguous()
    res["vis.grid"] = grid.to(torch.int32)
    save_file({k: v.contiguous() for k, v in res.items()}, out / "oracle_vision.safetensors",
              metadata={"image_rows": str(rows), "decode_steps": str(IMAGE_DECODE_STEPS),
                        "generator": "tests/dump_hy4_fixtures.py --vision: transformers c2ec8505, llama.cpp patches"})
    print(f"wrote {out / 'oracle_vision.safetensors'}: {len(res)} tensors, ids {res['vis.ids'].tolist()[-IMAGE_DECODE_STEPS:]}")


def dump_processor(rel: Path, out: Path):
    """One 70 x 45 RGB image through transformers' PIL image processor at the release's processor_config.json: the
    bicubic resize up to the pixel floor and the (C, P, P) raster patches the tower reads."""
    import numpy as np
    from PIL import Image
    from transformers.models.hy_v4_flash.image_processing_pil_hy_v4_flash import HYV4FlashImageProcessorPil
    ip = json.load(open(rel / "processor_config.json"))["image_processor"]
    proc = HYV4FlashImageProcessorPil(
        size={"shortest_edge": ip["min_pixels"], "longest_edge": ip["max_pixels"]},
        image_mean=ip["image_mean"], image_std=ip["image_std"], patch_size=ip["patch_size"], merge_size=ip["merge_size"])
    h, w = 45, 70
    y, x = np.mgrid[0:h, 0:w]
    g = np.random.default_rng(SEED)
    rgb = np.stack([(x * 255 // (w - 1)), (y * 255 // (h - 1)), ((x + y) * 7) % 256], -1) + g.integers(-12, 13, (h, w, 3))
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    got = proc(images=[Image.fromarray(rgb)], return_tensors="pt")
    res = {"proc.rgb": torch.from_numpy(rgb.copy()), "proc.pixels": got["pixel_values"].float(),
           "proc.grid": got["image_grid_thw"].float()}
    save_file({k: v.contiguous() for k, v in res.items()}, out / "oracle_processor.safetensors",
              metadata={"format": "pt", "generator": "tests/dump_hy4_fixtures.py --processor: transformers c2ec8505 "
                        "HYV4FlashImageProcessorPil at the release's processor_config.json"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, help="the Hy4-Flash checkout: config.json, tokenizer and template")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vision", action="store_true", help="dump the image case beside the pack already in --out")
    ap.add_argument("--processor", action="store_true", help="dump the image processor's case into --out")
    ap.add_argument("--geometry", choices=("oracle", "stq"), default="oracle")
    args = ap.parse_args()
    rel, out = Path(args.release), Path(args.out)
    if args.processor:
        dump_processor(rel, out)
        return
    if args.vision:
        dump_vision(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    tiny = tiny_config(json.load(open(rel / "config.json")), args.geometry)
    if args.geometry == "stq":
        write_stq_source(rel, out, tiny)
        return
    json.dump(tiny, open(out / "config.json", "w"), indent=1)
    cfg = AutoConfig.from_pretrained(out)
    # The first seed whose run holds no indexer tie at a pick's edge.
    for seed in range(SEED, SEED + 50):
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        model = AutoModelForImageTextToText.from_config(cfg).float().eval()
        randomize(model, g)
        round_to_disk(model)
        ids = prompt_ids(g, cfg)
        patch(model, LLAMA)
        ties = boundary_ties(model, ids, DECODE_STEPS)
        if not ties:
            print(f"seed {seed}: no indexer tie decides a pick")
            break
        print(f"seed {seed}: indexer ties at {ties[:3]}, next seed")
    else:
        sys.exit("no seed in 50 is free of indexer ties")

    # Each llama.cpp choice changes what it should: the full-patch output differs from the one without that patch.
    patch(model, LLAMA)
    ref = run(model, ids, 0)
    for name in LLAMA:
        patch(model, {k: v for k, v in LLAMA.items() if k != name})
        alt = run(model, ids, 0)
        delta = (alt["l1.engram"] - ref["l1.engram"]).abs().max().item()
        if delta == 0.0:
            sys.exit(f"the llama.cpp patch '{name}' changes nothing on this prompt")
        print(f"patch {name}: Engram delta moves by {delta:.3e}")
    patch(model, LLAMA)

    # The pack: BF16 bytes (the f32 keep list as f32), the release's names, its tokenizer and template.
    disk = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in disk.named_parameters():
            if not keep_f32(name):
                p.data = p.data.to(torch.bfloat16)
    disk.save_pretrained(out, safe_serialization=True)
    as_release(out / "model.safetensors")
    # save_pretrained writes transformers' own spellings; the pack keeps the release's (and the read-back below loads
    # them, so transformers is held to reading the release's form too).
    json.dump(tiny, open(out / "config.json", "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        shutil.copy(rel / f, out / f)

    # Round trip: transformers reads the pack back to the same model.
    back = AutoModelForImageTextToText.from_pretrained(out, dtype=torch.float32).eval()
    for mod_a, mod_b in zip(model.modules(), back.modules()):
        if isinstance(mod_a, hy.HYV4FlashEngramLayer):  # non-persistent: drawn from the loaded map
            mod_b.hash_multipliers.copy_(mod_b.draw_hash_multipliers(int(mod_b.lookup_table.max()) + 1))
            assert torch.equal(mod_a.hash_multipliers, mod_b.hash_multipliers)
    patch(back, LLAMA)
    res = run(back, ids, DECODE_STEPS)
    again = run(model, ids, DECODE_STEPS)
    # Every capture is equal bit for bit; the head's matmul over the reloaded weight's layout may differ in the last
    # bits, so the logits are held to 1e-5.
    for k in res:
        if k != "logits" and not torch.equal(res[k], again[k]):
            sys.exit(f"the pack read back gives a different {k}")
    if not torch.allclose(res["logits"], again["logits"], rtol=0, atol=1e-5):
        sys.exit("the pack read back does not give the model's logits")
    save_file({k: v.contiguous() for k, v in res.items()}, out / "oracle.safetensors",
              metadata={"prompt_len": str(PROMPT_LEN), "decode_steps": str(DECODE_STEPS),
                        "generator": "tests/dump_hy4_fixtures.py: transformers c2ec8505, llama.cpp patches"})
    print(f"wrote {out}: {len(res)} oracle tensors, ids {res['ids'].tolist()[-DECODE_STEPS:]}")


main()
