#!/usr/bin/env python3
# sushi-test: tier=tool
"""A tiny deepseek_v41 STQ pack for the serving tests (`DSV41_STQ_TINY`), from the `--geometry stq` source that
`tests/dump_dsv41_fixtures.py` writes.

  prep:      python -I tests/build_dsv41_stq_tiny.py prep --src <source> --work <dir>
  quantize:  SUSHI_ENCODER_STQ_TODO=<dir> zig-out/tests/encoder-test   (filter "todo directory")
  assemble:  python -I tests/build_dsv41_stq_tiny.py assemble --src <source> --work <dir> --out <pack>

Routed experts: PAIR1 in three rate groups, the router rows reordered into group order. DSpark stage experts: PAIR1
n32 (2bpw), one group. Trunk linears on the 128 grid: SPEC1 n64 (K4), the head n96 (K6). The embedding: MLX affine
4-bit, group 64. Everything else is the source's own tensor. Identity Hessian: a serving fixture, not a quality pack.
"""

import argparse
import json
import os
import re
import shutil

import numpy as np
import torch
from safetensors.torch import load_file, save_file

# new index i holds source expert PERM[i]; groups take consecutive new indices
PERM = [5, 2, 7, 0, 3, 6, 1, 4]
GROUPS = [(3, (24, 24, 32)), (3, (32, 32, 32)), (2, (48, 48, 64))]
STAGE_N = 32
TRUNK_N = 64
HEAD_N = 96
PROJ = (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2"))
TRUNK = ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "ffn.shared_experts.w1", "ffn.shared_experts.w3",
         "ffn.shared_experts.w2", "attn.indexer.wq_b", "attn.indexer.wk", "attn.compressor.wkv",
         "attn.compressor.wgate", "engram.wkv", "main_proj")


def source(src):
    t = {}
    for f in sorted(os.listdir(src)):
        if f.startswith("model") and f.endswith(".safetensors"):
            t.update(load_file(os.path.join(src, f)))
    return t


def deq(t, base):
    """`<base>.weight` in f32 [out, in]: fp8 x the 32x32 e8m0 block scale, fp4 x the e8m0 per 32 columns, or dense."""
    w = t[base + ".weight"]
    s = t.get(base + ".scale")
    if s is None:
        return w.float()
    sc = s.view(torch.float8_e8m0fnu).float() if s.dtype == torch.uint8 else s.float()
    if w.dtype == torch.int8:
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
        b = w.view(torch.uint8).int()
        v = torch.stack([lut[b & 15], lut[b >> 4]], -1).reshape(w.shape[0], -1)
        return v * sc.repeat_interleave(32, 1)
    v = w.view(torch.float8_e4m3fn).float() if w.dtype == torch.uint8 else w.float()
    return v * sc.repeat_interleave(32, 0)[: v.shape[0]].repeat_interleave(32, 1)[:, : v.shape[1]]


def on_grid(w):
    return w.shape[0] % 128 == 0 and w.shape[1] % 128 == 0


def todo_list(t, cfg):
    """(todo name, [out, in] f32, n, codebook) for every tensor the pack quantizes."""
    jobs = []
    stages = cfg["num_nextn_predict_layers"]
    blocks = [(f"layers.{li}", cfg["n_routed_experts"], True) for li in range(cfg["num_hidden_layers"])]
    blocks += [(f"mtp.{si}", cfg["dspark_n_routed_experts"], False) for si in range(stages)]
    for pfx, E, routed in blocks:
        for tail in TRUNK:
            base = f"{pfx}.{tail}"
            if base + ".weight" in t and on_grid(deq(t, base)):
                jobs.append((base, deq(t, base), TRUNK_N, "spec1"))
        wo = f"{pfx}.attn.wo_a"
        w = deq(t, wo)
        og = cfg["o_groups"]
        for g in range(og):
            jobs.append((f"{wo}@{g}", w[g * (w.shape[0] // og):(g + 1) * (w.shape[0] // og)], TRUNK_N, "spec1"))
        for e in range(E):
            src_e = PERM[e] if routed else e
            n_of = group_rates(e) if routed else (STAGE_N,) * 3
            for (pn, wn), n in zip(PROJ, n_of):
                jobs.append((f"{pfx}.x{e}.{pn}", deq(t, f"{pfx}.ffn.experts.{src_e}.{wn}"), n, "pair1"))
    jobs.append(("head", t["head.weight"].float(), HEAD_N, "spec1"))
    return jobs


def group_rates(e):
    at = 0
    for count, rates in GROUPS:
        if e < at + count:
            return rates
        at += count
    raise SystemExit("expert outside the groups")


def prep(a):
    t = source(a.src)
    cfg = json.load(open(os.path.join(a.src, "config.json")))["text_config"]
    os.makedirs(a.work, exist_ok=True)
    with open(os.path.join(a.work, "manifest.txt"), "w") as m:
        for name, w, n, cb in todo_list(t, cfg):
            out, inp = w.shape
            w.t().contiguous().numpy().astype(np.float32).tofile(os.path.join(a.work, name + ".f32"))
            m.write(f"{name} {inp} {out} {n} {cb}\n")


def bank(work, name, inp, out, n):
    tr = np.fromfile(os.path.join(work, name + ".trellis.bin"), np.uint16).reshape(inp // 16, out // 16, n)
    su = np.fromfile(os.path.join(work, name + ".suh.bin"), np.float16)
    sv = np.fromfile(os.path.join(work, name + ".svh.bin"), np.float16)
    return torch.from_numpy(tr.astype(np.int16)).view(torch.uint16), torch.from_numpy(su), torch.from_numpy(sv)


def put_bank(out_t, base, parts):
    tr, su, sv = (torch.stack(x) for x in zip(*parts))
    out_t[base + ".trellis"] = tr
    out_t[base + ".suh"] = su
    out_t[base + ".svh"] = sv


def affine4(w):
    """MLX affine 4-bit, group 64 (`mx.quantize`): w ~ scale * q + bias, eight codes to a u32, low first."""
    g = w.float().reshape(w.shape[0], -1, 64)
    lo, hi = g.min(-1, keepdim=True).values, g.max(-1, keepdim=True).values
    scale = ((hi - lo) / 15).clamp_min(1e-8)
    q = ((g - lo) / scale).round().clamp(0, 15).to(torch.int64).reshape(w.shape[0], -1, 8)
    packed = sum(q[..., i] << (4 * i) for i in range(8)).to(torch.int64)
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed).to(torch.int32).view(torch.uint32)
    return packed, scale.squeeze(-1).to(torch.bfloat16), lo.squeeze(-1).to(torch.bfloat16)


def assemble(a):
    t = source(a.src)
    cfg_all = json.load(open(os.path.join(a.src, "config.json")))
    cfg = cfg_all["text_config"]
    manifest = {}
    for line in open(os.path.join(a.work, "manifest.txt")):
        name, inp, out, n, cb = line.split()
        manifest[name] = (int(inp), int(out), int(n))
    out_t = {}
    used = set()
    for name, (inp, out, n) in manifest.items():
        if "@" in name or re.fullmatch(r".*\.x\d+\.\w+_proj", name) or name == "head":
            continue
        tr, su, sv = bank(a.work, name, inp, out, n)
        out_t[name + ".trellis"], out_t[name + ".suh"], out_t[name + ".svh"] = tr, su, sv
        used.update({name + ".weight", name + ".scale"})
    tr, su, sv = bank(a.work, "head", *manifest["head"])
    out_t["head.trellis"], out_t["head.suh"], out_t["head.svh"] = tr, su, sv
    used.add("head.weight")
    stages = cfg["num_nextn_predict_layers"]
    blocks = [(f"layers.{li}", cfg["n_routed_experts"], True) for li in range(cfg["num_hidden_layers"])]
    blocks += [(f"mtp.{si}", cfg["dspark_n_routed_experts"], False) for si in range(stages)]
    for pfx, E, routed in blocks:
        wo = f"{pfx}.attn.wo_a"
        put_bank(out_t, wo, [bank(a.work, f"{wo}@{g}", *manifest[f"{wo}@{g}"]) for g in range(cfg["o_groups"])])
        used.update({wo + ".weight", wo + ".scale"})
        nest = f"model.{pfx}.mlp.switch_mlp" if routed else f"{pfx}.mlp.switch_mlp"
        at = 0
        groups = GROUPS if routed else [(E, (STAGE_N,) * 3)]
        for gi, (count, _) in enumerate(groups):
            for pn, _ in PROJ:
                parts = [bank(a.work, f"{pfx}.x{e}.{pn}", *manifest[f"{pfx}.x{e}.{pn}"]) for e in range(at, at + count)]
                put_bank(out_t, f"{nest}.{pn}.g{gi}", parts)
            at += count
        for e in range(E):
            for _, wn in PROJ:
                used.update({f"{pfx}.ffn.experts.{e}.{wn}.weight", f"{pfx}.ffn.experts.{e}.{wn}.scale"})
        if routed:
            perm = torch.tensor(PERM)
            out_t[f"{pfx}.ffn.gate.weight"] = t[f"{pfx}.ffn.gate.weight"][perm].contiguous()
            out_t[f"{pfx}.ffn.gate.bias"] = t[f"{pfx}.ffn.gate.bias"][perm].contiguous()
            used.update({f"{pfx}.ffn.gate.weight", f"{pfx}.ffn.gate.bias", f"{pfx}.ffn.gate.bias_vl"})
    w, s, b = affine4(t["embed.weight"])
    out_t["embed.weight"], out_t["embed.scales"], out_t["embed.biases"] = w, s, b
    used.add("embed.weight")
    for k, v in t.items():
        if k not in used and k not in out_t:
            out_t[k] = v
    os.makedirs(a.out, exist_ok=True)
    fname = "model-00001-of-00001.safetensors"
    save_file({k: v.contiguous() for k, v in out_t.items()}, os.path.join(a.out, fname),
              metadata={"format": "stq", "k": "4", "codebook": "pair1", "window": "16"})
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": {k: fname for k in out_t}}, f, indent=1)
    widest = max(max(r) for _, r in GROUPS) / 16
    cfg_all["expert_quant"] = {"format": "stq", "k": widest, "codebook": "pair1", "window": 16,
                               "codebook_table": "pair1-w16-v1", "codebook_digest": "c089b4eb"}
    cfg_all["trunk_quant"] = {"format": "stq", "k": HEAD_N / 16, "codebook": "spec1", "window": 15,
                              "codebook_table": "spec1-w15", "codebook_digest": "37f72431"}
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg_all, f, indent=1)
    for extra in ("engram_token_map.json", "fixture.safetensors", "fixture_noqat.safetensors", "qat.safetensors"):
        shutil.copy(os.path.join(a.src, extra), os.path.join(a.out, extra))
    print(f"{a.out}: {len(out_t)} tensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prep", "assemble"))
    ap.add_argument("--src", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    prep(a) if a.step == "prep" else assemble(a)


if __name__ == "__main__":
    main()
