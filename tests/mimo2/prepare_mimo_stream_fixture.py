#!/usr/bin/env python3
"""Create a tiny mixed-precision pack from dump_mimo_v2_fixtures.py output."""
import argparse
import json
from pathlib import Path

import mlx.core as mx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if destination.exists() or source == destination or source in destination.parents:
        p.error("destination must be a fresh directory outside the source")
    cfg = json.loads((source / "config.json").read_text())
    weights = mx.load(str(source / "model.safetensors"))
    output = {}

    def affine(name, value):
        w, s, b = mx.quantize(value.astype(mx.bfloat16), group_size=64, bits=8)
        output[name + ".weight"] = w
        output[name + ".scales"] = s
        output[name + ".biases"] = b

    for name, value in weights.items():
        if ".mlp.experts." in name:
            continue
        if name.endswith(".self_attn.qkv_proj.weight"):
            layer = int(name.split(".")[2])
            swa = cfg["hybrid_layer_pattern"][layer] == 1
            h = cfg["swa_num_attention_heads" if swa else "num_attention_heads"]
            k = cfg["swa_num_key_value_heads" if swa else "num_key_value_heads"]
            d = cfg["swa_head_dim" if swa else "head_dim"]
            v = cfg["swa_v_head_dim" if swa else "v_head_dim"]
            base = name.removesuffix("qkv_proj.weight")
            qend, kend = h * d, h * d + k * d
            for projection, part in zip(("q", "k", "v"), (value[:qend], value[qend:kend], value[kend:kend + k * v])):
                affine(base + projection + "_proj", part)
        elif ".layers.0.mlp." in name and name.endswith("_proj.weight"):
            affine(name.removesuffix(".weight"), value)
        else:
            output[name] = value if name.endswith("e_score_correction_bias") else value.astype(mx.bfloat16)

    for layer, moe in enumerate(cfg["moe_layer_freq"]):
        if not moe:
            continue
        for projection in ("gate", "up", "down"):
            base = f"model.layers.{layer}.mlp"
            bank = mx.stack([weights[f"{base}.experts.{e}.{projection}_proj.weight"] for e in range(cfg["n_routed_experts"])])
            w, s = mx.quantize(bank.astype(mx.bfloat16), group_size=32, bits=4, mode="mxfp4")
            output[f"{base}.switch_mlp.{projection}_proj.weight"] = w
            output[f"{base}.switch_mlp.{projection}_proj.scales"] = s
    mx.eval(list(output.values()))
    cfg["attention_projection_layout"] = "split"
    cfg["quantization"] = {"bits": 4, "group_size": 32, "mode": "mxfp4"}
    destination.mkdir(parents=True)
    mx.save_safetensors(str(destination / "model.safetensors"), output)
    (destination / "config.json").write_text(json.dumps(cfg))
    (destination / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "model.safetensors" for key in output},
    }))


if __name__ == "__main__":
    main()
