#!/usr/bin/env python3
"""CPU oracle for DeepSeek-V4.1's image input: the release's own `inference/image_processor.py` and
`inference/vision.py`, run as the release runs them (bf16 default dtype, CPU) on two fixed images
with the pack's bf16 vision tensors. Only those tensors are read; peak RAM is about 3 GB.

Needs torch, numpy, Pillow and safetensors. Run

  python tests/dump_dsv41_vision_fixtures.py --ref-dir <release>/inference --pack <pack dir> [--out <dir>]

`--out` defaults to ${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/kld-teacher/dsv41-vision. It writes

  fixture.safetensors  per case: <case>_rgb (u8 HWC), <case>_grid (i32: vit_h vit_w llm_h llm_w),
                       <case>_patches / _embed / _block0 / _tower / _features / _span (f32 copies of the
                       reference's bf16 tensors), and <case>_tower_f32 (the same tower run in fp32)
  tower.safetensors    the pack's vision.*, aligner.* and image_* tensors, untouched (bf16)
  plan.json            plan_image_grid + ImageOps.pad geometry of a spread of source sizes
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

# The two fixed images. 301x203 letterboxes onto an odd margin after the min-pixel scale-up;
# 520x390 is a 4:3 photo-like canvas that needs bicubic down- and up-sampling on one axis each.
CASES = {"odd": (203, 301), "photo": (390, 520)}
# (width, height) sources whose processor geometry the Zig unit test pins.
PLAN_SIZES = [(730, 487), (97, 61), (7000, 1000), (4000, 3000), (14, 14), (1, 1), (5000, 50), (50, 5000),
              (1920, 1080), (640, 640), (1000, 7000), (300, 2900)]


def synth(h, w, seed):
    """A smooth gradient with edges and noise, so every resampling tap and channel matters."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([x / w, y / h, (x + y) / (w + h)], -1) * 200 + 20
    img[h // 4 : h // 2, w // 3 : w // 2] = [250, 30, 60]
    img[(x - w * 0.7) ** 2 + (y - h * 0.6) ** 2 < (min(h, w) * 0.2) ** 2] = [20, 220, 90]
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def vision_args(pack, text_dim):
    vc = json.load(open(pack / "config.json"))["vision_config"]
    return SimpleNamespace(
        dim=text_dim,
        vision_n_layers=vc["num_hidden_layers"],
        vision_dim=vc["hidden_size"],
        vision_n_heads=vc["num_attention_heads"],
        vision_inter_dim=vc["intermediate_size"],
        vision_patch_size=vc["patch_size"],
        vision_rope_theta=float(vc.get("rope_theta", 10000)),
        vision_downsample_ratio=vc["downsample_ratio"],
        vision_max_n_token=vc["max_image_tokens"],
        vision_min_pixels=vc["min_pixels"],
        vision_max_wh_ratio=vc.get("max_wh_ratio"),
    )


def load_tower_tensors(pack):
    """Every vision.*, aligner.* and image_* tensor of the pack, by name."""
    index = json.load(open(pack / "model.safetensors.index.json"))["weight_map"]
    wanted = [k for k in index if k.startswith(("vision.", "aligner.", "image_"))]
    out = {}
    for shard in sorted({index[k] for k in wanted}):
        with safe_open(str(pack / shard), "pt", device="cpu") as f:
            for k in f.keys():
                if k in wanted and index[k] == shard:
                    out[k] = f.get_tensor(k)
    assert len(out) == len(wanted), (len(out), len(wanted))
    return out


def tower_f32(args, t, patches, n_h, n_w):
    """The same reference tower in fp32 on the same patches: the yardstick a bf16 run is read against."""
    import vision as ref

    torch.set_default_dtype(torch.float32)
    try:
        vit = ref.ViT(args)
        vit.load_state_dict({k[len("vision."):]: v.float() for k, v in t.items() if k.startswith("vision.")})
        with torch.inference_mode():
            return vit(patches.float(), n_h, n_w).float()
    finally:
        torch.set_default_dtype(torch.bfloat16)


def plan_table(ip, args):
    rows = []
    for w, h in PLAN_SIZES:
        n_llm_h, n_llm_w, bh, bw = ip.plan_image_grid(w, h, args)
        # ImageOps.pad's geometry without resampling the (possibly huge) image: contain's size, then the paste box
        im_ratio, dest_ratio = w / h, bw / bh
        rw, rh = bw, bh
        if im_ratio != dest_ratio:
            if im_ratio > dest_ratio:
                rh = round(h / w * bw)
            else:
                rw = round(w / h * bh)
        x = round((bw - rw) * 0.5) if rw != bw else 0
        y = round((bh - rh) * 0.5) if rw == bw and rh != bh else 0
        rows.append(dict(w=w, h=h, bw=bw, bh=bh, lh=n_llm_h, lw=n_llm_w, rw=rw, rh=rh, x=x, y=y))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True, help="the release's inference/ directory")
    ap.add_argument("--pack", required=True, help="a pack (or the release) holding the vision tensors")
    ap.add_argument("--out", default=None)
    opts = ap.parse_args()
    out = Path(opts.out) if opts.out else Path(os.environ.get("SUSHI_MODELS_DIR") or Path.home() / ".sushi/models") / "kld-teacher/dsv41-vision"
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, opts.ref_dir)
    import image_processor as ip
    import vision as ref

    pack = Path(opts.pack)
    text_dim = json.load(open(pack / "config.json"))["text_config"]["hidden_size"]
    args = vision_args(pack, text_dim)
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.bfloat16)  # the release's default

    t = load_tower_tensors(pack)
    vit, aligner = ref.ViT(args), ref.Aligner(args)
    vit.load_state_dict({k[len("vision."):]: v for k, v in t.items() if k.startswith("vision.")})
    aligner.load_state_dict({k[len("aligner."):]: v for k, v in t.items() if k.startswith("aligner.")})
    vit.eval()
    aligner.eval()
    start, end, newline = (t[f"image_{n}"] for n in ("start", "end", "newline"))

    tensors = {}
    for name, (h, w) in CASES.items():
        rgb = synth(h, w, seed=len(name))
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")  # lossless: the reference decodes exactly `rgb`
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image({"data": buf.getvalue()}, args)
        types = ip.image_token_types(n_llm_h, n_llm_w)
        with torch.inference_mode():
            x0 = vit.patch_embed(patches)
            cos, sin = ref.get_vision_cos_sin(n_vit_h, n_vit_w, vit.rope_dim, vit.rope_theta)
            block0 = vit.blocks[0](x0, cos, sin)
            hidden = vit(patches, n_vit_h, n_vit_w)
            feats = aligner(hidden, n_vit_h, n_vit_w)
            span = torch.empty(types.numel(), text_dim, dtype=torch.bfloat16)
            span[types == ip.IMAGE_START] = start
            span[types == ip.IMAGE_END] = end
            span[types == ip.IMAGE_NEW_LINE] = newline
            span[types == ip.IMAGE] = feats
        print(f"{name}: {w}x{h} -> vit {n_vit_h}x{n_vit_w}, llm {n_llm_h}x{n_llm_w}, span {types.numel()}", file=sys.stderr)
        tensors[f"{name}_rgb"] = torch.from_numpy(rgb)
        tensors[f"{name}_grid"] = torch.tensor([n_vit_h, n_vit_w, n_llm_h, n_llm_w], dtype=torch.int32)
        tensors[f"{name}_patches"] = patches.flatten(1).float()
        tensors[f"{name}_embed"] = x0.float()
        tensors[f"{name}_block0"] = block0.float()
        tensors[f"{name}_tower_f32"] = tower_f32(args, t, patches, n_vit_h, n_vit_w)
        tensors[f"{name}_tower"] = hidden.float()
        tensors[f"{name}_features"] = feats.float()
        tensors[f"{name}_span"] = span.float().unsqueeze(0)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "fixture.safetensors"))
    save_file({k: v.contiguous() for k, v in t.items()}, str(out / "tower.safetensors"))
    json.dump(plan_table(ip, args), open(out / "plan.json", "w"), indent=1)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
