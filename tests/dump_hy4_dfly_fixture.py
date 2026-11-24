#!/usr/bin/env python3
# sushi-test: tier=tool
"""Hy4-Flash's DFly2 drafter oracle on the tiny, from AngelSpec's own modules.

Reference = Tencent/AngelSpec at 0505773af6dea129706003412100c541d3a39ce4, the released drafter's training code:
`DFly2DraftModel` (DFlash backbone with Hy4's gated MLA over the target's own latents), its `MTPFusion` at slot 0,
the block-causal mask (`_create_dflash_mask_mod`, `dflash_causal_attn: true`), the hidden correction's served chain
(`DFlashModel._served_draft_chain`) and the confidence head over the corrected hidden. Run on CPU in f32.

The target is never run here: the engine's own state for the tiny's prompt (its last hidden state before the final
norm, each drafter layer's shared latent rows and picks) is the input, so the oracle holds the drafter alone.

    export HY4_BF16_TINY_STQ=<tiny dir> MLX_ENABLE_TF32=0
    SUSHI_HY4_DFLY_STATE_DUMP=<abs state.json> zig build test -Doptimize=ReleaseFast -Dmodel=hy4flash \\
        -Dtest-filter="hy4 dfly oracle"
    uv run --no-project --with torch --with transformers --with safetensors --with numpy --with numba \\
        --with huggingface_hub python -I tests/dump_hy4_dfly_fixture.py --angelspec <AngelSpec checkout> \\
        --tiny "$HY4_BF16_TINY_STQ" --state <abs state.json> --out src/hy4flash/fixtures/hy4_dfly_oracle.json
"""

import argparse
import json
import struct
import sys
from types import SimpleNamespace

SPREAD = 96  # logits kept per slot at fixed ids across the vocab, beside the oracle's top ones
TOP = 32


def read_safetensors(path, want):
    """The tensors of `path` whose names `want` accepts, as f32 torch tensors (BF16 widened exactly)."""
    import torch

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__" or not want(name):
                continue
            lo, hi = meta["data_offsets"]
            f.seek(base + lo)
            raw = bytearray(f.read(hi - lo))
            dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[meta["dtype"]]
            out[name] = torch.frombuffer(raw, dtype=dtype).reshape(meta["shape"]).to(torch.float32)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angelspec", required=True)
    ap.add_argument("--tiny", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.angelspec)
    import torch
    from angelspec.models.draft.dfly2 import DFly2Config, DFly2DraftModel
    from angelspec.models.draft.dflash import DFlashRMSNorm
    from angelspec.models.draft.mtp import _interleaved_to_split_rope
    from angelspec.models.dflash import DFlashModel, _create_dflash_mask_mod
    from torch.nn.attention.flex_attention import create_mask

    torch.manual_seed(0)
    full = json.load(open(f"{args.tiny}/config.json"))
    text = full["text_config"]
    state = json.load(open(args.state))
    n_layers = int(full["dfly_num_hidden_layers"])
    block = int(full["speculative_config"]["num_speculative_tokens"]) + 1
    sparse_rope = text["rope_parameters"]["deepseek_sparse_attention"]
    weights = read_safetensors(f"{args.tiny}/model.safetensors", lambda k: k.startswith("model.dfly_layers.") or k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"))
    # The drafter's geometry is the target's sparse layers' (AngelSpec's dfly2_hy_v4_flash_draft_config.json).
    config = DFly2Config(
        vocab_size=text["vocab_size"],
        hidden_size=text["hidden_size"],
        intermediate_size=weights["model.dfly_layers.0.mlp.gate_proj.weight"].shape[0],
        num_hidden_layers=n_layers,
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        rms_norm_eps=text["rms_norm_eps"],
        max_position_embeddings=1 << 20,
        num_target_layers=n_layers,
        target_hidden_size=text["hidden_size"],
        # Training-side checks only (the captured hidden below each target layer, a layer above it); the tiny reads
        # its last layer, and neither reaches the forward.
        target_layer_ids=[t - 1 for t in full["dfly_target_kv_layers"]],
        target_num_hidden_layers=max(text["num_hidden_layers"], max(full["dfly_target_kv_layers"]) + 2),
        mask_token_id=full["dfly_mask_token_id"],
        context_local_window=full["dfly_context_local_window"],
        target_index_layer=full["dfly_target_index_layer"],
        target_index_topk=full["dfly_target_index_topk"],
        target_kv_layers=full["dfly_target_kv_layers"],
        mtp_fusion=full["dfly_mtp_fusion"],
        mtp_fusion_init="random",
        enable_hidden_correction=full["dfly_enable_hidden_correction"],
        enable_confidence_head=True,
        # As the released draft config (dfly2_hy_v4_flash_draft_config.json) states them.
        model_arch="dflash",
        layer_arch="dflash",
        attention_type="mla",
        gated_mla=True,
        gating_type="elementwise",
        learnable_sink=True,
        attn_post_proj_norm=True,
        q_lora_rank=text["q_lora_rank"],
        kv_lora_rank=text["kv_lora_rank"],
        qk_nope_head_dim=text["qk_nope_head_dim"],
        qk_rope_head_dim=text["qk_rope_head_dim"],
        v_head_dim=text["v_head_dim"],
        rope_parameters=sparse_rope,
        dsa_backend="torch",
        image_start_token_id=full["image_start_token_id"],
        image_end_token_id=full["image_end_token_id"],
        image_newline_token_id=full["image_newline_token_id"],
    )
    model = DFly2DraftModel(config).float().eval()
    sd = {}
    for k, v in weights.items():
        if k.startswith("model.dfly_layers."):
            name = k[len("model.dfly_layers.") :]
            sd[("layers." + name) if name[0].isdigit() else name] = v
    sd["embed_tokens.weight"] = weights["model.embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"weights: missing {missing}, unexpected {unexpected}")
    for layer in model.layers:
        rope = layer.self_attn.rotary_emb
        if not layer.self_attn.uses_mrope or rope.base != float(sparse_rope["rope_theta"]):
            raise SystemExit("the drafter's rope is not the sparse layers' M-RoPE")
    target_norm = DFlashRMSNorm(text["hidden_size"], eps=text["rms_norm_eps"])
    target_norm.weight.data.copy_(weights["model.norm.weight"])
    lm_head = weights["lm_head.weight"]

    t1, p0 = state["t1"], state["pos"]
    with torch.no_grad():
        # The block [t1, mask x draft] and the slot-0 fusion with the target's final hidden, put through its final norm.
        ids = torch.tensor([[t1] + [config.mask_token_id] * (block - 1)])
        noise = model.embed_tokens(ids)
        prev_hidden = target_norm(torch.tensor(state["hidden"]).view(1, 1, -1))
        noise = noise + model.mtp_fusion(noise[:, 0:1], prev_hidden, block)[:, 0]
        latents, masks = [], []
        for ly in state["layers"]:
            c = torch.tensor(ly["c"])
            # The engine keeps the rope rows interleaved; AngelSpec's latent holds them split.
            k_pe = _interleaved_to_split_rope(torch.tensor(ly["k_pe"]))
            latents.append(torch.cat([c, k_pe], dim=-1).unsqueeze(0))
            n = c.shape[0]
            keep = torch.ones(n, dtype=torch.bool) if ly["keep"] is None else torch.tensor(ly["keep"])
            mod = _create_dflash_mask_mod(
                anchor_positions=torch.tensor([[n]]),
                block_keep_mask=torch.tensor([[True]]),
                ctx_len=n,
                block_size=block,
                causal=True,
                ctx_candidates=keep.view(1, 1, n),
            )
            masks.append(SimpleNamespace(mask_mod=mod))
        if len({lat.shape[1] for lat in latents}) != 1:
            raise SystemExit("the oracle runs one context length for every layer")
        n = latents[0].shape[1]
        hidden = model(
            draft_input_ids=None,
            context_feature=None,
            draft_position_ids=(p0 + torch.arange(block)).view(1, -1),
            context_position_ids=torch.arange(n).view(1, -1),
            block_mask=masks,
            noise_embedding=noise,
            context_latents=latents,
        )
        # The served chain: slot 1 corrects with the anchor's embedding, slot j > 1 with slot j-1's draft.
        chain = SimpleNamespace(_acorr=False, hidden_correction_topk=0, _path_selector=lambda: None, draft_model=model, block_size=block)
        prev = torch.tensor([[[t1, t1] + [0] * (block - 2)]])
        prev, proposals = DFlashModel._served_draft_chain(chain, hidden, prev, lm_head, 1)
        corrected = model.hidden_correction(hidden, model.embed_tokens(prev.view(1, -1)))
        logits = (corrected @ lm_head.T)[0]
        conf = torch.sigmoid(model.confidence_head(corrected))[0]

    vocab = lm_head.shape[0]
    spread = [(i * 1259 + 7) % vocab for i in range(SPREAD)]
    slots = []
    for j in range(1, block):
        top = torch.topk(logits[j], TOP).indices.tolist()
        keep_ids = top + [v for v in spread if v not in top]
        slots.append({"ids": keep_ids, "values": [float(logits[j, v]) for v in keep_ids]})
    fixture = {
        "reference": "Tencent/AngelSpec 0505773af6dea129706003412100c541d3a39ce4",
        "state": state,
        "want": {
            "draft": proposals[0, 0, 1:].tolist(),
            "conf": [float(v) for v in conf[1:]],
            "logits": slots,
        },
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
