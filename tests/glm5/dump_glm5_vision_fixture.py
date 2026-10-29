#!/usr/bin/env python3
"""CPU oracle from the original HF GLM-5.3 vision model and image processor.

Run with torch, torchvision, safetensors and a transformers version containing
models.glm5_next. The fixture records the HF source hash for reproducibility.
No MPS/GPU operations are used. Regenerate:
  python tests/glm5/dump_glm5_vision_fixture.py src/fixtures/glm5_vision_tiny.safetensors
"""
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel


def main():
    torch.set_num_threads(2)
    torch.manual_seed(53014)
    cfg = Glm5NextVisionConfig(depth=2, hidden_size=32, num_heads=4, intermediate_size=48,
        out_hidden_size=64, patch_size=4, spatial_merge_size=2, temporal_patch_size=2,
        projection_intermediate_size=80, swiglu_limit=0.7,
        rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
        attn_implementation="eager")
    tower = Glm5NextVisionModel(cfg).float().cpu().eval()
    with torch.no_grad():
        for name, p in tower.named_parameters():
            if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name.endswith("q_norm.weight") or name.endswith("k_norm.weight") or name.endswith("post_layernorm.weight") or name.endswith("post_projection_norm.weight"):
                p.copy_(1 + torch.randn_like(p) * 0.04)
            else:
                p.copy_(torch.randn_like(p) * (0.12 if p.ndim > 1 else 0.03))
        pv = torch.randn(320, 3 * 2 * 4 * 4, device="cpu")
        grid = torch.tensor([[1, 16, 20]], dtype=torch.int32, device="cpu")
        output = tower(pv, grid).pooler_output
        # The reference isolates attention per temporal group.
        video_pv = torch.cat([pv[:48], pv[48:96]], dim=0)
        video_grid = torch.tensor([[2, 6, 8]], dtype=torch.int32, device="cpu")
        video_output = tower(video_pv, video_grid).pooler_output
    tensors = {"model.visual." + k: v.detach().contiguous() for k, v in tower.state_dict().items()}
    tensors.update({"fixture.pixel_values": pv, "fixture.features": output,
        "fixture.video_pixel_values": video_pv, "fixture.video_features": video_output})
    processor = Glm5NextImageProcessor(patch_size=4, temporal_patch_size=2, merge_size=2,
        min_image_tokens=16, max_image_tokens=32)
    rgb = (np.arange(55 * 81 * 3).reshape(55, 81, 3) * 17 % 256).astype(np.uint8)
    processed = processor.preprocess(rgb, return_tensors="pt")
    tensors["fixture.rgb"] = torch.from_numpy(rgb.copy())
    tensors["fixture.processed_pixels"] = processed.pixel_values
    tensors["fixture.processed_grid"] = processed.image_grid_thw
    output_path = Path(sys.argv[1])
    save_file(tensors, str(output_path), metadata={"source": "HF Glm5NextVisionModel on CPU", "seed": "53014"})
    metadata = {"transformers_source_sha256": hashlib.sha256(Path(inspect.getfile(Glm5NextVisionModel)).read_bytes()).hexdigest(),
        "image_processor_source_sha256": hashlib.sha256(Path(inspect.getfile(Glm5NextImageProcessor)).read_bytes()).hexdigest(),
        "config": cfg.to_dict(), "fixture_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "preprocess_grid": processed.image_grid_thw.tolist()}
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"fixture": str(output_path), "features": list(output.shape), "preprocess_grid": metadata["preprocess_grid"]}))


if __name__ == "__main__":
    main()
