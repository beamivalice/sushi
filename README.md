<p align="center"><img src="docs/assets/sushi-logo.png" alt="sushi" width="256"></p>

# SUSHI

A local AI server for low-bit, very high quality Sushi quants, using custom EXL3-Affine hybrid converter. It is a detached fork of [ddalcu's mlx-serve](https://github.com/ddalcu/mlx-serve). Sushi runs stand-alone and also as a guest engine inside mlx-serve, Apple Silicon M1+ is required.

## Model support list

Usage see [recommended launch commands and coding agent usage](docs/usage.md).

* DeepSeek 4.1 Flash 1017 STQ 1.75bpw - COMING SOON!
* [Qwen3.8-Flash-Next-Sushi-2bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-2bpw) (requires 48 GB+, or 32 GB [streamed](#streaming-on-a-32-gb-mac))
* [Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw) (requires 64 GB+ and sushi 1.3.0 or later)
* [Qwen3.8-Flash-Next-Sushi-4bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw) (requires 96 GB+)
* [MiMo-V2.6-Flash-Sushi-2.3bpw](https://huggingface.co/beamster/MiMo-V2.6-Flash-Sushi-2.3bpw) (requires 128 GB, text and image input)
* [GLM-5.3-Flash-Sushi-2bpw](https://huggingface.co/beamster/GLM-5.3-Flash-Sushi-2bpw) (requires 96 GB+, text, image and video input)
* [GLM-5.3-Flash-Sushi-2.4bpw](https://huggingface.co/beamster/GLM-5.3-Flash-Sushi-2.4bpw) (requires 128 GB, text, image and video input)

## Quality

<p align="center"><img src="docs/assets/dsv41-1.75bpw-kld.png" alt="DeepSeek-V4.1-Flash KLD by category" width="100%"></p>

<p align="center"><img src="docs/assets/kld-chart.png" alt="KLD vs size" width="100%"></p>

* MiMo-V2.6-Flash-Sushi-2.3bpw scores KLD 0.0860 (top-1 agreement 91.95%)
* GLM-5.3-Flash-Sushi-2.4bpw scores KLD 0.0742 (top-1 agreement 90.33%)
* GLM-5.3-Flash-Sushi-2bpw scores KLD 0.1037 (top-1 agreement 88.18%)

## Install

```bash
brew install beamivalice/tap/sushi
```

Update with `brew upgrade sushi`.

* The server listens on `127.0.0.1:12345`.
* Support speculative decoding: MTP for Qwen and MiMo, DFlash2 for GLM.
* 8-bit KV cache are on by default.

## Memory

GPU memory in GiB to serve one prompt that fills the whole context (8-bit KV, MTP on with `--mtp-head-kv-quant`; GLM
with its DFlash2 assistant and vision tower). The prompt cache and the n-gram table stay on the SSD and are not counted.
The Qwen columns are at the default f32 GatedDeltaNet state; `--qwen-gdn 16` (bf16) takes about 0.2 GiB off each Qwen cell.

| context | Sushi-2bpw | Sushi-STQ-2.75bpw | Sushi-4bpw | MiMo-2.3bpw | GLM-2bpw | GLM-2.4bpw |
|---|---:|---:|---:|---:|---:|---:|
| weights only | 35.2 | 46.4 | 63.9 | 89.7 | 78.3 | 89.8 |
| 128k | 40.3 | 51.4 | 69.0 | 93.7 | 82.6 | 94.1 |
| 256k | 43.2 | 54.3 | 71.9 | 95.8 | 83.8 | 95.2 |
| 512k | 48.2 | 59.4 | 76.9 | 99.9 | 86.0 | 97.5 |
| 1M | 58.4 | 69.5 | 87.1 | 108.2 | 90.5 | 102.0 |

A context fits when its number is below the GPU limit you set with `sudo sysctl iogpu.wired_limit_mb`. Max context is
the largest multiple of 8192 tokens that fits, at 8-bit / 4-bit KV, with 256 MiB spare and capped at 1M (GLM has no
4-bit KV):

| Mac | GPU limit | Sushi-2bpw | Sushi-STQ-2.75bpw | Sushi-4bpw | MiMo-2.3bpw | GLM-2bpw | GLM-2.4bpw |
|---|---|---|---|---|---|---|---|
| 48 GB † | 43,000 MB (42.0 GiB) | 200k / 304k | — | — | — | — | — |
| 64 GB † | 59,000 MB (57.6 GiB) | 976k / 1M | 416k / 664k | — | — | — | — |
| 96 GB | 90,000 MB (87.9 GiB) | 1M / 1M | 1M / 1M | 1M / 1M | — | 696k / — | — |
| 128 GB | 120,000 MB (117.2 GiB) | 1M / 1M | 1M / 1M | 1M / 1M | 1M / 1M | 1M / — | 1M / — |

† At `--qwen-gdn 16` (bf16 GatedDeltaNet state), which the 32, 48 and 64 GB launch lines use. The f32 default gives
less: 192k / 288k at 48 GB, and 968k / 1M (Sushi-2bpw) and 408k / 648k (Sushi-STQ-2.75bpw) at 64 GB.

A Mac with less memory than a Sushi pack can still serve it: `--ssd-budget-gb N` keeps N GiB resident and streams
the routed experts from the SSD, with the same replies as a resident load, at a speed set by the SSD. A streamed load
serves text unless `--vision` loads the vision tower, whose weights then come out of the N GiB (DeepSeek-V4.1-Flash loads it by default; `--no-vision` skips it).

## Streaming on a 32 GB Mac

Sushi-2bpw streams on a 32 GB Mac with MTP on, at about 20 tok/s decode:

```bash
sudo sysctl iogpu.wired_limit_mb=27000
sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-2bpw --ssd-budget-gb 18 --ctx-size 66000 \
  --qwen-gdn 16 --expert-pick-tolerance 0.3
```

`--qwen-gdn 16` stores Qwen's recurrent state in bf16 instead of the f32 default, which halves the state in every cache
checkpoint (about 54 MB each) and is what the 32, 48 and 64 GB recipes run.

`--ssd-budget-gb 18` keeps 18 GiB resident (the trunk, the MTP head and an expert cache) and reads the remaining experts
from the SSD; the KV cache is billed on top of it. `--expert-pick-tolerance 0.3` is lossy: when a routed expert is not cached, it uses the best cached expert
whose router probability is at least 0.7 of the missed one's, which saves an SSD read; leave it out for replies
identical to a resident load. The plan may come within 1 GiB of the GPU limit
(`--wired-margin-gib N`, 1 to 32). The GPU limit resets at reboot.

## Benchmarks

| Sushi 🍣 1.4.0 | M5 Max 128 GB | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen 3.8 Flash-Next-STQ-2.75bpw** | Prefill | 1,122 | 1,993 | 2,049 | 1,976 | 1,906 | 1,877 |
| | Decode | 168.6 | 151.1 | 146.1 | 146.8 | 151.0 | 138.2 |
| **Qwen 3.8 Flash-Next-4bpw** | Prefill | 2,079 | 2,379 | 2,326 | 2,231 | 2,147 | 2,120 |
| | Decode | 146.2 | 150.1 | 136.2 | 149.8 | 153.5 | 122.0 |
| **GLM 5.3 Flash-2.4bpw** | Prefill | 928 | 775 | 814 | 805 | 688 | 573 |
| | Decode | 71.5 | 69.4 | 67.3 | 65.8 | 71.8 | 63.8 |
| **DeepSeek 4.1 Flash-STQ-1.75bpw** (`--ssd-budget-gb 108`) | Prefill | 707 | 1,024 | 998 | 993 | 950 | 829 |
| | Decode | 76.6 | 84.2 | 76.2 | 85.3 | 78.1 | 63.7 |

`sushi bench` code arm (tok/s, shipping defaults with the default speculative mode, prefill is the 8k window ending at
the rung). The prose arm and each model's report are in [docs/bench/v1.4.0](docs/bench/v1.4.0/summary.md).

Community reports, using llmprobe `--bench-only`:

| Class | Typical RAM | Pack | Prefill tok/s | Gen tok/s |
|---|---|---|---:|---:|
| M1 Max | 64 GB | 3bpw | ~350 | ~32 |
| M2 Max | 64 GB | 3bpw | ~400 | ~38-40 |
| M3 Ultra 60c | 256 GB | 4bpw | ~420 | ~60 |
| M4 Max | 64 GB | 3bpw | ~690 | ~60-65 |
| M5 Pro | 64 GB | 2.6bpw | ~900 | ~50-55 |
| M5 Max | 128 GB | 3bpw | ~1,900 | ~95 |
| M5 Max | 128 GB | 4bpw | ~1,750 | ~90 |
| M5 Max | 128 GB | MiMo 2.3bpw | ~1,130 | ~70 |
| M5 Max | 128 GB | GLM 2.4bpw | ~860 | ~55 |

## License

MIT, for sushi and the mlx-serve code it forks ([LICENSE](LICENSE)); ported kernels and vendored code are listed in
[NOTICE](NOTICE). The Qwen packs follow the Qwen Community License and the MiMo pack Xiaomi's MIT license, stated on
each Hugging Face page.
