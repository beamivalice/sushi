# Launch and agent usage

[Back to the README](../README.md)

## Install

Apple Silicon M1+ and macOS 26.2 or later are required.

```bash
brew install beamivalice/tap/sushi
# or the release binary (ad-hoc signed; curl does not quarantine it):
curl -L https://github.com/beamivalice/sushi/releases/latest/download/sushi-bin-macos-arm64.tar.gz | tar xz
```

A browser download is quarantined: clear it with `xattr -dr com.apple.quarantine sushi-macos-arm64`. GPU memory and the
largest context per Mac: [README, Memory](../README.md#memory).

## `sushi pull` and `sushi list`

`sushi pull <model>` downloads a pack into `~/.sushi/models/beamster/<repo>`, where `sushi serve` and `sushi run` find it
by name. `<model>` is a short name (`qwen3.8-flash-next[:tag]`, `mimo-v2.6-flash`, `glm-5.3-flash[:tag]`,
`deepseek-v4.1-flash`) or `beamster/<repo>`; packs from any other org are refused. A rerun downloads only what is missing
or short, so it finishes an interrupted pull; until then the folder holds `.pull-incomplete`. A private repo pulls after
`hf auth login` or with `HF_TOKEN`.

`sushi list` shows each pack's size, whether it is downloaded and whether this Mac meets its tier, and marks the one it
recommends. A Mac's tier is its installed RAM in whole GB. Below a pack's tier it reads `may stream (--ssd-budget-gb)`:
it may still run streamed ([README, Streaming](../README.md#streaming)), with no promise.

| pack | runs in memory from | recommended from |
|---|---:|---:|
| Qwen3.8-Flash-Next-Sushi-2bpw | 48 GB | 32 GB (streamed) |
| Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw | 64 GB | 64 GB |
| Qwen3.8-Flash-Next-Sushi-4bpw | 96 GB | 96 GB |
| GLM-5.3-Flash-Sushi-2bpw | 96 GB | 96 GB |
| GLM-5.3-Flash-Sushi-2.4bpw | 128 GB | 128 GB |
| MiMo-V2.6-Flash-Sushi-2.3bpw | 128 GB | 128 GB |
| DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw | 96 GB (experts streamed) | 96 GB |

Sizes are the whole download; a Qwen pack's n-gram table and DeepSeek's 189 GiB of Engram tables stay on the SSD and are
never loaded. Offline, `sushi list` shows only the models on disk.

## Recommended launch

Every GPU limit below is set with `sudo sysctl iogpu.wired_limit_mb=<N>` and resets at reboot.

**48 GB Mac, Qwen Sushi-2bpw** (limit 43000)

```bash
sushi pull qwen3.8-flash-next:2bpw
sushi serve --model ~/.sushi/models/beamster/Qwen3.8-Flash-Next-Sushi-2bpw \
  --mtp --kv-quant 8 --mtp-head-kv-quant --qwen-gdn 16 --ctx-size 131072 \
  --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1
```

**64 GB Mac, Qwen Sushi-STQ-2.75bpw** (limit 59000; above it macOS runs out of memory first and the kernel panics)

```bash
sushi pull qwen3.8-flash-next:2.75bpw
# 8-bit KV, the default quality
sushi serve --model ~/.sushi/models/beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw \
  --mtp --kv-quant 8 --mtp-head-kv-quant --qwen-gdn 16 --ctx-size 250000 \
  --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1
# or 4-bit KV for 1.8 times the context: --kv-quant 4 --ctx-size 450000 --max-tokens 64000
```

The 32 to 64 GB Qwen recipes pass `--qwen-gdn 16`: a bf16 recurrent state halves every cache checkpoint (about 54 MB
each). 96 GB and up keep the f32 default. Context is set explicitly because auto-context reads free memory.

**96 GB+ Mac, Qwen Sushi-4bpw** (limit 90000 on 96 GB, 120000 on 128 GB)

```bash
sushi pull qwen3.8-flash-next:4bpw
sushi serve --model ~/.sushi/models/beamster/Qwen3.8-Flash-Next-Sushi-4bpw \
  --mtp --kv-quant 8 --mtp-head-kv-quant --ctx-size 500000 --prefill-chunk 2048 \
  --max-tokens 64000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1
```

**DFlash2 on Qwen (opt-in).** MTP stays the default. `--drafter <dir>` loads Inco.ai's
[DFlash2 assistant](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (Apache-2.0, 1.2 GB); rounds verify
between `--drafter-min-depth` (2) and `--drafter-max-depth` (8) rows, and greedy output equals MTP's byte for byte.
Speed per pack and context: [qwen4-perf](qwen4-perf.md#dflash2).

**128 GB Mac, MiMo-V2.6-Flash-Sushi-2.3bpw** (limit 120000)

```bash
sushi pull mimo-v2.6-flash
sushi serve --model ~/.sushi/models/beamster/MiMo-V2.6-Flash-Sushi-2.3bpw --ctx-size 1048576
```

MTP, 8-bit KV and thinking are on by default.

**96 GB Mac, GLM-5.3-Flash-Sushi-2bpw** (limit 90000) and **128 GB Mac, GLM-5.3-Flash-Sushi-2.4bpw** (limit 120000)

```bash
sushi pull glm-5.3-flash:2bpw      # or glm-5.3-flash for 2.4bpw
sushi serve --model ~/.sushi/models/beamster/GLM-5.3-Flash-Sushi-2bpw
```

Images, video, 8-bit KV and DFlash2 are on by default, and up to four requests decode together. Auto context on 2bpw is
512K at 8-bit KV, 300K with `--kv-quant 16`. The pack ships Inco.ai's
[DFlash2 assistant](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) under CC BY-NC-ND 4.0 (non-commercial, no
derivatives); the smaller copy sushi builds from it in `dflash2/` on first load is a derivative, so keep it local.
`--no-drafter` turns it off.

<a id="deepseek-launch"></a>
**96 GB and 128 GB Macs, DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** (limit 125000 on 128 GB, 93000 on 96 GB)

```bash
sushi pull deepseek-v4.1-flash
sushi serve --model ~/.sushi/models/beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw \
  --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000
# 128 GB Mac: 108 (or 96 to leave memory to other apps); 96 GB Mac: 80
```

- **`--ssd-budget-gb` is required.** It is everything the load keeps resident (trunk, DSpark, vision tower, workspace)
  plus the expert cache, which gets the rest; more budget means more expert slots and fewer SSD reads. Budget, the
  request's state (3.7 GB billed at 1M) and an image encode (1.6 GB) must fit under the GPU limit.

  | `--ssd-budget-gb` | expert slots a layer (of 384) | expert hits code / prose | decode 4k code / prose | decode 128k code / prose | prefill 8k window code / prose |
  |---|---:|---:|---:|---:|---:|
  | 80 | 182 | 96.5% / 99.2% | 44.7 / 44.5 | 36.6 / 39.4 | 760 / 547 |
  | 96 | 227 | 98.7% / 99.6% | 65.8 / 48.2 | 51.8 / 45.4 | 963 / 700 |
  | 108 | 260 | 99.6% / 99.9% | 76.6 / 49.3 | 63.7 / 43.5 | 1,024 / 821 |

  One `sushi bench` run each on an M5 Max 128 GB (DSpark on, one request); 108 is the
  [v1.4.0 bench](bench/v1.4.0/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw.md).
- **111 on a 128 GB Mac needs the user wire limit raised.** macOS caps the memory a process may wire at
  `vm.global_user_wire_limit`, 116.8 GB by default on 128 GB, separately from `iogpu.wired_limit_mb`. Past 108 the load
  crosses it and prefill slows down (7% at 111) with no error. Raising it brings 111's prefill back to 108's speed:
  ```bash
  sudo sysctl vm.global_user_wire_limit=125000000000 vm.user_wire_limit=125000000000   # resets at reboot
  ```
- **Smaller Macs** are plan numbers only, not measured: 32 GiB (48 slots) from a 48 GB Mac at limit 43000, 48 GiB
  (93 slots) from 64 GB at 59000. The floor is 17 GiB (16 with `--no-vision`). A shorter `--ctx-size` lowers the Mac a
  budget needs.
- **Disk:** 332.1 GB, of which 189 GiB are Engram tables read a row at a time; the SSD prefix cache adds up to 20 GB.
- **Context** runs to 1M tokens (890 B a token). One request decodes at a time; seen prompts restore from the SSD.
- **DSpark** (the checkpoint's drafter) is on by default; `--no-mtp` turns it off. Greedy output is stable across round
  widths and identical between stream and non-stream.
- **Thinking** is `high` by default; `reasoning_effort` `low`, `high` or `max`, or `off` / `enable_thinking: false`.
- **Images** are on by default; `--no-vision` frees the 0.50 GB tower for the expert cache; `--vision-bf16` keeps the stored bf16 tower.
- **`--drop-expert n`** (multiple of 8) skips the `n` least-used experts a layer, freeing 2.1–2.5 GB at 8 and 4.3–5.0 GB
  at 16. It is lossy: check it with `sushi kld compare` ([dsv41-arch](dsv41-arch.md#drop-expert)).
- **Refused flags:** `--kv-quant`, `--qwen-gdn`, `--mtp-head-kv-quant`, `--mtp-typical`, `--mtp-tokenv3` and
  `--mtp-greedy-tail` stop the load, since the model has none of those parts ([dsv41-parity](dsv41-parity.md)).

### Common flags

- `--prefix-cache-disk 20GB` sizes the SSD prefix cache (on by default, capped at 20 GB); `--prefix-cache-entries 1`
  keeps one conversation, raise it to 4-8 for several agents; `--prefix-cache-mem 1GB` also keeps prefixes in RAM.
- A restore point sits every 8192 tokens and at each prompt's end, so a 1M-token conversation takes about 0.9 GB of
  SSD (DeepSeek), 7-14 GB of points plus 16 GB of rows (Qwen) or 24 GB (GLM): raise `--prefix-cache-disk` for long
  contexts. A conversation past the budget drops its oldest points first.
- `--preserve-thinking off` keeps only the latest turn's thinking in the prompt.
- `--prefill-chunk 2048` caps the prompt step per forward; a request that does not fit steps down.

## Coding agents

With the server running, `sushi launch <agent>` starts claude, pi, omp, opencode, codex, grok, hermes or aider against it:

```bash
sushi launch omp
```

pi, omp, codex, grok and hermes run from their own home under `~/.sushi/<agent>/`, so your usual config is untouched;
claude, opencode and aider reach the server through environment variables. `--print` writes the config and prints the
launch script instead of running it.
