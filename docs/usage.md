# Launch and agent usage

[Back to the README](../README.md)

## Install

Apple Silicon M1+ and macOS 26.2 or later are required.

```bash
brew install beamivalice/tap/sushi
```

Or the release binary (ad-hoc signed; curl does not quarantine it):
```bash
curl -L https://github.com/beamivalice/sushi/releases/latest/download/sushi-bin-macos-arm64.tar.gz | tar xz
```
A browser download is quarantined by macOS: clear it with `xattr -dr com.apple.quarantine sushi-macos-arm64`.
GPU memory and the largest context per Mac for each pack: [README, Memory](../README.md#memory).
To download a pack and see which one suits your Mac: [`sushi pull` and `sushi list`](#sushi-pull-and-sushi-list).

## `sushi pull` and `sushi list`

`sushi pull <model>` downloads a Sushi pack into `~/.sushi/models/beamster/<repo>`; `sushi serve` and `sushi run`
then load it by name. `<model>` is a short name (`qwen3.8-flash-next[:tag]`, `mimo-v2.6-flash`, `glm-5.3-flash[:tag]`, `deepseek-v4.1-flash`)
or `beamster/<repo>`. Only packs of the Hugging Face org `beamster` are accepted; any other `org/repo` exits with an
error before anything downloads (`sushi run <model>` follows the same rule when it has to download).
`sushi pull` checks every file of the pack against its Hugging Face listing and downloads only what is missing or short,
so rerunning it finishes an interrupted download. Until it finishes, the pack folder holds a `.pull-incomplete` file and
`sushi list` does not show the pack as downloaded.

`sushi list` shows each published pack with its size, whether it is downloaded and whether this Mac meets the pack's
memory tier, and marks the pack it recommends for this Mac; the other models on disk follow. A Mac's tier is its
installed RAM in whole GB (a 64 GB Mac reports 68.7e9 bytes and counts as 64):

| pack | runs in memory from | recommended from |
|---|---:|---:|
| Qwen3.8-Flash-Next-Sushi-2bpw | 48 GB | 32 GB (streamed) |
| Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw | 64 GB | 64 GB |
| Qwen3.8-Flash-Next-Sushi-4bpw | 96 GB | 96 GB |
| GLM-5.3-Flash-Sushi-2bpw | 96 GB | 96 GB |
| GLM-5.3-Flash-Sushi-2.4bpw | 128 GB | 128 GB |
| MiMo-V2.6-Flash-Sushi-2.3bpw | 128 GB | 128 GB |
| DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw | 96 GB (experts streamed) | 96 GB |

Each family recommends its largest pack the Mac reaches in the last column. `fits` means the Mac meets the pack's tier
(DeepSeek's routed experts stream on every Mac: its tier is the Mac that admits the `--ssd-budget-gb 80` launch
[below](#deepseek-launch)).
Below it the pack is labelled `may stream (--ssd-budget-gb)`: it may still run with `--ssd-budget-gb`
([README, Streaming](../README.md#streaming)), and nothing promises it does.
`sushi pull` prints that warning before it downloads a pack below its tier, then pulls anyway. A pack whose repo
Hugging Face does not return yet is listed as `(not yet published)` and `sushi pull` of it says so. That includes a
private repo for anyone without access; a user granted access sees its size and pulls it after
`hf auth login` or with `HF_TOKEN`: both commands send `HF_TOKEN`, else the token file `hf auth login` saved
(`HF_TOKEN_PATH`, else `$HF_HOME/token`, else `~/.cache/huggingface/token`).

On a 64 GB Mac:

```
$ sushi list
This Mac: 64 GB

PACK                                SIZE  DOWNLOADED  THIS MAC
qwen3.8-flash-next:2bpw          64.8 GB  no          fits
qwen3.8-flash-next:2.75bpw       75.9 GB  yes         fits  <- recommended
qwen3.8-flash-next:4bpw         159.0 GB  no          may stream (--ssd-budget-gb)
mimo-v2.6-flash:2.3bpw          100.7 GB  no          may stream (--ssd-budget-gb)
glm-5.3-flash:2bpw               81.5 GB  no          may stream (--ssd-budget-gb)
glm-5.3-flash:2.4bpw             93.2 GB  no          may stream (--ssd-budget-gb)
deepseek-v4.1-flash:1.75bpw     332.1 GB  no          may stream (--ssd-budget-gb)
```

Sizes are the whole Hugging Face download and differ per release. A Qwen pack's size includes its n-gram table, which
stays on the SSD and is not loaded into memory; the DeepSeek pack's includes 189 GiB of Engram tables, read from the SSD
the same way. With Hugging Face unreachable (no network, or no answer within a few
seconds), `sushi list` prints one line saying so and lists only the models on disk, without sizes or recommendations.

## Recommended launch

**48 GB Mac, Sushi-2bpw**

Set the GPU memory limit first (it resets at reboot):
```bash
sudo sysctl iogpu.wired_limit_mb=43000
```

```bash
hf download beamster/Qwen3.8-Flash-Next-Sushi-2bpw --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-2bpw
# or: sushi pull qwen3.8-flash-next:2bpw   (lands in ~/.sushi/models/beamster/Qwen3.8-Flash-Next-Sushi-2bpw)

# images, 8-bit KV, 128k context; --qwen-gdn 16 keeps the GatedDeltaNet state in bf16 (see below)
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-2bpw \
  --mtp --kv-quant 8 --mtp-head-kv-quant --qwen-gdn 16 --ctx-size 131072 \
  --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1
```

Every 32 GB, 48 GB and 64 GB Qwen recipe passes `--qwen-gdn 16` (the 32 GB one is in the
[README](../README.md#streaming)): the recurrent state is f32 by default, and bf16 halves it in every
cache checkpoint (about 54 MB each on Flash-Next), the memory these recipes are sized to. The 96 GB and larger recipes
keep the f32 default.

**DFlash2 on Qwen3.8-Flash-Next (opt-in).** The recipes above decode with MTP, which stays the default. Inco.ai's
[DFlash2 assistant](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (Apache-2.0, 1.2 GB, loaded at 8 bits) is used only
when `--drafter <dir>` names it; a copy beside the pack is ignored. Download it with
`hf download incoai/Qwen3.8-Flash-Next-DFlash2 --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-DFlash2`. Its rounds verify
between `--drafter-min-depth` (default 2) and `--drafter-max-depth` (default 8) rows, as many as the drafter's confidence
pays for; greedy output equals the MTP and serial output byte for byte. Its speed against MTP per pack and context is in
[qwen4-perf](qwen4-perf.md#dflash2); `--no-drafter` or `enable_drafter:false` keeps MTP.

**64 GB Mac, Sushi-STQ-2.75bpw**

Set the GPU memory limit first (it resets at reboot). 59,000 MB is the ceiling for this box: above it macOS runs out
of memory before the model does, and the kernel panics rather than the server refusing.
```bash
sudo sysctl iogpu.wired_limit_mb=59000
```

Then pick one of the two. They differ only in KV width; the context is set explicitly because auto-context reads free
memory, so its answer is not the same on two 64 GB machines.

```bash
hf download beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw
# or: sushi pull qwen3.8-flash-next:2.75bpw   (lands in ~/.sushi/models/beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw)

# 1. images, 8-bit KV — the default quality
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw \
  --mtp --kv-quant 8 --mtp-head-kv-quant --qwen-gdn 16 --ctx-size 250000 \
  --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1

# 2. images, 4-bit KV — 1.8 times the context
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw \
  --mtp --kv-quant 4 --mtp-head-kv-quant --qwen-gdn 16 --ctx-size 450000 \
  --max-tokens 64000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1
```

**96 GB+ Mac, Sushi-4bpw**

```bash
hf download beamster/Qwen3.8-Flash-Next-Sushi-4bpw --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-4bpw
# or: sushi pull qwen3.8-flash-next:4bpw   (lands in ~/.sushi/models/beamster/Qwen3.8-Flash-Next-Sushi-4bpw)
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-4bpw \
  --mtp --kv-quant 8 --mtp-head-kv-quant --ctx-size 500000 \
  --prefill-chunk 2048 --max-tokens 64000 --prefix-cache-disk 20GB \
  --prefix-cache-entries 1 --temp 1
```

Set the GPU memory limit before serving (it resets at reboot):
```bash
sudo sysctl iogpu.wired_limit_mb=90000    # 96 GB Mac
sudo sysctl iogpu.wired_limit_mb=120000   # 128 GB Mac
```

**128 GB Mac, MiMo-V2.6-Flash-Sushi-2.3bpw**

```bash
sudo sysctl iogpu.wired_limit_mb=120000
hf download beamster/MiMo-V2.6-Flash-Sushi-2.3bpw --local-dir ~/.sushi/models/MiMo-V2.6-Flash-Sushi-2.3bpw
# or: sushi pull mimo-v2.6-flash   (lands in ~/.sushi/models/beamster/MiMo-V2.6-Flash-Sushi-2.3bpw)

# images, 8-bit KV, MTP, the full 1M context
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/MiMo-V2.6-Flash-Sushi-2.3bpw --ctx-size 1048576
```

MTP and the 8-bit KV cache are on by default for MiMo, and thinking is on by default, as in Xiaomi's chat template.

- `--mtp-head-kv-quant` stores the MTP head's own KV at 8 bits too.
- `--preserve-thinking off` keeps only the latest turn's thinking in the prompt. Agents running long sessions may prefer
  it for the shorter context; each new instruction then re-processes the prompt from the first dropped thought.
- `--prefill-chunk 2048` is the widest prompt step per forward; a wider one costs memory without prefilling faster,
  and a request that does not fit steps down to a narrower chunk.
- `--prefix-cache-mem 1GB` also keeps seen prompt prefixes hot in RAM, faster than the SSD (RAM retention is off by default).
- Seen prompt prefixes live on the SSD by default, so a repeated prompt skips its prefill; the budget is sized per model and capped at 20GB, and `--prefix-cache-disk 20GB` sets it.
- `--prefix-cache-entries 1` keeps one conversation's prefix; raise it to 4-8 when several agents share the server.

**96 GB Mac, GLM-5.3-Flash-Sushi-2bpw**

```bash
sudo sysctl iogpu.wired_limit_mb=90000
hf download beamster/GLM-5.3-Flash-Sushi-2bpw --local-dir ~/.sushi/models/GLM-5.3-Flash-Sushi-2bpw
# or: sushi pull glm-5.3-flash:2bpw   (lands in ~/.sushi/models/beamster/GLM-5.3-Flash-Sushi-2bpw)

# images and video, 8-bit KV, DFlash2 when present
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/GLM-5.3-Flash-Sushi-2bpw
```

- The auto context is 512K at 8-bit KV and 300K with `--kv-quant 16` (BF16 latent).

**128 GB Mac, GLM-5.3-Flash-Sushi-2.4bpw**

```bash
sudo sysctl iogpu.wired_limit_mb=120000
hf download beamster/GLM-5.3-Flash-Sushi-2.4bpw --local-dir ~/.sushi/models/GLM-5.3-Flash-Sushi-2.4bpw
# or: sushi pull glm-5.3-flash   (lands in ~/.sushi/models/beamster/GLM-5.3-Flash-Sushi-2.4bpw)

# images and video, 8-bit KV, DFlash2 when present
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/GLM-5.3-Flash-Sushi-2.4bpw
```

- The pack ships Inco.ai's [DFlash2 assistant](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) unmodified in
  `GLM-5.3-Flash-DFlash2/`, under its own CC BY-NC-ND 4.0 license (non-commercial, no derivatives). On its first load
  sushi builds a smaller copy for this Mac in the pack's `dflash2/` folder: a derivative of that work, so keep it local
  and never share it.
- `--drafter <dir>` points at an assistant stored elsewhere and `--no-drafter` turns it off; without it GLM decodes
  one token at a time.
- Up to four requests decode together.

<a id="deepseek-launch"></a>
**96 GB and 128 GB Macs, DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw**

```bash
sudo sysctl iogpu.wired_limit_mb=125000   # 96 GB Mac: 93000
hf download beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw --local-dir ~/.sushi/models/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw
# or: sushi pull deepseek-v4.1-flash   (lands in ~/.sushi/models/beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw)

# images, DSpark, thinking at high, the full 1M context; the routed experts stream from the SSD
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw \
  --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000
# 128 GB Mac: --ssd-budget-gb 108, or 96 to leave more memory to other apps; 96 GB Mac: --ssd-budget-gb 80
```

- **Disk.** The pack is 332.1 GB: 120 GiB of weights and 189 GiB of Engram tables, which are read from the SSD a row at
  a time and never loaded. The routed experts (119.6 GB of the weights) stream from the SSD within `--ssd-budget-gb`.
  The prefix cache's SSD tier, up to 20GB, comes on top.
- **`--ssd-budget-gb` is required** (without it the load fails with `ExpertStreamingRequired`). It is the total GiB the
  load keeps resident: the trunk, the DSpark stages, the vision tower, the prefill's expert workspace and the expert
  cache, which is what is left. A higher budget gives more expert slots a layer and fewer misses; a miss is an SSD
  read. The budget and the request's own state (0.93 GB at 1M tokens, 3.7 GB billed to a 1M request) plus the largest
  image's encode (1.6 GB) must sit under the GPU limit, so the budget tops out about 5 GiB under it at 1M context.
- **Recommended budgets, with `--ctx-size 1000000 --max-tokens 128000`:** on a 128 GB Mac 108, or 96 to leave more
  memory to other apps, at `sudo sysctl iogpu.wired_limit_mb=125000`; on a 96 GB Mac 80, at
  `sudo sysctl iogpu.wired_limit_mb=93000`.

  | `--ssd-budget-gb` | expert slots a layer (of 384) | expert hits code / prose | decode 4k code / prose | decode 128k code / prose | prefill 8k window code / prose |
  |---|---:|---:|---:|---:|---:|
  | 80 | 182 | 96.5% / 99.2% | 44.7 / 44.5 | 36.6 / 39.4 | 760 / 547 |
  | 96 | 227 | 98.7% / 99.6% | 65.8 / 48.2 | 51.8 / 45.4 | 963 / 700 |
  | 108 | 260 | 99.6% / 99.9% | 76.6 / 49.3 | 63.7 / 43.5 | 1,024 / 821 |

  Slots are the plan from the pack's headers. Every row is one `sushi bench` run of the same engine (tok/s, DSpark on,
  one request, code and prose corpora, 256 and 128 decoded tokens a sample); the 108 row is the
  [v1.4.0 bench](bench/v1.4.0/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw.md). Hits count the decode forwards' expert
  picks found in the cache.
- **Long prompts.** A long prefill may lend decoder layers and read them back before decode; a lower budget leaves more
  headroom, so it borrows later ([dsv41-memory](dsv41-memory.md#borrow)).
- **Other Macs.** Only the 128 GB Mac has been run; the 80 row above is the 96 GB Mac's budget measured there. Below it the ledger admits a budget as low as 17 GiB (16 with
  `--no-vision`; a lower one is refused, `ssd_budget_below_resident` or `ssd_budget_below_vision`), and the slots a budget
  buys are fixed by the pack, so these are plan numbers and no speed is promised:

  | `--ssd-budget-gb` | expert slots a layer | smallest Mac at 1M context | GPU limit to set |
  |---|---:|---|---|
  | 32 | 48 | 48 GB | `iogpu.wired_limit_mb=43000` |
  | 48 | 93 | 64 GB | `iogpu.wired_limit_mb=59000` |
  | 80 | 182 | 96 GB | `iogpu.wired_limit_mb=93000` |

  A shorter `--ctx-size` shrinks the request bill and lowers the Mac that admits a budget; at 17 to 20 GiB the cache
  holds 6 to 14 slots of 384 and nearly every expert is an SSD read.
- **Context** runs to 1M tokens (`--ctx-size 1000000`; set it, since auto-context reads free memory). The state is 890 B
  a token and the prefix cache keeps seen prompts on the SSD by default, so a repeated prompt skips its prefill. One
  request decodes at a time.
- **DSpark**, the checkpoint's own drafter, is on by default; `--no-mtp` turns it off. It accepts drafts by the exact rule
  only. Greedy output is the same at every round-width mix, can differ from `--no-mtp` at a near-tie, and never differs between a stream and a non-stream reply.
- **Thinking** is `high` by default. A request names `reasoning_effort` `low`, `high` or `max` (`medium` is served as
  `high`); `enable_thinking: false` or effort `off` answers without thinking.
- **Images** are served by default. `--no-vision` leaves the tower out, saves its 0.97 GB for the expert cache and turns
  an image into a 400.
- **`--drop-expert n`** (a multiple of 8) never loads or picks the `n` least-used experts of every layer, by the pack's
  `expert_usage.json`: it frees 2.1 to 2.5 GB at 8 and 4.3 to 5.0 GB at 16, enough to hold every kept expert resident at
  a budget of 116. It is lossy: score it with `sushi kld compare` first ([dsv41-arch](dsv41-arch.md#drop-expert)).
- **Refused flags.** The model has no KV cache, GatedDeltaNet layers or MTP head to configure, so `--kv-quant`,
  `--qwen-gdn`, `--mtp-head-kv-quant`, `--mtp-typical`, `--mtp-tokenv3` and `--mtp-greedy-tail` stop the load with a
  message naming the flag ([dsv41-parity](dsv41-parity.md)). A `model-settings.json` `kv_quant`, `mtp_acceptance` or
  `mtp_greedy_tail` is ignored, with one log line.

## Coding agents

With the server running, `sushi launch <agent>` starts claude, pi, omp, opencode, codex, grok, hermes or aider against it:

```bash
sushi launch omp
```

pi, omp, codex, grok and hermes run from their own home under `~/.sushi/<agent>/`, so your usual config is untouched and
the session does not see your other providers, settings or history; claude, opencode and aider reach the server
through environment variables. `--print` writes the config and prints the launch script instead of running it.
