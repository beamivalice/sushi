<!-- Written by Samuel Reed (@STRML), author of PR #128, and copied with credit from his gist: https://gist.github.com/STRML/1bea3e2e165d99873a854513ef7db903 (2026-11-26). His measurements, on an M5 Ultra constrained to the budgets named. -->

# Running DeepSeek-V4.1-Flash on 64 GB and 96 GB Macs with Sushi

This guide shows how to build [Sushi](https://github.com/beamivalice/sushi) with the DS4.1 expert-prefetcher fixes from [PR #128](https://github.com/beamivalice/sushi/pull/128) and serve `DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw` on 64 GB or 96 GB Apple Silicon Macs (plus a 48 GB stretch config) using SSD expert streaming, `--expert-cache-bias`, and DSpark speculative decoding.

On a 64 GB budget (`--ssd-budget-gb 42`, 47 cached experts/layer), this setup keeps exact 7,498-position KLD at `0.1188` (`bias=0.6`) or `0.1612` (`bias=0.9`) against the FP8 teacher while reaching `48-82 tok/s` across prose, code, echo, and rename workloads on a fast internal NVMe SSD. On a 96 GB M5 Ultra Studio (`--ssd-budget-gb 80`, ~120 cached experts/layer), you get ~2.5x the expert cache, letting you lower `--expert-cache-bias` to `0.3-0.4` for near-resident 2.7bpw quality (`~0.08-0.10` KLD) with 256K-1M context.

## Why build from `main` instead of stock `brew install sushi`

[PR #128](https://github.com/beamivalice/sushi/pull/128) is merged on `beamivalice/sushi` `main`. Until it ships in a tagged Homebrew release, building from `main` gives you three fixes that matter directly for 64 GB streaming decode:

1. **Fractional-share prefetch pacing (`PrefetchCredit.nextAffords` + enqueue reservation).** On 40-layer DS4.1 packs (`37` target layers at `lead = 3`, ~`11.7 MB` per expert), normal decode rounds grant less than one expert per layer, which left `--ssd-prefetch` at `0` queued jobs during decode. Carrying skipped layers' shares and reserving `min_bytes` on enqueue keeps prefetching active without burst-queuing into the worker.
2. **Pre-faulted SSD read-rate probe (`Engine.probeReadRate`).** Previously, probing newly mapped virtual slab pages across every layer and dividing by loop wall time let first-touch VM page faults drag readings down to `3.5-3.6 GB/s` on a `12 GB/s` SSD. Pre-faulting one layer's probe slots and measuring `fill_ns` keeps the probe steady at true NVMe speed.
3. **Cache-bias-aware prefetch ranking (`Engine.prefetchPenalty`).** Under `--expert-cache-bias`, the prefetch predictor now scores all 384 experts with the layer's host cache penalty (`cachePenaltyHost`) over `top_k` candidates instead of queuing uncached experts that the biased router will skip.

Once a release newer than `1.8.0` ships on Homebrew, you can install with `brew install beamivalice/tap/sushi` and skip the build section.

## Prerequisites

- **Hardware:** Apple Silicon Mac with 64 GB unified memory (48 GB stretch config is included below) and at least **420 GB free on the internal NVMe SSD**. Streaming only reads from disk and never writes to it, but every cache miss is an SSD read, so an external USB drive will bottleneck decode.
- **OS and SDK:** macOS 26.2 or later, Xcode 26 with macOS SDK >= 26.2, and the Xcode Metal Toolchain component.
- **Build tools:** `cmake` and Zig `0.17.1` (the minimum version pinned in `build.zig.zon`).

Install the missing build tools if you do not already have them:

```bash
xcodebuild -downloadComponent MetalToolchain
brew install cmake zig
zig version   # verify 0.17.1 or newer
```

## 1. Clone and build Sushi from `main`

Clone `beamivalice/sushi`, initialize the pinned MLX submodules, stage `lib/mlx`, and compile a `ReleaseFast` binary:

```bash
git clone https://github.com/beamivalice/sushi.git
cd sushi
git submodule update --init lib/mlx-src lib/mlxc-src
./scripts/build-mlx.sh
zig build -Doptimize=ReleaseFast
```

Why `./scripts/build-mlx.sh` instead of Homebrew's `mlx` bottle: Homebrew builds MLX with `MACOSX_DEPLOYMENT_TARGET=26.0`, which silently disables M5 neural-accelerator (`*_nax`) kernels in the metallib and skips Sushi's transient-cap and host-overhead MLX patches. `build-mlx.sh` builds at deployment target `26.2`, applies Sushi's patches, and fails loudly if NAX symbols are missing.

Verify the binary and unit tests:

```bash
./zig-out/bin/sushi --version
zig test src/prefetch_plan.zig
zig build lint
```

Optional: symlink or copy `./zig-out/bin/sushi` into `~/.local/bin/sushi` so you can run `sushi` directly.

## 2. Raise macOS wired memory limits

By default, macOS caps GPU wired memory well below installed RAM. Raise `iogpu.wired_limit_mb` together with both `vm.*_user_wire_limit` sysctls in bytes (`limit_mb * 1048576`). Setting only `iogpu.wired_limit_mb` leaves macOS's user-wire cap at its default, which slows prefill and can stall the system.

Do not exceed the per-tier ceiling below; higher values leave too little memory for macOS and risk a kernel panic. All three limits reset on reboot:

| Mac RAM | `iogpu.wired_limit_mb` | `vm.global_user_wire_limit` & `vm.user_wire_limit` |
| --- | ---: | ---: |
| 48 GB | `43000` | `45088768000` |
| 64 GB | `59000` | `61865984000` |
| 96 GB | `93000` | `97517568000` |

```bash
# 64 GB Mac
sudo sysctl \
  iogpu.wired_limit_mb=59000 \
  vm.global_user_wire_limit=61865984000 \
  vm.user_wire_limit=61865984000

# 96 GB Mac Studio (M5 Ultra)
sudo sysctl \
  iogpu.wired_limit_mb=93000 \
  vm.global_user_wire_limit=97517568000 \
  vm.user_wire_limit=97517568000

# 48 GB Mac (stretch)
sudo sysctl \
  iogpu.wired_limit_mb=43000 \
  vm.global_user_wire_limit=45088768000 \
  vm.user_wire_limit=45088768000

# Verify all three took effect
sysctl iogpu.wired_limit_mb vm.global_user_wire_limit vm.user_wire_limit
```

Sushi reads these limits at startup. If you change them while Sushi is running, restart the server.

## 3. Download the `STQ-2.7bpw` pack

Pull `DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw` (~373 GB on disk; ~186 GB routed experts, ~11 GB resident trunk/MTP/vision, and ~189 GiB Engram tables that stay on SSD and are never loaded into RAM):

```bash
./zig-out/bin/sushi pull deepseek-v4.1-flash:2.7bpw
```

An interrupted pull resumes from `.partial` checksums when you rerun the command. The pack downloads to:

```text
~/.sushi/models/beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw
```

## 4. Launch profiles

`--ssd-budget-gb` budgets weights and expert streaming buffers only; the KV cache and RAM prefix cache sit on top of it, and everything together must fit under the wired limit.

### 64 GB Mac - Profile A: Balanced quality (recommended default, KLD `0.1188`)

Use `--expert-cache-bias 0.6` when you want near-2.7bpw quality (`KLD = 0.1188`, `91.72%` top-1 agreement vs FP8 teacher, well below `STQ-1.75bpw`'s `0.1525-0.169`):

```bash
./zig-out/bin/sushi serve \
  --model ~/.sushi/models/beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw \
  --host 127.0.0.1 --port 12345 \
  --ctx-size 65536 \
  --max-concurrent 1 \
  --ssd-budget-gb 42 \
  --expert-cache-bias 0.6 \
  --ssd-prefetch 65 \
  --mtp \
  --prefill-chunk-max 8192 \
  --prefix-cache-mem 8GB \
  --prefix-cache-disk 20GB
```

### 64 GB Mac - Profile B: Max speed within `<= 0.16` KLD (`bias=0.9`, KLD `0.1612`)

Raise `--expert-cache-bias` to `0.9` for higher cache hit rates and `46-82 tok/s` DSpark decode while still beating `STQ-1.75bpw` (`bias=0.3`: KLD `0.1662`):

```bash
./zig-out/bin/sushi serve \
  --model ~/.sushi/models/beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw \
  --host 127.0.0.1 --port 12345 \
  --ctx-size 65536 \
  --max-concurrent 1 \
  --ssd-budget-gb 42 \
  --expert-cache-bias 0.9 \
  --ssd-prefetch 65 \
  --mtp \
  --prefill-chunk-max 8192 \
  --prefix-cache-mem 8GB \
  --prefix-cache-disk 20GB
```

### 96 GB M5 Ultra Mac Studio (`STQ-2.7bpw`, ~120 slots/layer, 256K-1M context)

Keep `STQ-2.7bpw` on 96 GB rather than dropping to `1.75bpw`. At `--ssd-budget-gb 80` (`85.9 GB`), fixed overhead (`18.4 GB`) leaves ~`67.5 GB` for the expert cache (~`120 slots/layer`, ~31% of all 384 experts) plus ~`7 GB` under the `93000` MB wired limit for KV, prefill transient, and prefix cache. Because DS4.1 KV state is only `890 B/token` (`0.93 GB` at 1M tokens), long contexts fit easily:

```bash
./zig-out/bin/sushi serve \
  --model ~/.sushi/models/beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw \
  --host 127.0.0.1 --port 12345 \
  --ctx-size 262144 \
  --max-concurrent 1 \
  --ssd-budget-gb 80 \
  --expert-cache-bias 0.35 \
  --ssd-prefetch 65 \
  --mtp \
  --prefill-chunk-max 8192 \
  --prefix-cache-mem 4GB \
  --prefix-cache-disk 20GB
```

Tuning the 96 GB M5 Ultra launch:
- **1M context (`--ctx-size 1000000`):** Keep `--ssd-budget-gb 80` with `--prefix-cache-mem 0` (using `--prefix-cache-disk 20GB`), or step down to `--ssd-budget-gb 78` (~116 slots/layer).
- **`--expert-cache-bias` dial at ~120 slots/layer:**
  - `0.3-0.4` (daily driver): near-resident 2.7bpw quality (`~0.08-0.10` KLD) with a large hit-rate bump over `0`.
  - `0.5-0.6` (max speed): ~`0.10-0.11` KLD with near-resident decode speed.
  - `0` (exact): bit-exact 2.7bpw (`KLD 0.0772`), slower than `0.35` but practical at 120 slots/layer on a ~12 GB/s SSD.
- **M5 Ultra acceleration:** Building `lib/mlx` with `./scripts/build-mlx.sh` (SDK >= 26.2) enables M5 NAX kernels alongside Sushi Cooler (`ds41_cooler`, on by default).

### General tuning knobs

- **Text-only workloads (`--no-vision`):** DeepSeek loads the vision tower by default (`0.51 GB`). Pass `--no-vision` if you do not send images; that memory returns directly to the expert cache.
- **Reclaiming more cache slots on 64 GB:** If you set `--prefix-cache-mem 0` (relying on `--prefix-cache-disk 20GB`) and close heavy desktop apps, you can raise `--ssd-budget-gb` from `42` toward `46-48` to add more cached experts per layer. If a budget is too large for current free RAM, Sushi refuses to load and prints the largest `--ssd-budget-gb` that fits.
- **48 GB Mac stretch:** With the `43000` MB sysctl limit above, pass `--ssd-budget-gb 30 --expert-cache-bias 0.6 --ssd-prefetch 65 --no-vision --prefix-cache-mem 0` (exact 7,498-pos KLD `0.1383`, `91.18%` top-1 agreement).

## Memory layout (`42 GiB` vs `80 GiB` budget)

With `--mtp` and vision enabled (`--no-vision` frees `0.51 GB` back into the expert cache):

| Component | 64 GB Mac (`--ssd-budget-gb 42`) | 96 GB Studio (`--ssd-budget-gb 80`) |
| --- | ---: | ---: |
| Resident trunk | `6.06 GB` | `6.06 GB` |
| MTP / DSpark head | `4.77 GB` | `4.77 GB` |
| Vision tower (`0.00 GB` with `--no-vision`) | `0.51 GB` | `0.51 GB` |
| Streaming workspace | `5.37 GB` | `5.37 GB` |
| Kept scales | `0.68 GB` | `0.68 GB` |
| Bounce + MLX pool + selected buffers | `1.02 GB` | `1.02 GB` |
| **Routed expert cache** | **`26.28 GB` (`47 slots/layer`, profile `39..93`)** | **~`67.5 GB` (~`120 slots/layer`)** |

KV cache (`890 B/token`, `0.06 GB` at 64K, `0.93 GB` at 1M) and optional RAM prefix cache sit beside this budget under the wired limit (`59000` MB on 64 GB, `93000` MB on 96 GB).

## 5. Verify the running server

Check `/health` and confirm the startup logs show the SSD probe, cache warm, and active prefetching:

```bash
curl -s http://127.0.0.1:12345/health
# {"status":"ok"}
```

Look for these lines in the server log (`~/.sushi/logs/sushi-12345.log` or stdout with `--log-file off`):

```text
[expert-stream] ssd budget 42 GiB: ... -> expert cache 26.28 GB = 47 slots/layer
[dsv41] expert prefetch on: 65% of 12.1 GB/s (probe k2/k4/k8 10.7/11.8/12.1 GB/s)
[expert-stream] cache warm complete: slots=47/47 per layer, layers=40, ...
```

After a few generations, confirm prefetch jobs are actually running (`jobs queued > 0`, `fill_gbps` near your internal NVMe's read speed):

```text
[expert-stream] prefetch: 3733 jobs queued, 3428 read 3840 experts (867 skipped for lack of credit) ...
```

Open the local chat/dashboard UI at `http://127.0.0.1:12345/chat`, or point any OpenAI- or Anthropic-compatible client at `http://127.0.0.1:12345/v1`.

## Measured results and alternatives ruled out

All KLD scores are exact 7,498-position to-first-EOS readings against the FP8 teacher (`MLX_ENABLE_TF32=0`, `--ctx-size 2048`). Throughput numbers were measured in `ReleaseFast` on a 256 GB M5 Ultra Mac Studio constrained to `--ssd-budget-gb 42` (`47 slots/layer`, cold internal NVMe ~`12 GB/s`). Expect lower absolute tok/s on M4/M5 Pro or Max chips with slower internal SSD bandwidth or fewer GPU cores; the relative gains from `--expert-cache-bias`, DSpark, and PR #128 still hold.

### `STQ-2.7bpw` at `--ssd-budget-gb 42` with DSpark (`--mtp`)

| Setup | Exact KLD | Top-1 | `prose` | `code` | `echo` | `rename` | `32k_code` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bias=0.6`, PR #128 `pf=0` | `0.1188` | `91.72%` | `35.8` | `37.3` | `50.0` | `53.7` | `29.4` |
| `bias=0.6`, PR #128 `pf=65` | `0.1188` | `91.72%` | `35.0` | `38.8` | `50.1` | `55.6` | `28.9` |
| `bias=0.9`, upstream `pf=65` | `0.1612` | `90.17%` | `40.4` | `48.1` | `52.0` | `82.0` | `38.9` |
| `bias=0.9`, PR #128 `pf=0` | `0.1612` | `90.17%` | `46.4` | `52.6` | `73.6` | `83.8` | `39.4` |
| **`bias=0.9`, PR #128 `pf=65`** | **`0.1612`** | **`90.17%`** | **`48.4`** | **`54.4`** | **`75.0`** | **`82.4`** | **`45.9`** |

### Why other 64 GB approaches were rejected

| Approach | Exact KLD | Verdict |
| --- | ---: | --- |
| `STQ-2.7bpw` + `bias=0.6` / `0.9` (this guide) | `0.1188` / `0.1612` | **Best quality/speed Pareto frontier** at `<= 0.16` KLD. |
| `STQ-1.75bpw` + `bias=0.3` (42g) | `0.1662` | Strictly dominated by `STQ-2.7bpw` `bias=0.9` (worse KLD and slower). |
| Hybrid hot-80 `2.7bpw` / cold-304 `1.75bpw` (`133 GB`) | `0.1543` resident / `0.1929` (42g `b=0.6`) | Faster (`54-111 tok/s` at 50g `--no-vision`), but long-tailed MoE routing leaves KLD past `0.19` once streamed. |
| Hard pruning (`--drop-expert` to 48 GiB) | `1.073` | Severe quality collapse; soft `--expert-cache-bias` avoids permanently dropping cold experts. |
| Sub-1-bit LittleBit-2 (`0.32-0.50 bpp`) | `>= 0.723` (FP32 SVD lower bound) | Mathematically ruled out even at infinite-precision low-rank factors. |

## Troubleshooting

- **`build-mlx.sh` fails on Metal compiler or SDK < 26.2:** Run `xcodebuild -downloadComponent MetalToolchain` and verify `xcrun --sdk macosx --show-sdk-version` is `26.2` or higher.
- **Sushi refuses to load with a `sudo sysctl` message:** Run the exact three-variable `sysctl` command it prints (always including `vm.global_user_wire_limit` and `vm.user_wire_limit`), or lower `--ssd-budget-gb` / `--prefix-cache-mem` / `--ctx-size`.
- **Unknown flag `--prefill-chunk`:** Newer Sushi commits (including PR #128) replaced `--prefill-chunk` with `--prefill-chunk-max` and removed `--prefix-cache-entries`.
- **Low `probeReadRate` (`< 3 GB/s`) even with PR #128:** Check that the model folder lives on the internal NVMe SSD and not an external drive or network share.

---



*Guide by [Samuel Reed (@STRML)](https://github.com/STRML), who contributed the DeepSeek-V4.1 prefetch fixes in [PR #128](https://github.com/beamivalice/sushi/pull/128). Source: [his gist](https://gist.github.com/STRML/1bea3e2e165d99873a854513ef7db903).*
