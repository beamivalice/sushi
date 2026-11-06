# Choosing a pack, and running one bigger than your Mac's memory

This page answers two questions: which pack suits your Mac, and when to stream a pack from the SSD
(`--ssd-budget-gb`). For launch commands see [usage](usage.md#recommended-launch); for the exact memory each pack needs
at each context see [README, Memory](../README.md#memory).

## The short answer

| Your Mac | Pick | Also possible |
|---|---|---|
| 32 GB | Qwen Sushi-2bpw, streamed | |
| 48 GB | Qwen Sushi-2bpw | |
| 64 GB | **Qwen Sushi-STQ-2.75bpw** | Qwen Sushi-4bpw, streamed: closer to the original model, about half the speed |
| 96 GB | Qwen Sushi-4bpw, GLM Sushi-2bpw, DeepSeek (streamed) | |
| 128 GB | any pack; GLM Sushi-2.4bpw and DeepSeek are the largest | |

A pack that fits in memory is always the faster choice. Stream only when you want a pack that does not fit and you
accept that it runs slower.

## How the Qwen packs compare

All three are the same model, Qwen3.8-Flash-Next, stored at different precision. Lower precision means smaller and
faster, and further from the original.

| | Sushi-2bpw | Sushi-STQ-2.75bpw | Sushi-4bpw |
|---|---:|---:|---:|
| GPU memory for the weights (GiB) | 35.2 | 44.9 | 63.9 |
| Distance from the original model (KLD, lower is better) | 0.0673 | 0.0250 | 0.0135 |
| Next token agrees with the original | 91.90% | 94.73% | 96.14% |
| Tool calling (tool-eval-bench 3.0.0, hard mode) | 88 / 100 | 94 / 100 | 90 / 100 |

- KLD is measured the same way for all three (eight prompt categories, against the original BF16 weights), so this row
  compares them directly. Older model cards used different readings; this table supersedes them.
- STQ ("Sushi Trellis Quant") stores weights with a codebook rather than plain rounding. That is why the 2.75-bit STQ
  pack lands much closer to Sushi-4bpw than to Sushi-2bpw, though its bit count sits nearer 2.
- Tool-eval scores move by a few points from run to run, so read 88-94 as "all good at tool calling", not as a ranking.

## The two memory numbers

A Mac with Apple silicon shares one pool of memory between the CPU and the GPU. Two numbers matter:

1. **Installed memory**: what the Mac has (32, 48, 64 GB and so on).
2. **GPU limit**: how much of that macOS lets the GPU hold at once. A model runs on the GPU, so this is the number that
   decides what fits. macOS keeps it well below the installed memory by default.

   Raise it **together with the two user-wire limits, always**. macOS also caps how much memory one program may lock in
   place (`vm.global_user_wire_limit`, `vm.user_wire_limit`), and that cap stays at its lower default when you raise
   only the GPU limit: sushi then loads, but prompt reading slows down and a large load can make the Mac unresponsive.
   Set all three to the same value, the GPU limit in megabytes and the other two in bytes (add six zeros):

   ```bash
   sudo sysctl iogpu.wired_limit_mb=59000 vm.global_user_wire_limit=59000000000 vm.user_wire_limit=59000000000
   sysctl iogpu.wired_limit_mb vm.global_user_wire_limit vm.user_wire_limit   # shows them; 0 means the default
   ```

   All three reset at reboot. Leave room for macOS and your other apps. The values sushi's docs assume:

   | Installed | GPU limit (MB) | user-wire limits (bytes) |
   |---|---:|---:|
   | 32 GB | 27,000 | 27000000000 |
   | 48 GB | 43,000 | 43000000000 |
   | 64 GB | 59,000 (higher risks a kernel panic when macOS runs out of memory) | 59000000000 |
   | 96 GB | 90,000 (93,000 for DeepSeek) | 90000000000 (93000000000) |
   | 128 GB | 120,000 (125,000 for DeepSeek) | 120000000000 (125000000000) |

Everything sushi loads (the weights, plus the memory the conversation's context takes) must fit under the GPU limit.
When it does not, sushi refuses to load and prints the `sudo sysctl` command, all three limits, that would fit.

## What streaming does

The Qwen, GLM and DeepSeek models are mixture-of-experts models: each layer holds hundreds of "experts", and each
token uses only a few of them. Most of a pack's size is experts.

With `--ssd-budget-gb N`, sushi keeps N GiB on the GPU: the always-used parts of the model, the speculative-decoding
head, and as many experts as fit, as a cache. When a token needs an expert that is not cached, sushi reads it from the
SSD. The answers are byte-for-byte the same as with the whole pack in memory; only the speed changes.

- **A bigger budget is faster**: more experts stay cached, so fewer SSD reads. At 48 GiB, Qwen Sushi-4bpw caches 362 of
  each layer's 512 experts.
- **The budget is for the weights only.** The context's memory comes on top, so the budget plus the context must fit
  under the GPU limit. If you do not pass `--ctx-size`, sushi picks the largest context that fits.
- **A fast internal SSD matters.** Every cache miss is an SSD read; an external drive is slower.
- **On Qwen and GLM, image input is off while streaming** (it would take room from the expert cache). Add `--vision` to
  keep it: on Qwen Sushi-4bpw it costs 0.6 GB of the budget. DeepSeek keeps it on.
- Streaming only reads from the SSD; it does not write to it.

Set the budget with `--ssd-budget-gb N` on the command line, or per model on the web page: Models tab, pick the model, and
set "Memory for SSD streaming (GiB)" in its Settings. It applies at the next load.

## Example setups

| Mac | Pack | `--ssd-budget-gb` | GPU limit (MB) | Context | Generation speed at 4k tokens |
|---|---|---:|---:|---|---|
| 32 GB | Qwen Sushi-2bpw | 18 | 27,000 | 66k | about 20 tok/s ¹ |
| 64 GB | Qwen Sushi-4bpw | 48 | 59,000 | what fits (auto) | 72 code / 74 prose ² |
| 96 GB | DeepSeek Sushi-STQ-1.75bpw | 80 | 93,000 | 1M | 45 code / 45 prose ² |
| 128 GB | DeepSeek Sushi-STQ-1.75bpw | 108 (or 96) | 125,000 | 1M | 77 code / 49 prose ² |

- **`--ssd-budget-gb`**: GiB of GPU memory for the model's weights (see above).
- **GPU limit**: the value for `iogpu.wired_limit_mb`, in megabytes; set the two user-wire limits with it ([above](#the-two-memory-numbers)).
- **Context**: the longest conversation, in tokens, that fits beside that budget.
- ¹ measured on an M1 Max. ² measured on an M5 Max with sushi 1.5.1; an M4 or M5 Pro reads its SSD and computes more
  slowly, so expect less. DeepSeek streams on 96 and 128 GB Macs ([usage](usage.md#deepseek-launch) has its launch
  lines and why a budget above 96 needs one more `sysctl`).

### Streaming against a smaller pack: Qwen on a 64 GB Mac

Measured on an M5 Max, sushi 1.5.1, tokens per second:

| | STQ-2.75bpw, in memory | 4bpw, streamed at 48 GiB |
|---|---:|---:|
| Generation, code, 4k / 32k tokens of context | 169 / 147 | 72 / 61 |
| Generation, prose, 4k / 32k | 77 / 102 | 74 / 74 |
| Reading the prompt, 4k / 32k | 1,453 / 2,090 | 831 / 866 |

Streamed Sushi-4bpw generates code at about half the speed and reads long prompts at under half, for answers closer to
the original model. Coding agents send long prompts and write a lot of code, so STQ suits them better; streamed 4bpw
suits chat where you want the extra precision and do not mind the wait.

## When to stream, and when not to

Stream when:
- the pack you want is bigger than your GPU limit, and
- you prefer its precision to the speed of a smaller pack that fits, and
- the Mac has a fast internal SSD.

Do not stream when:
- a pack you are happy with fits in memory: it is always faster;
- you need image input at no extra cost (on Qwen and GLM, streaming turns it off unless you pass `--vision`);
- the SSD is external or slow.

How it works inside: [engine-expert-streaming](engine-expert-streaming.md).
