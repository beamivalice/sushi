# GLM-5.3-Flash: KLD of the served packs

GLM-5.3-Flash KLD tables, split out of [quality-kld](quality-kld.md); everything shared by the served architectures stays there.



## GLM-5.3-Flash: native BF16 teacher, 4x512 (2026-10-04)

GLM's release reading is this 4x512 screen, with the two code prompts (2x512) reported apart: a 16x512 BF16 teacher
would stream the BF16 experts and is too slow to capture. Shipped Sushi-2.4bpw (K2.25/K2.5 W14 experts, A6 g128 trunk)
against the NAX-path teacher: KLD 0.0742 / top-1 90.3%; code 0.0429, prose 0.1055. The same
screen on Sushi-2.5bpw (kv8, FP32 decode attention, first teacher): KLD 0.0716 / top-1 90.3%; code 0.0324 / 95.8%,
prose 0.1107 / 84.9%.

Teacher (every row of the table below was scored against the reference-arm capture, whose `identity.json` has no `nax_arms`): the BF16 source checkpoint through the native forward, `MLX_ENABLE_TF32=0 sushi kld capture --prompts
standard4 --tokens 512 --no-template --kv-quant off --ssd-budget-gb 100` (streamed BF16 experts, dense prefill in
chunks of at most 512, synchronous layers, BF16 MLA cache, FP32 KDA state). Native prompt lengths 242/261/190/183; no
EOS in the 2,048 rows, so first-EOS and all-positions readings are the same. Students carry MCG EXL3
experts at W12 (K2.25 unless the row says K2.5; the shipped row is W14), BF16 MLA and FP32 KDA state, one resident target, MTP and DFlash2 off.

| Pack | Trunk | KLD | Top-1 | NLL | Code / prose KLD | Peak active | Commit, settings |
|---|---|---:|---:|---:|---:|---:|---|
| Sushi-2.3bpw | A6 g128 | 0.092950 | 88.96% | 0.4323 | 0.0468 / 0.1391 | 94.08 GB | `3ed533d7`+WIP, TF32 on, fast target kernels |
| A8 experiment (not shipped) | A8 g128 | 0.091519 | 88.87% | 0.4318 | 0.0441 / 0.1389 | 96.30 GB | same binary |
| Sushi-2.45bpw | raw FP8 block-128 | 0.091266 | 88.62% | 0.4304 | 0.0439 / 0.1386 | 102.51 GB | `56017748`, TF32 off, `sushi kld compare` |
| Sushi-2.5bpw (K2.5) | A6 g128 | 0.072071 | 89.94% | 0.4075 | 0.0314 / 0.1127 | 103.59 GB | `00668fcb`, `sushi kld compare` defaults; 2.3bpw reproduces its row bit for bit there |

Against the NAX-path teacher (`glm5_model.enterTeacher()`, binary `177c526f`, TF32 off; decode attention is the only
arm that changes this capture), students at BF16 latent, same 4x512 prompts:

| Pack | Experts | Trunk | KLD | Top-1 | NLL | Code / prose KLD | Peak active | Commit, settings |
|---|---|---|---:|---:|---:|---:|---:|---|
| Sushi-2.3bpw | K2.25 L3-45, W14 | A6 g128 | 0.078609 | 89.94% | 0.4147 | 0.0460 / 0.1112 | 94.07 GB | `177c526f`, `sushi kld compare --kv-quant 16` |
| **Sushi-2.4bpw (shipped)** | K2.25 L3-36 + MTP, K2.5 L37-44, W14 | A6 g128 | 0.074213 | 90.33% | 0.4160 | 0.0429 / 0.1055 | 95.88 GB | same |
| Sushi-2.5bpw | K2.5 L3-45, W14 | A6 g128 | 0.057151 | 91.06% | 0.3920 | 0.0316 / 0.0827 | 103.58 GB | same |

The rows above were scored against the first teacher, so they do not rank against this one.

K2.5 experts cut KLD 22.5% from K2.25 on the same A6 trunk (byte-identical trunk tensors). The three trunks sit within 2% of each other under two numerical profiles; this four-prompt screen does not rank them
and is not the 16x512 release reading. Code scores about 3x lower than prose on every pack.

GLM kv8 latent (`--kv-quant 8`, Sushi-2.5bpw, binary `694e36a3`, `taskpolicy -a`, GPU lock):
- Same 4x512 teacher: KLD 0.071762 / top-1 89.70% against the BF16 latent's 0.072071 / 89.94% (-0.43%, inside the
  noise floor); peak active unchanged at 103.59 GB.
- Long context, against the pack's own BF16-latent reference (2 prompts of 59k and 66k tokens x 2048 teacher-forced
  tokens): KLD to first EOS 0.00715, top-1 96.8% over 496 positions; per-256-position means stay at 0.0006-0.0094 on
  the prose prompt. Past EOS the code prompt's continuation degenerates and its KLD climbs to 0.18, outside the
  scored window.
- Greedy free-run at those prompts diverges early (token 16 and 118) at near ties; the answers reword the same facts.
- FP32 composite B1/B3 decode attention against the fused NAX SDPA, one binary (`cc56be2b` diag arms): KLD 0.071569 /
  top-1 90.33% against 0.071762 / 89.70% (−0.27%, inside the noise floor); with MLX's TF32 GEMMs 0.071602 / 89.84%.
