# MiMo-V2.6-Flash: KLD of the served packs

MiMo KLD tables, split out of [quality-kld](quality-kld.md); everything shared by the served architectures stays there.


<a id="mimo"></a>
## MiMo (16x512, first EOS, student kv8)

| pack | KLD | top-1 | positions | binary |
|---|---|---|---|---|
| MiMo-V2.6-Flash-Sushi-2.3bpw | 0.0860 | 91.95% | 8067 | 83dc9b6c (v1.1.0 gate) |

Against the 2026-09-30 MOPD teacher. Readings against an earlier teacher capture generate different continuations and
are not comparable.

The FP8-native teacher against the bf16-rounded teacher: 0.0034 nats.

The FP8 trunk's matrix-unit tile ([engine-kernels](engine-kernels.md#prefill-kernels)) serves 9-128 rows, so it never
touches these 295-363-token prompt forwards. Forced onto every prompt forward as a stress arm (binary `01319267`, same
settings, same binary as the off arm): KLD 0.086674 / top-1 91.78% (7404) / NLL 0.360444 against 0.086035 / 91.95%
(7418) / 0.357721 off, +0.74% KLD, inside the rounding-flip floor (the same bf16 operands summed in another order).
