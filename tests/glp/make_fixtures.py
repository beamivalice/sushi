#!/usr/bin/env python3
"""Write the synthetic GLP GGUF v3 fixtures `src/glp.zig` embeds.

Stdlib only: the `gguf` package is not a dependency of this repo, and the fixtures must be
reproducible offline. The container is the one `docs/spec/GLP.md` specifies — GGUF v3, a
`glp.*` metadata namespace and `direction.N` F32 tensors — and each file after `valid.gguf`
exercises one refusal path of the reader contract.

Run from the repo root: python3 tests/glp/make_fixtures.py
"""
import hashlib
import os
import struct

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "glp", "fixtures")

# gguf_metadata_value_type
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
# ggml_type
F32, F16 = 0, 1


def s(text):
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def u32(v):
    return struct.pack("<I", v)


def u64(v):
    return struct.pack("<Q", v)


def f32(v):
    return struct.pack("<f", v)


def boolean(v):
    return struct.pack("<B", 1 if v else 0)


def kv_str(key, value):
    return s(key) + u32(STRING) + s(value)


def kv_u32(key, value):
    return s(key) + u32(UINT32) + u32(value)


def kv_f32(key, value):
    return s(key) + u32(FLOAT32) + f32(value)


def kv_bool(key, value):
    return s(key) + u32(BOOL) + boolean(value)


def tensor_info(name, dims, typ, offset):
    out = s(name) + u32(len(dims))
    for d in dims:
        out += u64(d)
    return out + u32(typ) + u64(offset)


def write(path, kvs, tensors, alignment=32):
    """tensors: [(name, dims, ggml_type, payload_bytes)]; offsets are data-section relative."""
    header = b"GGUF" + u32(3) + u64(len(tensors)) + u64(len(kvs))
    for kv in kvs:
        header += kv
    infos = b""
    data = b""
    off = 0
    for name, dims, typ, payload in tensors:
        pad = (-off) % alignment
        data += b"\0" * pad
        off += pad
        infos += tensor_info(name, dims, typ, off)
        data += payload
        off += len(payload)
    body = header + infos
    body += b"\0" * ((-len(body)) % alignment)
    with open(path, "wb") as f:
        f.write(body + data)


def f32s(values):
    return b"".join(f32(v) for v in values)


def base_meta(alpha=4.0, spec=1, mode="project", hook="residual_stream_post_layer", rank=1):
    return [
        kv_u32("general.alignment", 32),
        kv_u32("glp.spec_version", spec),
        kv_str("glp.mode", mode),
        kv_f32("glp.alpha_default", alpha),
        kv_u32("glp.rank", rank),
        kv_bool("glp.orthonormal", True),
        kv_str("glp.hook_point", hook),
        kv_str("glp.derived_at", hook),
        kv_str("glp.base_model", "synthetic/glp-fixture"),
        kv_str("glp.base_revision", "0000000"),
    ]


def content_sha(tensors):
    h = hashlib.sha256()
    for _, _, _, payload in tensors:
        h.update(payload)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    n_embd = 8
    d1 = f32s([3, 4, 0, 0, 0, 0, 0, 0])
    d2 = f32s([0, 0, 1, 0, 0, 0, 0, 0])
    valid_tensors = [
        ("direction.1", [n_embd], F32, d1),
        ("direction.2", [n_embd], F32, d2),
    ]
    valid_meta = base_meta() + [
        kv_str("glp.layer_ids_zero_based", "1,2"),
        kv_str("glp.content_sha256", content_sha(valid_tensors)),
    ]
    write(os.path.join(OUT, "valid.gguf"), valid_meta, valid_tensors)

    # One file per refusal path.
    write(os.path.join(OUT, "mode_add.gguf"), base_meta(mode="add"), valid_tensors)
    write(os.path.join(OUT, "mode_unknown.gguf"), base_meta(mode="projectile"), valid_tensors)
    write(os.path.join(OUT, "spec_v2.gguf"), base_meta(spec=2), valid_tensors)
    write(os.path.join(OUT, "hook_hc.gguf"), base_meta(hook="hc_fold_pre_engram"), valid_tensors)
    write(
        os.path.join(OUT, "rank2.gguf"),
        base_meta(rank=2),
        [
            ("direction.1", [n_embd], F32, d1),
            ("direction.1.1", [n_embd], F32, d2),
        ],
    )
    write(
        os.path.join(OUT, "direction0.gguf"),
        base_meta(),
        [("direction.0", [n_embd], F32, d1)],
    )
    write(
        os.path.join(OUT, "no_glp.gguf"),
        [kv_u32("general.alignment", 32), kv_str("general.name", "legacy")],
        valid_tensors,
    )
    write(
        os.path.join(OUT, "bad_dtype.gguf"),
        base_meta(),
        [("direction.1", [n_embd], F16, b"\0" * (n_embd * 2))],
    )
    write(
        os.path.join(OUT, "bad_shape.gguf"),
        base_meta(),
        [("direction.1", [2, 4], F32, d1)],
    )
    write(
        os.path.join(OUT, "bad_norm.gguf"),
        base_meta(),
        [("direction.1", [n_embd], F32, f32s([0] * n_embd))],
    )
    write(
        os.path.join(OUT, "bad_sha.gguf"),
        base_meta() + [kv_str("glp.content_sha256", "0" * 64)],
        valid_tensors,
    )
    write(
        os.path.join(OUT, "layer_oob.gguf"),
        base_meta(),
        [("direction.9", [n_embd], F32, d1)],
    )
    write(
        os.path.join(OUT, "dir_scales.gguf"),
        base_meta() + [kv_str("glp.dir_scales", "1.0")],
        valid_tensors,
    )


if __name__ == "__main__":
    main()
