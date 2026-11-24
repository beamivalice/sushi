#!/usr/bin/env python3
# sushi-test: tier=tool
"""Build the DeepSeek-V4.1 KLD prompt files (token ids) from the Flash-Next prompt sets.

  python3 -I tests/build_dsv41_kld_prompts.py --checkpoint <DeepSeek-V4.1 dir> \
      --raw16 <jsonl of {id,prompt}> --cat8 <jsonl of the 8-category set> \
      --out-raw16 raw16.jsonl --out-8cat 8cat.jsonl

Every prompt is `prompt_ids`, so `sushi kld capture` needs no template flag. The checkpoint's tokenizer.json (HF
`tokenizers`) is the oracle `src/tokenizer_deepseek.zig` is pinned to, and its own encoding/encoding.py renders the chat
formats. DeepSeek's tokenizer adds no BOS, and the model's template always opens with one, so a raw prompt is BOS + text.
"""
import argparse
import importlib.util
import json
import os

from tokenizers import Tokenizer

THINK_EFFORT = "high"  # what sushi renders for a thinking request that names no effort


def load_encoder(checkpoint):
    path = os.path.join(checkpoint, "encoding", "encoding.py")
    spec = importlib.util.spec_from_file_location("dsv41_encoding", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render(enc, mode, text):
    if mode == "raw":
        return enc.bos_token + text
    messages = [{"role": "user", "content": text}]
    if mode == "think":
        return enc.encode_messages(messages, thinking_mode="thinking", reasoning_effort=THINK_EFFORT)
    if mode == "nothink":
        return enc.encode_messages(messages, thinking_mode="chat")
    raise SystemExit(f"unknown mode {mode!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--raw16", required=True)
    ap.add_argument("--cat8", required=True)
    ap.add_argument("--out-raw16", required=True)
    ap.add_argument("--out-8cat", required=True)
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(args.checkpoint, "tokenizer.json"))
    enc = load_encoder(args.checkpoint)
    bos = tok.token_to_id(enc.bos_token)

    def ids_of(rendered):
        ids = tok.encode(rendered, add_special_tokens=False).ids
        if ids[0] != bos:
            raise SystemExit("rendered prompt does not open with BOS")
        return ids

    n = 0
    with open(args.raw16) as f, open(args.out_raw16, "w") as out:
        for line in f:
            d = json.loads(line)
            if not d["id"].startswith("wikitext2-test-"):
                continue
            rendered = render(enc, "raw", d["prompt"])
            out.write(json.dumps({"id": d["id"], "prompt_ids": ids_of(rendered)}) + "\n")
            n += 1
    if n != 16:
        raise SystemExit(f"expected 16 raw prompts, got {n}")

    n = 0
    with open(args.cat8) as f, open(args.out_8cat, "w") as out:
        for line in f:
            d = json.loads(line)
            rendered = render(enc, d["mode"], d["text"])
            row = {k: v for k, v in d.items() if k not in ("rendered", "prompt_ids")}
            row["rendered"] = rendered
            row["prompt_ids"] = ids_of(rendered)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    if n != 16:
        raise SystemExit(f"expected 16 category prompts, got {n}")


if __name__ == "__main__":
    main()
