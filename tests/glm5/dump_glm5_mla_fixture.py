"""Capture tiny complete BF16 MLA source outputs and an absorbed-rounding comparison.

This is a reference fixture utility. It does not load or convert a checkpoint.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from dump_glm5_layer_fixture import reference
import mlx.core as mx
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--omlx', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('src/fixtures/glm5_mla.safetensors'))
    args = ap.parse_args()
    language = reference(args.omlx.resolve())
    from mlx_lm.models.cache import PoolingCache
    cfg = SimpleNamespace(hidden_size=128, num_attention_heads=2, q_lora_rank=128,
                          qk_rope_head_dim=0, kv_lora_rank=512, v_head_dim=256,
                          qk_nope_head_dim=256, mla_use_nope=True, attention_bias=False,
                          rms_norm_eps=1e-5, index_n_heads=2, index_head_dim=32,
                          index_topk=2048, index_kpool=4, index_kpool_always_select_tail=True)
    layer = language.Glm5NextSparseAttention(cfg)
    layer.eval()
    rng = np.random.default_rng(53256)
    values = {}
    def rand(shape, scale=.035):
        return mx.array(rng.normal(0, scale, shape).astype(np.float32)).astype(mx.bfloat16)
    for path in ('q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'o_proj',
                 'indexer.wq_b', 'indexer.wk', 'indexer.weights_proj'):
        owner = layer
        for field in path.split('.')[:-1]: owner = getattr(owner, field)
        linear = getattr(owner, path.split('.')[-1])
        linear.weight = rand(linear.weight.shape)
        values[f'm.{path}.weight'] = linear.weight
    for path in ('q_a_layernorm', 'kv_a_layernorm', 'indexer.k_norm'):
        owner = layer
        for field in path.split('.')[:-1]: owner = getattr(owner, field)
        norm = getattr(owner, path.split('.')[-1])
        norm.weight = (1 + rand(norm.weight.shape, .05)).astype(mx.bfloat16)
        values[f'm.{path}.weight'] = norm.weight
        if hasattr(norm, 'bias'):
            norm.bias = rand(norm.bias.shape, .02)
            values[f'm.{path}.bias'] = norm.bias
    for path in ('index_kpool_compress_gate', 'index_kpool_compress_ape'):
        value = rand(getattr(layer.indexer, path).shape)
        setattr(layer.indexer, path, value)
        values[f'm.indexer.{path}'] = value
    bank = rand((2, 512, 512), .025)
    values['m.kv_b_proj.weight'] = bank.reshape(1024, 512)
    layer.embed_q.weight = mx.contiguous(bank[:, :256].transpose(0, 2, 1))
    layer.unembed_out.weight = mx.contiguous(bank[:, 256:])
    values['input'] = rand((1, 33, 128), .7)
    for label, chunks in [('full', [33]), ('irregular', [17, 1, 15]), ('serial', [1]*33)]:
        cache = language.CacheList(language.KVCache(), PoolingCache(4))
        outputs, start = [], 0
        for n in chunks:
            mask = None if n == 1 else (mx.arange(start+n)[None, None, None, :] <= (start+mx.arange(n))[None, None, :, None])
            outputs.append(layer(values['input'][:, start:start+n], mask=mask, cache=cache))
            mx.eval(outputs[-1])
            start += n
        values[f'{label}.output'] = mx.concatenate(outputs, axis=1)
    # Prime only the source projection/pooling cache, avoiding quadratic
    # prefix attention. Decode then crosses the 512-pool selection boundary.
    prefix_input = rand((1, 2047, 128), .7)
    prefix_latent = layer.kv_a_layernorm(layer.kv_a_proj_with_mqa(prefix_input))
    prefix_keys = layer.indexer.k_norm(layer.indexer.wk(prefix_input))
    prefix_gates = prefix_input @ layer.indexer.index_kpool_compress_gate.T
    values.update({'boundary.latent': prefix_latent[0], 'boundary.keys': prefix_keys[0],
                   'boundary.gates': prefix_gates[0], 'boundary.input': rand((1, 6, 128), .7)})
    for label, chunks in [('boundary.serial', [1]*6), ('boundary.chunk', [3, 3])]:
        cache = language.CacheList(language.KVCache(), PoolingCache(4))
        cache[0].update_and_fetch(prefix_latent[:, None], mx.zeros((1, 1, 2047, 0), dtype=mx.bfloat16))
        keys, gates, _ = cache[1].accumulate_windows(prefix_keys, prefix_gates, 0)
        cache[1].update_and_fetch(layer.indexer._compress_windows(keys, gates))
        outputs, start = [], 0
        for n in chunks:
            offset = 2047 + start
            mask = None if n == 1 else (mx.arange(offset+n)[None, None, None, :] <= (offset+mx.arange(n))[None, None, :, None])
            outputs.append(layer(values['boundary.input'][:, start:start+n], mask=mask, cache=cache))
            mx.eval(outputs[-1])
            start += n
        values[f'{label}.output'] = mx.concatenate(outputs, axis=1)
    values['boundary.prefill.input'] = rand((1, 17, 128), .7)
    for label, past in [('boundary.dense', 2034), ('boundary.fallback', 2035)]:
        cache = language.CacheList(language.KVCache(), PoolingCache(4))
        cache[0].update_and_fetch(prefix_latent[:, None, :past], mx.zeros((1, 1, past, 0), dtype=mx.bfloat16))
        keys, gates, _ = cache[1].accumulate_windows(prefix_keys[:, :past], prefix_gates[:, :past], 0)
        cache[1].update_and_fetch(layer.indexer._compress_windows(keys, gates))
        mask = mx.arange(past+17)[None, None, None, :] <= (past+mx.arange(17))[None, None, :, None]
        values[f'{label}.output'] = layer(values['boundary.prefill.input'], mask=mask, cache=cache)
        mx.eval(values[f'{label}.output'])
    # Same source weights and operation primitives; only reassociate attention
    # into latent space, matching the native path's BF16 rounding boundaries.
    x = values['input']
    qr = layer.q_a_layernorm(layer.q_a_proj(x))
    q = layer.q_b_proj(qr).reshape(1, 33, 2, 256).transpose(0, 2, 1, 3)
    kv = layer.kv_a_layernorm(layer.kv_a_proj_with_mqa(x))[:, None]
    q_latent = layer.embed_q(q)
    mask = mx.arange(33)[None, None, None, :] <= mx.arange(33)[None, None, :, None]
    y = mx.fast.scaled_dot_product_attention(q_latent, kv, kv, scale=1/16, mask=mask)
    y = layer.unembed_out(y).transpose(0, 2, 1, 3).reshape(1, 33, 512)
    values['absorbed.output'] = layer.o_proj(y)
    values['latent'] = kv.reshape(33, 512)
    mx.eval(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), values)
    reference_output = values['full.output'].astype(mx.float32)
    delta = values['absorbed.output'].astype(mx.float32) - reference_output
    relative_l2 = mx.sqrt(mx.sum(delta*delta)/mx.sum(reference_output*reference_output)).item()
    source_hashes = {str(Path(inspect.getfile(t)).relative_to(args.omlx.resolve())):
                     hashlib.sha256(Path(inspect.getfile(t)).read_bytes()).hexdigest()
                     for t in (language.Glm5NextSparseAttention, type(layer.embed_q), PoolingCache)}
    manifest = {'source_hashes': source_hashes, 'omlx_commit': subprocess.check_output(['git', '-C', str(args.omlx), 'rev-parse', 'HEAD'], text=True).strip(),
                'seed': 53256, 'geometry': vars(cfg), 'dtype': 'BF16',
                'settings': {'decode_fusion': False, 'tf32': False},
                'absorbed_vs_expanded': {'max_abs': mx.max(mx.abs(delta)).item(), 'relative_l2': relative_l2},
                'fixture_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['absorbed_vs_expanded']))

if __name__ == '__main__':
    main()
