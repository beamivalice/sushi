# sushi-test: tier=tool
"""Capture small deterministic GLM layer oracles; no model checkpoint is required.

Run with an oMLX environment: python tests/glm5/dump_glm5_layer_fixture.py --omlx REPO.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

os.environ.setdefault('MLX_ENABLE_TF32', '0')
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def reference(repo):
    sys.path.insert(0, str(repo))
    for name in ('mlx_vlm.models.fast_ops', 'mlx_vlm.models.linear'):
        if importlib.util.find_spec(name) is None:
            stub = ModuleType(name)
            if name.endswith('linear'):
                stub.DECODE_BLOCK_SIZE = 8
            else:
                def exact_hc_norm(connection, norm, x, mixes):
                    collapsed, post, comb = connection(x)
                    return norm(collapsed), post, comb
                stub.exact_hc_norm = exact_hc_norm
            sys.modules[name] = stub
    from omlx.patches.mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch
    assert apply_mlx_vlm_glm5_next_compat_patch()
    import mlx_vlm.models.glm5_next.language as language
    language._DECODE_FUSION = False
    return language


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--omlx', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('src/fixtures/glm5_layers.safetensors'))
    args = parser.parse_args()
    language = reference(args.omlx.resolve())
    rng = np.random.default_rng(53128)
    values = {}
    def rand(shape, scale=.06, dtype=mx.bfloat16):
        return mx.array(rng.normal(0, scale, shape).astype(np.float32)).astype(dtype)
    cfg = SimpleNamespace(hidden_size=128, linear_num_heads=1, linear_head_dim=128,
                          linear_conv_kernel_dim=4, linear_lower_bound=-5., rms_norm_eps=1e-5,
                          hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)
    layer = language.Glm5NextLinearAttention(cfg)
    layer.fuse_in = False
    layer.eval()
    for name in ('q_proj', 'k_proj', 'v_proj', 'f_a_proj', 'f_b_proj', 'g_a_proj', 'g_b_proj', 'b_proj', 'o_proj'):
        owner = layer.forget_gate if name.startswith('f_') else layer
        linear = getattr(owner, name)
        linear.weight = rand(linear.weight.shape)
        values[f'a.{name}.weight'] = linear.weight
    convs = [rand((128, 1, 4), .2) for _ in range(3)]
    for name, w in zip(('q', 'k', 'v'), convs):
        values[f'a.{name}_conv1d.weight'] = w
    layer.conv1d.weight = mx.concatenate(convs, axis=0).transpose(0, 2, 1)
    layer.forget_gate.A_log = rand((1,), .3, mx.float32)
    layer.forget_gate.dt_bias = rand((128,), .2, mx.float32)
    layer.o_norm.weight = (1 + rand((128,), .1)).astype(mx.bfloat16)
    values.update({'a.A_log': layer.forget_gate.A_log, 'a.dt_bias': layer.forget_gate.dt_bias,
                   'a.o_norm.weight': layer.o_norm.weight})
    values['input'] = rand((2, 5, 128), .7)
    values['initial.conv'] = rand((2, 3, 384), .2)
    values['initial.state'] = rand((2, 1, 128, 128), .003, mx.float32)
    class Cache:
        lengths = None
        def __init__(self): self.values = [values['initial.conv'], values['initial.state']]
        def __getitem__(self, i): return self.values[i]
        def __setitem__(self, i, x): self.values[i] = x
        def advance(self, n): pass
    for label, chunks in [('full', [5]), ('serial', [1]*5), ('irregular', [2, 1, 2]), ('cold', [5])]:
        cache, outputs, start = Cache(), [], 0
        if label == 'cold': cache.values = [None, None]
        for n in chunks:
            outputs.append(layer(values['input'][:, start:start+n], cache=cache))
            mx.eval(outputs[-1], cache[0], cache[1])
            start += n
        values[f'{label}.output'] = mx.concatenate(outputs, axis=1)
        values[f'{label}.conv'] = cache[0]
        values[f'{label}.state'] = cache[1]
    hc = language.HyperConnection(cfg)
    hc.eval()
    hc.fn = rand((24, 512), .025, mx.bfloat16)
    hc.base = rand((24,), .2, mx.float32)
    hc.scale = mx.array([.7, 1.2, .4], dtype=mx.float32)
    values.update({'h.hc_attn_fn': hc.fn, 'h.hc_attn_base': hc.base, 'h.hc_attn_scale': hc.scale})
    values['hc.input'] = rand((2, 5, 4, 128), .7)
    mixed, post, comb = hc(values['hc.input'])
    values.update({'hc.mixed': mixed, 'hc.post': post, 'hc.comb': comb,
                   'hc.expanded': language.hc_expand(mixed, values['hc.input'], post, comb)})
    sx = (mx.arange(-1000, 1000, dtype=mx.float32) / 100).astype(mx.bfloat16)
    up = mx.sin(sx).astype(mx.bfloat16)
    values.update({'silu.input': sx, 'silu.up': up, 'silu.conv': nn.silu(sx),
                   'silu.dense': language._clamped_swiglu(up, sx, 10.)})
    mx.eval(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), values)
    files = {str(Path(inspect.getfile(x)).relative_to(args.omlx.resolve())): hashlib.sha256(Path(inspect.getfile(x)).read_bytes()).hexdigest()
             for x in (language.Glm5NextLinearAttention, language.gated_delta_update, language._HyperConnection)}
    manifest = {'omlx_commit': subprocess.check_output(['git', '-C', str(args.omlx), 'rev-parse', 'HEAD'], text=True).strip(),
                'source_hashes': files, 'seed': 53128, 'geometry': {'batch': 2, 'tokens': 5, 'hidden': 128, 'heads': 1, 'head_dim': 128},
                'dtype': 'BF16 weights and activations; FP32 decay and recurrent state',
                'settings': {'decode_fusion': False, 'fuse_in': False, 'tf32': False},
                'fixture_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'bytes': args.output.stat().st_size, 'omlx_commit': manifest['omlx_commit']}))

if __name__ == '__main__':
    main()
