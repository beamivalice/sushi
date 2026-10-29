#!/usr/bin/env python3
# sushi-test: tier=hermetic
import pathlib
import subprocess
import tempfile


def main():
    suite = pathlib.Path(__file__).with_name('test_mtp_equivalence.sh').read_text()
    check = suite[suite.index('check() {'):suite.index('\necho "── baseline server')]
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        expected = root / 'expected'
        actual = root / 'actual'
        expected.write_text('same prefix the end')
        for gap in ('0.0000', '0.1250', '0.1500', '0.2461', 'none'):
            actual.write_text('same prefix humans end')
            script = '\n'.join([
                'PASS=0; FAIL=0; PREFIX_CHARS=100',
                f'tie_gap_at_divergence() {{ echo {gap}; }}', check,
                'check regression "$1" "$2" no',
                'test "$PASS" = 0 && test "$FAIL" = 1',
            ])
            result = subprocess.run(['bash', '-c', script, 'strict', str(expected), str(actual)], capture_output=True, text=True)
            if result.returncode or gap not in result.stdout:
                raise AssertionError(f'Divergence acquitted or gap hidden at {gap}: {result.stdout!r} {result.stderr!r}')
        expected.write_bytes(b'x' * 100 + b'expected tail')
        actual.write_bytes(b'x' * 100 + b'different tail')
        script = '\n'.join([
            'PASS=0; FAIL=0; PREFIX_CHARS=100',
            'tie_gap_at_divergence() { echo none; }', check,
            'check full-output "$1" "$2" no',
            'test "$PASS" = 0 && test "$FAIL" = 1',
        ])
        result = subprocess.run(['bash', '-c', script, 'strict', str(expected), str(actual)], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(f'Full-output divergence acquitted: {result.stdout!r} {result.stderr!r}')
        expected.write_bytes(b'same prefix')
        actual.write_bytes(b'same prefix\n')
        script = '\n'.join([
            'PASS=0; FAIL=0; PREFIX_CHARS=100',
            'tie_gap_at_divergence() { echo none; }', check,
            'check newline "$1" "$2" no',
            'test "$PASS" = 0 && test "$FAIL" = 1',
        ])
        result = subprocess.run(['bash', '-c', script, 'strict', str(expected), str(actual)], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(f'Trailing newline divergence acquitted: {result.stdout!r} {result.stderr!r}')
        for same in (expected.read_bytes(), b"\n", b"a\x00b"):
            expected.write_bytes(same)
            actual.write_bytes(same)
            script = '\n'.join([
                'PASS=0; FAIL=0; PREFIX_CHARS=100',
                'tie_gap_at_divergence() { echo unexpected-gap-replay; }', check,
                'check regression "$1" "$2" no',
                'test "$PASS" = 1 && test "$FAIL" = 0',
            ])
            result = subprocess.run(['bash', '-c', script, 'strict', str(expected), str(actual)], capture_output=True, text=True)
            if result.returncode or result.stdout or result.stderr:
                raise AssertionError(f'Identical output must pass silently: {result.stdout!r} {result.stderr!r}')


if __name__ == '__main__':
    main()
