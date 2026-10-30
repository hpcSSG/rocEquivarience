import os
from pathlib import Path
import subprocess
import sys
base = Path(__file__).resolve().parent
env = dict(os.environ, CUDA_VISIBLE_DEVICES='5', PYTHONPATH=str(Path.cwd()))
env.pop('FASTEQ_INFERENCE', None)
for op in ['fctp', 'linear', 'cwtp', 'stc']:
    for mode in ['off', 'on']:
        print(f'START {op} {mode}', flush=True)
        with (base / f'{op}-{mode}.log').open('w') as log:
            result = subprocess.run([sys.executable, str(base / 'run.py'), mode, op], env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f'END {op} {mode} exit={result.returncode}', flush=True)
