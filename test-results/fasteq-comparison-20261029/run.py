import inspect
import os
import sys
from pathlib import Path
import torch
import cuequivariance_torch as cuet
import pytest

mode, operation, *extra_args = sys.argv[1:]
assert torch.cuda.is_available()
assert os.environ['CUDA_VISIBLE_DEVICES'] == '5'
assert Path(cuet.__file__).resolve().is_relative_to(Path.cwd())
enabled = mode == 'on'
for cls in [cuet.FullyConnectedTensorProduct, cuet.Linear, cuet.ChannelWiseTensorProduct, cuet.SymmetricContraction, cuet.SegmentedPolynomial]:
    init = cls.__init__
    sig = inspect.signature(init)
    params = list(sig.parameters.values())
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if 'use_fasteq' in [p.name for p in positional]:
        defaults = list(init.__defaults__)
        idx = [p.name for p in positional].index('use_fasteq') - (len(positional) - len(defaults))
        defaults[idx] = enabled
        init.__defaults__ = tuple(defaults)
    else:
        init.__kwdefaults__['use_fasteq'] = enabled
    print(cls.__name__, inspect.signature(init), flush=True)
torch.manual_seed(20261029)
print('SOURCE', cuet.__file__, 'GPU', torch.cuda.get_device_name(0), 'MODE', mode, flush=True)
files = dict(fctp='fully_connected', linear='linear', cwtp='channel_wise', stc='symmetric_contraction')
base = Path(__file__).resolve().parent
sys.exit(pytest.main([f'tests/operations/{files[operation]}_test.py', '-ra', '--tb=short', f'--junitxml={base}/{operation}-{mode}.xml'] + extra_args))
