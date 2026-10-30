# layers / operations 单元测试统计

日期：2026-10-29。测试当前工作区（包含运行前已有的未提交改动），未修改源码或测试。

环境：Python 3.10.18，PyTorch 2.7.0+cu128，pytest 8.3.4；CUDA 不可用，GPU 数量为 0，nvidia-smi 无法连接驱动。

共 697 个参数化用例：112 通过、25 失败、560 跳过，0 error；pytest 执行耗时 35.10 秒，退出码 1。实际执行 137 个，通过率 81.75%；跳过比例 80.34%。

| 测试文件 | 总数 | 通过 | 失败 | 跳过 |
|---|---:|---:|---:|---:|
| tests.layers.batchnorm_test | 21 | 21 | 0 | 0 |
| tests.layers.tp_conv_fully_connected_test | 18 | 11 | 0 | 7 |
| tests.operations.channel_wise_test | 74 | 0 | 24 | 50 |
| tests.operations.fully_connected_test | 81 | 21 | 0 | 60 |
| tests.operations.linear_test | 335 | 38 | 0 | 297 |
| tests.operations.rotation_test | 7 | 2 | 1 | 4 |
| tests.operations.spherical_harmonics_test | 96 | 18 | 0 | 78 |
| tests.operations.symmetric_contraction_test | 65 | 1 | 0 | 64 |

layers 合计：39 个，32 通过、0 失败、7 跳过。operations 合计：658 个，80 通过、25 失败、553 跳过。

## 失败原因

- channel_wise：24 个 CPU fallback 前向用例在 `cuequivariance_torch/operations/tp_channel_wise.py:268` 无条件调用 `torch.cuda.synchronize()`，报 `RuntimeError: No CUDA GPUs are available`。
- rotation：`test_vector_to_euler_angles` 的调用链进入 CUDA stream，报同一错误；详见 pytest.log 的完整堆栈。
- 当前结果不能证明 GPU 路径的正确性。

## 跳过原因

| 原因 | 数量 |
|---|---:|
| CUDA is not available | 464 |
| Skipping slow JIT compilation test | 22 |
| Skipping symmetric contraction test for speed - takes 1.3+ seconds | 16 |
| Skipping script mode for speed | 12 |
| Skipping batch=1 test for speed | 8 |
| Skipping redundant layout test with fallback=True | 8 |
| Skipping compile mode for speed | 8 |
| Skipping redundant layout test | 6 |
| Skipping compile mode for speed - takes 1+ seconds | 4 |
| Skipping high degree multi-l tests for speed | 4 |
| Skipping all rotation export tests for speed - they take 10+ seconds each | 3 |
| indexed_linear does not support non-None or CUBLAS math_dtype | 2 |
| Skipping complex MLP test without batch_norm | 1 |
| Skipping channel_wise_bwd_bwd test for speed - Covered by test_channel_wise_indexing | 1 |
| fused_tp does not support float32 math_dtype with FP64 inputs | 1 |

## 行覆盖率

覆盖率在预先导入 torch 和 cuequivariance_torch 后开始采集，因此不包含导入阶段执行的语句；这是本次运行的有限覆盖统计。未采集分支覆盖率。

| 模块 | 命中行 / 可执行行 | 覆盖率 |
|---|---:|---:|
| cuequivariance_torch/layers/__init__.py | 0 / 3 | 0.00% |
| cuequivariance_torch/layers/batchnorm.py | 102 / 113 | 90.27% |
| cuequivariance_torch/layers/tp_conv_fully_connected.py | 57 / 73 | 78.08% |
| cuequivariance_torch/operations/linear.py | 28 / 63 | 44.44% |
| cuequivariance_torch/operations/rotation.py | 62 / 94 | 65.96% |
| cuequivariance_torch/operations/spherical_harmonics.py | 14 / 27 | 51.85% |
| cuequivariance_torch/operations/symmetric_contraction.py | 41 / 87 | 47.13% |
| cuequivariance_torch/operations/tp_channel_wise.py | 45 / 92 | 48.91% |
| cuequivariance_torch/operations/tp_fully_connected.py | 44 / 61 | 72.13% |
| 合计 | 393 / 613 | 64.11% |

## 复现方式

直接 pytest 在导入 torch 时出现重复 docstring 错误；仅预加载 torch 后启用覆盖率又在原生扩展导入时 abort。最终预加载两个模块后完成测试，并关闭无关插件自动加载。启动崩溃日志见 coverage-startup-error.log。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python - <<'PYTEST'
import torch
import cuequivariance_torch
import pytest
raise SystemExit(pytest.main([
    "-p", "pytest_cov", "tests/layers", "tests/operations",
    "--junitxml=test-results/layers-operations-20261029/junit.xml",
    "--cov=cuequivariance_torch.layers", "--cov=cuequivariance_torch.operations",
    "--cov-report=term-missing",
    "--cov-report=json:test-results/layers-operations-20261029/coverage.json",
    "--cov-report=html:test-results/layers-operations-20261029/html",
    "--durations=15", "-ra",
]))
PYTEST
```

产物：pytest.log、junit.xml、coverage.json、html/index.html。
