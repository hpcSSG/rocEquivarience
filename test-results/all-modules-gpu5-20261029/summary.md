# layers、operations、primitives GPU 单元测试汇总

日期：2026-10-29。三类测试在同一个 pytest 进程中执行，测试当前工作区已有代码和测试；未修改源码或测试。

环境：物理 5 号 NVIDIA H100 80GB HBM3；CUDA_VISIBLE_DEVICES=5，进程内 cuda:0；Python 3.10.18，PyTorch 2.7.0+cu128，pytest 8.3.4。测试在可访问 GPU 的沙箱外执行。

共 3206 个参数化用例：538 通过、1 失败、2667 跳过、0 error；耗时 231.34 秒（3 分 51 秒），退出码 1。实际执行 539 个用例，通过率 99.81%；跳过占比 83.19%。

## 按模块汇总

| 模块 | 总数 | 通过 | 失败 | 跳过 | 行覆盖率 |
|---|---:|---:|---:|---:|---:|
| layers | 39 | 32 | 0 | 7 | 159/189 (84.13%) |
| operations | 738 | 216 | 1 | 521 | 270/424 (63.68%) |
| primitives | 2429 | 290 | 0 | 2139 | 1033/1588 (65.05%) |
| 合计 | 3206 | 538 | 1 | 2667 | 1462/2201 (66.42%) |

覆盖率说明：为绕过当前环境的 PyTorch 导入错误，在启动 pytest-cov 前预加载 torch 与 cuequivariance_torch，因此不包含导入阶段执行的语句；本次统计为行覆盖率，未启用分支覆盖率。各模块覆盖率包含本轮全部测试对该模块的调用。

## 按测试文件汇总

| 测试文件 | 总数 | 通过 | 失败 | 跳过 |
|---|---:|---:|---:|---:|
| tests.layers.batchnorm_test | 21 | 21 | 0 | 0 |
| tests.layers.tp_conv_fully_connected_test | 18 | 11 | 0 | 7 |
| tests.operations.channel_wise_test | 74 | 36 | 1 | 37 |
| tests.operations.fully_connected_test | 81 | 29 | 0 | 52 |
| tests.operations.linear_test | 335 | 82 | 0 | 253 |
| tests.operations.rotation_test | 7 | 4 | 0 | 3 |
| tests.operations.spherical_harmonics_test | 176 | 56 | 0 | 120 |
| tests.operations.symmetric_contraction_test | 65 | 9 | 0 | 56 |
| tests.primitives.equivariant_tensor_product_test | 214 | 78 | 0 | 136 |
| tests.primitives.primitive_export_test | 17 | 12 | 0 | 5 |
| tests.primitives.segmented_polynomial_test | 1265 | 173 | 0 | 1092 |
| tests.primitives.symmetric_tensor_product_test | 50 | 2 | 0 | 48 |
| tests.primitives.tensor_product_test | 858 | 0 | 0 | 858 |
| tests.primitives.transpose_test | 22 | 22 | 0 | 0 |
| tests.primitives.triangle_test | 3 | 3 | 0 | 0 |

## 失败详情

唯一失败为 `tests/operations/channel_wise_test.py::test_channel_wise_indexing`，与前一轮 GPU 测试相同，索引路径与显式 scatter_add 路径的权重梯度比较失败。当前轮次错误：

```
AssertionError: Tensor-likes are not close!

Mismatched elements: 32 / 32 (100.0%)
Greatest absolute difference: 312.74222534977093 at index (0, 24) (up to 1e-05 allowed)
Greatest relative difference: 35076493.788248554 at index (0, 20) (up to 1e-05 allowed)
```

primitives 本轮无失败。测试代码原有跳过规则保持原样；跳过用例未得到验证。

## 跳过原因

| 原因 | 数量 |
|---|---:|
| Skipping tests with many options that are not default | 1048 |
| Skipping d3 and degree 3 tests for speed - they take 5+ seconds each | 640 |
| Skipping slow JIT compilation test | 262 |
| Skipping script mode for speed | 102 |
| Skipping compile mode for speed - takes 1+ seconds | 78 |
| Skipping use_fallback=False for speed | 72 |
| Skipping redundant layout test with fallback=True | 44 |
| Skipping compile mode for speed - takes 1.5+ seconds | 40 |
| Skipping script mode for speed - takes 1+ seconds | 40 |
| Skipping redundant layout test | 32 |
| Skipping batch=1 test for speed | 32 |
| Skipping compile mode for speed | 26 |
| Skipping fp16/bf16 tests with CUDA backend | 20 |
| Skipping complex descriptors d29-d32 for speed - they take 1.3+ seconds each | 20 |
| Skipping shared_weights=False for speed | 18 |
| Skipping complex irreps combinations for speed | 16 |
| Skipping symmetric contraction test for speed - takes 1.3+ seconds | 16 |
| Linear is not supported for uniform_1d | 16 |
| Skipping batch_size=0 test for speed | 14 |
| This method does not work for non-uniform irreps. | 12 |
| Skipping high degree multi-l tests for speed | 12 |
| Skipping fused TP for float32 math_dtype with float64 inputs | 12 |
| Skipping fp16/bf16 tests for speed | 12 |
| Skipping complex irreps combination for speed | 9 |
| Skipping non-shared weights test for speed | 9 |
| Skipping fp16/bf16 performance tests for speed | 8 |
| indexed_linear does not support math_dtype | 8 |
| Skipping float32 test for fallback=False (fused_tp) | 4 |
| Skipping mixed precision performance tests for speed | 4 |
| Skipping high degrees test for speed - takes 2+ seconds | 4 |
| Skipping fused TP for symmetric contraction: unsupported | 4 |
| Skipping mixed precision tests for speed | 4 |
| Skipping script mode for speed - takes 2+ seconds | 4 |
| Skipping all rotation export tests for speed - they take 10+ seconds each | 3 |
| This is deprecated and will be removed soon. | 3 |
| torch.jit.trace does not work with inline autograd | 3 |
| indexed_linear does not support non-None or CUBLAS math_dtype | 2 |
| ONNX not available! | 2 |
| Skipping float64 tests for speed | 2 |
| Skipping complex descriptor combinations for speed | 2 |
| Skipping fallback=True tests for speed | 2 |
| Skipping slow compile mode with fallback=True | 2 |
| Skipping complex MLP test without batch_norm | 1 |
| Skipping channel_wise_bwd_bwd test for speed - Covered by test_channel_wise_indexing | 1 |
| fused_tp does not support float32 math_dtype with FP64 inputs | 1 |
| Skipping export mode for naive method | 1 |

## 按源码文件的行覆盖率

| 文件 | 命中行 / 总行数 | 覆盖率 |
|---|---:|---:|---:|
| cuequivariance_torch/layers/__init__.py | 0/3 | 0.00% |
| cuequivariance_torch/layers/batchnorm.py | 102/113 | 90.27% |
| cuequivariance_torch/layers/tp_conv_fully_connected.py | 57/73 | 78.08% |
| cuequivariance_torch/operations/linear.py | 44/63 | 69.84% |
| cuequivariance_torch/operations/rotation.py | 62/94 | 65.96% |
| cuequivariance_torch/operations/spherical_harmonics.py | 14/27 | 51.85% |
| cuequivariance_torch/operations/symmetric_contraction.py | 47/87 | 54.02% |
| cuequivariance_torch/operations/tp_channel_wise.py | 58/92 | 63.04% |
| cuequivariance_torch/operations/tp_fully_connected.py | 45/61 | 73.77% |
| cuequivariance_torch/primitives/equivariant_tensor_product.py | 72/114 | 63.16% |
| cuequivariance_torch/primitives/fctp.py | 13/44 | 29.55% |
| cuequivariance_torch/primitives/segmented_polynomial.py | 213/300 | 71.00% |
| cuequivariance_torch/primitives/segmented_polynomial_fused_tp.py | 74/106 | 69.81% |
| cuequivariance_torch/primitives/segmented_polynomial_indexed_linear.py | 107/150 | 71.33% |
| cuequivariance_torch/primitives/segmented_polynomial_naive.py | 147/202 | 72.77% |
| cuequivariance_torch/primitives/segmented_polynomial_uniform_1d.py | 77/101 | 76.24% |
| cuequivariance_torch/primitives/symmetric_tensor_product.py | 97/144 | 67.36% |
| cuequivariance_torch/primitives/tensor_product.py | 175/323 | 54.18% |
| cuequivariance_torch/primitives/transpose.py | 49/74 | 66.22% |
| cuequivariance_torch/primitives/triangle.py | 9/30 | 30.00% |

## 复现命令

```bash
CUDA_VISIBLE_DEVICES=5 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
COVERAGE_FILE=test-results/all-modules-gpu5-20261029/.coverage python - <<'PYTEST'
import torch
import cuequivariance_torch
import pytest
raise SystemExit(pytest.main([
    "-p", "pytest_cov", "tests/layers", "tests/operations", "tests/primitives",
    "--junitxml=test-results/all-modules-gpu5-20261029/junit.xml",
    "--cov=cuequivariance_torch.layers",
    "--cov=cuequivariance_torch.operations",
    "--cov=cuequivariance_torch.primitives",
    "--cov-report=term-missing",
    "--cov-report=json:test-results/all-modules-gpu5-20261029/coverage.json",
    "--cov-report=html:test-results/all-modules-gpu5-20261029/html",
    "--durations=20", "-ra",
]))
PYTEST
```

产物：summary.md、summary.json、pytest.log、junit.xml、coverage.json、html/index.html。
