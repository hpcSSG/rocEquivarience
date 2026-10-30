# layers / operations GPU 单元测试统计

日期：2026-10-29；使用物理 5 号 NVIDIA H100 80GB HBM3，CUDA_VISIBLE_DEVICES=5（进程内 cuda:0）。此前 CPU 测试受沙箱设备隔离影响，不能代表机器无 GPU。

环境：Python 3.10.18，PyTorch 2.7.0+cu128，pytest 8.3.4。测试当前工作区，未修改源码或测试。

777 个用例：248 通过、1 失败、528 跳过，0 error；耗时 55.07 秒，退出码 1。已执行 249 个用例，通过率 99.60%。

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

layers：39 个，32 通过、7 跳过；operations：738 个，216 通过、1 失败、521 跳过。

## 失败

`tests/operations/channel_wise_test.py:289`，`test_channel_wise_indexing`：索引路径与显式 scatter_add 路径的权重梯度 Igrad3 / grad3 不一致。32/32 元素不匹配；最大绝对差 294.24541877299635，容差 atol=rtol=1e-5。完整堆栈见 pytest.log。

## 跳过原因

| 原因 | 数量 |
|---|---:|
| Skipping slow JIT compilation test | 152 |
| Skipping use_fallback=False for speed | 72 |
| Skipping redundant layout test with fallback=True | 44 |
| Skipping script mode for speed | 36 |
| Skipping redundant layout test | 32 |
| Skipping batch=1 test for speed | 32 |
| Skipping compile mode for speed | 24 |
| Skipping fp16/bf16 tests with CUDA backend | 20 |
| Skipping shared_weights=False for speed | 18 |
| Skipping complex irreps combinations for speed | 16 |
| Skipping symmetric contraction test for speed - takes 1.3+ seconds | 16 |
| This method does not work for non-uniform irreps. | 12 |
| Skipping compile mode for speed - takes 1+ seconds | 12 |
| Skipping high degree multi-l tests for speed | 12 |
| Skipping complex irreps combination for speed | 9 |
| Skipping non-shared weights test for speed | 9 |
| Skipping float32 test for fallback=False (fused_tp) | 4 |
| Skipping all rotation export tests for speed - they take 10+ seconds each | 3 |
| indexed_linear does not support non-None or CUBLAS math_dtype | 2 |
| Skipping complex MLP test without batch_norm | 1 |
| Skipping channel_wise_bwd_bwd test for speed - Covered by test_channel_wise_indexing | 1 |
| fused_tp does not support float32 math_dtype with FP64 inputs | 1 |

## 行覆盖率

为绕过当前环境的 PyTorch 导入错误，先导入 torch 和 cuequivariance_torch，再启动 pytest-cov；因此未统计导入阶段执行的语句。未启用分支覆盖。

| 文件 | 命中 / 总行数 | 行覆盖率 |
|---|---:|---:|
| cuequivariance_torch/layers/__init__.py | 0/3 | 0.00% |
| cuequivariance_torch/layers/batchnorm.py | 102/113 | 90.27% |
| cuequivariance_torch/layers/tp_conv_fully_connected.py | 57/73 | 78.08% |
| cuequivariance_torch/operations/linear.py | 44/63 | 69.84% |
| cuequivariance_torch/operations/rotation.py | 62/94 | 65.96% |
| cuequivariance_torch/operations/spherical_harmonics.py | 14/27 | 51.85% |
| cuequivariance_torch/operations/symmetric_contraction.py | 47/87 | 54.02% |
| cuequivariance_torch/operations/tp_channel_wise.py | 58/92 | 63.04% |
| cuequivariance_torch/operations/tp_fully_connected.py | 45/61 | 73.77% |
| 合计 | 429/613 | 69.98% |

运行设置：`CUDA_VISIBLE_DEVICES=5 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`；Python 先导入 torch、cuequivariance_torch，再调用 `pytest.main`；参数为 `-p pytest_cov tests/layers tests/operations --cov=cuequivariance_torch.layers --cov=cuequivariance_torch.operations --durations=15 -ra`，并输出 JUnit XML、coverage JSON 和 HTML。

产物：junit.xml、pytest.log、coverage.json、html/index.html；单例复跑见 recheck.log。

单独复跑：同一 5 号 GPU 上仍失败（1 failed，1.65 秒），确认可复现；见 recheck.log。
