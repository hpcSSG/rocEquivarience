# FastEq GPU 单元测试实测（2026-10-29）

使用物理 GPU 5：NVIDIA H100 80GB HBM3。运行前显存占用 0 MiB、GPU 利用率 0%。`CUDA_VISIBLE_DEVICES=5`，进程内 `cuda:0` 对应物理 GPU 5。Python 3.10.18、PyTorch 2.7.0+cu128、pytest 8.3.4。

测试的是当前工作区（基准提交 e579a1dfd9ee761348dbc75c1903c2a93b7057de，包含用户已有的未提交修改），不是干净的上游版本。没有改动被测源码及测试文件。

## 统计

| 操作 | FastEq | collected | passed | skipped | failed | errors | warnings | pytest 耗时/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FCTP | off | 81 | 29 | 52 | 0 | 0 | 32 | 8.24 |
| FCTP | on | 81 | 29 | 52 | 0 | 0 | 32 | 6.04 |
| Linear | off | 335 | 82 | 253 | 0 | 0 | 74 | 6.45 |
| Linear | on | 335 | 82 | 253 | 0 | 0 | 74 | 9.19 |
| CWTP | off | 74 | 37 | 37 | 0 | 0 | 38 | 12.11 |
| CWTP | on | 74 | 36 | 37 | 1 | 0 | 38 | 6.18 |
| STC | off | 65 | 9 | 56 | 0 | 0 | 1 | 11.31 |
| STC | on | 65 | 9 | 56 | 0 | 0 | 1 | 10.19 |
| 合计 | off | 555 | 157 | 398 | 0 | 0 | 145 | — |
| 合计 | on | 555 | 156 | 398 | 1 | 0 | 145 | — |

无 xfail/xpass、无 deselected（八次主测试）。耗时为单元测试耗时，受编译缓存和运行顺序影响，不能当作性能基准。

## 测试口径与复现

对应 `tests/operations/fully_connected_test.py`、`linear_test.py`、`channel_wise_test.py`、`symmetric_contraction_test.py`，每个文件分别在独立 Python 进程运行 off/on，全量收集，保留所有原有 skip。

`run.py` 在进程内修改四个操作类及 SegmentedPolynomial 的 `use_fasteq` 默认参数，不增加包装函数、不编辑源码。测试显式传入的 `use_fasteq` 仍优先，`use_fallback` 和 method 不变；两组使用相同初始随机种子。未开启额外的 `FASTEQ_INFERENCE` 投影缓存优化。

因此是“切换默认 FastEq 路由后的现有测试文件统计”，并非每个 case 都强制进入同一内核。特别是 FCTP 的 4 个 `test_fctp_sparse_terms_and_gradients` 直接调用 FastEq 内核，5 个 `test_fctp_backend_priority` 显式验证开关优先级，两组都保留这些测试。关闭组并不完全排除 FastEq 代码，也仍需要能导入 FastEq 依赖。

复现全部八组：在仓库根目录运行 `python test-results/fasteq-comparison-20261029/run_all.py`（会覆盖同名日志/XML）。`summarize.py` 从 XML 提取统计，`summary.json` 保存 skip 原因及失败摘要。每组 `.log` 含完整 pytest 输出，`.xml` 含每个 case 结果。

## 为什么统计不同

实际只有 CWTP 的 passed/failed 不同；四类的收集数、skip 数和 errors 都相同。

CWTP 开启组的 `test_channel_wise_indexing` 在 `tests/operations/channel_wise_test.py:289` 失败：`assert_close(Igrad3, grad3)`。这里比较两种数学等价表达的权重一阶梯度：一条使用 `indices_1` 与 `indices_out` 的融合索引/输出累加，另一条显式执行 `x1[indices]` 并用 PyTorch `scatter_add` 聚合输出。测试同时要求计算图可求二阶梯度。

关闭组使用参考实现，两者一致。开启组在 `use_fallback=False` 时进入 `SegmentedPolynomial.forward` 的 CWTP `uniform_1d` 路由，调用外部 `/home/malixian/repos/FastEq/fasteq/ops/uniform1d_jit.py` 的自定义 autograd 内核。此时 32/32 个权重梯度元素不匹配，最大绝对差 282.43816162749556，容差仅 1e-5。这是 FastEq 索引/输出累加反向路径的一致性缺陷，不能解释为一般浮点舍入误差。当前证据定位到这条路径，尚未确定生成内核中具体哪条索引或归约指令错误，也没有修改实现。失败发生在一阶权重梯度断言，不能据此声称所有二阶梯度均正确或仅二阶梯度有问题。

`use_fallback` 不等于 `use_fasteq`：前者选择 naive/fused_tp/uniform_1d 等 method，后者控制对应 method 是否走 FastEq 实现。

其他操作的结果相同有具体代码原因及覆盖范围限制：

- FCTP 只有解析后的 method 为 `fused_tp` 时才启用 FastEq，naive 仍走参考实现；现有导出测试全部被原有条件跳过。
- Linear 的 FastEq 分支在 `segmented_polynomial.py` 中直接 `return self.m(...)`，源码注释说明未导入 `fast_equi_linear`。所以目前开启开关并未执行独立的 FastEq Linear 内核，统计相同不能证明该内核正确。
- STC 的 eager uniform_1d 路由会调用 FastEq，但 TorchScript 分支明确使用 `self.f` 参考后端。因此通过 script 测试不代表 FastEq 内核已经可以被 TorchScript 导出。

## 为什么 skip 很多

这些 skip 是测试文件已有的条件，与 FastEq 开关无关，不是此次为了通过测试新增：

- FCTP：float32/fused_tp 组合跳过 4 个；导出相关共跳过 48 个。
- Linear：普通 FX/组合裁剪跳过 36 个；导出相关裁剪跳过 214 个；math_dtype/method 不支持组合跳过 3 个。
- CWTP：非均匀 irreps 且不 fallback 跳过 12 个；导出参数实际仅包含 batch=1，因此 24 个导出测试全部跳过；独立 bwd_bwd 测试无条件跳过 1 个。
- STC：普通前向测试 16 个全部无条件跳过；导出中的 batch/layout/JIT 条件再跳过 40 个。

warnings 都来自现有 `use_fallback` 弃用提示，不属于 pytest errors。大量 skip 意味着这些结果只说明当前执行到的用例表现，不能推断全部形状、精度和导出模式均受支持。

## 独立复跑确认

在同一 GPU 上单独运行开启组的 `test_channel_wise_indexing`，仍在同一行权重梯度断言失败：1 failed、73 deselected、2 warnings。32/32 个元素不匹配，最大绝对差 338.21507930670475。独立运行与整文件运行消耗随机数的顺序不同，因此数值不同，但失败位置和性质一致。结果见 `cwtp-indexing-recheck.log/xml`，不计入上表八组主测试合计。
