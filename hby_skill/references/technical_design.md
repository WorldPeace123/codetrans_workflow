# HBY Operator Skill 技术说明

## 目标

`hby_skill` 把 `workflow.py` 中已经验证过的算子蒸馏闭环抽成可复用技能：给定结构化算子契约，使用 DeepSeek V4 Pro 自动生成 Triton 实现、功能测试和 benchmark，并通过静态校验、运行校验和自动修复把生成结果推进到可落地状态。

它面向两类常见输入：

- OpenBLAS/BLAS 风格算子，例如 AXPY、SCAL、DOT、COPY、GEMV 等。
- Elementwise 算子，例如 add、mul、relu、where、abs 等。

对 OpenBLAS Level-2 还支持 interface-only txt 输入。用户只给 `cblas_sgemv float32`、`gemv: s,d`、`cblas_dgbmv` 这类行时，脚本会用内置 Level-2 catalog 补齐 family、dtype、signature、semantics、edge cases 和 kernel policy。

## 架构

核心执行文件是 `scripts/operator_distiller.py`，主要模块如下：

```text
OperatorSpec       算子契约，描述 API、dtype、语义、边界条件和输出路径
WorkflowConfig     项目根目录、写入目录、DeepSeek 参数和验证参数
TraceLogger        JSONL 事件、模型请求响应、生成文件、失败日志和 repair diff
DeepSeekClient     DeepSeek chat completions JSON 客户端
StaticValidator    静态门禁，写文件前拒绝明显错误
PromptFactory      implementation/test/benchmark/repair 四类 prompt
FileManager        路径解析、原子写入和 __init__.py 导出更新
CommandRunner      py_compile、pytest 和 benchmark smoke 执行器
DistillWorkflow    端到端编排
```

四个技能点在代码中的对应关系：

```text
解析技能      load_operator_txt_specs / load_operator_specs / operator_specs_to_data
生成技能      generate_operator / generate_stage_with_retries
检验技能      validate_existing_operator / run_validations
修复技能      repair_existing_operator / repair_operator
```

OpenBLAS Level-2 catalog 覆盖：

```text
gemv gbmv symv sbmv spmv hemv hbmv hpmv
trmv tbmv tpmv trsv tbsv tpsv
ger geru gerc syr spr syr2 spr2 her hpr her2 hpr2
```

接口名前缀映射：

```text
s=float32, d=float64, c=complex64, z=complex128
```

## 工作流

一次生成分为固定阶段：

1. 读取 txt/YAML/JSON operator request。
2. 如果输入是 OpenBLAS Level-2 compact spec 或 txt，先按接口名和 dtype 分组并展开成完整 `OperatorSpec`。
3. 解析目标路径，确认 operator/test/benchmark 文件都在允许写入根目录内。
4. 调用 DeepSeek V4 Pro 生成 implementation JSON。
5. 对 implementation 做静态校验，通过后原子写入。
6. 调用 DeepSeek 生成 functional test JSON。
7. 对 test 做静态校验，通过后原子写入。
8. 调用 DeepSeek 生成 benchmark JSON。
9. 对 benchmark 做静态校验，通过后原子写入。
10. 更新导出文件。
11. 运行 `py_compile`、pytest 功能测试、benchmark smoke。
12. 如果失败，收集静态报告、编译日志、pytest 日志和 benchmark 日志，交给 DeepSeek repair。
13. repair 返回完整替换文件，再次静态校验、写入和运行验证。

## DeepSeek 协议

默认模型：

```text
deepseek-v4-pro
```

请求使用 JSON 输出协议：

```json
{
  "model": "deepseek-v4-pro",
  "response_format": {"type": "json_object"},
  "thinking": {"type": "disabled"},
  "stream": true
}
```

生产默认建议关闭 thinking 并开启 SSE streaming：

```text
--thinking disabled --stream --temperature 0.0
```

原因是算子生成要求严格 JSON，流式响应可以降低长响应被 HTTP 连接截断的概率。仍然保留 `--thinking enabled --reasoning-effort high|max`，用于需要更强推理的复杂 repair，但这会增加响应长度和等待时间。

implementation/test/benchmark 阶段要求返回：

```json
{
  "path": "relative/output/path.py",
  "content": "complete file content",
  "rationale": {
    "kernel_strategy": "...",
    "dtype_strategy": "...",
    "known_risks": []
  }
}
```

repair 阶段要求返回：

```json
{
  "files": [
    {
      "path": "relative/output/path.py",
      "content": "complete replacement content"
    }
  ],
  "rationale": {
    "root_cause": "...",
    "repair_strategy": "..."
  }
}
```

这种协议确保模型输出可解析、路径可检查、内容可直接写入，并且 rationale 可留档审查。

## 静态门禁

`StaticValidator` 的设计目标是把常见模型失败模式变成硬规则。当前检查包括：

- Python 语法必须可 `ast.parse`。
- implementation 必须包含 spec 声明的所有 public functions。
- implementation 必须包含至少一个 `@triton.jit` kernel。
- implementation 禁止导入 `cupy`、`cublas`、`numpy.ctypeslib`。
- implementation 禁止把 `tensor.data_ptr()` 传入 Triton kernel。
- OpenBLAS Level-2 implementation 必须使用 signature 中出现的 `m/n/lda/incx/incy/uplo/trans/diag/kl/ku/k/beta` 等参数。
- Triton kernel 名不能按 dtype 或 public API 拆分，例如 `saxpy_kernel`、`daxpy_kernel`。
- Triton kernel 签名不应使用 `tl.float32`/`tl.float64` 区分精度。
- Triton kernel 声明 `BLOCK_SIZE: tl.constexpr` 时，launch 必须显式传 `BLOCK_SIZE=`。
- functional test 禁止定义 Triton kernel，必须有 pytest 和断言。
- benchmark 禁止定义 Triton kernel。
- FlagBLAS benchmark 必须使用 `benchmark.performance_utils.Benchmark` 并调用 `bench.run()`。
- benchmark `get_input_iter` 不应 yield `(args_tuple), kwargs`，应 yield 一个 flat tuple，例如 `yield (x,)` 或 `yield (n, x, 1, out, 1)`。

这些门禁不能替代运行测试，但可以在写入前拦截大量明显错误，减少 repair 轮数。

## 工程健壮性设计

### 路径安全

所有模型返回路径都会通过 `ensure_within` 校验：

```text
operator   必须在 operator-root 下
test       必须在 test-root 下
benchmark  必须在 benchmark-root 下
```

写文件使用临时文件加原子替换，避免生成中断留下半文件。

### 超时与重试

DeepSeek 请求有三层保护：

```text
--request-timeout       单次 HTTP 请求超时
--api-retries           retry 次数
--api-retry-base-delay  指数退避基准
```

本地验证命令使用：

```text
--command-timeout
```

批处理脚本额外有：

```text
--distiller-timeout
```

用于避免某个 operator 的 distiller 子进程无限挂起。

### Trace 和状态

单次 distiller run 保存：

```text
ledger.jsonl
state.json
manifest.json
ops/<op>/attempt_*/...
```

批处理 run 保存：

```text
batch_state.json                    跨 run 的 passed/failed 状态
batch_runs/<run_id>/parsed_specs.json
batch_runs/<run_id>/summary.json
```

`batch_state.json` 让十个或更多算子可以断点式推进：已经通过的算子默认跳过，失败算子可 repair 或 regenerate。

### GPU 预检

`scripts/preflight.py` 不参与生成逻辑，但用于生产前检查：

```text
Python 包
DEEPSEEK_API_KEY
写入目录
nvidia-smi
torch.cuda
GPU 空闲显存
可选 DeepSeek 最小 JSON 请求
```

这能区分三类问题：

```text
宿主机没有 GPU
当前沙箱/容器没有透传 GPU
GPU 存在但空闲显存不足
```

## 项目 Profile

支持两个 profile：

```text
flagblas     默认 profile，适配当前 FlagBLAS 风格
standalone   通用 PyTorch/Triton 项目
```

`flagblas` profile 默认：

```text
operator-root: src/flag_blas/ops
test-root: tests
benchmark-root: benchmark
benchmark-style: flagblas
package-import-root: flag_blas.ops
```

如果 spec 不提供 `paths.operator`，默认写入：

```text
src/flag_blas/ops/generated/<module>.py
```

如果要覆盖现有 Level-1 算子，可以在 spec 中显式指定：

```yaml
paths:
  operator: src/flag_blas/ops/level1/axpy.py
  functional_test: tests/test_axpy.py
  benchmark: benchmark/test_axpy_perf.py
```

## 输出标准

生成的 implementation 应满足：

- 一个算子族一个文件。
- public functions 与 spec 完全一致。
- Python wrapper 负责 dtype/device/shape/stride/result 校验。
- Triton kernel 按数学形态拆分，不按 float32/float64 拆分。
- complex pointer arithmetic 使用 real view。
- 线性 pointer arithmetic 读取非连续输入前必须 `.contiguous()`，或显式传入 stride。
- in-place autotune kernel 必须设置 `restore_value` 或避免 autotune。
- 热路径避免 `.cpu()`、`.item()` 和不必要的 device allocation。

生成的 test 应满足：

- 覆盖所有 public functions 和 dtype。
- 覆盖边界条件、stride、in-place 修改范围和 result dtype。
- reference 优先用 torch 或本地纯 Python 实现。
- quick-mode shape 足够小，适合自动 repair。
- 复杂数 sqrt/division 类结果使用 dtype 合理容差；real sign/index 类结果可使用零容差。

生成的 benchmark 应满足：

- baseline 必须做等价计算。
- in-place 算子在 timed wrapper 内 clone 输入。
- 参数传递使用 flat args + kwargs。
- smoke shapes 保守，避免 OOM 或超时。
- 报告 latency，并在适用时报告 GB/s 或 TFLOPS。

## Trace 结构

默认 trace 位于：

```text
hby_operator_trace/runs/<timestamp>/
```

典型内容：

```text
ledger.jsonl
manifest.json
state.json
ops/<op>/attempt_001/implementation_try01_request.json
ops/<op>/attempt_001/implementation_try01_response.json
ops/<op>/attempt_001/generated_files/<module>.py
ops/<op>/attempt_001/combined_static_report.json
ops/<op>/attempt_001/compile.log
ops/<op>/attempt_001/functional.log
ops/<op>/attempt_001/benchmark.log
ops/<op>/attempt_002/repair_request.json
ops/<op>/attempt_002/repair.diff
```

trace 是调试和复现生成质量的核心材料，不应在失败后只看终端最后一行。

## GPU 验证注意事项

FlagBLAS 在 import 阶段会探测设备。如果当前执行环境看不到 GPU，会在 pytest collection 阶段失败，而不是在单个算子测试里失败。

推荐验证顺序：

```bash
python hby_skill/scripts/preflight.py --require-gpu --min-free-gpu-mb 512
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_<module>.py -q
CUDA_VISIBLE_DEVICES=0 python -m pytest benchmark/test_<module>_perf.py -q --mode=kernel --level=core --warmup=1 --iter=1
```

如果 benchmark OOM，但 functional 通过，优先检查 `nvidia-smi` 空闲显存。Triton benchmark 会额外申请 cache buffer，GPU 只剩几百 MiB 时结果会不稳定。

## 与 workflow.py 的对应关系

`operator_distiller.py` 保留了 `workflow.py` 的关键思想：

- `OperatorSpec` 对应 manifest 中的算子契约。
- `PromptFactory` 对应分阶段 prompt。
- `StaticValidator` 对应静态质量门。
- `TraceLogger` 对应可追溯生成。
- `CommandRunner` 对应 py_compile/pytest/benchmark 执行。
- repair loop 对应失败日志驱动自动修复。

差异是 `hby_skill` 不把算子清单写死在 Python 里，而是通过 YAML/JSON spec 输入，因此可以支持 OpenBLAS、Elementwise 和自定义算子族。
