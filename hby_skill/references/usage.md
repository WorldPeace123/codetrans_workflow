# HBY Operator Skill 使用文档

本文档面向日常生产使用：从一个 txt/YAML/JSON 算子清单出发，自动生成 Triton 实现、pytest 功能测试和 benchmark，并通过验证和修复闭环收敛。

## 目录结构

```text
hby_skill/
  SKILL.md
  scripts/
    operator_distiller.py      核心四技能流水线
    run_operator_batch.py      多算子一键批处理入口
    preflight.py               环境/GPU/API/路径预检
  assets/examples/
    openblas_level2_interfaces.txt
    openblas_level2_ten_interfaces.yaml
    elementwise_sign.yaml
  references/
    usage.md
    technical_design.md
    pipeline_architecture.md
    operator_spec_schema.md
```

## 环境准备

在目标项目根目录运行命令。最小依赖：

```text
Python 3.9+
torch
triton
pytest
PyYAML
DeepSeek API Key
```

设置 API Key：

```bash
export DEEPSEEK_API_KEY=<your_key>
```

默认模型是：

```text
deepseek-v4-pro
```

CLI 也接受 `deepseekV4pro`，会自动归一到 `deepseek-v4-pro`。

## 环境预检

先检查 Python 包、写入目录、API Key、GPU 和空闲显存：

```bash
python hby_skill/scripts/preflight.py \
  --require-gpu \
  --min-free-gpu-mb 512
```

只检查本地环境，不检查 API 连通性：

```bash
python hby_skill/scripts/preflight.py --skip-api-key
```

连同 DeepSeek 最小 JSON 请求一起检查：

```bash
python hby_skill/scripts/preflight.py \
  --require-gpu \
  --min-free-gpu-mb 512 \
  --check-deepseek
```

机器可读输出：

```bash
python hby_skill/scripts/preflight.py --json
```

### GPU 与沙箱说明

Codex 默认命令沙箱可能看不到 `/dev/nvidia*`，即使宿主机有 GPU。表现通常是：

```text
nvidia-smi failed because it could not communicate with the NVIDIA driver
torch.cuda.is_available() == False
FlagBLAS: No device were detected on your machine
```

这表示当前执行环境没有透传 GPU，不等于宿主机没有 GPU。需要在能看到 GPU 的环境中运行验证命令，或显式用可见 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_<module>.py -q
```

benchmark smoke 还需要额外空闲显存。Triton benchmark 可能分配 256MiB cache buffer；如果所有 GPU 都接近满载，功能测试可能通过，但 benchmark 或 full validate 会 OOM。先查空闲显存：

```bash
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader,nounits
```

## 推荐 DeepSeek 参数

结构化代码生成建议使用：

```text
--thinking disabled
--stream
--temperature 0.0
--max-tokens 8192 到 32768
```

原因：

- `--stream` 能避免长响应在非流式 HTTP 读取中断。
- `--thinking disabled` 更适合严格 JSON 输出。
- `--temperature 0.0` 便于复现和 repair。
- `--max-tokens 8192` 适合较小算子；复杂 BLAS family 可放宽到 `32768`。

## 一键批处理

日常推荐入口是 `run_operator_batch.py`。它会解析一次 spec，然后按算子逐个执行：

```text
已通过 -> 跳过
缺失/失败 -> 生成 -> 验证
验证失败 -> repair/regenerate/skip/stop
```

十个 OpenBLAS Level-2 family 一键运行：

```bash
python hby_skill/scripts/run_operator_batch.py \
  --spec hby_skill/assets/examples/openblas_level2_interfaces.txt \
  --all \
  --on-fail ask \
  --thinking disabled \
  --stream \
  --max-tokens 8192 \
  --max-repair-attempts 3 \
  --stage-retries 3 \
  --distiller-timeout 1800
```

非交互环境直接自动修复：

```bash
python hby_skill/scripts/run_operator_batch.py \
  --spec hby_skill/assets/examples/openblas_level2_interfaces.txt \
  --all \
  --on-fail repair \
  --thinking disabled \
  --stream
```

指定 GPU 运行验证：

```bash
python hby_skill/scripts/run_operator_batch.py \
  --spec hby_skill/assets/examples/openblas_level2_interfaces.txt \
  --all \
  --cuda-visible-devices 0 \
  --on-fail repair \
  --thinking disabled \
  --stream
```

只处理某几个接口或 family：

```bash
python hby_skill/scripts/run_operator_batch.py \
  --spec hby_skill/assets/examples/openblas_level2_interfaces.txt \
  --ops cblas_sgemv,dgbmv,trmv \
  --on-fail ask \
  --thinking disabled \
  --stream
```

失败时重新生成，不走 repair：

```bash
python hby_skill/scripts/run_operator_batch.py \
  --spec hby_skill/assets/examples/openblas_level2_interfaces.txt \
  --all \
  --on-fail regenerate \
  --thinking disabled \
  --stream
```

演练解析和 prompt 生成，不调用 DeepSeek：

```bash
python hby_skill/scripts/run_operator_batch.py \
  --spec hby_skill/assets/examples/openblas_level2_interfaces.txt \
  --ops gemv \
  --dry-run
```

批处理输出：

```text
<trace-root>/batch_runs/<run_id>/parsed_specs.json
<trace-root>/batch_runs/<run_id>/summary.json
<trace-root>/batch_state.json
<trace-root>/batch_runs/<run_id>/operator_trace/<op>_<action>_*/runs/<run_id>/
```

`batch_state.json` 用于跳过已经通过的算子；如果要强制重跑：

```bash
python hby_skill/scripts/run_operator_batch.py \
  --spec <spec> \
  --all \
  --force-passed \
  --force-generate
```

## 四技能分阶段运行

### 1. 解析技能

从 txt/YAML/JSON 解析成规范化 full spec：

```bash
python hby_skill/scripts/operator_distiller.py \
  --spec hby_skill/assets/examples/openblas_level2_interfaces.txt \
  --parse-only \
  --parsed-spec-out /tmp/hby_parsed_specs.json
```

检查输出：

```bash
python -m json.tool /tmp/hby_parsed_specs.json
```

### 2. 生成技能

只生成 implementation/test/benchmark，不运行测试：

```bash
python hby_skill/scripts/operator_distiller.py \
  --spec /tmp/hby_parsed_specs.json \
  --ops gemv,gbmv \
  --generate-only \
  --thinking disabled \
  --stream \
  --stage-retries 3
```

### 3. 检验技能

只验证已有文件：

```bash
CUDA_VISIBLE_DEVICES=0 python hby_skill/scripts/operator_distiller.py \
  --spec /tmp/hby_parsed_specs.json \
  --ops gemv,gbmv \
  --validate-only
```

只跑静态和编译，不跑 GPU pytest/benchmark：

```bash
python hby_skill/scripts/operator_distiller.py \
  --spec /tmp/hby_parsed_specs.json \
  --ops gemv,gbmv \
  --validate-only \
  --skip-tests \
  --skip-benchmark
```

### 4. 修复技能

根据验证日志和当前文件自动修复：

```bash
CUDA_VISIBLE_DEVICES=0 python hby_skill/scripts/operator_distiller.py \
  --spec /tmp/hby_parsed_specs.json \
  --ops gemv,gbmv \
  --repair-only \
  --thinking disabled \
  --stream \
  --max-repair-attempts 3
```

## 端到端单算子示例

生成并验证 `sign/dsign/csign/zsign`：

```bash
python hby_skill/scripts/run_operator_batch.py \
  --spec hby_skill/assets/examples/elementwise_sign.yaml \
  --ops sign \
  --on-fail repair \
  --thinking disabled \
  --stream \
  --max-tokens 8192 \
  --cuda-visible-devices 0
```

单独跑生成产物测试：

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_sign.py -q
CUDA_VISIBLE_DEVICES=0 python -m pytest benchmark/test_sign_perf.py -q --mode=kernel --level=core --warmup=1 --iter=1
```

## Spec 编写步骤

1. 确定算子类别：OpenBLAS/BLAS、Elementwise、Reduction、Data movement 或自定义。
2. 写清楚 public functions。BLAS dtype 变体应放在同一个 module 中，例如 `saxpy/daxpy/caxpy/zaxpy`。
3. 写清楚 dtype、signature、semantics、reference。
4. 把边界条件写到 `edge_cases`。
5. 把 kernel 设计要求写到 `kernel_policy`，尤其是是否允许 real/complex 分 kernel、是否禁止 dtype-specific kernels。
6. 如果要覆盖现有文件，写 `paths`；否则默认写到 `src/flag_blas/ops/generated/<module>.py`。

Elementwise 示例：

```yaml
operators:
  - name: sign
    module: sign
    kind: elementwise
    public_functions: [sign, dsign, csign, zsign]
    dtypes: [float32, float64, complex64, complex128]
    signature: "(x, out=None) -> torch.Tensor"
    semantics: "real variants match torch.sign; complex variants match torch.sgn"
    reference: "torch.sign and torch.sgn"
    kernel_policy: "one generic real kernel and one generic complex real-view kernel"
```

OpenBLAS Level-2 txt 示例：

```text
kind: openblas_level2
api_style: cblas
min_precisions: 2

cblas_sgemv float32
cblas_dgemv float64
cblas_sgbmv float32,float64
symv: s,d
trmv fp32 fp64
```

完整 schema 见 `operator_spec_schema.md`。

## 输出文件

默认输出：

```text
src/flag_blas/ops/generated/<module>.py
tests/test_<module>.py
benchmark/test_<module>_perf.py
```

OpenBLAS Level-2 compact spec 默认写到：

```text
src/flag_blas/ops/level2/<family>.py
tests/test_<family>.py
benchmark/test_<family>_perf.py
```

路径安全规则：

```text
paths.operator         必须在 --operator-root 下
paths.functional_test  必须在 --test-root 下
paths.benchmark        必须在 --benchmark-root 下
```

写入方式：

- 原子替换。
- trace 中保存 diff。
- 默认更新同目录 `__init__.py` 的自动导出块。
- 可通过 `--extra-export-init src/flag_blas/ops/__init__.py` 更新额外导出文件。
- 可通过 `--no-update-exports` 禁用导出更新。

## 常用参数

```text
--spec                         txt/YAML/JSON 算子规格文件
--ops                          逗号分隔的 operator name/module/public function
--all                          处理 spec 中全部 operators
--project-root                 目标项目根目录，默认当前目录
--operator-root                默认 src/flag_blas/ops
--default-operator-subdir      默认 generated
--test-root                    默认 tests
--benchmark-root               默认 benchmark
--trace-root                   默认 hby_operator_trace
--profile                      flagblas 或 standalone
--benchmark-style              flagblas 或 standalone
--model                        默认 deepseek-v4-pro
--base-url                     默认 https://api.deepseek.com
--temperature                  默认 0.0
--max-tokens                   默认 32768，网络不稳可降到 8192
--thinking                     enabled 或 disabled，默认 disabled
--reasoning-effort             high 或 max，仅 thinking enabled 时发送
--stream                       使用 SSE 流式响应，推荐开启
--request-timeout              DeepSeek 单请求超时，默认 240 秒
--api-retries                  DeepSeek 单阶段重试次数，默认 4
--api-retry-base-delay         DeepSeek 重试退避基准秒数，默认 4.0
--max-repair-attempts          自动修复次数
--stage-retries                单阶段静态拒绝后的重试次数
--command-timeout              py_compile/pytest/benchmark 子命令超时
--parse-only                   只解析输入
--parsed-spec-out              parse-only 输出路径
--generate-only                只生成，不验证和修复
--validate-only                只验证已有文件
--repair-only                  先验证，再按失败日志修复
--dry-run                      保存 prompt，不调用 DeepSeek
--skip-tests                   跳过 pytest 功能测试
--skip-benchmark               跳过 benchmark smoke
--resume-run                   继续已有 trace run
--force                        已完成状态下强制重生成
```

批处理额外参数：

```text
--on-fail                      ask/repair/regenerate/skip/stop
--force-passed                 不跳过 batch_state 中已 passed 的算子
--force-generate               即使文件存在也重新生成
--distiller-timeout            每个 distiller 子进程超时，默认 1800 秒
--cuda-visible-devices         为子进程设置 CUDA_VISIBLE_DEVICES
```

## Trace 排查

失败时终端会打印 trace directory。优先看：

```text
<trace>/ledger.jsonl
<trace>/state.json
<trace>/ops/<op>/attempt_*/combined_static_report.json
<trace>/ops/<op>/attempt_*/compile.log
<trace>/ops/<op>/attempt_*/functional.log
<trace>/ops/<op>/attempt_*/benchmark.log
<trace>/ops/<op>/attempt_*/repair_request.json
<trace>/ops/<op>/attempt_*/repair_rationale.md
<trace>/ops/<op>/attempt_*/repair.diff
```

流式响应会额外保留：

```text
<trace>/ops/<op>/attempt_*/implementation_try*_stream.sse
<trace>/ops/<op>/attempt_*/benchmark_try*_stream.sse
<trace>/ops/<op>/attempt_*/repair_stream.sse
```

## 常见问题

`DEEPSEEK_API_KEY is not set`：

```bash
export DEEPSEEK_API_KEY=<your_key>
```

`IncompleteRead(0 bytes read)`：

```bash
--thinking disabled --stream --max-tokens 8192 --api-retries 4
```

GPU 检测失败：

```bash
python hby_skill/scripts/preflight.py --require-gpu
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

benchmark OOM：

```bash
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
CUDA_VISIBLE_DEVICES=<free_gpu> python -m pytest benchmark/test_<module>_perf.py -q --mode=kernel --level=core --warmup=1 --iter=1
```

static validation failed：

查看 `*_static_report.json`，常见原因包括 dtype-specific kernel、漏传 `BLOCK_SIZE`、benchmark 格式不符合 `Benchmark`、测试里定义 Triton kernel。

复杂数误差：

复杂数 `sqrt/division` 不应使用零容差；测试应按 dtype 使用合理 `rtol/atol`。

非连续输入错误：

如果 Triton kernel 用线性 pointer arithmetic，wrapper 必须先 `.contiguous()`，或显式传 stride 并在 kernel 内按 stride 读取。

## 生产建议

1. 先运行 `preflight.py`。
2. 对 txt 输入先 `--parse-only`，确认 family/dtype 被正确解析。
3. 先跑 1-2 个算子，确认生成质量。
4. 稳定后使用 `run_operator_batch.py --all`。
5. 批量运行时保留 trace，不要只看终端最后一行。
6. GPU 资源紧张时先 `--skip-benchmark` 收敛功能正确性，再在空闲 GPU 上补跑 benchmark。
