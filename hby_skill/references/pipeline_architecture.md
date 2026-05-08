# HBY 四技能流水线

`hby_skill` 现在按四个可组合技能点组织：

```text
解析技能 -> 生成技能 -> 检验技能 -> 修复技能
```

四个技能点共享同一份 `scripts/operator_distiller.py`，避免 spec schema、OpenBLAS catalog、trace 格式和静态校验规则在多份脚本之间漂移。

生产运行前建议先执行预检：

```bash
python hby_skill/scripts/preflight.py \
  --require-gpu \
  --min-free-gpu-mb 512
```

如果只想做解析或生成 prompt，可以加 `--skip-api-key`，避免因为未设置 DeepSeek Key 阻塞本地检查。

## 1. 解析技能

输入：用户维护的 txt、YAML 或 JSON。

典型 txt：

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

运行：

```bash
python hby_skill/scripts/operator_distiller.py \
  --spec hby_skill/assets/examples/openblas_level2_interfaces.txt \
  --parse-only \
  --parsed-spec-out /tmp/hby_parsed_specs.json
```

输出：规范化 full spec JSON：

```text
operators[*].name
operators[*].module
operators[*].kind
operators[*].public_functions
operators[*].dtypes
operators[*].signature
operators[*].semantics
operators[*].edge_cases
operators[*].kernel_policy
operators[*].result_policy
```

解析能力：

- `cblas_sgemv float32`
- `cblas_sgemv float32,float64`
- `gemv: s,d`
- `trmv fp32 fp64`
- `cblas_strsv, cblas_dtrsv`
- `kind/api_style/min_precisions/precisions` 指令

## 2. 生成技能

输入：解析技能输出的 full spec JSON，或原始 txt/YAML/JSON。

运行：

```bash
python hby_skill/scripts/operator_distiller.py \
  --spec /tmp/hby_parsed_specs.json \
  --generate-only \
  --ops gemv,gbmv \
  --thinking disabled \
  --stream \
  --stage-retries 3
```

职责：

- 调 DeepSeek V4 Pro 生成 implementation。
- 调 DeepSeek V4 Pro 生成 functional test。
- 调 DeepSeek V4 Pro 生成 benchmark。
- 做写入前静态校验。
- 原子写入文件。
- 更新导出。
- 不运行 pytest，不进入 repair。

## 3. 检验技能

输入：生成技能写入的文件。

运行：

```bash
python hby_skill/scripts/operator_distiller.py \
  --spec /tmp/hby_parsed_specs.json \
  --validate-only \
  --ops gemv,gbmv \
  --skip-benchmark
```

职责：

- 组合静态校验。
- `python -m py_compile`。
- pytest functional。
- benchmark smoke。
- 将失败写入 trace：
  - `combined_static_report.json`
  - `compile.log`
  - `functional.log`
  - `benchmark.log`

完整 GPU + benchmark 验证：

```bash
CUDA_VISIBLE_DEVICES=0 python hby_skill/scripts/operator_distiller.py \
  --spec /tmp/hby_parsed_specs.json \
  --validate-only \
  --ops gemv,gbmv
```

## 4. 修复技能

输入：检验技能留下的失败日志和当前生成文件。

运行：

```bash
python hby_skill/scripts/operator_distiller.py \
  --spec /tmp/hby_parsed_specs.json \
  --repair-only \
  --ops gemv,gbmv \
  --thinking disabled \
  --stream \
  --max-repair-attempts 3
```

职责：

- 先运行一次检验，生成 attempt_000 失败日志。
- 汇总失败信息。
- 把当前 implementation/test/benchmark 和失败日志交给 DeepSeek。
- 接收完整替换文件。
- 静态校验、写入、重跑检验。
- 直到通过或 repair 次数耗尽。

## 一条命令全流程

面向日常使用，推荐批处理入口：

```bash
python hby_skill/scripts/run_operator_batch.py \
  --spec hby_skill/assets/examples/openblas_level2_interfaces.txt \
  --all \
  --on-fail ask \
  --thinking disabled \
  --stream \
  --max-repair-attempts 3 \
  --distiller-timeout 1800
```

批处理入口会为每个算子执行：

```text
already passed? skip
else validate existing files
  passed -> mark passed
  failed/missing -> generate -> validate
    passed -> mark passed
    failed -> ask/repair/regenerate/skip/stop
```

`--on-fail` 可选：

```text
ask          交互选择 repair/regenerate/skip/stop
repair       直接进入修复技能
regenerate   重新生成后再检验
skip         标记失败跳过
stop         遇到失败即停止
```

批处理新增两个生产参数：

```text
--cuda-visible-devices  为每个 distiller 子进程设置 CUDA_VISIBLE_DEVICES
--distiller-timeout     为每个 distiller 子进程设置总超时，默认 1800 秒
```

每次 batch run 会写：

```text
batch_runs/<run_id>/summary.json
```

其中包含每个 operator 的状态、最后动作和 attempt 数。跨 run 的状态保存在：

```text
batch_state.json
```

底层仍然支持 all-in-one：

```bash
python hby_skill/scripts/operator_distiller.py \
  --spec hby_skill/assets/examples/openblas_level2_interfaces.txt \
  --all \
  --batch-size 2 \
  --max-repair-attempts 3
```

这等价于：

```text
parse -> generate -> validate -> repair
```

## 推荐生产节奏

1. 先 parse-only，确认 txt 被解析成正确 family 和 dtype。
2. 对 1-2 个 family generate-only。
3. validate-only 看真实失败日志。
4. repair-only 收敛。
5. 单算子稳定后再扩大到 `--all --batch-size 2`。

GPU 资源紧张时，把第 3 步拆成：

```text
先 --validate-only --skip-benchmark 确认功能正确
再挑空闲 GPU 单独跑 benchmark smoke
```
