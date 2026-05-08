# Operator Spec Schema

Spec 文件可以是 YAML 或 JSON。顶层可以是单个 operator，也可以是：

```yaml
operators:
  - ...
  - ...
```

## Required Fields

```yaml
name: string
module: string
kind: string
public_functions:
  - string
dtypes:
  - string
signature: string
semantics: string
reference: string
```

字段说明：

```text
name              算子契约 id，可用于 --ops
module            生成的 Python 模块名，可用于 --ops
kind              算子类别，例如 openblas_level1、blas、elementwise、reduction
public_functions  对外导出的 Python API 名
dtypes            支持 dtype，例如 float32、float64、complex64、float16
signature         public API 签名
semantics         精确语义，包括是否 in-place、stride、broadcast、result 行为
reference         参考语义来源，例如 OpenBLAS、BLAS、torch.add
```

## Optional Fields

```yaml
edge_cases:
  - string
kernel_policy: string
result_policy: string
notes:
  - string
paths:
  operator: string
  functional_test: string
  benchmark: string
```

字段说明：

```text
edge_cases       必测边界条件
kernel_policy    Triton kernel 设计约束
result_policy    输出 dtype、shape、index、in-place 结果规则
notes            额外工程注意事项
paths            精确输出路径；不写时使用 CLI 默认路径
```

## Path Defaults

默认 CLI 参数下：

```text
operator-root = src/flag_blas/ops
default-operator-subdir = generated
test-root = tests
benchmark-root = benchmark
```

没有写 `paths` 时输出为：

```text
src/flag_blas/ops/generated/<module>.py
tests/test_<module>.py
benchmark/test_<module>_perf.py
```

写了 `paths` 时，路径必须仍在允许根目录内：

```text
paths.operator         必须在 operator-root 下
paths.functional_test  必须在 test-root 下
paths.benchmark        必须在 benchmark-root 下
```

## OpenBLAS/BLAS Example

```yaml
operators:
  - name: axpy
    module: axpy
    kind: openblas_level1
    public_functions: [saxpy, daxpy, caxpy, zaxpy]
    dtypes: [float32, float64, complex64, complex128]
    signature: "(n, alpha, x, incx, y, incy) -> None"
    semantics: "BLAS AXPY: y_i = alpha * x_i + y_i, mutating y in place."
    reference: "OpenBLAS cblas_*axpy"
    edge_cases:
      - "n <= 0 is a no-op"
      - "incx and incy must be positive"
    kernel_policy: "one generic real kernel and one generic complex kernel"
    result_policy: "return None; mutate y"
```

## OpenBLAS Level-2 Compact Interface Spec

如果只提供接口列表，可以使用 txt 或 YAML compact spec。脚本会按 Level-2 family 自动分组，并根据 `s/d/c/z` 前缀或显式 dtype 推断 dtype。

Txt 输入示例：

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

支持的 txt 行格式：

```text
cblas_sgemv float32
cblas_sgemv float32,float64
gemv: s,d
trmv fp32 fp64
cblas_strsv, cblas_dtrsv
```

YAML compact spec：

```yaml
openblas_level2:
  api_style: cblas
  min_precisions: 2
  interfaces:
    - cblas_sgemv
    - cblas_dgemv
    - cblas_sgbmv
    - cblas_dgbmv
    - cblas_ssymv
    - cblas_dsymv
```

也可以按 family 写：

```yaml
openblas_level2:
  api_style: flagblas
  operators:
    - family: gemv
      precisions: [s, d, c, z]
    - family: ger
      precisions: [float32, float64]
    - family: hemv
      precisions: [complex64, complex128]
```

compact spec 展开后等价于 full `OperatorSpec`，会自动补齐：

```text
kind            openblas_level2
module          family 名，例如 gemv
public_functions 例如 sgemv/dgemv
dtypes          由 s/d/c/z 推断
signature       由 api_style 和 family 推断
semantics       OpenBLAS Level-2 语义
edge_cases      lda/incx/incy/layout/uplo/trans/band/packed 等通用边界
kernel_policy   family 专项 Triton 设计约束
paths           默认写到 src/flag_blas/ops/level2/<family>.py
```

`api_style` 可选：

```text
cblas      使用 OpenBLAS CBLAS 风格签名，包含 order 参数
flagblas   使用当前 FlagBLAS/cuBLAS 风格签名，通常不含 order 参数
```

内置 Level-2 family 覆盖：

```text
gemv gbmv symv sbmv spmv hemv hbmv hpmv
trmv tbmv tpmv trsv tbsv tpsv
ger geru gerc syr spr syr2 spr2 her hpr her2 hpr2
```

前缀和 dtype 映射：

```text
s -> float32
d -> float64
c -> complex64
z -> complex128
```

`--ops` 可以选择 family、module 或 public interface：

```bash
python hby_skill/scripts/operator_distiller.py \
  --spec hby_skill/assets/examples/openblas_level2_ten_interfaces.yaml \
  --ops cblas_sgemv,dgbmv
```

## Elementwise Example

```yaml
operators:
  - name: elementwise_relu
    module: elementwise_relu
    kind: elementwise
    public_functions: [elementwise_relu]
    dtypes: [float16, bfloat16, float32, float64]
    signature: "(x, out=None) -> torch.Tensor"
    semantics: "out = max(x, 0) for same-shaped output."
    reference: "torch.relu"
    edge_cases:
      - "zero-element tensors"
      - "out tensor validation"
      - "non-contiguous inputs are compacted or explicitly strided"
    kernel_policy: "one generic elementwise kernel across floating dtypes"
    result_policy: "return out when provided, otherwise allocate output"
```

## Writing High-Quality Semantics

Good `semantics` text should answer:

- Which tensors are mutated?
- Are strides positive, negative, or unsupported?
- Are inputs one-dimensional or arbitrary shaped?
- Is broadcasting supported?
- Are non-contiguous inputs supported by compaction or explicit stride math?
- What happens for zero-sized inputs?
- What dtype/shape should result tensors have?
- Does BLAS require 1-based indexes?
- What reference function defines correctness?

Good `kernel_policy` text should answer:

- Is this data movement, elementwise, reduction, or scalar?
- Can real and complex share a kernel?
- Is a staged reduction allowed?
- Are dtype-specific kernels forbidden?
- Should complex use real views?
- Should complex tests use dtype-specific tolerances?
- Are allocations or host sync forbidden in the hot path?
