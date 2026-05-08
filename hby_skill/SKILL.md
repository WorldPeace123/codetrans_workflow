---
name: hby-skill
description: Use this skill when generating production Triton operators from structured operator specs, including OpenBLAS/BLAS-style routines and Elementwise operators. It drives DeepSeek V4 Pro to create implementation, pytest tests, and benchmark files, then validates and repairs them with a workflow modeled after FlagBLAS workflow.py.
---

# HBY Operator Skill

## Purpose

Use this skill to turn one txt/YAML/JSON operator request into production-ready Triton artifacts:

- parsed full operator specs
- implementation module
- functional pytest file
- pytest benchmark smoke file
- trace logs, static reports, validation logs, and repair diffs

The executable workflow is `scripts/operator_distiller.py`. It defaults to `deepseek-v4-pro` and the DeepSeek chat completions API.

For one-command batch usage, prefer `scripts/run_operator_batch.py`. It parses once, skips operators that already pass validation, generates missing/failing operators, and lets the user choose repair or regeneration for failures.

## Four Skill Points

The skill is intentionally split into four composable points:

```text
Parser -> Generator -> Validator -> Repairer
```

- **Parser** reads a txt/YAML/JSON operator request, infers operator families and dtypes, and writes normalized specs.
- **Generator** uses DeepSeek V4 Pro to generate Triton implementation, pytest, and benchmark files.
- **Validator** runs static checks, `py_compile`, pytest, and benchmark smoke.
- **Repairer** feeds validation logs and current files back to DeepSeek V4 Pro until the generated operator passes or attempts are exhausted.

Read `references/pipeline_architecture.md` when the user wants staged execution or starts from a txt file.

## Quick Start

1. Write or reuse an operator spec. Start from:
   - `assets/examples/openblas_axpy.yaml`
   - `assets/examples/openblas_level2_ten_interfaces.yaml`
   - `assets/examples/openblas_level2_interfaces.txt`
   - `assets/examples/elementwise_add.yaml`
   - `assets/examples/elementwise_sign.yaml`
2. Export `DEEPSEEK_API_KEY`.
3. Run a local preflight when using GPU validation:

```bash
python hby_skill/scripts/preflight.py \
  --require-gpu \
  --min-free-gpu-mb 512
```

4. Run the distiller from the target repo root:

```bash
python hby_skill/scripts/operator_distiller.py \
  --spec hby_skill/assets/examples/openblas_level2_interfaces.txt \
  --all \
  --model deepseek-v4-pro \
  --thinking disabled \
  --stream \
  --max-repair-attempts 3
```

5. Inspect the trace directory printed by the command when a generation or repair fails.

For the ten-operator Level-2 example:

```bash
python hby_skill/scripts/run_operator_batch.py \
  --spec hby_skill/assets/examples/openblas_level2_interfaces.txt \
  --all \
  --on-fail ask \
  --thinking disabled \
  --stream
```

## Workflow

The distiller follows the same production pattern as `workflow.py`:

1. Parse txt/YAML/JSON into normalized operator specs.
2. Resolve implementation/test/benchmark output paths and reject paths outside configured roots.
3. Ask DeepSeek V4 Pro for implementation JSON.
4. Static-check syntax, public functions, Triton kernel presence, banned imports, dtype-generic kernel names, `.data_ptr()` misuse, test structure, and benchmark structure.
5. Ask for functional tests and benchmark files with the same JSON protocol.
6. Write files atomically and record diffs.
7. Run `py_compile`, pytest functional tests, and benchmark smoke tests.
8. Feed failure logs and current files back to DeepSeek for repair until validation passes or attempts are exhausted.

## Operator Spec Rules

Full specs need:

- `name`: stable contract id.
- `module`: Python module name.
- `kind`: `openblas_level1`, `blas`, `elementwise`, or another descriptive category.
- `public_functions`: exported API names.
- `dtypes`: supported dtype families.
- `signature`: public API signature.
- `semantics`: exact math and mutation semantics.
- `reference`: reference source, such as OpenBLAS, BLAS, or torch.
- Optional `paths` for exact output files.

For full schema details, read `references/operator_spec_schema.md`.

For OpenBLAS Level-2, compact interface-only specs are supported. Example:

```yaml
openblas_level2:
  api_style: cblas
  min_precisions: 2
  interfaces:
    - cblas_sgemv
    - cblas_dgemv
    - cblas_sgbmv
    - cblas_dgbmv
```

The distiller groups interfaces by family, infers dtypes from `s/d/c/z`, fills signatures, semantics, edge cases, kernel policy, and default Level-2 paths.

Txt input is also supported:

```text
kind: openblas_level2
api_style: cblas
min_precisions: 2

cblas_sgemv float32
cblas_dgemv float64
cblas_sgbmv float32,float64
symv: s,d
```

## Generation Standards

- Implementations must use pure Triton kernels plus Python wrappers.
- Do not call OpenBLAS/cuBLAS/CuPy/ctypes in generated implementation files.
- Kernel names must describe algorithmic shape, not dtype-specific public APIs.
- Tests should use pure torch or local Python references unless a provider comparison is explicitly required.
- Benchmarks must compare against equivalent work, never clone-only or empty baselines.
- In-place benchmark wrappers must clone inside timed functions.
- Complex kernels should use flattened real views when pointer arithmetic is required.
- OpenBLAS Level-2 compact specs must preserve layout/order, uplo, trans, diag, lda, band width, packed storage, and in-place x/y/A/AP mutation semantics.
- DeepSeek output must be valid JSON with complete file contents.

## References

- `references/usage.md`: CLI usage, examples, troubleshooting.
- `references/technical_design.md`: architecture and validation design.
- `references/operator_spec_schema.md`: full operator spec schema.
- `references/pipeline_architecture.md`: parser/generator/validator/repairer staged workflow.
