#!/usr/bin/env python3
"""DeepSeek V4 Pro driven Triton operator distillation workflow.

This script is the executable part of the hby-skill Codex skill. It turns a
structured operator spec into three production artifacts:

* Triton implementation
* pytest functional tests
* pytest benchmark smoke tests

It intentionally mirrors the control pattern used by FlagBLAS workflow.py:
structured specs, staged JSON generation, path-checked writes, static guards,
runtime validation, trace logs, and repair from failure logs.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import difflib
import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - json specs still work without PyYAML.
    yaml = None


DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TRACE_DIR = "hby_operator_trace"
AUTO_EXPORT_BEGIN = "# BEGIN HBY OPERATOR DISTILLER EXPORTS"
AUTO_EXPORT_END = "# END HBY OPERATOR DISTILLER EXPORTS"

OPENBLAS_PRECISION_DTYPES = {
    "s": "float32",
    "d": "float64",
    "c": "complex64",
    "z": "complex128",
}

OPENBLAS_LEVEL2_COMMON_EDGE_CASES = (
    "m and n dimensions may be zero; zero-sized operations must return without invalid kernel launches",
    "increments incx and incy must be non-zero; generated wrappers may restrict to positive increments unless negative increments are explicitly requested",
    "leading dimensions lda must satisfy the OpenBLAS/CBLAS contract for the selected storage/order and band/packed layout",
    "float64 and complex128 tests must skip when the target device does not support fp64",
    "tests must include unit-stride and non-unit-stride vectors",
    "benchmarks must use modest smoke shapes and equivalent torch references",
)

OPENBLAS_LEVEL2_CATALOG: Dict[str, Dict[str, Any]] = {
    "gemv": {
        "title": "general matrix-vector multiply",
        "allowed_precisions": ("s", "d", "c", "z"),
        "cblas_signature": "(order, trans, m, n, alpha, A, lda, x, incx, beta, y, incy) -> None",
        "flagblas_signature": "(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) -> None",
        "semantics": "General matrix-vector multiply: y = alpha * op(A) * x + beta * y. Support no-transpose, transpose, and conjugate-transpose for complex variants.",
        "kernel_policy": "Use generic real and complex matrix-vector kernels. Split by transpose/conjugation or reduction strategy when needed, not by precision.",
        "result_policy": "Mutates y in place and returns None.",
    },
    "gbmv": {
        "title": "general band matrix-vector multiply",
        "allowed_precisions": ("s", "d", "c", "z"),
        "cblas_signature": "(order, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) -> None",
        "flagblas_signature": "(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) -> None",
        "semantics": "General band matrix-vector multiply: y = alpha * op(A_band) * x + beta * y with kl lower and ku upper diagonals.",
        "kernel_policy": "Exploit banded storage to avoid reading outside the kl/ku diagonal band; keep dtype-generic kernels.",
        "result_policy": "Mutates y in place and returns None.",
    },
    "symv": {
        "title": "symmetric matrix-vector multiply",
        "allowed_precisions": ("s", "d", "c", "z"),
        "cblas_signature": "(order, uplo, n, alpha, A, lda, x, incx, beta, y, incy) -> None",
        "flagblas_signature": "(uplo, n, alpha, A, lda, x, incx, beta, y, incy) -> None",
        "semantics": "Symmetric matrix-vector multiply: y = alpha * A * x + beta * y, reading only the triangle selected by uplo and mirroring the other triangle.",
        "kernel_policy": "Use triangular-aware matrix-vector kernels; for complex symv do not conjugate mirrored values.",
        "result_policy": "Mutates y in place and returns None.",
    },
    "sbmv": {
        "title": "symmetric band matrix-vector multiply",
        "allowed_precisions": ("s", "d"),
        "cblas_signature": "(order, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) -> None",
        "flagblas_signature": "(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) -> None",
        "semantics": "Real symmetric band matrix-vector multiply using k stored off-diagonals.",
        "kernel_policy": "Read only the stored symmetric band and mirror logically; do not materialize the dense matrix.",
        "result_policy": "Mutates y in place and returns None.",
    },
    "spmv": {
        "title": "symmetric packed matrix-vector multiply",
        "allowed_precisions": ("s", "d"),
        "cblas_signature": "(order, uplo, n, alpha, AP, x, incx, beta, y, incy) -> None",
        "flagblas_signature": "(uplo, n, alpha, AP, x, incx, beta, y, incy) -> None",
        "semantics": "Real symmetric packed matrix-vector multiply using packed AP storage for the selected triangle.",
        "kernel_policy": "Index packed triangular storage directly; avoid dense unpacking in the timed path.",
        "result_policy": "Mutates y in place and returns None.",
    },
    "hemv": {
        "title": "Hermitian matrix-vector multiply",
        "allowed_precisions": ("c", "z"),
        "cblas_signature": "(order, uplo, n, alpha, A, lda, x, incx, beta, y, incy) -> None",
        "flagblas_signature": "(uplo, n, alpha, A, lda, x, incx, beta, y, incy) -> None",
        "semantics": "Hermitian matrix-vector multiply: y = alpha * A * x + beta * y. Mirror the unstored triangle with conjugation and treat diagonal imaginary parts as zero.",
        "kernel_policy": "Use complex real-view arithmetic where needed; mirror Hermitian entries with conjugation.",
        "result_policy": "Mutates y in place and returns None.",
    },
    "hbmv": {
        "title": "Hermitian band matrix-vector multiply",
        "allowed_precisions": ("c", "z"),
        "cblas_signature": "(order, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) -> None",
        "flagblas_signature": "(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) -> None",
        "semantics": "Hermitian band matrix-vector multiply using k stored off-diagonals and conjugate mirroring.",
        "kernel_policy": "Read only the stored Hermitian band; do not materialize dense A.",
        "result_policy": "Mutates y in place and returns None.",
    },
    "hpmv": {
        "title": "Hermitian packed matrix-vector multiply",
        "allowed_precisions": ("c", "z"),
        "cblas_signature": "(order, uplo, n, alpha, AP, x, incx, beta, y, incy) -> None",
        "flagblas_signature": "(uplo, n, alpha, AP, x, incx, beta, y, incy) -> None",
        "semantics": "Hermitian packed matrix-vector multiply using packed AP storage and conjugate mirroring.",
        "kernel_policy": "Index packed Hermitian storage directly; avoid dense unpacking in the timed path.",
        "result_policy": "Mutates y in place and returns None.",
    },
    "trmv": {
        "title": "triangular matrix-vector multiply",
        "allowed_precisions": ("s", "d", "c", "z"),
        "cblas_signature": "(order, uplo, trans, diag, n, A, lda, x, incx) -> None",
        "flagblas_signature": "(uplo, trans, diag, n, A, lda, x, incx) -> None",
        "semantics": "Triangular matrix-vector multiply: x = op(A) * x. Respect uplo, trans/conjugate-transpose, and unit/non-unit diagonal.",
        "kernel_policy": "Use triangular masks and clone or stage x when needed so in-place updates do not consume already-mutated elements.",
        "result_policy": "Mutates x in place and returns None.",
    },
    "tbmv": {
        "title": "triangular band matrix-vector multiply",
        "allowed_precisions": ("s", "d", "c", "z"),
        "cblas_signature": "(order, uplo, trans, diag, n, k, A, lda, x, incx) -> None",
        "flagblas_signature": "(uplo, trans, diag, n, k, A, lda, x, incx) -> None",
        "semantics": "Triangular band matrix-vector multiply: x = op(A_band) * x with k stored off-diagonals.",
        "kernel_policy": "Use triangular band masks and avoid dense materialization.",
        "result_policy": "Mutates x in place and returns None.",
    },
    "tpmv": {
        "title": "triangular packed matrix-vector multiply",
        "allowed_precisions": ("s", "d", "c", "z"),
        "cblas_signature": "(order, uplo, trans, diag, n, AP, x, incx) -> None",
        "flagblas_signature": "(uplo, trans, diag, n, AP, x, incx) -> None",
        "semantics": "Triangular packed matrix-vector multiply: x = op(A_packed) * x.",
        "kernel_policy": "Index packed triangular storage directly; avoid dense unpacking in timed wrappers.",
        "result_policy": "Mutates x in place and returns None.",
    },
    "trsv": {
        "title": "triangular solve",
        "allowed_precisions": ("s", "d", "c", "z"),
        "cblas_signature": "(order, uplo, trans, diag, n, A, lda, x, incx) -> None",
        "flagblas_signature": "(uplo, trans, diag, n, A, lda, x, incx) -> None",
        "semantics": "Triangular solve: solve op(A) * x_out = x_in in place. Respect uplo, trans/conjugate-transpose, and unit/non-unit diagonal.",
        "kernel_policy": "Correctness-first triangular solve. Use blocked forward/back substitution; avoid parallel updates that violate dependencies.",
        "result_policy": "Mutates x in place and returns None.",
    },
    "tbsv": {
        "title": "triangular band solve",
        "allowed_precisions": ("s", "d", "c", "z"),
        "cblas_signature": "(order, uplo, trans, diag, n, k, A, lda, x, incx) -> None",
        "flagblas_signature": "(uplo, trans, diag, n, k, A, lda, x, incx) -> None",
        "semantics": "Triangular band solve using k stored off-diagonals.",
        "kernel_policy": "Correctness-first triangular band solve; preserve dependency order and use band indexing.",
        "result_policy": "Mutates x in place and returns None.",
    },
    "tpsv": {
        "title": "triangular packed solve",
        "allowed_precisions": ("s", "d", "c", "z"),
        "cblas_signature": "(order, uplo, trans, diag, n, AP, x, incx) -> None",
        "flagblas_signature": "(uplo, trans, diag, n, AP, x, incx) -> None",
        "semantics": "Triangular packed solve using packed triangular AP storage.",
        "kernel_policy": "Correctness-first triangular packed solve; index packed storage directly.",
        "result_policy": "Mutates x in place and returns None.",
    },
    "ger": {
        "title": "real general rank-1 update",
        "allowed_precisions": ("s", "d"),
        "cblas_signature": "(order, m, n, alpha, x, incx, y, incy, A, lda) -> None",
        "flagblas_signature": "(m, n, alpha, x, incx, y, incy, A, lda) -> None",
        "semantics": "Real general rank-1 update: A = alpha * x * y^T + A.",
        "kernel_policy": "Use a 2-D tiled update kernel; do not duplicate kernels by precision.",
        "result_policy": "Mutates A in place and returns None.",
    },
    "geru": {
        "title": "complex unconjugated general rank-1 update",
        "allowed_precisions": ("c", "z"),
        "cblas_signature": "(order, m, n, alpha, x, incx, y, incy, A, lda) -> None",
        "flagblas_signature": "(m, n, alpha, x, incx, y, incy, A, lda) -> None",
        "semantics": "Complex unconjugated rank-1 update: A = alpha * x * y^T + A.",
        "kernel_policy": "Use complex real-view arithmetic; do not conjugate y.",
        "result_policy": "Mutates A in place and returns None.",
    },
    "gerc": {
        "title": "complex conjugated general rank-1 update",
        "allowed_precisions": ("c", "z"),
        "cblas_signature": "(order, m, n, alpha, x, incx, y, incy, A, lda) -> None",
        "flagblas_signature": "(m, n, alpha, x, incx, y, incy, A, lda) -> None",
        "semantics": "Complex conjugated rank-1 update: A = alpha * x * conj(y)^T + A.",
        "kernel_policy": "Use complex real-view arithmetic and conjugate y in the update.",
        "result_policy": "Mutates A in place and returns None.",
    },
    "syr": {
        "title": "symmetric rank-1 update",
        "allowed_precisions": ("s", "d", "c", "z"),
        "cblas_signature": "(order, uplo, n, alpha, x, incx, A, lda) -> None",
        "flagblas_signature": "(uplo, n, alpha, x, incx, A, lda) -> None",
        "semantics": "Symmetric rank-1 update: A = alpha * x * x^T + A, updating only the triangle selected by uplo.",
        "kernel_policy": "Use triangular 2-D masks and update only stored triangle.",
        "result_policy": "Mutates A in place and returns None.",
    },
    "spr": {
        "title": "symmetric packed rank-1 update",
        "allowed_precisions": ("s", "d"),
        "cblas_signature": "(order, uplo, n, alpha, x, incx, AP) -> None",
        "flagblas_signature": "(uplo, n, alpha, x, incx, AP) -> None",
        "semantics": "Real symmetric packed rank-1 update on packed AP storage.",
        "kernel_policy": "Index packed triangular storage directly; update only stored triangle.",
        "result_policy": "Mutates AP in place and returns None.",
    },
    "syr2": {
        "title": "symmetric rank-2 update",
        "allowed_precisions": ("s", "d", "c", "z"),
        "cblas_signature": "(order, uplo, n, alpha, x, incx, y, incy, A, lda) -> None",
        "flagblas_signature": "(uplo, n, alpha, x, incx, y, incy, A, lda) -> None",
        "semantics": "Symmetric rank-2 update: A = alpha*x*y^T + alpha*y*x^T + A, updating selected triangle.",
        "kernel_policy": "Use triangular 2-D masks and update only stored triangle.",
        "result_policy": "Mutates A in place and returns None.",
    },
    "spr2": {
        "title": "symmetric packed rank-2 update",
        "allowed_precisions": ("s", "d"),
        "cblas_signature": "(order, uplo, n, alpha, x, incx, y, incy, AP) -> None",
        "flagblas_signature": "(uplo, n, alpha, x, incx, y, incy, AP) -> None",
        "semantics": "Real symmetric packed rank-2 update on packed AP storage.",
        "kernel_policy": "Index packed triangular storage directly; update only stored triangle.",
        "result_policy": "Mutates AP in place and returns None.",
    },
    "her": {
        "title": "Hermitian rank-1 update",
        "allowed_precisions": ("c", "z"),
        "cblas_signature": "(order, uplo, n, alpha, x, incx, A, lda) -> None",
        "flagblas_signature": "(uplo, n, alpha, x, incx, A, lda) -> None",
        "semantics": "Hermitian rank-1 update: A = alpha*x*conj(x)^T + A. Alpha is real; diagonal imaginary parts remain zero.",
        "kernel_policy": "Use triangular 2-D masks, complex real-view arithmetic, and Hermitian diagonal handling.",
        "result_policy": "Mutates A in place and returns None.",
    },
    "hpr": {
        "title": "Hermitian packed rank-1 update",
        "allowed_precisions": ("c", "z"),
        "cblas_signature": "(order, uplo, n, alpha, x, incx, AP) -> None",
        "flagblas_signature": "(uplo, n, alpha, x, incx, AP) -> None",
        "semantics": "Hermitian packed rank-1 update on packed AP storage. Alpha is real; diagonal imaginary parts remain zero.",
        "kernel_policy": "Index packed Hermitian storage directly and preserve Hermitian diagonal semantics.",
        "result_policy": "Mutates AP in place and returns None.",
    },
    "her2": {
        "title": "Hermitian rank-2 update",
        "allowed_precisions": ("c", "z"),
        "cblas_signature": "(order, uplo, n, alpha, x, incx, y, incy, A, lda) -> None",
        "flagblas_signature": "(uplo, n, alpha, x, incx, y, incy, A, lda) -> None",
        "semantics": "Hermitian rank-2 update: A = alpha*x*conj(y)^T + conj(alpha)*y*conj(x)^T + A.",
        "kernel_policy": "Use triangular 2-D masks, complex real-view arithmetic, and Hermitian diagonal handling.",
        "result_policy": "Mutates A in place and returns None.",
    },
    "hpr2": {
        "title": "Hermitian packed rank-2 update",
        "allowed_precisions": ("c", "z"),
        "cblas_signature": "(order, uplo, n, alpha, x, incx, y, incy, AP) -> None",
        "flagblas_signature": "(uplo, n, alpha, x, incx, y, incy, AP) -> None",
        "semantics": "Hermitian packed rank-2 update on packed AP storage.",
        "kernel_policy": "Index packed Hermitian storage directly and preserve Hermitian diagonal semantics.",
        "result_policy": "Mutates AP in place and returns None.",
    },
}


class DistillError(RuntimeError):
    """Controlled workflow failure."""


@dataclasses.dataclass(frozen=True)
class TargetPaths:
    operator: Path
    functional_test: Path
    benchmark: Path

    def as_relative_dict(self, project_root: Path) -> Dict[str, str]:
        return {
            "operator": str(self.operator.resolve().relative_to(project_root.resolve())),
            "functional_test": str(self.functional_test.resolve().relative_to(project_root.resolve())),
            "benchmark": str(self.benchmark.resolve().relative_to(project_root.resolve())),
        }


@dataclasses.dataclass(frozen=True)
class OperatorSpec:
    """A complete generation contract for one operator module."""

    name: str
    module: str
    kind: str
    public_functions: Tuple[str, ...]
    dtypes: Tuple[str, ...]
    signature: str
    semantics: str
    reference: str
    edge_cases: Tuple[str, ...] = ()
    kernel_policy: str = ""
    result_policy: str = ""
    notes: Tuple[str, ...] = ()
    paths: Mapping[str, str] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OperatorSpec":
        required = [
            "name",
            "module",
            "kind",
            "public_functions",
            "dtypes",
            "signature",
            "semantics",
            "reference",
        ]
        missing = [key for key in required if key not in raw]
        if missing:
            raise DistillError(f"operator spec missing required fields: {missing}")
        public_functions = tuple(str(x) for x in raw["public_functions"])
        dtypes = tuple(str(x) for x in raw["dtypes"])
        if not public_functions:
            raise DistillError(f"{raw.get('name', '<unknown>')} has no public_functions")
        if not dtypes:
            raise DistillError(f"{raw.get('name', '<unknown>')} has no dtypes")
        return cls(
            name=str(raw["name"]),
            module=str(raw["module"]),
            kind=str(raw["kind"]),
            public_functions=public_functions,
            dtypes=dtypes,
            signature=str(raw["signature"]),
            semantics=str(raw["semantics"]),
            reference=str(raw["reference"]),
            edge_cases=tuple(str(x) for x in raw.get("edge_cases", ())),
            kernel_policy=str(raw.get("kernel_policy", "")),
            result_policy=str(raw.get("result_policy", "")),
            notes=tuple(str(x) for x in raw.get("notes", ())),
            paths={str(k): str(v) for k, v in dict(raw.get("paths", {})).items()},
        )

    def to_prompt_dict(self, paths: TargetPaths, project_root: Path) -> Dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "kind": self.kind,
            "public_functions": list(self.public_functions),
            "dtypes": list(self.dtypes),
            "signature": self.signature,
            "semantics": self.semantics,
            "reference": self.reference,
            "edge_cases": list(self.edge_cases),
            "kernel_policy": self.kernel_policy,
            "result_policy": self.result_policy,
            "notes": list(self.notes),
            "paths": paths.as_relative_dict(project_root),
        }


@dataclasses.dataclass
class WorkflowConfig:
    project_root: Path
    operator_root: Path
    test_root: Path
    benchmark_root: Path
    trace_root: Path
    src_root: Path
    default_operator_subdir: str = "generated"
    package_import_root: str = "flag_blas.ops"
    profile: str = "flagblas"
    benchmark_style: str = "flagblas"
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 32768
    thinking: str = "disabled"
    reasoning_effort: str = "high"
    stream: bool = False
    request_timeout: int = 240
    api_retries: int = 4
    api_retry_base_delay: float = 4.0
    stage_retries: int = 2
    max_repair_attempts: int = 3
    command_timeout: int = 600
    batch_size: int = 2
    cooldown: float = 5.0
    dry_run: bool = False
    validate_only: bool = False
    operation_mode: str = "all"
    skip_tests: bool = False
    skip_benchmark: bool = False
    update_exports: bool = True
    extra_export_inits: Tuple[Path, ...] = ()
    hard_generic_kernel_guard: bool = True
    continue_on_failure: bool = True

    @property
    def allowed_write_roots(self) -> Tuple[Path, Path, Path]:
        return (self.operator_root, self.test_root, self.benchmark_root)


@dataclasses.dataclass
class CommandResult:
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def combined_output(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text_if_exists(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n# ... truncated by hby operator distiller ...\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def ensure_within(path: Path, allowed_roots: Sequence[Path]) -> Path:
    resolved = path.resolve()
    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    roots = ", ".join(str(root.resolve()) for root in allowed_roots)
    raise DistillError(f"refusing path outside allowed roots: {resolved}; allowed roots: {roots}")


class TraceLogger:
    """Run-scoped trace writer."""

    def __init__(self, trace_root: Path, run_path: Optional[Path] = None):
        self.trace_root = trace_root.resolve()
        self.run_dir = run_path.resolve() if run_path else self.trace_root / "runs" / local_run_id()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.run_dir / "ledger.jsonl"
        self.state_path = self.run_dir / "state.json"

    def event(self, stage: str, status: str, op: Optional[str] = None, **data: Any) -> None:
        record = {"ts": utc_now_iso(), "stage": stage, "status": status, "op": op, **data}
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def save_json(self, relative_path: str, data: Any) -> Path:
        path = self.run_dir / relative_path
        atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def save_text(self, relative_path: str, text: str) -> Path:
        path = self.run_dir / relative_path
        atomic_write(path, text)
        return path

    def attempt_dir(self, op: str, attempt: int) -> Path:
        path = self.run_dir / "ops" / op / f"attempt_{attempt:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"completed_ops": [], "failed_ops": {}, "current_batch": 0}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save_state(self, state: Mapping[str, Any]) -> None:
        atomic_write(self.state_path, json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


class DeepSeekClient:
    """Small stdlib HTTP client for DeepSeek chat completions."""

    def __init__(self, config: WorkflowConfig, trace: TraceLogger):
        self.config = config
        self.trace = trace
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key and not config.dry_run and not config.validate_only:
            raise DistillError("DEEPSEEK_API_KEY is not set")

    def chat_json(
        self,
        *,
        messages: List[Dict[str, str]],
        stage: str,
        op: str,
        attempt_dir: Path,
    ) -> Dict[str, Any]:
        request_payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": self.config.stream,
            "response_format": {"type": "json_object"},
            "thinking": {"type": self.config.thinking},
        }
        if self.config.thinking == "enabled":
            request_payload["reasoning_effort"] = self.config.reasoning_effort
        request_path = attempt_dir / f"{stage}_request.json"
        atomic_write(request_path, json.dumps(request_payload, indent=2, ensure_ascii=False) + "\n")

        if self.config.dry_run:
            self.trace.event(stage, "dry_run", op=op, request_path=str(request_path))
            raise DistillError("dry-run mode does not call DeepSeek")

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(request_payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error = ""
        for api_attempt in range(1, self.config.api_retries + 1):
            started = time.monotonic()
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.config.request_timeout) as response:
                    if self.config.stream:
                        response_data = self._read_stream_response(response, stage, op, attempt_dir)
                    else:
                        raw = response.read().decode("utf-8", errors="replace")
                        response_data = json.loads(raw)
                duration = time.monotonic() - started
                response_path = attempt_dir / f"{stage}_response.json"
                atomic_write(response_path, json.dumps(response_data, indent=2, ensure_ascii=False) + "\n")
                self.trace.event(
                    stage,
                    "api_success",
                    op=op,
                    api_attempt=api_attempt,
                    duration_s=round(duration, 3),
                    usage=response_data.get("usage"),
                    response_path=str(response_path),
                )
                return self._extract_json_payload(response_data, stage, op, attempt_dir)
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {raw[:2000]}"
                retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                http.client.IncompleteRead,
                ConnectionResetError,
            ) as exc:
                last_error = repr(exc)
                retryable = True
            except Exception as exc:  # pragma: no cover - defensive catch for provider SDK behavior.
                last_error = repr(exc)
                retryable = False

            self.trace.event(stage, "api_error", op=op, api_attempt=api_attempt, error=last_error, retryable=retryable)
            if not retryable or api_attempt == self.config.api_retries:
                break
            time.sleep(self.config.api_retry_base_delay * (2 ** (api_attempt - 1)))

        raise DistillError(f"DeepSeek request failed for {op}/{stage}: {last_error}")

    def _read_stream_response(
        self, response: Any, stage: str, op: str, attempt_dir: Path
    ) -> Dict[str, Any]:
        raw_lines: List[str] = []
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        finish_reason: Optional[str] = None
        chunk_count = 0

        while True:
            line = response.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            raw_lines.append(text)
            if not text.startswith("data:"):
                continue
            data = text[len("data:") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            chunk_count += 1
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])

        atomic_write(attempt_dir / f"{stage}_stream.sse", "\n".join(raw_lines) + ("\n" if raw_lines else ""))
        content = "".join(content_parts)
        if not content:
            raise DistillError(f"DeepSeek stream returned empty content for {op}/{stage}; chunks={chunk_count}")
        return {
            "object": "chat.completion",
            "model": self.config.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason or "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": "".join(reasoning_parts) or None,
                    },
                }
            ],
        }

    def _extract_json_payload(
        self, response_data: Mapping[str, Any], stage: str, op: str, attempt_dir: Path
    ) -> Dict[str, Any]:
        choices = response_data.get("choices") or []
        if not choices:
            raise DistillError(f"DeepSeek returned no choices for {op}/{stage}")
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason not in {"stop", "tool_calls"}:
            raise DistillError(f"DeepSeek finish_reason={finish_reason!r} for {op}/{stage}")
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content")
        if reasoning:
            atomic_write(attempt_dir / f"{stage}_provider_reasoning.md", reasoning)

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise
            payload = json.loads(match.group(0))
        atomic_write(
            attempt_dir / f"{stage}_payload.json",
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        return payload


class StaticValidator:
    """Static checks that reject common bad generated artifacts before writing."""

    def __init__(self, config: WorkflowConfig):
        self.config = config

    def validate_python_syntax(self, content: str, label: str) -> List[str]:
        try:
            ast.parse(content)
            return []
        except SyntaxError as exc:
            return [f"{label}: SyntaxError line {exc.lineno}: {exc.msg}"]

    def validate_operator(self, spec: OperatorSpec, content: str) -> List[str]:
        errors = self.validate_python_syntax(content, f"{spec.module}.py")
        if errors:
            return errors
        tree = ast.parse(content)
        functions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = [name for name in spec.public_functions if name not in functions]
        if missing:
            errors.append(f"missing public functions: {missing}")

        banned_imports = ("cupy", "cublas", "numpy.ctypeslib")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in banned_imports:
                        errors.append(f"operator implementation must not import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in banned_imports or "cublas" in node.module:
                    errors.append(f"operator implementation must not import from {node.module}")

        kernel_names = self._triton_kernel_names(tree)
        if not kernel_names:
            errors.append("operator implementation must contain at least one @triton.jit kernel")
        errors.extend(self._validate_triton_runtime_contract(content))
        if self.config.hard_generic_kernel_guard:
            errors.extend(self._validate_generic_kernel_policy(spec, content, kernel_names))
        if spec.kind.lower().startswith("elementwise"):
            errors.extend(self._validate_elementwise_contract(spec, content))
        if spec.kind == "openblas_level2":
            errors.extend(self._validate_openblas_level2_contract(spec, content))
        if "complex" in " ".join(spec.dtypes).lower():
            errors.extend(self._validate_complex_contract(content))
        return errors

    def validate_test(self, spec: OperatorSpec, content: str) -> List[str]:
        errors = self.validate_python_syntax(content, f"test_{spec.module}.py")
        if errors:
            return errors
        if "@triton.jit" in content:
            errors.append("functional tests must not define Triton kernels")
        if "pytest" not in content:
            errors.append("functional tests must use pytest")
        if "torch.testing.assert_close" not in content and re.search(r"\bassert\b", content) is None:
            errors.append("functional tests must assert correctness")
        for name in spec.public_functions:
            if name not in content:
                errors.append(f"functional tests should cover public function {name}")
        return errors

    def validate_benchmark(self, spec: OperatorSpec, content: str) -> List[str]:
        errors = self.validate_python_syntax(content, f"test_{spec.module}_perf.py")
        if errors:
            return errors
        if "@triton.jit" in content:
            errors.append("benchmark files must not define Triton kernels")
        if "pytest" not in content:
            errors.append("benchmarks must be pytest-runnable")
        if self.config.benchmark_style == "flagblas":
            if "Benchmark" not in content or "benchmark.performance_utils" not in content:
                errors.append("FlagBLAS benchmarks must use benchmark.performance_utils.Benchmark")
            if "bench.run()" not in content:
                errors.append("FlagBLAS benchmark tests must create bench and call bench.run()")
            if re.search(r"def\s+test_\w+\s*\([^)]*\bbench\b", content):
                errors.append("benchmark tests must not use a pytest-benchmark fixture named bench")
        else:
            if "do_bench" not in content and "time.perf_counter" not in content:
                errors.append("standalone benchmarks must measure latency with triton.testing.do_bench or time.perf_counter")
        if re.search(r"yield\s*\([^)]*\)\s*,", content):
            errors.append("Benchmark.get_input_iter must yield flat args followed by kwargs, not (args_tuple), kwargs")
        return errors

    def _validate_triton_runtime_contract(self, content: str) -> List[str]:
        errors: List[str] = []
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "data_ptr":
                    errors.append("do not pass tensor.data_ptr() into Triton kernels; pass tensors/views directly")
            if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Attribute):
                if self._attribute_name(node.annotation) == "tl.constexpr":
                    errors.append("do not annotate module globals as tl.constexpr; pass constexpr kernel arguments")
        if "tl.make_block_ptr" in content and ".data_ptr()" in content:
            errors.append("tl.make_block_ptr bases must be pointer arguments, not integer data_ptr values")
        if re.search(r"BLOCK_SIZE\s*:\s*tl\.constexpr", content) and "BLOCK_SIZE=" not in content:
            errors.append("Triton kernel launches must pass BLOCK_SIZE when kernels declare BLOCK_SIZE: tl.constexpr")
        return errors

    def _validate_generic_kernel_policy(
        self, spec: OperatorSpec, content: str, kernel_names: Sequence[str]
    ) -> List[str]:
        errors: List[str] = []
        public_kernel_names = {f"{name}_kernel" for name in spec.public_functions}
        for kernel in kernel_names:
            lower = kernel.lower()
            if kernel in public_kernel_names:
                errors.append(f"{kernel} is a dtype/API-specific kernel; use generic math-shape names")
            if re.search(r"(float32|float64|fp32|fp64|f32|f64)", lower):
                errors.append(f"{kernel} encodes precision in the kernel name")
            for public_name in spec.public_functions:
                if lower.startswith(public_name.lower()):
                    errors.append(f"{kernel} starts with public dtype/API variant {public_name}")
        signature_blocks = re.findall(
            r"@triton\.jit\s*\ndef\s+\w+\s*\((.*?)\)\s*:",
            content,
            flags=re.DOTALL,
        )
        for signature in signature_blocks:
            if "tl.float32" in signature or "tl.float64" in signature:
                errors.append("Triton kernel signatures must not specialize scalar parameters as tl.float32/tl.float64")
        normalized = [self._normalize_kernel_name_for_duplicates(name) for name in kernel_names]
        if len(normalized) != len(set(normalized)):
            errors.append(f"kernel names suggest duplicated dtype-specific implementations: {list(kernel_names)}")
        return errors

    def _validate_elementwise_contract(self, spec: OperatorSpec, content: str) -> List[str]:
        errors: List[str] = []
        if "BLOCK_SIZE" not in content:
            errors.append(f"{spec.module} elementwise implementation should expose BLOCK_SIZE as a kernel constexpr")
        if "torch.empty_like" not in content and "torch.empty" not in content and "out" in spec.signature:
            errors.append(f"{spec.module} should allocate/validate output tensors in the Python wrapper")
        return errors

    def _validate_openblas_level2_contract(self, spec: OperatorSpec, content: str) -> List[str]:
        errors: List[str] = []
        for token in ("m", "n", "lda", "incx", "incy", "uplo", "trans", "diag", "kl", "ku", "k", "beta"):
            if re.search(rf"\b{token}\b", spec.signature) and not re.search(rf"\b{token}\b", content):
                errors.append(f"{spec.module} signature includes {token}; implementation should validate/use it")
        if "packed" in spec.semantics.lower() and "AP" not in content and "ap" not in content:
            errors.append(f"{spec.module} packed-storage operator should use AP/ap packed input naming and indexing")
        if "band" in spec.semantics.lower() and not any(name in content for name in ("kl", "ku", " k,", " k)")):
            errors.append(f"{spec.module} band operator should use band-width parameters")
        if any(word in spec.result_policy.lower() for word in ("mutates x", "mutates y", "mutates a", "mutates ap")):
            if "@triton.autotune" in content and "restore_value" not in content:
                errors.append(f"{spec.module} mutates inputs; autotuned kernels must set restore_value or avoid autotune")
        return errors

    def _validate_complex_contract(self, content: str) -> List[str]:
        errors: List[str] = []
        if "tl.load" in content and "complex" in content and "view_as_real" not in content and ".view(" not in content:
            errors.append("complex Triton kernels should use flattened real views instead of native complex pointers")
        return errors

    def _attribute_name(self, node: ast.Attribute) -> str:
        parts = [node.attr]
        value = node.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))

    def _triton_kernel_names(self, tree: ast.AST) -> List[str]:
        result: List[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if self._is_triton_jit_decorator(decorator):
                    result.append(node.name)
                    break
        return result

    def _is_triton_jit_decorator(self, node: ast.AST) -> bool:
        target = node.func if isinstance(node, ast.Call) else node
        if isinstance(target, ast.Attribute):
            return target.attr == "jit"
        if isinstance(target, ast.Name):
            return target.id == "jit"
        return False

    def _normalize_kernel_name_for_duplicates(self, name: str) -> str:
        normalized = name.lower()
        normalized = re.sub(r"^(s|d|c|z)(?=[a-z_]+kernel$)", "", normalized)
        normalized = re.sub(r"(float32|float64|fp32|fp64|f32|f64)", "float", normalized)
        return normalized


class PromptFactory:
    """Build compact prompts for staged operator generation."""

    SYSTEM = textwrap.dedent(
        """
        You are an expert Triton, OpenBLAS/BLAS, and PyTorch operator engineer.
        Generate production Python code for the target repository.

        Return only valid JSON. Do not include Markdown fences. The JSON must match
        the schema requested by the user message. Include concise, human-reviewable
        engineering rationale. Do not write hidden chain-of-thought; write concrete
        decisions, invariants, risks, and validation expectations.

        Hard rule: Triton arithmetic is type-generic. Do not duplicate kernels only
        for float32 versus float64. Split kernels by algorithmic math shape only,
        for example real versus complex, vectorized versus reduction, or stage1
        versus stage2 when a true staged reduction is required.
        """
    ).strip()

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.impl_style = self._style_context(self._default_impl_examples(), max_chars_each=5000)
        self.test_style = self._style_context(self._default_test_examples(), max_chars_each=5000)
        self.bench_style = self._style_context(self._default_bench_examples(), max_chars_each=5000)

    def implementation_messages(
        self,
        spec: OperatorSpec,
        paths: TargetPaths,
        feedback: Optional[str] = None,
        current_files: Optional[Mapping[str, str]] = None,
    ) -> List[Dict[str, str]]:
        schema = {
            "path": paths.as_relative_dict(self.config.project_root)["operator"],
            "content": "complete Python source for the operator module",
            "rationale": {
                "kernel_strategy": "short explanation",
                "dtype_strategy": "how public wrappers share generic kernels",
                "semantic_invariants": ["..."],
                "edge_cases": ["..."],
                "known_risks": ["..."],
            },
        }
        user = f"""
        Generate the Triton operator implementation file.

        Target project profile:
        {json.dumps(self._project_prompt_dict(), indent=2)}

        Operator spec:
        {json.dumps(spec.to_prompt_dict(paths, self.config.project_root), indent=2, ensure_ascii=False)}

        Required JSON schema:
        {json.dumps(schema, indent=2)}

        Project implementation style examples:
        {self.impl_style}

        Kind-specific requirements:
        {self._kind_specific_requirements(spec)}

        Implementation requirements:
        - Output path must be exactly {schema["path"]}.
        - Use pure Python wrappers plus Triton kernels. Do not use CuPy, cuBLAS,
          OpenBLAS FFI, ctypes, or numpy.ctypeslib in the implementation.
        - Pass torch tensors or tensor views directly into Triton launches. Never
          pass tensor.data_ptr() to a Triton kernel.
        - Public functions must be exactly: {", ".join(spec.public_functions)}.
        - Validate dtype, device, dimensions, positive strides, output/result shape,
          and n <= 0 behavior in Python wrappers.
        - Use generic kernel names such as _{spec.module}_kernel,
          _{spec.module}_real_kernel, _{spec.module}_complex_kernel, or
          _{spec.module}_stage1_kernel. Do not name kernels after dtype-specific
          public APIs.
        - For OpenBLAS/BLAS semantics, match the BLAS contract exactly, including
          stride interpretation, in-place behavior, scalar result behavior, and
          1-based index behavior for BLAS index returns when applicable.
        - For Elementwise semantics, support broadcasting only if the spec says so;
          otherwise require compatible contiguous 1-D/flat logical views.
        - If an elementwise kernel uses linear pointer arithmetic, compact
          non-contiguous inputs with `.contiguous()` before flattening, or pass
          explicit strides and validate them. Never silently read a strided tensor
          as if it were contiguous.
        - For complex tensors, prefer torch.view_as_real(...).reshape(-1) or an
          equivalent real dtype view when Triton kernels need pointer arithmetic.
        - Avoid host synchronizations and timed-path allocations: no .cpu(), .item(),
          or per-call device scalar tensors unless the operator is scalar-only and
          the rationale explains why it is unavoidable.
        - For in-place autotuned kernels, set restore_value for every mutated pointer
          argument or avoid autotune.
        """
        user = self._append_optional_context(user, feedback, current_files)
        return [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": textwrap.dedent(user)}]

    def functional_test_messages(
        self,
        spec: OperatorSpec,
        paths: TargetPaths,
        feedback: Optional[str] = None,
        current_files: Optional[Mapping[str, str]] = None,
    ) -> List[Dict[str, str]]:
        schema = {
            "path": paths.as_relative_dict(self.config.project_root)["functional_test"],
            "content": "complete pytest source",
            "rationale": {
                "reference_strategy": "torch or local pure-python reference",
                "coverage": ["dtypes", "shapes", "strides", "edge cases"],
                "style_alignment": "how it follows the target project",
                "known_risks": ["..."],
            },
        }
        user = f"""
        Generate the functional pytest file for the operator.

        Target project profile:
        {json.dumps(self._project_prompt_dict(), indent=2)}

        Operator spec:
        {json.dumps(spec.to_prompt_dict(paths, self.config.project_root), indent=2, ensure_ascii=False)}

        Required JSON schema:
        {json.dumps(schema, indent=2)}

        Project test style examples:
        {self.test_style}

        Kind-specific requirements:
        {self._kind_specific_requirements(spec)}

        Functional test requirements:
        - Output path must be exactly {schema["path"]}.
        - Do not define Triton kernels in the test file.
        - Use pure torch or local Python reference implementations. Do not call
          OpenBLAS/cuBLAS/CuPy in functional tests unless the spec explicitly requires
          provider comparison and the project already depends on that provider.
        - Cover every public function: {", ".join(spec.public_functions)}.
        - Cover all requested dtype families: {", ".join(spec.dtypes)}.
        - Include edge cases from the spec, including n <= 0, strided views, in-place
          mutation boundaries, scalar/result dtypes, and non-contiguous logical views
          when relevant.
        - Keep quick-mode shapes modest so tests are suitable for CI and repair loops.
        - Use torch.testing.assert_close for numeric tensors and exact assertions for
          integer/index outputs.
        - For complex math involving sqrt/division, use dtype-appropriate tolerances
          rather than exact zero tolerance; real sign-style integer-valued outputs
          can use exact tolerance.
        """
        user = self._append_optional_context(user, feedback, current_files)
        return [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": textwrap.dedent(user)}]

    def benchmark_messages(
        self,
        spec: OperatorSpec,
        paths: TargetPaths,
        feedback: Optional[str] = None,
        current_files: Optional[Mapping[str, str]] = None,
    ) -> List[Dict[str, str]]:
        schema = {
            "path": paths.as_relative_dict(self.config.project_root)["benchmark"],
            "content": "complete pytest benchmark source",
            "rationale": {
                "benchmark_strategy": "Benchmark subclass or do_bench wrapper",
                "metrics": ["latency", "gbps or tflops when appropriate"],
                "dtype_coverage": ["..."],
                "style_alignment": "how it follows the target project",
                "known_risks": ["..."],
            },
        }
        style_requirement = (
            "Use benchmark.performance_utils.Benchmark, create a local bench variable, and call bench.run()."
            if self.config.benchmark_style == "flagblas"
            else "Use pytest plus triton.testing.do_bench or an equivalent synchronized timer."
        )
        user = f"""
        Generate the pytest benchmark file for the operator.

        Target project profile:
        {json.dumps(self._project_prompt_dict(), indent=2)}

        Operator spec:
        {json.dumps(spec.to_prompt_dict(paths, self.config.project_root), indent=2, ensure_ascii=False)}

        Required JSON schema:
        {json.dumps(schema, indent=2)}

        Project benchmark style examples:
        {self.bench_style}

        Kind-specific requirements:
        {self._kind_specific_requirements(spec)}

        Benchmark requirements:
        - Output path must be exactly {schema["path"]}.
        - {style_requirement}
        - Do not define Triton kernels in benchmark files.
        - For FlagBLAS Benchmark.get_input_iter, yield one flat tuple of call
          arguments, for example `yield (x,)` or `yield (n, x, 1, out, 1)`.
          If kwargs are unavoidable, include the kwargs dict as an item in that
          same tuple, for example `yield (x, {{"out": out}})`. Never write
          `yield (args_tuple), kwargs` or `yield (x,), {{}}`.
        - Benchmark the generated operator against equivalent torch work. Do not use
          clone-only, empty, precomputed, or non-equivalent baselines.
        - In-place operator wrappers and baselines must clone inputs inside timed
          functions before mutation so repeated do_bench calls do not reuse mutated
          inputs.
        - Benchmark wrappers must accept flat arguments and keyword arguments. Do not
          define wrappers as op(args, **kwargs), and do not yield (args_tuple), kwargs.
        - Use modest smoke shapes by default. Include larger shapes only behind
          project comprehensive modes if the target project has such a mode.
        - Report latency plus bandwidth or FLOP-derived metrics where meaningful.
        """
        user = self._append_optional_context(user, feedback, current_files)
        return [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": textwrap.dedent(user)}]

    def repair_messages(
        self,
        spec: OperatorSpec,
        paths: TargetPaths,
        failure_summary: str,
        current_files: Mapping[str, str],
    ) -> List[Dict[str, str]]:
        schema = {
            "files": [
                {
                    "path": "relative path for a changed operator, test, or benchmark file",
                    "content": "complete replacement file content",
                }
            ],
            "rationale": {
                "root_cause": "specific failure cause from logs",
                "repair_strategy": "specific changes made",
                "generic_kernel_compliance": "why dtype-duplicated kernels are not introduced",
                "validation_expectation": "what should pass after this repair",
            },
        }
        rel_paths = paths.as_relative_dict(self.config.project_root)
        user = f"""
        Repair the generated files using the failure logs.

        Operator spec:
        {json.dumps(spec.to_prompt_dict(paths, self.config.project_root), indent=2, ensure_ascii=False)}

        Required JSON schema:
        {json.dumps(schema, indent=2)}

        Allowed output paths:
        - {rel_paths["operator"]}
        - {rel_paths["functional_test"]}
        - {rel_paths["benchmark"]}

        Current files:
        {self._format_files(current_files)}

        Failure summary:
        {failure_summary}

        Repair constraints:
        - Return complete replacement content for every changed file.
        - Preserve project style and public API.
        - Do not define kernels in tests or benchmarks.
        - FlagBLAS Benchmark.get_input_iter must yield one flat tuple of call
          arguments, e.g. `yield (x,)`; never `yield (x,), {{}}` or
          `yield (args_tuple), kwargs`.
        - Do not create dtype-duplicated kernels.
        - Do not pass tensor.data_ptr() into Triton kernels.
        - Keep tests and benchmark smoke shapes modest.
        """
        return [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": textwrap.dedent(user)}]

    def _append_optional_context(
        self,
        user: str,
        feedback: Optional[str],
        current_files: Optional[Mapping[str, str]],
    ) -> str:
        if current_files:
            user += "\n\nCurrent generated files for context:\n" + self._format_files(current_files)
        if feedback:
            user += "\n\nFeedback to fix:\n" + feedback
        return user

    def _format_files(self, files: Mapping[str, str], max_chars_per_file: int = 12000) -> str:
        chunks = []
        for path, content in files.items():
            clipped = content
            if len(clipped) > max_chars_per_file:
                clipped = clipped[:max_chars_per_file] + "\n# ... truncated by hby operator distiller ...\n"
            chunks.append(f"### {path}\n```python\n{clipped}\n```")
        return "\n\n".join(chunks)

    def _kind_specific_requirements(self, spec: OperatorSpec) -> str:
        if spec.kind != "openblas_level2":
            return "- No extra kind-specific requirements beyond the operator spec."
        return textwrap.dedent(
            """
            - Treat this as an OpenBLAS/CBLAS Level-2 operator family.
            - Respect order/layout, uplo, trans/conjugate-transpose, diag, lda,
              band width, and packed-storage arguments when present in the signature.
            - Tests must cover both row-major/column-major or the target project
              layout convention, both unit and non-unit vector increments, beta=0
              and beta!=0 for y-updating routines, and transpose variants when present.
            - Matrix-vector routines must validate logical x/y lengths from trans.
            - Rank update routines must update only the intended dense, triangular,
              banded, or packed storage region.
            - Triangular solve/multiply routines mutate x in place; benchmark wrappers
              must clone x inside timed functions.
            - Hermitian routines must conjugate mirrored entries and keep diagonal
              imaginary parts semantically zero.
            - For complex Level-2 kernels, prefer flattened real views when Triton
              pointer arithmetic touches real and imaginary lanes.
            """
        ).strip()

    def _style_context(self, paths: Sequence[Path], max_chars_each: int) -> str:
        chunks: List[str] = []
        for path in paths:
            if path.exists():
                label = self._relative_label(path)
                chunks.append(f"# File: {label}\n{read_text_if_exists(path, max_chars_each)}")
        return "\n\n".join(chunks) or "# No local style examples found."

    def _relative_label(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.config.project_root.resolve()))
        except ValueError:
            return str(path)

    def _default_impl_examples(self) -> List[Path]:
        root = self.config.project_root
        candidates = [
            root / "src/flag_blas/ops/level2/gemv.py",
            root / "src/flag_blas/ops/level2/symv.py",
            root / "src/flag_blas/ops/level2/gbmv.py",
            root / "src/flag_blas/ops/level1/copy.py",
            root / "src/flag_blas/ops/level1/axpy.py",
            root / "src/flag_blas/ops/level1/asum.py",
        ]
        return [path for path in candidates if path.exists()]

    def _default_test_examples(self) -> List[Path]:
        root = self.config.project_root
        candidates = [
            root / "tests/test_gemv.py",
            root / "tests/test_symv.py",
            root / "tests/test_axpy.py",
            root / "tests/test_abs.py",
            root / "tests/conftest.py",
        ]
        return [path for path in candidates if path.exists()]

    def _default_bench_examples(self) -> List[Path]:
        root = self.config.project_root
        candidates = [
            root / "benchmark/test_gemv_perf.py",
            root / "benchmark/test_symv_perf.py",
            root / "benchmark/test_axpy_perf.py",
            root / "benchmark/performance_utils.py",
            root / "benchmark/conftest.py",
        ]
        return [path for path in candidates if path.exists()]

    def _project_prompt_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.config.profile,
            "benchmark_style": self.config.benchmark_style,
            "project_root": str(self.config.project_root),
            "operator_root": str(self.config.operator_root.relative_to(self.config.project_root)),
            "test_root": str(self.config.test_root.relative_to(self.config.project_root)),
            "benchmark_root": str(self.config.benchmark_root.relative_to(self.config.project_root)),
            "package_import_root": self.config.package_import_root,
        }


class FileManager:
    """Path resolution, controlled writes, and export integration."""

    def __init__(self, config: WorkflowConfig, trace: TraceLogger):
        self.config = config
        self.trace = trace

    def target_paths(self, spec: OperatorSpec) -> TargetPaths:
        paths = spec.paths
        default_subdir = self.config.default_operator_subdir
        if spec.kind == "openblas_level2" and default_subdir == "generated":
            default_subdir = "level2"
        operator = self._resolve_path(
            paths.get("operator"),
            self.config.operator_root / default_subdir / f"{spec.module}.py",
        )
        functional_test = self._resolve_path(
            paths.get("functional_test"),
            self.config.test_root / f"test_{spec.module}.py",
        )
        benchmark = self._resolve_path(
            paths.get("benchmark"),
            self.config.benchmark_root / f"test_{spec.module}_perf.py",
        )
        ensure_within(operator, (self.config.operator_root,))
        ensure_within(functional_test, (self.config.test_root,))
        ensure_within(benchmark, (self.config.benchmark_root,))
        return TargetPaths(operator=operator, functional_test=functional_test, benchmark=benchmark)

    def current_files(self, paths: TargetPaths) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for path in (paths.operator, paths.functional_test, paths.benchmark):
            if path.exists():
                result[self.relative_to_project(path)] = path.read_text(encoding="utf-8", errors="replace")
        return result

    def write_generated_file(self, path: Path, content: str, op: str, stage: str) -> Dict[str, Any]:
        ensure_within(path, self.config.allowed_write_roots)
        old_content = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        atomic_write(path, content.rstrip() + "\n")
        diff = "\n".join(
            difflib.unified_diff(
                old_content.splitlines(),
                content.rstrip().splitlines(),
                fromfile=f"a/{self.relative_to_project(path)}",
                tofile=f"b/{self.relative_to_project(path)}",
                lineterm="",
            )
        )
        info = {
            "path": self.relative_to_project(path),
            "sha256": sha256_text(content.rstrip() + "\n"),
            "diff_lines": diff.count("\n") + 1 if diff else 0,
        }
        self.trace.event(stage, "file_written", op=op, **info)
        return {"info": info, "diff": diff}

    def update_exports(self, specs_and_paths: Sequence[Tuple[OperatorSpec, TargetPaths]]) -> None:
        if not self.config.update_exports:
            return
        by_init: Dict[Path, List[Tuple[OperatorSpec, TargetPaths]]] = {}
        for spec, paths in specs_and_paths:
            init_path = paths.operator.parent / "__init__.py"
            by_init.setdefault(init_path, []).append((spec, paths))

        for init_path, entries in by_init.items():
            lines = [AUTO_EXPORT_BEGIN]
            names: List[str] = []
            for spec, paths in sorted(entries, key=lambda item: item[0].module):
                names.extend(spec.public_functions)
                lines.append(f"from .{paths.operator.stem} import {', '.join(spec.public_functions)}")
            lines.append("")
            lines.append("__all__ = sorted(set(globals().get('__all__', [])) | {")
            for name in sorted(set(names)):
                lines.append(f"    {name!r},")
            lines.append("})")
            lines.append(AUTO_EXPORT_END)
            self._replace_auto_block(init_path, "\n".join(lines) + "\n")
            self.trace.event("exports", "package_init_updated", path=self.relative_to_project(init_path))

        for extra_init in self.config.extra_export_inits:
            lines = [AUTO_EXPORT_BEGIN]
            names = []
            for spec, paths in sorted(specs_and_paths, key=lambda item: item[0].module):
                names.extend(spec.public_functions)
                module_path = self.module_path_for_operator(paths.operator)
                lines.append(f"from {module_path} import {', '.join(spec.public_functions)}")
            lines.append("")
            lines.append("try:")
            lines.append("    __all__")
            lines.append("except NameError:")
            lines.append("    __all__ = []")
            lines.append("for _hby_name in [")
            for name in sorted(set(names)):
                lines.append(f"    {name!r},")
            lines.append("]:")
            lines.append("    if _hby_name not in __all__:")
            lines.append("        __all__.append(_hby_name)")
            lines.append("del _hby_name")
            lines.append(AUTO_EXPORT_END)
            self._replace_auto_block(extra_init, "\n".join(lines) + "\n")
            self.trace.event("exports", "extra_init_updated", path=self.relative_to_project(extra_init))

    def module_path_for_operator(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            rel = resolved.relative_to(self.config.src_root.resolve())
            return ".".join(rel.with_suffix("").parts)
        except ValueError:
            try:
                rel = resolved.relative_to(self.config.operator_root.resolve())
                suffix = ".".join(rel.with_suffix("").parts)
                return f"{self.config.package_import_root}.{suffix}"
            except ValueError:
                return f"{self.config.package_import_root}.{path.stem}"

    def relative_to_project(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.config.project_root.resolve()))

    def path_from_payload(self, payload_path: str) -> Path:
        path = (self.config.project_root / payload_path).resolve()
        return ensure_within(path, self.config.allowed_write_roots)

    def _resolve_path(self, raw: Optional[str], default: Path) -> Path:
        if not raw:
            return default.resolve()
        path = Path(raw)
        if not path.is_absolute():
            path = self.config.project_root / path
        return path.resolve()

    def _replace_auto_block(self, path: Path, block: str) -> None:
        existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        pattern = re.compile(
            re.escape(AUTO_EXPORT_BEGIN) + r".*?" + re.escape(AUTO_EXPORT_END) + r"\n?",
            flags=re.DOTALL,
        )
        if pattern.search(existing):
            updated = pattern.sub(block, existing)
        else:
            separator = "\n\n" if existing.strip() else ""
            updated = existing.rstrip() + separator + block
        atomic_write(path, updated.rstrip() + "\n")


class CommandRunner:
    def __init__(self, config: WorkflowConfig, trace: TraceLogger):
        self.config = config
        self.trace = trace

    def run(self, cmd: List[str], *, stage: str, op: Optional[str], cwd: Path) -> CommandResult:
        self.trace.event(stage, "command_start", op=op, cmd=cmd)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.config.command_timeout,
                check=False,
            )
            result = CommandResult(
                cmd=cmd,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_s=time.monotonic() - started,
            )
        except subprocess.TimeoutExpired as exc:
            result = CommandResult(
                cmd=cmd,
                returncode=124,
                stdout=self._to_text(exc.stdout),
                stderr=self._to_text(exc.stderr) + f"\nCommand timed out after {self.config.command_timeout}s",
                duration_s=time.monotonic() - started,
            )
        self.trace.event(
            stage,
            "command_done" if result.ok else "command_failed",
            op=op,
            cmd=cmd,
            returncode=result.returncode,
            duration_s=round(result.duration_s, 3),
        )
        return result

    def _to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)


class DistillWorkflow:
    def __init__(self, config: WorkflowConfig, trace: TraceLogger):
        self.config = config
        self.trace = trace
        self.client = DeepSeekClient(config, trace)
        self.files = FileManager(config, trace)
        self.prompts = PromptFactory(config)
        self.validator = StaticValidator(config)
        self.runner = CommandRunner(config, trace)
        self.export_entries: Dict[str, Tuple[OperatorSpec, TargetPaths]] = {}

    def run(self, selected_specs: Sequence[OperatorSpec], force: bool = False) -> None:
        self.initialize(selected_specs)
        state = self.trace.load_state()
        completed = set(state.get("completed_ops", []))
        batches = list(chunked(selected_specs, self.config.batch_size))

        for batch_index, batch in enumerate(batches, start=1):
            state["current_batch"] = batch_index
            state["current_batch_ops"] = [spec.module for spec in batch]
            self.trace.save_state(state)
            generated: List[Tuple[OperatorSpec, TargetPaths]] = []
            self.trace.event("batch", "start", batch_index=batch_index, ops=[spec.module for spec in batch])

            for spec in batch:
                paths = self.files.target_paths(spec)
                if spec.module in completed and not force:
                    self.trace.event("operator", "skipped_completed", op=spec.module)
                    continue
                try:
                    if self.config.validate_only:
                        self.validate_existing_operator(spec, paths)
                    elif self.config.dry_run:
                        self.write_dry_run_prompts(spec, paths)
                    elif self.config.operation_mode == "generate":
                        self.generate_operator(spec, paths)
                    elif self.config.operation_mode == "repair":
                        self.repair_existing_operator(spec, paths)
                    else:
                        self.process_operator(spec, paths)
                    generated.append((spec, paths))
                    completed.add(spec.module)
                    failed_ops = dict(state.get("failed_ops", {}))
                    failed_ops.pop(spec.module, None)
                    state["completed_ops"] = sorted(completed)
                    state["failed_ops"] = failed_ops
                    self.trace.save_state(state)
                except Exception as exc:
                    failed_ops = dict(state.get("failed_ops", {}))
                    failed_ops[spec.module] = f"{type(exc).__name__}: {exc}"
                    state["failed_ops"] = failed_ops
                    self.trace.save_state(state)
                    self.trace.event(
                        "operator",
                        "failed",
                        op=spec.module,
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(limit=8),
                    )
                    if not self.config.continue_on_failure:
                        raise

            if generated and not self.config.dry_run:
                self.update_exports(generated)
            self.trace.event("batch", "done", batch_index=batch_index, ops=[spec.module for spec in batch])
            if batch_index < len(batches) and self.config.cooldown > 0:
                time.sleep(self.config.cooldown)

        if state.get("failed_ops"):
            raise DistillError(f"workflow completed with failures: {state['failed_ops']}")

    def initialize(self, selected_specs: Sequence[OperatorSpec]) -> None:
        for path in (self.config.operator_root, self.config.test_root, self.config.benchmark_root, self.config.trace_root):
            path.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for spec in selected_specs:
            paths = self.files.target_paths(spec)
            manifest[spec.module] = spec.to_prompt_dict(paths, self.config.project_root)
        self.trace.save_json("manifest.json", manifest)
        self.trace.event(
            "workflow",
            "initialized",
            project_root=str(self.config.project_root),
            model=self.config.model,
            base_url=self.config.base_url,
            trace_dir=str(self.trace.run_dir),
        )

    def write_dry_run_prompts(self, spec: OperatorSpec, paths: TargetPaths) -> None:
        attempt_dir = self.trace.attempt_dir(spec.module, 1)
        for stage in ("implementation", "functional_test", "benchmark"):
            messages = self._messages_for_stage(spec, paths, stage)
            self.trace.save_json(f"ops/{spec.module}/attempt_001/{stage}_messages.json", messages)
        self.trace.event("operator", "dry_run_prompts_written", op=spec.module, attempt_dir=str(attempt_dir))

    def validate_existing_operator(self, spec: OperatorSpec, paths: TargetPaths) -> None:
        self.trace.event("operator", "validate_existing_start", op=spec.module)
        errors = self._combined_static_errors(spec, paths)
        if errors:
            self.trace.save_json(f"ops/{spec.module}/validate_existing_static_errors.json", errors)
            raise DistillError(f"static validation failed for {spec.module}: {errors}")
        if not self.run_validations(spec, paths, attempt=0):
            raise DistillError(f"runtime validation failed for {spec.module}")
        self.trace.event("operator", "validate_existing_done", op=spec.module)

    def process_operator(self, spec: OperatorSpec, paths: TargetPaths) -> None:
        self.trace.event("operator", "start", op=spec.module)
        attempt = 1
        self.generate_operator(spec, paths, attempt=attempt)
        if self.run_validations(spec, paths, attempt=attempt):
            self.trace.event("operator", "done", op=spec.module, attempt=attempt)
            return

        for repair_attempt in range(1, self.config.max_repair_attempts + 1):
            attempt += 1
            self.repair_operator(spec, paths, repair_attempt=repair_attempt, attempt=attempt)
            self.update_exports([(spec, paths)])
            if self.run_validations(spec, paths, attempt=attempt):
                self.trace.event("operator", "done_after_repair", op=spec.module, attempt=attempt)
                return

        raise DistillError(f"{spec.module} failed after {self.config.max_repair_attempts} repair attempts")

    def generate_operator(self, spec: OperatorSpec, paths: TargetPaths, attempt: int = 1) -> None:
        self.trace.event("operator", "generate_start", op=spec.module, attempt=attempt)
        self.generate_stage_with_retries(spec, paths, "implementation", attempt)
        self.generate_stage_with_retries(spec, paths, "functional_test", attempt)
        self.generate_stage_with_retries(spec, paths, "benchmark", attempt)
        self.update_exports([(spec, paths)])
        self.trace.event("operator", "generate_done", op=spec.module, attempt=attempt)

    def repair_existing_operator(self, spec: OperatorSpec, paths: TargetPaths) -> None:
        self.trace.event("operator", "repair_existing_start", op=spec.module)
        if self.run_validations(spec, paths, attempt=0):
            self.trace.event("operator", "repair_existing_noop_valid", op=spec.module)
            return

        for repair_attempt in range(1, self.config.max_repair_attempts + 1):
            attempt = repair_attempt
            self.repair_operator(spec, paths, repair_attempt=repair_attempt, attempt=attempt)
            self.update_exports([(spec, paths)])
            if self.run_validations(spec, paths, attempt=attempt):
                self.trace.event("operator", "repair_existing_done", op=spec.module, attempt=attempt)
                return

        raise DistillError(f"{spec.module} failed after {self.config.max_repair_attempts} repair attempts")

    def update_exports(self, entries: Sequence[Tuple[OperatorSpec, TargetPaths]]) -> None:
        for spec, paths in entries:
            self.export_entries[self.files.relative_to_project(paths.operator)] = (spec, paths)
        self.files.update_exports(list(self.export_entries.values()))

    def generate_stage_with_retries(self, spec: OperatorSpec, paths: TargetPaths, stage: str, attempt: int) -> None:
        feedback: Optional[str] = None
        for stage_attempt in range(1, self.config.stage_retries + 1):
            attempt_dir = self.trace.attempt_dir(spec.module, attempt)
            messages = self._messages_for_stage(spec, paths, stage, feedback=feedback)
            api_stage = f"{stage}_try{stage_attempt:02d}"
            payload = self.client.chat_json(messages=messages, stage=api_stage, op=spec.module, attempt_dir=attempt_dir)
            try:
                self.apply_single_file_payload(spec, paths, stage, payload, attempt_dir)
                return
            except DistillError as exc:
                feedback = str(exc)
                self.trace.event(
                    stage,
                    "static_rejected",
                    op=spec.module,
                    stage_attempt=stage_attempt,
                    feedback=feedback,
                )
                if stage_attempt == self.config.stage_retries:
                    raise

    def _messages_for_stage(
        self,
        spec: OperatorSpec,
        paths: TargetPaths,
        stage: str,
        feedback: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        current = self.files.current_files(paths)
        if stage == "implementation":
            return self.prompts.implementation_messages(spec, paths, feedback=feedback, current_files=current or None)
        if stage == "functional_test":
            return self.prompts.functional_test_messages(spec, paths, feedback=feedback, current_files=current or None)
        if stage == "benchmark":
            return self.prompts.benchmark_messages(spec, paths, feedback=feedback, current_files=current or None)
        raise DistillError(f"unknown generation stage: {stage}")

    def apply_single_file_payload(
        self,
        spec: OperatorSpec,
        paths: TargetPaths,
        stage: str,
        payload: Mapping[str, Any],
        attempt_dir: Path,
    ) -> None:
        payload_path = payload.get("path")
        content = payload.get("content")
        rationale = payload.get("rationale", {})
        if not isinstance(payload_path, str) or not isinstance(content, str):
            raise DistillError(f"{stage} payload must contain string path and content")
        expected_path = {
            "implementation": paths.operator,
            "functional_test": paths.functional_test,
            "benchmark": paths.benchmark,
        }[stage]
        path = self.files.path_from_payload(payload_path)
        if path != expected_path.resolve():
            raise DistillError(f"{stage} path mismatch: expected {expected_path}, got {path}")

        errors = self._validate_stage_content(spec, stage, content)
        self.trace.save_json(
            f"ops/{spec.module}/{attempt_dir.name}/{stage}_static_report.json",
            {"errors": errors, "sha256": sha256_text(content)},
        )
        if errors:
            raise DistillError(f"{stage} static validation errors: {errors}")

        self.trace.save_text(
            f"ops/{spec.module}/{attempt_dir.name}/{stage}_rationale.md",
            render_rationale(stage, rationale),
        )
        generated_dir = attempt_dir / "generated_files"
        generated_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(generated_dir / path.name, content.rstrip() + "\n")
        write_result = self.files.write_generated_file(path, content, spec.module, stage)
        if write_result["diff"]:
            atomic_write(attempt_dir / f"{stage}.diff", write_result["diff"] + "\n")

    def repair_operator(self, spec: OperatorSpec, paths: TargetPaths, repair_attempt: int, attempt: int) -> None:
        attempt_dir = self.trace.attempt_dir(spec.module, attempt)
        previous_attempt_dir = self.trace.attempt_dir(spec.module, attempt - 1)
        failure_summary = self.build_failure_summary(spec, previous_attempt_dir)
        messages = self.prompts.repair_messages(spec, paths, failure_summary, self.files.current_files(paths))
        payload = self.client.chat_json(messages=messages, stage="repair", op=spec.module, attempt_dir=attempt_dir)
        rationale = payload.get("rationale", {})
        self.trace.save_text(
            f"ops/{spec.module}/{attempt_dir.name}/repair_rationale.md",
            render_rationale("repair", rationale),
        )
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            raise DistillError("repair payload must contain a non-empty files list")

        allowed = {paths.operator.resolve(), paths.functional_test.resolve(), paths.benchmark.resolve()}
        staged: List[Tuple[Path, str]] = []
        all_errors: List[str] = []
        for item in files:
            if not isinstance(item, dict):
                all_errors.append("repair file item is not an object")
                continue
            payload_path = item.get("path")
            content = item.get("content")
            if not isinstance(payload_path, str) or not isinstance(content, str):
                all_errors.append("repair file item missing string path/content")
                continue
            path = self.files.path_from_payload(payload_path)
            if path not in allowed:
                all_errors.append(f"repair path not allowed for {spec.module}: {path}")
                continue
            stage = self._stage_for_path(paths, path)
            all_errors.extend(self._validate_stage_content(spec, stage, content))
            staged.append((path, content))

        if all_errors:
            self.trace.save_json(f"ops/{spec.module}/{attempt_dir.name}/repair_static_report.json", {"errors": all_errors})
            raise DistillError(f"repair static validation failed: {all_errors}")

        generated_dir = attempt_dir / "generated_files"
        generated_dir.mkdir(parents=True, exist_ok=True)
        repair_diff_chunks = []
        for path, content in staged:
            atomic_write(generated_dir / path.name, content.rstrip() + "\n")
            write_result = self.files.write_generated_file(path, content, spec.module, "repair")
            if write_result["diff"]:
                repair_diff_chunks.append(write_result["diff"])
        if repair_diff_chunks:
            atomic_write(attempt_dir / "repair.diff", "\n\n".join(repair_diff_chunks) + "\n")
        self.trace.event("repair", "applied", op=spec.module, repair_attempt=repair_attempt, files=[str(p) for p, _ in staged])

    def run_validations(self, spec: OperatorSpec, paths: TargetPaths, attempt: int) -> bool:
        attempt_dir = self.trace.attempt_dir(spec.module, attempt)
        static_errors = self._combined_static_errors(spec, paths)
        self.trace.save_json(f"ops/{spec.module}/{attempt_dir.name}/combined_static_report.json", {"errors": static_errors})
        if static_errors:
            self.trace.event("validation", "static_failed", op=spec.module, errors=static_errors)
            return False
        if not self.run_compile_check(spec, paths, attempt_dir):
            return False
        if not self.config.skip_tests and not self.run_functional_tests(spec, paths, attempt_dir):
            return False
        if not self.config.skip_benchmark and not self.run_benchmark_smoke(spec, paths, attempt_dir):
            return False
        return True

    def run_compile_check(self, spec: OperatorSpec, paths: TargetPaths, attempt_dir: Path) -> bool:
        cmd = [
            sys.executable,
            "-m",
            "py_compile",
            str(paths.operator),
            str(paths.functional_test),
            str(paths.benchmark),
        ]
        result = self.runner.run(cmd, stage="compile", op=spec.module, cwd=self.config.project_root)
        self.trace.save_text(f"ops/{spec.module}/{attempt_dir.name}/compile.log", result.combined_output())
        return result.ok

    def run_functional_tests(self, spec: OperatorSpec, paths: TargetPaths, attempt_dir: Path) -> bool:
        cmd = [sys.executable, "-m", "pytest", str(paths.functional_test), "-q"]
        result = self.runner.run(cmd, stage="functional", op=spec.module, cwd=self.config.project_root)
        self.trace.save_text(f"ops/{spec.module}/{attempt_dir.name}/functional.log", result.combined_output())
        return result.ok

    def run_benchmark_smoke(self, spec: OperatorSpec, paths: TargetPaths, attempt_dir: Path) -> bool:
        cmd = [sys.executable, "-m", "pytest", str(paths.benchmark), "-q"]
        if self.config.benchmark_style == "flagblas":
            cmd.extend(["--mode=kernel", "--level=core", "--warmup=1", "--iter=1"])
        result = self.runner.run(cmd, stage="benchmark", op=spec.module, cwd=self.config.project_root)
        self.trace.save_text(f"ops/{spec.module}/{attempt_dir.name}/benchmark.log", result.combined_output())
        return result.ok

    def build_failure_summary(self, spec: OperatorSpec, attempt_dir: Path) -> str:
        parts = []
        for log_name in ("combined_static_report.json", "compile.log", "functional.log", "benchmark.log"):
            path = attempt_dir / log_name
            if path.exists():
                parts.append(f"## {log_name}\n{summarize_log(path.read_text(encoding='utf-8', errors='replace'))}")
        if not parts:
            parts.append("No validation logs were available. Inspect generated code for static and runtime failures.")
        return "\n\n".join(parts)

    def _combined_static_errors(self, spec: OperatorSpec, paths: TargetPaths) -> List[str]:
        errors: List[str] = []
        if paths.operator.exists():
            errors.extend(self.validator.validate_operator(spec, paths.operator.read_text(encoding="utf-8", errors="replace")))
        else:
            errors.append(f"missing {self.files.relative_to_project(paths.operator)}")
        if paths.functional_test.exists():
            errors.extend(self.validator.validate_test(spec, paths.functional_test.read_text(encoding="utf-8", errors="replace")))
        else:
            errors.append(f"missing {self.files.relative_to_project(paths.functional_test)}")
        if paths.benchmark.exists():
            errors.extend(self.validator.validate_benchmark(spec, paths.benchmark.read_text(encoding="utf-8", errors="replace")))
        else:
            errors.append(f"missing {self.files.relative_to_project(paths.benchmark)}")
        return errors

    def _validate_stage_content(self, spec: OperatorSpec, stage: str, content: str) -> List[str]:
        if stage == "implementation":
            return self.validator.validate_operator(spec, content)
        if stage == "functional_test":
            return self.validator.validate_test(spec, content)
        if stage == "benchmark":
            return self.validator.validate_benchmark(spec, content)
        raise DistillError(f"unknown stage: {stage}")

    def _stage_for_path(self, paths: TargetPaths, path: Path) -> str:
        if path == paths.operator.resolve():
            return "implementation"
        if path == paths.functional_test.resolve():
            return "functional_test"
        if path == paths.benchmark.resolve():
            return "benchmark"
        raise DistillError(f"unknown path for stage mapping: {path}")


def render_rationale(stage: str, rationale: Any) -> str:
    lines = [f"# {stage} rationale", ""]
    if isinstance(rationale, dict):
        for key, value in rationale.items():
            lines.append(f"## {key.replace('_', ' ').title()}")
            if isinstance(value, list):
                for item in value:
                    lines.append(f"- {item}")
            elif isinstance(value, dict):
                lines.append(json.dumps(value, indent=2, ensure_ascii=False))
            else:
                lines.append(str(value))
            lines.append("")
    else:
        lines.append(str(rationale))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def summarize_log(text: str, max_chars: int = 10000) -> str:
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    important = [
        line
        for line in lines
        if any(token in line for token in ("FAILED", "ERROR", "Traceback", "AssertionError", "RuntimeError", "SyntaxError"))
    ]
    head = "\n".join(lines[:120])
    tail = "\n".join(lines[-220:])
    middle = "\n".join(important[:120])
    summary = "\n\n".join(part for part in (head, "# important lines", middle, "# tail", tail) if part.strip())
    if len(summary) > max_chars:
        return summary[-max_chars:]
    return summary


def is_full_operator_mapping(raw: Mapping[str, Any]) -> bool:
    required = {
        "name",
        "module",
        "kind",
        "public_functions",
        "dtypes",
        "signature",
        "semantics",
        "reference",
    }
    return required.issubset(raw.keys())


def expand_openblas_level2_specs(raw: Any) -> List[OperatorSpec]:
    """Expand compact OpenBLAS Level-2 interface specs into full OperatorSpec objects."""

    if isinstance(raw, list):
        return _expand_openblas_level2_sequence(raw, defaults={})
    if not isinstance(raw, Mapping):
        raise DistillError("openblas_level2 compact spec must be a mapping or list")

    defaults = {
        "api_style": raw.get("api_style", "cblas"),
        "min_precisions": raw.get("min_precisions"),
        "precisions": raw.get("precisions"),
        "path_templates": raw.get("path_templates", {}),
    }
    result: List[OperatorSpec] = []
    if raw.get("interfaces"):
        result.extend(_expand_openblas_level2_interfaces(raw["interfaces"], defaults=defaults))
    if raw.get("operators"):
        result.extend(_expand_openblas_level2_sequence(raw["operators"], defaults=defaults))
    if raw.get("families"):
        result.extend(_expand_openblas_level2_sequence(raw["families"], defaults=defaults))
    if raw.get("family") or raw.get("public_functions"):
        result.extend(_expand_openblas_level2_sequence([raw], defaults=defaults))
    if not result:
        raise DistillError("openblas_level2 compact spec did not define interfaces, operators, families, or family")
    deduplicated = _deduplicate_level2_specs(result)
    min_precisions = defaults.get("min_precisions")
    if min_precisions is not None:
        enforce_min_precisions(deduplicated, int(min_precisions))
    return deduplicated


def _expand_openblas_level2_sequence(items: Sequence[Any], defaults: Mapping[str, Any]) -> List[OperatorSpec]:
    result: List[OperatorSpec] = []
    strings = [item for item in items if isinstance(item, str)]
    if strings and len(strings) == len(items):
        try:
            return _expand_openblas_level2_interfaces(strings, defaults=defaults)
        except DistillError:
            return [_make_openblas_level2_family_spec(item, {}, defaults) for item in strings]

    for item in items:
        if isinstance(item, str):
            result.append(_make_openblas_level2_family_spec(item, {}, defaults))
        elif isinstance(item, Mapping):
            if is_full_operator_mapping(item):
                result.append(OperatorSpec.from_mapping(item))
            elif item.get("interfaces") or item.get("public_functions"):
                merged_defaults = {**defaults, **{k: item[k] for k in ("api_style", "min_precisions", "precisions", "path_templates") if k in item}}
                result.extend(_expand_openblas_level2_interfaces(item.get("interfaces") or item.get("public_functions"), defaults=merged_defaults, item_defaults=item))
            elif item.get("family"):
                result.append(_make_openblas_level2_family_spec(str(item["family"]), item, defaults))
            else:
                raise DistillError(f"unknown openblas_level2 compact item: {item}")
        else:
            raise DistillError(f"openblas_level2 item must be a string or mapping, got {type(item).__name__}")
    return result


def _expand_openblas_level2_interfaces(
    interfaces: Sequence[Any],
    defaults: Mapping[str, Any],
    item_defaults: Optional[Mapping[str, Any]] = None,
) -> List[OperatorSpec]:
    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for interface in normalize_compact_list(interfaces):
        prefix, family, public_name = parse_openblas_level2_interface(str(interface))
        grouped.setdefault(family, []).append((prefix, public_name))

    result = []
    for family, entries in grouped.items():
        seen_prefixes: List[str] = []
        public_names: List[str] = []
        for prefix, public_name in entries:
            if prefix not in seen_prefixes:
                seen_prefixes.append(prefix)
            if public_name not in public_names:
                public_names.append(public_name)
        item = dict(item_defaults or {})
        item["precisions"] = seen_prefixes
        item["public_functions"] = public_names
        result.append(_make_openblas_level2_family_spec(family, item, defaults))
    return result


def _make_openblas_level2_family_spec(
    family: str,
    item: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> OperatorSpec:
    family = normalize_openblas_family_name(family)
    if family not in OPENBLAS_LEVEL2_CATALOG:
        raise DistillError(f"unknown OpenBLAS Level-2 family {family!r}; available: {sorted(OPENBLAS_LEVEL2_CATALOG)}")
    entry = OPENBLAS_LEVEL2_CATALOG[family]
    prefixes = normalize_openblas_precisions(item.get("precisions") or defaults.get("precisions") or entry["allowed_precisions"][:2])
    invalid = [prefix for prefix in prefixes if prefix not in entry["allowed_precisions"]]
    if invalid:
        raise DistillError(
            f"{family} does not support precision prefixes {invalid}; allowed={entry['allowed_precisions']}"
        )
    min_precisions = item.get("min_precisions", defaults.get("min_precisions"))
    if min_precisions is not None and not item.get("_allow_partial") and len(set(prefixes)) < int(min_precisions):
        raise DistillError(f"{family} requires at least {min_precisions} precision variants, got {prefixes}")

    public_functions = tuple(str(name) for name in item.get("public_functions", [prefix + family for prefix in prefixes]))
    api_style = str(item.get("api_style", defaults.get("api_style", "cblas")))
    signature_key = "flagblas_signature" if api_style == "flagblas" else "cblas_signature"
    signature = str(item.get("signature", entry[signature_key]))
    dtypes = tuple(OPENBLAS_PRECISION_DTYPES[prefix] for prefix in prefixes)
    extra_edge_cases = tuple(str(x) for x in item.get("edge_cases", ()))
    extra_notes = tuple(str(x) for x in item.get("notes", ()))
    kernel_policy = entry["kernel_policy"]
    if item.get("kernel_policy"):
        kernel_policy += " " + str(item["kernel_policy"])
    kernel_policy += " Keep one operator-family module and avoid duplicate float32/float64 Triton kernels."
    notes = (
        "Expanded from compact OpenBLAS Level-2 interface input.",
        f"API style: {api_style}.",
        "Precision mapping: " + ", ".join(f"{prefix}={OPENBLAS_PRECISION_DTYPES[prefix]}" for prefix in prefixes) + ".",
    ) + extra_notes
    paths = dict(item.get("paths", {}))
    paths.update(_render_path_templates(defaults.get("path_templates") or {}, family, item.get("module", family)))
    return OperatorSpec(
        name=str(item.get("name", family)),
        module=str(item.get("module", family)),
        kind="openblas_level2",
        public_functions=public_functions,
        dtypes=dtypes,
        signature=signature,
        semantics=str(item.get("semantics", f"OpenBLAS Level-2 {family.upper()} ({entry['title']}): {entry['semantics']}")),
        reference=str(item.get("reference", f"OpenBLAS CBLAS {', '.join('cblas_' + name for name in public_functions)}")),
        edge_cases=OPENBLAS_LEVEL2_COMMON_EDGE_CASES + tuple(str(x) for x in entry.get("edge_cases", ())) + extra_edge_cases,
        kernel_policy=kernel_policy,
        result_policy=str(item.get("result_policy", entry["result_policy"])),
        notes=notes,
        paths=paths,
    )


def _render_path_templates(path_templates: Mapping[str, Any], family: str, module: Any) -> Dict[str, str]:
    result: Dict[str, str] = {}
    values = {"family": family, "module": str(module), "name": family}
    for key in ("operator", "functional_test", "benchmark"):
        template = path_templates.get(key)
        if template:
            result[key] = str(template).format(**values)
    return result


def _deduplicate_level2_specs(specs: Sequence[OperatorSpec]) -> List[OperatorSpec]:
    merged: Dict[str, OperatorSpec] = {}
    for spec in specs:
        if spec.module not in merged:
            merged[spec.module] = spec
            continue
        previous = merged[spec.module]
        public_functions = tuple(dict.fromkeys(previous.public_functions + spec.public_functions))
        dtypes = tuple(dict.fromkeys(previous.dtypes + spec.dtypes))
        prefixes = tuple(_prefix_for_dtype(dtype) for dtype in dtypes)
        merged[spec.module] = dataclasses.replace(
            previous,
            public_functions=public_functions,
            dtypes=dtypes,
            notes=tuple(dict.fromkeys(previous.notes + spec.notes + ("Merged from multiple compact entries.",))),
            reference=f"{previous.reference}; {spec.reference}",
            kernel_policy=previous.kernel_policy,
        )
        family = spec.module
        if family in OPENBLAS_LEVEL2_CATALOG:
            allowed = OPENBLAS_LEVEL2_CATALOG[family]["allowed_precisions"]
            invalid = [prefix for prefix in prefixes if prefix not in allowed]
            if invalid:
                raise DistillError(f"{family} merged invalid precision prefixes {invalid}")
    return list(merged.values())


def enforce_min_precisions(specs: Sequence[OperatorSpec], min_precisions: int) -> None:
    for spec in specs:
        if spec.kind != "openblas_level2":
            continue
        prefixes = {_prefix_for_dtype(dtype) for dtype in spec.dtypes}
        if len(prefixes) < min_precisions:
            raise DistillError(f"{spec.module} requires at least {min_precisions} precision variants, got {sorted(prefixes)}")


def parse_openblas_level2_interface(interface: str) -> Tuple[str, str, str]:
    public_name = normalize_openblas_interface_name(interface)
    if len(public_name) < 2:
        raise DistillError(f"invalid OpenBLAS interface name: {interface!r}")
    prefix = public_name[0]
    family = public_name[1:]
    if prefix not in OPENBLAS_PRECISION_DTYPES:
        raise DistillError(f"{interface!r} does not start with s/d/c/z precision prefix")
    if family not in OPENBLAS_LEVEL2_CATALOG:
        raise DistillError(f"{interface!r} maps to unknown Level-2 family {family!r}")
    if prefix not in OPENBLAS_LEVEL2_CATALOG[family]["allowed_precisions"]:
        raise DistillError(f"{interface!r} uses precision {prefix!r}, but {family} supports {OPENBLAS_LEVEL2_CATALOG[family]['allowed_precisions']}")
    return prefix, family, public_name


def normalize_openblas_interface_name(interface: str) -> str:
    name = interface.strip().lower()
    name = name.split("(", 1)[0].strip()
    for prefix in ("cblas_", "openblas_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def normalize_openblas_family_name(family: str) -> str:
    name = family.strip().lower()
    for prefix in ("cblas_", "openblas_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name and name[0] in OPENBLAS_PRECISION_DTYPES and name[1:] in OPENBLAS_LEVEL2_CATALOG:
        name = name[1:]
    return name


def normalize_openblas_precisions(raw: Any) -> Tuple[str, ...]:
    parts = normalize_compact_list(raw)
    result: List[str] = []
    for part in parts:
        key = part.lower()
        prefix = {
            "single": "s",
            "double": "d",
            "complex_single": "c",
            "complex_double": "z",
            "float": "s",
            "double_complex": "z",
            "fp32": "s",
            "fp64": "d",
            "f32": "s",
            "f64": "d",
            "single_complex": "c",
            "float32": "s",
            "float64": "d",
            "complex64": "c",
            "complex128": "z",
        }.get(key, key)
        if prefix not in OPENBLAS_PRECISION_DTYPES:
            raise DistillError(f"unknown OpenBLAS precision {part!r}; use s/d/c/z or dtype names")
        if prefix not in result:
            result.append(prefix)
    return tuple(result)


def normalize_compact_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part.strip() for part in re.split(r"[, ]+", raw) if part.strip()]
    return [str(part).strip() for part in raw]


def _prefix_for_dtype(dtype: str) -> str:
    for prefix, value in OPENBLAS_PRECISION_DTYPES.items():
        if value == dtype:
            return prefix
    raise DistillError(f"unknown dtype for OpenBLAS precision mapping: {dtype}")


def load_operator_txt_specs(path: Path) -> List[OperatorSpec]:
    """Parse a human-maintained txt interface list into OperatorSpec objects.

    The parser is intentionally permissive because this file is usually written
    by humans before generation. Supported examples:

        cblas_sgemv float32
        cblas_sgemv float32,float64
        gemv: s,d
        trmv float32 float64
        cblas_strsv, cblas_dtrsv
    """

    config: Dict[str, Any] = {
        "api_style": "cblas",
        "min_precisions": None,
        "precisions": None,
        "interfaces": [],
        "operators": [],
    }
    full_specs: List[OperatorSpec] = []

    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = strip_txt_comment(raw_line).strip()
        if not line:
            continue
        directive = parse_txt_directive(line)
        if directive:
            key, value = directive
            if key in {"kind", "profile"}:
                if value not in {"openblas_level2", "level2"}:
                    raise DistillError(f"{path}:{lineno}: txt parser currently supports openblas_level2, got {value!r}")
                continue
            if key == "api_style":
                config["api_style"] = value
                continue
            if key == "min_precisions":
                config["min_precisions"] = int(value)
                continue
            if key in {"precisions", "types", "dtypes", "default_precisions"}:
                config["precisions"] = normalize_openblas_precisions(value)
                continue

        try:
            spec_items = parse_txt_operator_line(line, config)
        except DistillError as exc:
            raise DistillError(f"{path}:{lineno}: {exc}") from exc

        for item in spec_items:
            if isinstance(item, OperatorSpec):
                full_specs.append(item)
            elif item.get("interfaces"):
                config["interfaces"].extend(item["interfaces"])
            else:
                config["operators"].append(item)

    compact = {
        "api_style": config["api_style"],
        "interfaces": config["interfaces"],
        "operators": config["operators"],
    }
    if config.get("min_precisions") is not None:
        compact["min_precisions"] = config["min_precisions"]
    if config.get("precisions") is not None:
        compact["precisions"] = config["precisions"]

    specs = full_specs
    if compact["interfaces"] or compact["operators"]:
        specs.extend(expand_openblas_level2_specs(compact))
    if not specs:
        raise DistillError(f"{path} did not contain any parsable operator interfaces")
    return _deduplicate_level2_specs(specs)


def strip_txt_comment(line: str) -> str:
    for marker in ("#", "//"):
        idx = line.find(marker)
        if idx != -1:
            line = line[:idx]
    return line


def parse_txt_directive(line: str) -> Optional[Tuple[str, str]]:
    match = re.match(r"^\s*([A-Za-z_][\w-]*)\s*[:=]\s*(.+?)\s*$", line)
    if not match:
        return None
    key = match.group(1).strip().lower().replace("-", "_")
    value = match.group(2).strip()
    if key in {"kind", "profile", "api_style", "min_precisions", "precisions", "types", "dtypes", "default_precisions"}:
        return key, value
    return None


def parse_txt_operator_line(line: str, config: Mapping[str, Any]) -> List[Any]:
    head = ""
    rest = ""
    if ":" in line:
        head, rest = line.split(":", 1)
        raw_tokens = [head.strip()] + split_txt_tokens(rest)
    else:
        raw_tokens = split_txt_tokens(line)
    tokens = [token for token in (normalize_token(token) for token in raw_tokens) if token]
    if not tokens:
        return []
    if tokens[0] in {"openblas_level2", "level2"}:
        return []

    dtype_tokens = [token for token in tokens[1:] if is_precision_like(token)]
    symbol_tokens = [token for token in tokens if not is_precision_like(token)]
    result: List[Any] = []

    # A line with explicit dtype tokens lets the parser infer the full family,
    # even when the first token is a single-precision interface like sgemv.
    if dtype_tokens:
        precisions = normalize_openblas_precisions(dtype_tokens)
        for symbol in symbol_tokens:
            family = family_from_symbol(symbol)
            result.append(
                {
                    "family": family,
                    "precisions": precisions,
                    "api_style": config.get("api_style", "cblas"),
                    "_allow_partial": True,
                }
            )
        return result

    interfaces: List[str] = []
    for symbol in symbol_tokens:
        if maybe_openblas_interface(symbol):
            interfaces.append(normalize_openblas_interface_name(symbol))
        elif normalize_openblas_family_name(symbol) in OPENBLAS_LEVEL2_CATALOG:
            result.append({"family": normalize_openblas_family_name(symbol), "api_style": config.get("api_style", "cblas")})
        else:
            raise DistillError(f"cannot parse token {symbol!r}; use an OpenBLAS Level-2 interface or family name")
    if interfaces:
        result.append({"interfaces": interfaces})
    return result


def split_txt_tokens(value: str) -> List[str]:
    return [part for part in re.split(r"[\s,;|]+", value.strip()) if part]


def normalize_token(value: str) -> str:
    token = value.strip().lower()
    token = token.split("(", 1)[0].strip()
    return token


def is_precision_like(token: str) -> bool:
    try:
        normalize_openblas_precisions([token])
        return True
    except DistillError:
        return False


def maybe_openblas_interface(symbol: str) -> bool:
    try:
        parse_openblas_level2_interface(symbol)
        return True
    except DistillError:
        return False


def family_from_symbol(symbol: str) -> str:
    normalized = normalize_openblas_interface_name(symbol)
    if len(normalized) > 1 and normalized[0] in OPENBLAS_PRECISION_DTYPES and normalized[1:] in OPENBLAS_LEVEL2_CATALOG:
        return normalized[1:]
    family = normalize_openblas_family_name(normalized)
    if family in OPENBLAS_LEVEL2_CATALOG:
        return family
    raise DistillError(f"{symbol!r} does not identify a supported OpenBLAS Level-2 family")


def operator_specs_to_data(specs: Sequence[OperatorSpec]) -> Dict[str, Any]:
    return {
        "operators": [
            {
                "name": spec.name,
                "module": spec.module,
                "kind": spec.kind,
                "public_functions": list(spec.public_functions),
                "dtypes": list(spec.dtypes),
                "signature": spec.signature,
                "semantics": spec.semantics,
                "reference": spec.reference,
                "edge_cases": list(spec.edge_cases),
                "kernel_policy": spec.kernel_policy,
                "result_policy": spec.result_policy,
                "notes": list(spec.notes),
                "paths": dict(spec.paths),
            }
            for spec in specs
        ]
    }


def load_operator_specs(path: Path) -> List[OperatorSpec]:
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".txt", ".list", ".interfaces"}:
        return load_operator_txt_specs(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise DistillError("PyYAML is required for YAML specs; install PyYAML or use JSON")
        raw = yaml.safe_load(raw_text)
    else:
        raw = json.loads(raw_text)

    if isinstance(raw, Mapping):
        specs: List[OperatorSpec] = []
        if raw.get("openblas_level2") is not None:
            specs.extend(expand_openblas_level2_specs(raw["openblas_level2"]))
        if raw.get("interfaces") is not None and str(raw.get("kind", "openblas_level2")) == "openblas_level2":
            specs.extend(expand_openblas_level2_specs(raw))
        raw_ops = raw.get("operators")
        if raw_ops is not None:
            if not isinstance(raw_ops, list):
                raise DistillError("operators must be a list")
            for item in raw_ops:
                if isinstance(item, Mapping) and is_full_operator_mapping(item):
                    specs.append(OperatorSpec.from_mapping(item))
                elif isinstance(item, Mapping) and (item.get("family") or item.get("interfaces") or item.get("public_functions")):
                    specs.extend(expand_openblas_level2_specs({"operators": [item]}))
                else:
                    raise DistillError(f"unsupported operator item: {item}")
        elif not specs:
            if is_full_operator_mapping(raw):
                specs.append(OperatorSpec.from_mapping(raw))
            else:
                specs.extend(expand_openblas_level2_specs(raw))
        return specs
    elif isinstance(raw, list):
        if all(isinstance(item, str) for item in raw):
            return expand_openblas_level2_specs(raw)
        specs = []
        for item in raw:
            if isinstance(item, Mapping) and is_full_operator_mapping(item):
                specs.append(OperatorSpec.from_mapping(item))
            elif isinstance(item, Mapping):
                specs.extend(expand_openblas_level2_specs({"operators": [item]}))
            else:
                raise DistillError(f"unsupported operator list item: {item}")
        return specs
    else:
        raise DistillError("spec file must contain an operator object, an operators list, or a top-level list")


def select_specs(specs: Sequence[OperatorSpec], names_csv: Optional[str], all_ops: bool) -> List[OperatorSpec]:
    if all_ops or not names_csv:
        return list(specs)
    requested = [part.strip() for part in names_csv.split(",") if part.strip()]
    by_key: Dict[str, OperatorSpec] = {}
    for spec in specs:
        by_key[spec.name] = spec
        by_key[spec.module] = spec
        for public_function in spec.public_functions:
            by_key[public_function] = spec
            by_key[f"cblas_{public_function}"] = spec
            by_key[normalize_openblas_interface_name(public_function)] = spec
    unknown = [name for name in requested if name not in by_key]
    if unknown:
        for name in list(unknown):
            normalized = normalize_openblas_interface_name(name)
            if normalized in by_key:
                by_key[name] = by_key[normalized]
                unknown.remove(name)
    if unknown:
        available = sorted({spec.name for spec in specs} | {spec.module for spec in specs})
        raise DistillError(f"unknown operators: {unknown}. Available names/modules: {available}")
    selected = []
    seen = set()
    for name in requested:
        spec = by_key[name]
        if spec.module not in seen:
            selected.append(spec)
            seen.add(spec.module)
    return selected


def chunked(items: Sequence[OperatorSpec], size: int) -> Iterable[Sequence[OperatorSpec]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def normalize_model_name(model: str) -> str:
    compact = re.sub(r"[^a-z0-9]", "", model.lower())
    if compact in {"deepseekv4pro", "deepseekv4"}:
        return DEFAULT_MODEL
    return model


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="YAML or JSON operator spec file")
    parser.add_argument("--ops", help="comma-separated operator names/modules from the spec")
    parser.add_argument("--all", action="store_true", help="process all operators in the spec")
    parser.add_argument("--force", action="store_true", help="regenerate operators already marked completed in state")
    parser.add_argument("--resume-run", help="path to an existing trace run directory")
    parser.add_argument("--project-root", default=".", help="target repository root")
    parser.add_argument("--operator-root", default="src/flag_blas/ops", help="allowed operator write root")
    parser.add_argument("--default-operator-subdir", default="generated", help="default subdir under operator-root")
    parser.add_argument("--test-root", default="tests")
    parser.add_argument("--benchmark-root", default="benchmark")
    parser.add_argument("--trace-root", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--src-root", default="src")
    parser.add_argument("--package-import-root", default="flag_blas.ops")
    parser.add_argument("--profile", default="flagblas", choices=["flagblas", "standalone"])
    parser.add_argument("--benchmark-style", default=None, choices=["flagblas", "standalone"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument("--stream", action="store_true", help="use OpenAI-compatible SSE streaming for DeepSeek responses")
    parser.add_argument("--request-timeout", type=int, default=240, help="DeepSeek HTTP request timeout in seconds")
    parser.add_argument("--api-retries", type=int, default=4, help="DeepSeek HTTP retry count per stage")
    parser.add_argument("--api-retry-base-delay", type=float, default=4.0, help="DeepSeek retry base delay in seconds")
    parser.add_argument("--stage-retries", type=int, default=2)
    parser.add_argument("--max-repair-attempts", type=int, default=3)
    parser.add_argument("--command-timeout", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--extra-export-init", action="append", default=[], help="extra __init__.py path to update with imports")
    parser.add_argument("--parse-only", action="store_true", help="parse txt/YAML/JSON specs and write normalized operator specs, without generation")
    parser.add_argument("--parsed-spec-out", help="output path for --parse-only normalized JSON; stdout when omitted")
    parser.add_argument("--generate-only", action="store_true", help="generate implementation/test/benchmark files but do not run validation or repair")
    parser.add_argument("--repair-only", action="store_true", help="validate existing files, then repair failures without regenerating from scratch")
    parser.add_argument("--dry-run", action="store_true", help="write prompt payloads but do not call DeepSeek")
    parser.add_argument("--validate-only", action="store_true", help="validate existing files without generation")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--no-update-exports", action="store_true")
    parser.add_argument("--soft-generic-kernel-guard", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> WorkflowConfig:
    project_root = Path(args.project_root).resolve()
    benchmark_style = args.benchmark_style or ("flagblas" if args.profile == "flagblas" else "standalone")
    operation_mode = "all"
    requested_modes = [args.generate_only, args.repair_only, args.validate_only]
    if sum(1 for enabled in requested_modes if enabled) > 1:
        raise DistillError("choose only one of --generate-only, --validate-only, or --repair-only")
    if args.generate_only:
        operation_mode = "generate"
    elif args.repair_only:
        operation_mode = "repair"
    elif args.validate_only:
        operation_mode = "validate"
    return WorkflowConfig(
        project_root=project_root,
        operator_root=(project_root / args.operator_root).resolve(),
        test_root=(project_root / args.test_root).resolve(),
        benchmark_root=(project_root / args.benchmark_root).resolve(),
        trace_root=(project_root / args.trace_root).resolve(),
        src_root=(project_root / args.src_root).resolve(),
        default_operator_subdir=args.default_operator_subdir,
        package_import_root=args.package_import_root,
        profile=args.profile,
        benchmark_style=benchmark_style,
        model=normalize_model_name(args.model),
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        stream=args.stream,
        request_timeout=args.request_timeout,
        api_retries=args.api_retries,
        api_retry_base_delay=args.api_retry_base_delay,
        stage_retries=args.stage_retries,
        max_repair_attempts=args.max_repair_attempts,
        command_timeout=args.command_timeout,
        batch_size=args.batch_size,
        cooldown=args.cooldown,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        operation_mode=operation_mode,
        skip_tests=args.skip_tests,
        skip_benchmark=args.skip_benchmark,
        update_exports=not args.no_update_exports,
        extra_export_inits=tuple((project_root / path).resolve() for path in args.extra_export_init),
        hard_generic_kernel_guard=not args.soft_generic_kernel_guard,
        continue_on_failure=not args.stop_on_failure,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    specs = load_operator_specs(Path(args.spec))
    selected_specs = select_specs(specs, args.ops, args.all)
    if args.parse_only:
        payload = json.dumps(operator_specs_to_data(selected_specs), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        if args.parsed_spec_out:
            out_path = Path(args.parsed_spec_out)
            atomic_write(out_path, payload)
            print(f"Parsed {len(selected_specs)} operator specs -> {out_path}")
        else:
            print(payload, end="")
        return 0

    config = config_from_args(args)
    run_path = Path(args.resume_run).resolve() if args.resume_run else None
    trace = TraceLogger(config.trace_root, run_path=run_path)
    workflow = DistillWorkflow(config, trace)
    try:
        workflow.run(selected_specs, force=args.force)
    except Exception as exc:
        trace.event("workflow", "failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=10))
        print(f"HBY operator distiller failed: {exc}", file=sys.stderr)
        print(f"Trace directory: {trace.run_dir}", file=sys.stderr)
        return 1
    trace.event("workflow", "succeeded", ops=[spec.module for spec in selected_specs])
    print(f"HBY operator distiller succeeded. Trace directory: {trace.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
