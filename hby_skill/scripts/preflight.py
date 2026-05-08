#!/usr/bin/env python3
"""Preflight checks for the HBY operator generation workflow."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--operator-root", default="src/flag_blas/ops")
    parser.add_argument("--test-root", default="tests")
    parser.add_argument("--benchmark-root", default="benchmark")
    parser.add_argument("--trace-root", default="hby_operator_trace")
    parser.add_argument("--require-gpu", action="store_true", help="treat missing CUDA/NVIDIA GPU as a failure")
    parser.add_argument("--min-free-gpu-mb", type=int, default=512, help="minimum free GPU memory recommended for benchmark smoke")
    parser.add_argument("--skip-api-key", action="store_true", help="do not require DEEPSEEK_API_KEY")
    parser.add_argument("--check-deepseek", action="store_true", help="send a tiny DeepSeek JSON request")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    checks: List[Dict[str, Any]] = []

    add_check(checks, "python", "ok", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    add_import_check(checks, "yaml", "PyYAML is required for YAML specs")
    add_import_check(checks, "torch", "torch is required by generated tests and benchmarks")

    check_api_key(checks, skip=args.skip_api_key)
    check_paths(checks, project_root, args)
    check_gpu(checks, require=args.require_gpu, min_free_mb=args.min_free_gpu_mb, timeout=args.timeout)
    if args.check_deepseek:
        check_deepseek(checks, args)

    failed = any(item["status"] == "fail" for item in checks)
    if args.json:
        print(json.dumps({"ok": not failed, "checks": checks}, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print_human(checks)
    return 1 if failed else 0


def add_check(checks: List[Dict[str, Any]], name: str, status: str, message: str, **extra: Any) -> None:
    checks.append({"name": name, "status": status, "message": message, **extra})


def add_import_check(checks: List[Dict[str, Any]], module_name: str, help_text: str) -> None:
    try:
        module = __import__(module_name)
    except Exception as exc:
        add_check(checks, f"import:{module_name}", "fail", f"{help_text}: {exc!r}")
        return
    version = getattr(module, "__version__", "unknown")
    add_check(checks, f"import:{module_name}", "ok", f"available, version={version}")


def check_api_key(checks: List[Dict[str, Any]], skip: bool) -> None:
    if skip:
        add_check(checks, "deepseek_api_key", "warn", "skipped by --skip-api-key")
        return
    if os.environ.get("DEEPSEEK_API_KEY"):
        add_check(checks, "deepseek_api_key", "ok", "DEEPSEEK_API_KEY is set")
    else:
        add_check(checks, "deepseek_api_key", "fail", "DEEPSEEK_API_KEY is not set")


def check_paths(checks: List[Dict[str, Any]], project_root: Path, args: argparse.Namespace) -> None:
    for label, raw_path in (
        ("project_root", project_root),
        ("operator_root", project_root / args.operator_root),
        ("test_root", project_root / args.test_root),
        ("benchmark_root", project_root / args.benchmark_root),
        ("trace_root", project_root / args.trace_root),
    ):
        path = Path(raw_path).resolve()
        target = path if path.exists() else path.parent
        if target.exists() and os.access(target, os.W_OK):
            add_check(checks, f"path:{label}", "ok", str(path))
        else:
            add_check(checks, f"path:{label}", "fail", f"not writable or missing parent: {path}")


def check_gpu(checks: List[Dict[str, Any]], require: bool, min_free_mb: int, timeout: int) -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        add_check(checks, "nvidia_smi", "fail" if require else "warn", "nvidia-smi not found")
        return
    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        add_check(checks, "nvidia_smi", "fail" if require else "warn", f"nvidia-smi failed: {exc!r}")
        return
    if proc.returncode != 0:
        add_check(checks, "nvidia_smi", "fail" if require else "warn", proc.stderr.strip() or "nvidia-smi returned non-zero")
        return

    gpus = parse_nvidia_smi(proc.stdout)
    if not gpus:
        add_check(checks, "nvidia_smi", "fail" if require else "warn", "no GPU rows parsed")
        return
    best = max(gpus, key=lambda item: item.get("memory_free_mb", 0))
    add_check(checks, "nvidia_smi", "ok", f"{len(gpus)} GPU(s), best free memory={best['memory_free_mb']} MiB", gpus=gpus)
    if best["memory_free_mb"] >= min_free_mb:
        add_check(checks, "gpu_memory", "ok", f"best GPU has >= {min_free_mb} MiB free")
    else:
        add_check(
            checks,
            "gpu_memory",
            "fail" if require else "warn",
            f"best GPU has {best['memory_free_mb']} MiB free, below recommended {min_free_mb} MiB",
        )

    check_torch_cuda(checks, require=require)


def parse_nvidia_smi(stdout: str) -> List[Dict[str, Any]]:
    rows = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, total, used, free, util = parts
        try:
            rows.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_total_mb": int(total),
                    "memory_used_mb": int(used),
                    "memory_free_mb": int(free),
                    "utilization_percent": int(util),
                }
            )
        except ValueError:
            continue
    return rows


def check_torch_cuda(checks: List[Dict[str, Any]], require: bool) -> None:
    try:
        import torch

        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
        detail = f"available={available}, count={count}, torch={torch.__version__}, cuda={torch.version.cuda}"
        if available and count:
            detail += f", first={torch.cuda.get_device_name(0)}"
        add_check(checks, "torch_cuda", "ok" if available else ("fail" if require else "warn"), detail)
    except Exception as exc:
        add_check(checks, "torch_cuda", "fail" if require else "warn", repr(exc))


def check_deepseek(checks: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        add_check(checks, "deepseek_api", "fail", "cannot check API without DEEPSEEK_API_KEY")
        return
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Output JSON only."},
            {"role": "user", "content": "Return {\"ok\": true}."},
        ],
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        add_check(checks, "deepseek_api", "ok", f"HTTP OK, content={content[:80]}")
    except Exception as exc:
        add_check(checks, "deepseek_api", "fail", repr(exc))


def print_human(checks: Sequence[Dict[str, Any]]) -> None:
    label = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}
    for item in checks:
        print(f"[{label.get(item['status'], item['status']).rjust(4)}] {item['name']}: {item['message']}")


if __name__ == "__main__":
    raise SystemExit(main())
