#!/usr/bin/env python3
"""Convenience batch runner for the HBY operator pipeline.

This script is the daily-use wrapper around operator_distiller.py. It is meant
for the common workflow where a txt file lists many operator interfaces and
types. It parses once, then processes operators one by one:

1. already generated and validation-passing operators are skipped;
2. missing or failing operators are generated and validated;
3. failed operators can be repaired, regenerated, skipped, or stop the batch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DISTILLER_PATH = SCRIPT_DIR / "operator_distiller.py"
DEFAULT_SPEC = SKILL_DIR / "assets" / "examples" / "openblas_level2_interfaces.txt"
DEFAULT_TRACE_ROOT = "hby_operator_trace"


@dataclass
class StepResult:
    action: str
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "returncode": self.returncode,
            "stdout_tail": tail(self.stdout),
            "stderr_tail": tail(self.stderr),
            "duration_s": round(self.duration_s, 3),
            "timed_out": self.timed_out,
        }


@dataclass
class OperatorState:
    module: str
    status: str = "pending"
    attempts: int = 0
    last_action: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, result: StepResult) -> None:
        self.attempts += 1
        self.last_action = result.action
        self.history.append(result.to_dict())


def load_distiller_module():
    spec = importlib.util.spec_from_file_location("operator_distiller", DISTILLER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC), help="txt/YAML/JSON operator request file")
    parser.add_argument("--ops", help="comma-separated operators/interfaces to run; default is all")
    parser.add_argument("--all", action="store_true", help="run all parsed operators")
    parser.add_argument("--project-root", default=".", help="target repository root")
    parser.add_argument("--trace-root", default=DEFAULT_TRACE_ROOT, help="batch trace root")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument("--stream", action="store_true", help="use DeepSeek SSE streaming for generation and repair")
    parser.add_argument("--request-timeout", type=int, default=240, help="DeepSeek HTTP request timeout in seconds")
    parser.add_argument("--api-retries", type=int, default=4, help="DeepSeek HTTP retry count per stage")
    parser.add_argument("--api-retry-base-delay", type=float, default=4.0, help="DeepSeek retry base delay in seconds")
    parser.add_argument("--max-repair-attempts", type=int, default=3)
    parser.add_argument("--stage-retries", type=int, default=2)
    parser.add_argument("--command-timeout", type=int, default=600)
    parser.add_argument("--distiller-timeout", type=int, default=1800, help="timeout in seconds for each distiller subprocess; 0 disables")
    parser.add_argument("--cuda-visible-devices", help="set CUDA_VISIBLE_DEVICES for validation subprocesses, e.g. 0 or 4")
    parser.add_argument("--cooldown", type=float, default=0.0, help="seconds between operators")
    parser.add_argument("--on-fail", choices=["ask", "repair", "regenerate", "skip", "stop"], default="ask")
    parser.add_argument("--max-failure-rounds", type=int, default=3, help="max repair/regenerate decisions per operator")
    parser.add_argument("--force-passed", action="store_true", help="do not skip operators marked passed in batch_state.json")
    parser.add_argument("--force-generate", action="store_true", help="generate even when files already exist")
    parser.add_argument("--dry-run", action="store_true", help="write prompts but do not call DeepSeek")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--no-update-exports", action="store_true")
    parser.add_argument("--extra-export-init", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    spec_path = resolve_path(args.spec, project_root)
    trace_root = resolve_path(args.trace_root, project_root)
    run_dir = trace_root / "batch_runs" / local_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    distiller = load_distiller_module()
    specs = distiller.load_operator_specs(spec_path)
    selected = distiller.select_specs(specs, args.ops, args.all or not args.ops)
    parsed_spec = run_dir / "parsed_specs.json"
    atomic_write(parsed_spec, json.dumps(distiller.operator_specs_to_data(selected), indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    state_path = trace_root / "batch_state.json"
    state = load_batch_state(state_path)
    print(f"HBY batch run: {run_dir}")
    print(f"Parsed specs: {parsed_spec}")
    print(f"Operators: {', '.join(spec.module for spec in selected)}")
    if args.cuda_visible_devices:
        print(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    failed = False
    for index, spec in enumerate(selected, start=1):
        op_state = OperatorState(**state.get(spec.module, {"module": spec.module}))
        print(f"\n[{index}/{len(selected)}] {spec.module}")

        if op_state.status == "passed" and not args.force_passed:
            print("  skip: already passed in batch_state.json")
            continue

        if args.dry_run:
            generate = run_distiller(parsed_spec, spec.module, args, project_root, run_dir, "generate")
            op_state.record(generate)
            op_state.status = "dry_run" if generate.ok else "failed"
            state[spec.module] = op_state.__dict__
            save_batch_state(state_path, state)
            if not generate.ok:
                failed = True
            maybe_sleep(args.cooldown)
            continue

        if not args.force_generate:
            validate = run_distiller(parsed_spec, spec.module, args, project_root, run_dir, "validate")
            op_state.record(validate)
            if validate.ok:
                op_state.status = "passed"
                state[spec.module] = op_state.__dict__
                save_batch_state(state_path, state)
                print("  passed: existing generated files validate")
                continue

        generate = run_distiller(parsed_spec, spec.module, args, project_root, run_dir, "generate")
        op_state.record(generate)
        if generate.ok:
            validate = run_distiller(parsed_spec, spec.module, args, project_root, run_dir, "validate")
            op_state.record(validate)
            if validate.ok:
                op_state.status = "passed"
                state[spec.module] = op_state.__dict__
                save_batch_state(state_path, state)
                print("  passed: generated files validate")
                maybe_sleep(args.cooldown)
                continue

        op_state.status = "failed"
        state[spec.module] = op_state.__dict__
        save_batch_state(state_path, state)
        print("  failed: generation or validation did not pass")

        handled = handle_failed_operator(parsed_spec, spec.module, args, project_root, run_dir, op_state, state, state_path)
        if handled == "passed":
            print("  passed after follow-up action")
        elif handled == "stop":
            failed = True
            break
        else:
            failed = True
        maybe_sleep(args.cooldown)

    summary_path = write_batch_summary(run_dir, state, selected, failed)
    print_summary(state, selected, state_path, summary_path)
    return 1 if failed else 0


def handle_failed_operator(
    parsed_spec: Path,
    module: str,
    args: argparse.Namespace,
    project_root: Path,
    run_dir: Path,
    op_state: OperatorState,
    state: Dict[str, Dict[str, Any]],
    state_path: Path,
) -> str:
    for round_index in range(1, args.max_failure_rounds + 1):
        action = choose_failure_action(module, args.on_fail)
        if action == "stop":
            op_state.status = "failed"
            state[module] = op_state.__dict__
            save_batch_state(state_path, state)
            return "stop"
        if action == "skip":
            op_state.status = "skipped_failed"
            state[module] = op_state.__dict__
            save_batch_state(state_path, state)
            return "skipped"
        if action == "repair":
            result = run_distiller(parsed_spec, module, args, project_root, run_dir, "repair")
            op_state.record(result)
        elif action == "regenerate":
            result = run_distiller(parsed_spec, module, args, project_root, run_dir, "generate", force=True)
            op_state.record(result)
            if result.ok:
                result = run_distiller(parsed_spec, module, args, project_root, run_dir, "validate")
                op_state.record(result)
        else:
            raise RuntimeError(f"unexpected failure action: {action}")

        if result.ok:
            validate = run_distiller(parsed_spec, module, args, project_root, run_dir, "validate")
            op_state.record(validate)
            if validate.ok:
                op_state.status = "passed"
                state[module] = op_state.__dict__
                save_batch_state(state_path, state)
                return "passed"

        op_state.status = "failed"
        state[module] = op_state.__dict__
        save_batch_state(state_path, state)
        print(f"  still failing after {action} round {round_index}")

    return "failed"


def run_distiller(
    parsed_spec: Path,
    module: str,
    args: argparse.Namespace,
    project_root: Path,
    run_dir: Path,
    action: str,
    force: bool = False,
) -> StepResult:
    cmd = [
        sys.executable,
        str(DISTILLER_PATH),
        "--spec",
        str(parsed_spec),
        "--ops",
        module,
        "--project-root",
        str(project_root),
        "--trace-root",
        str(run_dir / "operator_trace" / f"{module}_{action}_{time.time_ns()}"),
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--thinking",
        args.thinking,
        "--reasoning-effort",
        args.reasoning_effort,
        "--request-timeout",
        str(args.request_timeout),
        "--api-retries",
        str(args.api_retries),
        "--api-retry-base-delay",
        str(args.api_retry_base_delay),
        "--stage-retries",
        str(args.stage_retries),
        "--max-repair-attempts",
        str(args.max_repair_attempts),
        "--command-timeout",
        str(args.command_timeout),
        "--cooldown",
        "0",
        "--stop-on-failure",
    ]
    if action == "generate":
        cmd.append("--generate-only")
    elif action == "validate":
        cmd.append("--validate-only")
    elif action == "repair":
        cmd.append("--repair-only")
    else:
        raise RuntimeError(f"unknown distiller action: {action}")
    if force:
        cmd.append("--force")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.stream:
        cmd.append("--stream")
    if args.skip_tests:
        cmd.append("--skip-tests")
    if args.skip_benchmark:
        cmd.append("--skip-benchmark")
    if args.no_update_exports:
        cmd.append("--no-update-exports")
    for init_path in args.extra_export_init:
        cmd.extend(["--extra-export-init", init_path])

    print(f"  {action}: {' '.join(shell_quote(part) for part in cmd)}")
    started = time.monotonic()
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=args.distiller_timeout if args.distiller_timeout > 0 else None,
            env=env,
        )
        timed_out = False
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = to_text(exc.stdout)
        stderr = to_text(exc.stderr)
        stderr = (stderr + "\n" if stderr else "") + f"distiller subprocess timed out after {args.distiller_timeout}s"
        returncode = 124
    duration = time.monotonic() - started
    if stdout.strip():
        print(indent_tail(stdout, "    stdout: "))
    if stderr.strip():
        print(indent_tail(stderr, "    stderr: "), file=sys.stderr)
    return StepResult(action=action, returncode=returncode, stdout=stdout, stderr=stderr, duration_s=duration, timed_out=timed_out)


def choose_failure_action(module: str, configured: str) -> str:
    if configured != "ask":
        return configured
    if not sys.stdin.isatty():
        print("  non-interactive stdin: defaulting failed action to repair")
        return "repair"
    prompt = (
        f"  {module} failed. Choose action "
        "[r]epair / re[g]enerate / [s]kip / s[t]op: "
    )
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"r", "repair", ""}:
            return "repair"
        if answer in {"g", "regen", "regenerate"}:
            return "regenerate"
        if answer in {"s", "skip"}:
            return "skip"
        if answer in {"t", "stop", "q", "quit"}:
            return "stop"


def load_batch_state(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def save_batch_state(path: Path, state: Dict[str, Dict[str, Any]]) -> None:
    atomic_write(path, json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_batch_summary(run_dir: Path, state: Dict[str, Dict[str, Any]], specs: Sequence[Any], failed: bool) -> Path:
    counts: Dict[str, int] = {}
    operators = []
    for spec in specs:
        item = state.get(spec.module, {})
        status = str(item.get("status", "pending"))
        counts[status] = counts.get(status, 0) + 1
        operators.append(
            {
                "module": spec.module,
                "status": status,
                "last_action": item.get("last_action", ""),
                "attempts": item.get("attempts", 0),
            }
        )
    summary = {
        "ok": not failed,
        "counts": counts,
        "operators": operators,
        "run_dir": str(run_dir),
    }
    path = run_dir / "summary.json"
    atomic_write(path, json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def print_summary(state: Dict[str, Dict[str, Any]], specs: Sequence[Any], state_path: Path, summary_path: Path) -> None:
    print(f"\nBatch state: {state_path}")
    print(f"Batch summary: {summary_path}")
    for spec in specs:
        item = state.get(spec.module, {})
        print(f"  {spec.module}: {item.get('status', 'pending')} ({item.get('last_action', '-')})")


def resolve_path(path: str, project_root: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    if value.exists():
        return value.resolve()
    return (project_root / value).resolve()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def local_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"


def maybe_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def indent_tail(text: str, prefix: str, max_chars: int = 2000) -> str:
    return "\n".join(prefix + line for line in tail(text.strip(), max_chars).splitlines())


def shell_quote(value: str) -> str:
    if re_safe_arg(value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def re_safe_arg(value: str) -> bool:
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:=+-")
    return bool(value) and all(ch in safe for ch in value)


if __name__ == "__main__":
    raise SystemExit(main())
