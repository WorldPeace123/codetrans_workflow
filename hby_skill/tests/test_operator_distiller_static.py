import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "hby_skill" / "scripts" / "operator_distiller.py"
BATCH_SCRIPT = ROOT / "hby_skill" / "scripts" / "run_operator_batch.py"
PREFLIGHT_SCRIPT = ROOT / "hby_skill" / "scripts" / "preflight.py"


def load_distiller():
    spec = importlib.util.spec_from_file_location("operator_distiller", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_batch_runner():
    spec = importlib.util.spec_from_file_location("run_operator_batch", BATCH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_preflight():
    spec = importlib.util.spec_from_file_location("preflight", PREFLIGHT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_openblas_example_spec():
    distiller = load_distiller()
    specs = distiller.load_operator_specs(ROOT / "hby_skill" / "assets" / "examples" / "openblas_axpy.yaml")
    assert len(specs) == 1
    assert specs[0].module == "axpy"
    assert "saxpy" in specs[0].public_functions
    assert distiller.normalize_model_name("deepseekV4pro") == "deepseek-v4-pro"


def test_batch_runner_default_spec_points_to_txt_example():
    batch = load_batch_runner()
    args = batch.parse_args([])
    assert args.spec.endswith("openblas_level2_interfaces.txt")
    assert args.on_fail == "ask"
    assert args.model == "deepseek-v4-pro"
    assert args.max_tokens == 32768
    assert args.thinking == "disabled"
    assert args.stream is False
    assert args.request_timeout == 240
    assert args.api_retries == 4
    assert args.distiller_timeout == 1800


def test_expand_openblas_level2_ten_interface_example():
    distiller = load_distiller()
    specs = distiller.load_operator_specs(
        ROOT / "hby_skill" / "assets" / "examples" / "openblas_level2_ten_interfaces.yaml"
    )
    assert len(specs) == 10
    by_module = {spec.module: spec for spec in specs}
    assert by_module["gemv"].public_functions == ("sgemv", "dgemv")
    assert by_module["gemv"].dtypes == ("float32", "float64")
    assert by_module["gbmv"].kind == "openblas_level2"
    assert "order" in by_module["gemv"].signature
    selected = distiller.select_specs(specs, "cblas_sgemv,dgbmv", all_ops=False)
    assert [spec.module for spec in selected] == ["gemv", "gbmv"]


def test_parse_txt_interface_and_type_file():
    distiller = load_distiller()
    specs = distiller.load_operator_specs(
        ROOT / "hby_skill" / "assets" / "examples" / "openblas_level2_interfaces.txt"
    )
    assert len(specs) == 10
    by_module = {spec.module: spec for spec in specs}
    assert by_module["gemv"].public_functions == ("sgemv", "dgemv")
    assert by_module["gbmv"].dtypes == ("float32", "float64")
    assert by_module["trmv"].public_functions == ("strmv", "dtrmv")
    data = distiller.operator_specs_to_data([by_module["gemv"]])
    assert data["operators"][0]["kind"] == "openblas_level2"


def test_openblas_level2_default_paths_use_level2(tmp_path):
    distiller = load_distiller()
    project = tmp_path
    config = distiller.WorkflowConfig(
        project_root=project,
        operator_root=project / "src" / "flag_blas" / "ops",
        test_root=project / "tests",
        benchmark_root=project / "benchmark",
        trace_root=project / "trace",
        src_root=project / "src",
    )
    trace = distiller.TraceLogger(config.trace_root)
    manager = distiller.FileManager(config, trace)
    spec = distiller.expand_openblas_level2_specs({"interfaces": ["sgemv", "dgemv"]})[0]
    paths = manager.target_paths(spec)
    assert paths.operator == project / "src" / "flag_blas" / "ops" / "level2" / "gemv.py"


def test_target_paths_stay_under_configured_roots(tmp_path):
    distiller = load_distiller()
    project = tmp_path
    config = distiller.WorkflowConfig(
        project_root=project,
        operator_root=project / "src" / "flag_blas" / "ops",
        test_root=project / "tests",
        benchmark_root=project / "benchmark",
        trace_root=project / "trace",
        src_root=project / "src",
    )
    trace = distiller.TraceLogger(config.trace_root)
    manager = distiller.FileManager(config, trace)
    spec = distiller.OperatorSpec(
        name="elementwise_add",
        module="elementwise_add",
        kind="elementwise",
        public_functions=("elementwise_add",),
        dtypes=("float32",),
        signature="(x, y, out=None) -> torch.Tensor",
        semantics="out = x + y",
        reference="torch.add",
    )
    paths = manager.target_paths(spec)
    assert paths.operator == project / "src" / "flag_blas" / "ops" / "generated" / "elementwise_add.py"
    assert paths.functional_test == project / "tests" / "test_elementwise_add.py"
    assert paths.benchmark == project / "benchmark" / "test_elementwise_add_perf.py"


def test_static_validator_rejects_data_ptr():
    distiller = load_distiller()
    project = ROOT
    config = distiller.WorkflowConfig(
        project_root=project,
        operator_root=project / "src" / "flag_blas" / "ops",
        test_root=project / "tests",
        benchmark_root=project / "benchmark",
        trace_root=project / "trace",
        src_root=project / "src",
    )
    validator = distiller.StaticValidator(config)
    spec = distiller.OperatorSpec(
        name="bad",
        module="bad",
        kind="elementwise",
        public_functions=("bad",),
        dtypes=("float32",),
        signature="(x) -> torch.Tensor",
        semantics="identity",
        reference="torch.clone",
    )
    content = """
import triton
import triton.language as tl

@triton.jit
def _bad_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    return

def bad(x):
    _bad_kernel[(1,)](x.data_ptr(), BLOCK_SIZE=1)
    return x
"""
    errors = validator.validate_operator(spec, content)
    assert any("data_ptr" in error for error in errors)


def test_static_validator_requires_block_size_launch():
    distiller = load_distiller()
    config = distiller.WorkflowConfig(
        project_root=ROOT,
        operator_root=ROOT / "src" / "flag_blas" / "ops",
        test_root=ROOT / "tests",
        benchmark_root=ROOT / "benchmark",
        trace_root=ROOT / "trace",
        src_root=ROOT / "src",
    )
    validator = distiller.StaticValidator(config)
    spec = distiller.OperatorSpec(
        name="missing_block",
        module="missing_block",
        kind="elementwise",
        public_functions=("missing_block",),
        dtypes=("float32",),
        signature="(x) -> torch.Tensor",
        semantics="identity",
        reference="torch.clone",
    )
    content = """
import torch
import triton
import triton.language as tl

@triton.jit
def _missing_block_kernel(x, out, n, BLOCK_SIZE: tl.constexpr):
    return

def missing_block(x):
    out = torch.empty_like(x)
    _missing_block_kernel[(1,)](x, out, x.numel())
    return out
"""
    errors = validator.validate_operator(spec, content)
    assert any("BLOCK_SIZE" in error for error in errors)


def test_preflight_parses_nvidia_smi_rows():
    preflight = load_preflight()
    rows = preflight.parse_nvidia_smi(
        "0, NVIDIA H100 80GB HBM3, 81559, 80737, 353, 100\n"
        "1, NVIDIA H100 80GB HBM3, 81559, 1000, 80559, 0\n"
    )
    assert rows[0]["index"] == 0
    assert rows[0]["memory_free_mb"] == 353
    assert rows[1]["memory_free_mb"] == 80559


def test_workflow_export_updates_are_accumulative(tmp_path):
    distiller = load_distiller()
    project = tmp_path
    config = distiller.WorkflowConfig(
        project_root=project,
        operator_root=project / "src" / "flag_blas" / "ops",
        test_root=project / "tests",
        benchmark_root=project / "benchmark",
        trace_root=project / "trace",
        src_root=project / "src",
        dry_run=True,
    )
    trace = distiller.TraceLogger(config.trace_root)
    workflow = distiller.DistillWorkflow(config, trace)
    spec_a = distiller.OperatorSpec(
        name="a",
        module="a",
        kind="elementwise",
        public_functions=("a",),
        dtypes=("float32",),
        signature="(x) -> torch.Tensor",
        semantics="identity",
        reference="torch.clone",
    )
    spec_b = distiller.OperatorSpec(
        name="b",
        module="b",
        kind="elementwise",
        public_functions=("b",),
        dtypes=("float32",),
        signature="(x) -> torch.Tensor",
        semantics="identity",
        reference="torch.clone",
    )
    paths_a = workflow.files.target_paths(spec_a)
    paths_b = workflow.files.target_paths(spec_b)
    workflow.update_exports([(spec_a, paths_a)])
    workflow.update_exports([(spec_b, paths_b)])
    init_text = (paths_a.operator.parent / "__init__.py").read_text(encoding="utf-8")
    assert "from .a import a" in init_text
    assert "from .b import b" in init_text
