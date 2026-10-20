"""AlphaResearch problems (github.com/answers111/alpha-research), pinned as a git submodule, as program tasks.

The ideator's proposal contains a complete Python program defining solve(problem). The program is run as is on
every instance of the task, and its construction is scored with the upstream evaluator functions (extracted from
the pinned files; module-level calls dropped). Where an upstream evaluator is weaker than its own problem statement,
the statement's rule is enforced as well (STRICTER).

Each task has a simple baseline that a proposal must beat; its pseudocode and measured score are the only numbers
the ideator sees. REFERENCE (best-known values) is internal: it only normalizes the performance score.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]           # verifiable
BENCH = ROOT / "third_party" / "alpha-research" / "benchmark"
REPO = "https://github.com/answers111/alpha-research"
PIN = "e6d64542ff2b87f75108b229d1a68da024a56f5a"
GRACE_S = 10         # interpreter start-up and output; the process is killed at time_limit_s + GRACE_S
MEMORY_MB = 4096
MAX_STEPS = 20000     # step-function length cap (keeps exact scoring fast)

TASKS = {
    "ar_packing_circles": {
        "dir": "packing_circles", "title": "Circle packing in the unit square (maximize the sum of radii)",
        "definition": "Place n circles inside the unit square [0,1]^2. Every circle must lie entirely inside the square "
                      "and the circles must be pairwise disjoint (they may touch, but the check is exact in floating point: "
                      "center distance >= r_i + r_j and r <= x <= 1 - r, r <= y <= 1 - r, so keep a tiny margin such as "
                      "1e-12). Maximize the sum of the radii.",
        "instances": [{"n": 26}, {"n": 32}], "direction": "max", "metric": "sum of radii",
        "output": "{'circles': [[x, y, r], ...]} with exactly n rows and r > 0",
        "baseline": "Square grid: k = ceil(sqrt(n)); r = 1/(2k) - 1e-12; put circle i (i = 0..n-1) at "
                    "((i mod k + 0.5)/k, (floor(i/k) + 0.5)/k) with radius r.",
        "reference": [2.634, 2.936],
    },
    "ar_max_min_distance_ratio": {
        "dir": "minizing_raio_max_min_distance", "title": "Max-min distance ratio (maximize (min distance / max distance)^2)",
        "definition": "Place n distinct points in R^d. Maximize (smallest pairwise Euclidean distance / largest pairwise "
                      "Euclidean distance)^2. The score does not change under translation or scaling.",
        "instances": [{"n": 16, "d": 2}, {"n": 14, "d": 3}], "direction": "max", "metric": "(min distance / max distance)^2",
        "output": "{'points': [[x_1, ..., x_d], ...]} with exactly n distinct rows of length d",
        "baseline": "Grid: g = ceil(n^(1/d)); list the g^d points with coordinates in {0, 1/(g-1), ..., 1} in lexicographic "
                    "order and return the first n.",
        "reference": [1 / 12.89, 1 / 4.165849767],
    },
    "ar_third_autocorrelation": {
        "dir": "third_autocorrelation_inequality", "title": "Third autocorrelation inequality (minimize the C3 bound)",
        "definition": "Choose a step function with L equal-width steps and real heights h_0..h_(L-1) (negative values "
                      "allowed, nonzero sum). Minimize C = 2 * L * max_k |(h*h)_k| / (sum_i h_i)^2, where h*h = "
                      "numpy.convolve(h, h) is the discrete autoconvolution. C upper-bounds the third autocorrelation "
                      "constant.",
        "instances": [{}], "direction": "min", "metric": "C (smaller is better)",
        "output": f"{{'heights': [h_0, ..., h_(L-1)]}} with 1 <= L <= {MAX_STEPS} finite values and a nonzero sum",
        "baseline": "Constant function: L = 64 and every h_i = 1 (then max_k (h*h)_k = 64 and C = 2).",
        "reference": [1.4581],
    },
    "ar_spherical_code": {
        "dir": "spherical_code", "title": "Spherical code (30 points on S^2, maximize the minimum angle)",
        "definition": "Choose 30 points on the unit sphere in R^3 (each row is normalized to unit length by the checker; "
                      "rows must be nonzero). Maximize the minimum pairwise angle arccos(<p_i, p_j>), in radians.",
        "instances": [{"n": 30, "d": 3}], "direction": "max", "metric": "minimum pairwise angle (radians)",
        "output": "{'points': [[x, y, z], ...]} with exactly n nonzero rows",
        "baseline": "Fibonacci sphere: for i = 0..n-1, z = 1 - (2i + 1)/n, rho = sqrt(1 - z^2), phi = i * pi * (3 - sqrt(5)); "
                    "point (rho cos phi, rho sin phi, z).",
        "reference": [0.6736467551690225],
    },
    "ar_autoconvolution_peak": {
        "dir": "autoconvolution_peak_minimization", "title": "Autoconvolution peak minimization (minimize max f*f)",
        "definition": "Choose a nonnegative step function f on [-1/2, 1/2] with K equal steps and heights h_0..h_(K-1) "
                      "(positive sum). f is normalized to unit integral (h scaled by 1/(sum h * dx), dx = 1/K). Minimize "
                      "mu = max_t (f*f)(t), computed as dx * max of the discrete autoconvolution of the normalized heights. "
                      "(f*f)(t) = integral of f(x) f(t - x) dx is the autoconvolution, not the autocorrelation: the "
                      "discrete values are c_k = sum_i h_i h_(k-i) for k = 0..2K-2, so the middle value (t = 0) is "
                      "sum_i h_i h_(K-1-i), which pairs each step with its mirror image and is not sum_i h_i^2.",
        "instances": [{}], "direction": "min", "metric": "mu (smaller is better)",
        "output": f"{{'heights': [h_0, ..., h_(K-1)]}} with 1 <= K <= {MAX_STEPS} nonnegative finite values and a positive sum",
        "baseline": "Constant function: K = 64 and every h_i = 1 (f = 1 on [-1/2, 1/2], f*f is a triangle with peak 1, mu = 1).",
        "reference": [0.75496],
    },
    "ar_littlewood": {
        "dir": "littlewood_polynomials", "title": "Littlewood polynomial of length 512 (minimize the sup norm)",
        "definition": "Choose coefficients c_0..c_511, each +1 or -1, for P(z) = sum_k c_k z^k. Minimize the sup norm of "
                      "|P| on the unit circle, measured as max |numpy.fft.fft(c zero-padded to length 16384)|.",
        "instances": [{"n": 512}], "direction": "min", "metric": "sup norm (smaller is better)",
        "output": "{'coefficients': [c_0, ..., c_(n-1)]} with every c_k equal to the integer 1 or -1",
        "baseline": "Random signs: for seed s = 0..49, draw c with random.Random(s).choice([1, -1]) for each k; return the "
                    "draw with the smallest sup norm.",
        "reference": [32.0],
    },
    "ar_mstd": {
        "dir": "MSTD", "title": "More sums than differences (maximize |A+A| / |A-A| in {0, ..., N-1})",
        "definition": "Choose a nonempty set A of integers in {0, ..., N-1}. Maximize |A+A| / |A-A|, where A+A = "
                      "{a + b : a, b in A} and A-A = {a - b : a, b in A} (pairs with a = b included).",
        "instances": [{"N": 30}], "direction": "max", "metric": "|A+A| / |A-A|",
        "output": "{'indicators': [a_0, ..., a_(N-1)]} with every a_i equal to the integer 0 or 1 and at least one 1",
        "baseline": "Whole interval: A = {0, ..., N-1} (|A+A| = |A-A| = 2N - 1, ratio 1).",
        "reference": [1.04],
    },
    "ar_heilbronn": {
        "dir": "heilbronn_in_the_unit_square", "title": "Heilbronn problem (16 points in the unit square, maximize the smallest triangle)",
        "definition": "Place n points in the unit square [0,1]^2. Maximize the minimum area over all triangles formed by "
                      "three of the points (three collinear points give area 0).",
        "instances": [{"n": 16}], "direction": "max", "metric": "minimum triangle area",
        "output": "{'points': [[x, y], ...]} with exactly n points, all coordinates in [0, 1]",
        "baseline": "Regular polygon: point i (i = 0..n-1) at (0.5 + R cos(2 pi i / n), 0.5 + R sin(2 pi i / n)) with "
                    "R = 0.5 - 1e-9.",
        "reference": [7 / 341],
    },
    "ar_kissing_number": {
        "dir": "kissing_number", "title": "Kissing configuration in 11 dimensions (maximize the number of vectors)",
        "definition": "Construct as many vectors in R^d as possible such that every vector has unit Euclidean norm "
                      "(within 1e-9) and every two distinct vectors have inner product at most 0.5 (+1e-9).",
        "instances": [{"d": 11}], "direction": "max", "metric": "number of vectors",
        "output": "{'vectors': [[v_1, ..., v_d], ...]} (any number of rows of length d)",
        "baseline": "D_d root system: all vectors (s e_i + t e_j) / sqrt(2) for i < j and signs s, t in {+1, -1} "
                    "(2 d (d - 1) = 220 vectors for d = 11).",
        "reference": [592],
    },
}
SUITES = {"alpha_research": tuple(TASKS)}
from verifiable.program_qd import alebench  # noqa: E402
if alebench.DATA.is_dir():                                   # ALE-Bench tasks: when its data and built tools are
    TASKS.update({task: alebench.spec(task) for task in alebench.PROBLEMS})
    SUITES["ale_bench"] = tuple(alebench.PROBLEMS)
MODULES = {"ale_bench": alebench}                           # the other suites' task modules
LANGUAGE = "python"      # program language of this process's tasks: "cpp" for ALE-Bench (use_language)
STRICTER = {
    "ar_packing_circles": "the sum of radii is recomputed from the circles; radii must be positive; exactly n circles",
    "ar_max_min_distance_ratio": "exactly n distinct points of dimension d; distances computed with numpy (upstream uses scipy)",
    "ar_third_autocorrelation": "the peak is max|h*h| (upstream abs(max(h*h)) understates it for signed heights); heights "
                                "are rescaled to max |h| = 1 first (C is scale-invariant; tiny heights would underflow "
                                "h*h to 0 and score C = 0)",
    "ar_spherical_code": "exactly 30 nonzero points in 3 dimensions",
    "ar_autoconvolution_peak": f"at most {MAX_STEPS} steps",
    "ar_littlewood": "exactly 512 coefficients, each the integer 1 or -1",
    "ar_mstd": "exactly 30 indicators, each the integer 0 or 1",
    "ar_heilbronn": "exactly 16 points inside [0,1]^2",
    "ar_kissing_number": "finite rows of length 11",
}
UPSTREAM_FUNCTIONS = {
    "ar_packing_circles": ["verify_circles"],
    "ar_spherical_code": ["evaluate_spherical_code_min_angle"],
    "ar_autoconvolution_peak": ["evaluate_C1_upper_std"],
    "ar_littlewood": ["evaluate_littlewood_supnorm"],
    "ar_mstd": ["_to_index_set", "mstd_ratio"],
    "ar_heilbronn": ["_triangle_area", "evaluate_min_triangle_area"],
    "ar_kissing_number": ["verify_kissing_configuration"],
}

BASELINE_CODE = {
    "ar_packing_circles": '''import math
def solve(problem):
    n = problem["n"]; k = math.ceil(math.sqrt(n)); r = 1 / (2 * k) - 1e-12
    return {"circles": [[(i % k + 0.5) / k, (i // k + 0.5) / k, r] for i in range(n)]}
''',
    "ar_max_min_distance_ratio": '''import itertools, math
def solve(problem):
    n, d = problem["n"], problem["d"]; g = math.ceil(n ** (1 / d) - 1e-9)
    grid = [[c / (g - 1) for c in p] for p in itertools.product(range(g), repeat=d)]
    return {"points": grid[:n]}
''',
    "ar_third_autocorrelation": '''def solve(problem):
    return {"heights": [1.0] * 64}
''',
    "ar_spherical_code": '''import math
def solve(problem):
    n = problem["n"]; points = []
    for i in range(n):
        z = 1 - (2 * i + 1) / n; rho = math.sqrt(1 - z * z); phi = i * math.pi * (3 - math.sqrt(5))
        points.append([rho * math.cos(phi), rho * math.sin(phi), z])
    return {"points": points}
''',
    "ar_autoconvolution_peak": '''def solve(problem):
    return {"heights": [1.0] * 64}
''',
    "ar_littlewood": '''import random
import numpy as np
def solve(problem):
    n = problem["n"]; best = None
    for seed in range(50):
        rng = random.Random(seed); c = [rng.choice([1, -1]) for _ in range(n)]
        pad = np.zeros(16384); pad[:n] = c; norm = float(np.max(np.abs(np.fft.fft(pad))))
        if best is None or norm < best[0]:
            best = (norm, c)
    return {"coefficients": best[1]}
''',
    "ar_mstd": '''def solve(problem):
    return {"indicators": [1] * problem["N"]}
''',
    "ar_heilbronn": '''import math
def solve(problem):
    n = problem["n"]; R = 0.5 - 1e-9
    return {"points": [[0.5 + R * math.cos(2 * math.pi * i / n), 0.5 + R * math.sin(2 * math.pi * i / n)] for i in range(n)]}
''',
    "ar_kissing_number": '''import itertools, math
def solve(problem):
    d = problem["d"]; s = 1 / math.sqrt(2); vectors = []
    for i, j in itertools.combinations(range(d), 2):
        for a, b in itertools.product((1, -1), repeat=2):
            v = [0.0] * d; v[i] = a * s; v[j] = b * s; vectors.append(v)
    return {"vectors": vectors}
''',
    **{task: code for task, code in alebench.BASELINE_CODE.items() if task in TASKS},
}


def suite(task: str) -> str:
    return TASKS[task].get("suite", "alpha_research")


def use_language(task_list) -> None:
    """Programs are C++ when every task of this process is an ALE-Bench task; Python otherwise."""
    global LANGUAGE
    LANGUAGE = "cpp" if task_list and all(suite(t) == "ale_bench" for t in task_list) else "python"


def is_task(task: str) -> bool:
    return task in TASKS


def source_url(task: str) -> str:
    if suite(task) in MODULES:
        return MODULES[suite(task)].source_url(task)
    return f"{REPO}/tree/{PIN}/benchmark/{TASKS[task]['dir']}"


def source_files(task: str) -> dict[str, str]:
    if suite(task) in MODULES:
        return MODULES[suite(task)].source_files(task)
    folder = BENCH / TASKS[task]["dir"]
    files = [folder / "evaluator.py", folder / "initial_proposal.txt"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


_LOADED: dict = {}


def _upstream(task: str) -> dict:
    """The named upstream evaluator functions with their own imports and constants (no scipy)."""
    if task not in _LOADED:
        path = BENCH / TASKS[task]["dir"] / "evaluator.py"
        wanted = set(UPSTREAM_FUNCTIONS.get(task, []))
        keep = [n for n in ast.parse(path.read_text()).body
                if (isinstance(n, (ast.Import, ast.ImportFrom)) and "scipy" not in ast.unparse(n))
                or (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets))
                or (isinstance(n, ast.FunctionDef) and n.name in wanted)]
        namespace: dict = {"__name__": f"upstream_{task}"}
        exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), namespace)
        if wanted - set(namespace):
            raise RuntimeError(f"Upstream functions missing for {task}: {sorted(wanted - set(namespace))}")
        _LOADED[task] = namespace
    return _LOADED[task]


def instances(task: str, budget_s: float) -> list[dict]:
    """Instance problems with time_limit_s; the program budget is split evenly across instances."""
    if suite(task) in MODULES:
        return MODULES[suite(task)].instances(task)       # the suite's fixed time limit per instance
    spec = TASKS[task]
    return [{**instance, "time_limit_s": round(budget_s / len(spec["instances"]), 3)} for instance in spec["instances"]]


def _matrix(value, rows, cols, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or (rows is not None and array.shape[0] != rows) or (cols is not None and array.shape[1] != cols):
        raise ValueError(f"{name} must have shape ({rows if rows is not None else 'N'}, {cols}); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _steps(value, *, signed: bool) -> np.ndarray:
    h = np.asarray(value, dtype=float)
    if h.ndim != 1 or not 1 <= h.size <= MAX_STEPS or not np.all(np.isfinite(h)):
        raise ValueError(f"heights must be a flat list of 1..{MAX_STEPS} finite numbers")
    if signed and h.sum() == 0:
        raise ValueError("heights must have a nonzero sum")
    if not signed and (np.any(h < 0) or not h.sum() > 0):
        raise ValueError("heights must be nonnegative with a positive sum")
    return h


def score(task: str, problem: dict, solution) -> float:
    """Upstream score of a construction; raises ValueError when it is invalid."""
    if not isinstance(solution, dict):
        raise ValueError("solve must return a dict with the documented key")
    up = _upstream(task) if task in UPSTREAM_FUNCTIONS else {}
    if task == "ar_packing_circles":
        circles = _matrix(solution.get("circles"), problem["n"], 3, "circles")
        if np.any(circles[:, 2] <= 0):
            raise ValueError("radii must be positive")
        if not up["verify_circles"](circles):
            raise ValueError("circles overlap or leave the unit square")
        return float(circles[:, 2].sum())
    if task == "ar_max_min_distance_ratio":
        points = _matrix(solution.get("points"), problem["n"], problem["d"], "points")
        i, j = np.triu_indices(len(points), 1)
        distances = np.sqrt(((points[i] - points[j]) ** 2).sum(axis=1))
        if not distances.min() > 0:
            raise ValueError("points must be pairwise distinct")
        return float((distances.min() / distances.max()) ** 2)
    if task == "ar_third_autocorrelation":
        h = _steps(solution.get("heights"), signed=True)
        h = h / np.max(np.abs(h))     # C is scale-invariant; this keeps tiny or huge heights from under/overflowing
        return float(2 * len(h) * np.max(np.abs(np.convolve(h, h))) / (np.sum(h) ** 2))
    if task == "ar_spherical_code":
        points = _matrix(solution.get("points"), problem["n"], problem["d"], "points")
        if np.any(np.linalg.norm(points, axis=1) <= 1e-12):
            raise ValueError("points must be nonzero")
        return float(up["evaluate_spherical_code_min_angle"](points)["min_angle"])
    if task == "ar_autoconvolution_peak":
        result = up["evaluate_C1_upper_std"](_steps(solution.get("heights"), signed=False))
        if result["valid"] != 1.0 or not math.isfinite(result["mu_inf"]) or result["mu_inf"] <= 0:
            raise ValueError("invalid step function")
        return float(result["mu_inf"])
    if task == "ar_littlewood":
        c = solution.get("coefficients")
        if not isinstance(c, list) or len(c) != problem["n"] or any(isinstance(x, bool) or x not in (1, -1) for x in c):
            raise ValueError(f"coefficients must be a list of exactly {problem['n']} values, each 1 or -1")
        return float(up["evaluate_littlewood_supnorm"](c, num_grid=16384)["supnorm"])
    if task == "ar_mstd":
        a = solution.get("indicators")
        if not isinstance(a, list) or len(a) != problem["N"] or any(isinstance(x, bool) or x not in (0, 1) for x in a) or not any(a):
            raise ValueError(f"indicators must be a list of exactly {problem['N']} values, each 0 or 1, not all 0")
        return float(up["mstd_ratio"](up["_to_index_set"](a), None))
    if task == "ar_heilbronn":
        points = _matrix(solution.get("points"), problem["n"], 2, "points")
        if np.any(points < 0) or np.any(points > 1):
            raise ValueError("points must lie in [0,1]^2")
        return float(up["evaluate_min_triangle_area"](points)["min_area"])
    if task == "ar_kissing_number":
        vectors = _matrix(solution.get("vectors"), None, problem["d"], "vectors")
        if len(vectors) < 1:
            raise ValueError("no vectors")
        try:
            up["verify_kissing_configuration"](vectors)
        except AssertionError as exc:
            raise ValueError(str(exc)) from None
        return float(len(vectors))
    raise ValueError(f"Unknown task {task}")


def validate(task: str, problem: dict, solution) -> dict:
    try:
        value = score(task, problem, solution)
    except Exception as exc:  # any malformed construction is invalid, never a harness error
        return {"valid": False, "score": None, "reason": f"{type(exc).__name__}: {exc}"[:300]}
    if not math.isfinite(value):
        return {"valid": False, "score": None, "reason": "nonfinite score"}
    return {"valid": True, "score": value, "reason": ""}


BETTER_REL_TOL = 1e-9   # a score must beat the reference by more than float noise (reproducing the baseline is no gain)


def better(task: str, a: float, b: float) -> bool:
    margin = BETTER_REL_TOL * max(1.0, abs(b))
    return a > b + margin if TASKS[task]["direction"] == "max" else a < b - margin


def performance(task: str, scores: list[float | None], baseline: list[float]) -> int:
    """0 if any instance is invalid; 20 for a valid program with no progress over the baseline; then 20 + 80 x the mean
    progress from the baseline to the best-known value (progress 1 = 100). Not capped: beating the best-known value
    scores above 100, so pushing further is always rewarded."""
    if not scores or any(s is None for s in scores):
        return 0
    progress = []
    for value, base, best in zip(scores, baseline, TASKS[task]["reference"]):
        span = (best - base) if TASKS[task]["direction"] == "max" else (base - best)
        gain = (value - base) if TASKS[task]["direction"] == "max" else (base - value)
        progress.append(max(0.0, gain / span) if span > 0 else float(gain > 0))
    mean = sum(progress) / len(progress)
    return 20 if mean <= 0 else round(20 + 80 * mean)


PROGRAM_BLOCK = re.compile(r"```(?:python|py)[^\n]*\n(.*?)```", re.DOTALL)


def extract_program(text: str) -> str | None:
    """The last fenced Python block that defines solve (C++ block with main for ALE-Bench, use_language)."""
    if LANGUAGE == "cpp":
        return alebench.extract_program(text)
    blocks = [b for b in PROGRAM_BLOCK.findall(text or "") if re.search(r"^\s*def solve\s*\(", b, re.MULTILINE)]
    return blocks[-1] if blocks else None


def replace_program(text: str, code: str) -> str:
    """The proposal with its program (the block extract_program returns) replaced by code; the prose is untouched."""
    if LANGUAGE == "cpp":
        return alebench.replace_program(text, code)
    matches = [m for m in PROGRAM_BLOCK.finditer(text or "") if re.search(r"^\s*def solve\s*\(", m.group(1), re.MULTILINE)]
    if not matches:
        return text
    last = matches[-1]
    return text[:last.start(1)] + (code if code.endswith("\n") else code + "\n") + text[last.end(1):]


SHIM = '''import importlib.util as _u
_spec = _u.spec_from_file_location("candidate_program", {path!r})
_mod = _u.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _plain(value):
    if hasattr(value, "tolist") and callable(value.tolist):
        return _plain(value.tolist())
    if hasattr(value, "item") and callable(value.item) and type(value).__module__ == "numpy":
        return value.item()
    if isinstance(value, dict):
        return {{(k.item() if type(k).__module__ == "numpy" else k): _plain(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def solve(problem):
    # numpy containers are an equivalent representation of the JSON construction
    return _plain(_mod.solve(problem))
'''


def run_program(task: str, code: str, folder: Path, budget_s: float) -> dict:
    """Run the program on every instance (budget split evenly) and score each construction."""
    if suite(task) in MODULES:
        return MODULES[suite(task)].run_program(task, code, folder)
    from verifiable.execution import run_solution
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "program.py").write_text(code)
    (folder / "entry.py").write_text(SHIM.format(path=str((folder / "program.py").resolve())))
    cases = []
    for index, problem in enumerate(instances(task, budget_s)):
        execution = run_solution(task, problem, folder / "entry.py", timeout_s=problem["time_limit_s"] + GRACE_S,
                                 memory_mb=MEMORY_MB)
        if execution.get("status") == "ok":
            solution = execution.pop("solution")
            # Keep the construction itself, so any score can be audited later (randomized programs do not replay).
            (folder / f"solution-{index}.json").write_text(json.dumps(solution))
            check = validate(task, problem, solution)
        else:
            check = {"valid": False, "score": None, "reason": execution.get("reason", "execution error")}
        trace = (execution.get("stderr_tail") or "").strip()
        status = execution.get("status")
        cases.append({"problem": problem, "valid": check["valid"], "score": check["score"], "reason": check["reason"],
                      "runtime_s": execution.get("solve_seconds"), "wall_s": round(execution.get("wall_seconds", 0.0), 2),
                      "traceback_tail": trace[-600:] if not check["valid"] else "",
                      # "ok" (the program returned; invalid means a bad construction), "error" (it raised) or "timeout"
                      "status": status, "error_trace": trace[-3000:] if status == "error" else ""})
    return {"task": task, "budget_s": budget_s, "cases": cases, "all_valid": all(c["valid"] for c in cases),
            "scores": [c["score"] for c in cases]}


def background(task: str, baseline_scores: list[float], budget_s: float, final_budget_s: float) -> str:
    """Problem definition, I/O contract, program rules and the baseline to beat (no other reference values)."""
    if suite(task) in MODULES:
        return MODULES[suite(task)].background(task, baseline_scores)
    spec = TASKS[task]
    shown = "; ".join(f"{', '.join(f'{k}={v}' for k, v in inst.items()) or 'the instance'}: {value:.6g}"
                      for inst, value in zip(spec["instances"], baseline_scores))
    return "\n".join([
        f"PROBLEM: {spec['title']}.",
        spec["definition"],
        f"Score: {spec['metric']}; {'larger' if spec['direction'] == 'max' else 'smaller'} is better.",
        "",
        "PROGRAM CONTRACT. Your proposal must contain the complete program, as one ```python block defining "
        "solve(problem). It is executed as is, once per instance, with problem equal to one of "
        f"{instances(task, budget_s)} (instance fields plus time_limit_s, the wall-clock seconds for that call). "
        f"solve must return {spec['output']}. Plain lists or numpy arrays are accepted. The process is killed "
        f"{GRACE_S} s after time_limit_s, so "
        f"{spec.get('return_rule', 'return the best valid construction found before time_limit_s')}. One CPU "
        f"thread, {MEMORY_MB} MiB, Python 3 standard library and numpy (networkx and sympy also available; no scipy), "
        "no files, network or subprocesses. Any invalid output, exception or timeout scores zero.",
        "",
        "BASELINE (a simple, obvious method; an idea is accepted only if it scores above it on every instance):",
        spec["baseline"],
        f"Baseline score: {shown}.",
        "",
        f"Every program you write is run on the instance during the search ({budget_s:g} s per program, split across "
        "instances). Its validity, score and runtime become part of the idea's quality, and are reported back to you. "
        f"The final evaluation runs the frozen program once with {final_budget_s:g} s per program, split the same way.",
    ])
