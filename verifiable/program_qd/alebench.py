"""ALE-Bench Lite problems (github.com/SakanaAI/ALE-Bench, data huggingface.co/datasets/SakanaAI/ALE-Bench), pinned, as
program tasks, judged natively (no Docker).

The ideator's program is a complete C++ program (stdin -> stdout), as in the AtCoder Heuristic Contests. It is
compiled with ALE-Bench's cpp20 judge flags (judge version 202301, without Boost/GMP/Eigen) against the AtCoder
Library, run on each case with ALE-Bench's limits (wall `timeout ceil(TL+0.1)+0.2`, CPU `prlimit ceil(TL+0.1)+0.1`,
TLE when the run takes longer than TL), and scored with the problem's own Rust tester (`tester in out` for batch
problems, `tester program < in > out` for interactive ones): the last stderr line `Score = N`, wrong answer on a
non-zero exit or "wrong answer:". The tools are AtCoder's, built with Rust 1.79 (ALE-Bench's rust:1.79.0 image).

Cases: the problem's 50 public seeds (ALE-Bench's public evaluation), for the search and the final run alike. Quality
per case is the progress from the simple baseline program (BASELINE_CODE) towards the mean per-case score of the
contest winner on the private Lite seeds (standings_scores_lite.csv): it normalizes only. Problems: the 7 Lite problems
with absolute scores (ahc016, ahc025 and ahc027 score relative to other submissions).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]           # verifiable
REPO = "https://github.com/SakanaAI/ALE-Bench"
PIN = "b77a9a54ded08e410e789471145925b1eaaf62b1"
DATA_REPO = "SakanaAI/ALE-Bench"
DATA_PIN = "0f426173b4e4e73b09b2b3631ae0490f66b75f99"
DATA = Path(os.environ.get("ALE_BENCH_DATA", ROOT / "third_party" / "ale-bench"))   # setup_alebench.sh
ACL = Path(os.environ.get("ALE_BENCH_ACL", ROOT / "third_party" / "ac-library"))
CXX = os.environ.get("ALE_BENCH_CXX", "g++")
COMPILE = ["-std=gnu++20", "-O2", "-DONLINE_JUDGE", "-DATCODER", "-mtune=native", "-march=native",
           "-fconstexpr-depth=2147483647", "-fconstexpr-loop-limit=2147483647", "-fconstexpr-ops-limit=2147483647"]
COMPILE_TIMEOUT_S = 120
MEMORY_BYTES = 2 * 1024 ** 3    # ALE-Bench's container memory limit
PARALLEL = 8                    # cases run at the same time (one CPU each)
# Crash diagnostics (fix-loop report only, never scored): the first crashing case is re-run with a sanitizer build.
DEBUG_COMPILE = ["-g", "-O1", "-fno-omit-frame-pointer", "-fsanitize=address,undefined"]
DEBUG_TIME_FACTOR = 3           # sanitizer builds run 2-3x slower
SANITIZER_ENV = {"ASAN_OPTIONS": "detect_leaks=0:abort_on_error=0:symbolize=1",
                 "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=0"}

PROBLEMS = {"ale_ahc008": "ahc008", "ale_ahc011": "ahc011", "ale_ahc015": "ahc015", "ale_ahc024": "ahc024",
            "ale_ahc026": "ahc026", "ale_ahc039": "ahc039", "ale_ahc046": "ahc046"}

BASELINE_TEXT = {
    "ale_ahc008": "Every person does nothing for all 300 turns.",
    "ale_ahc011": "Perform no operation (output an empty line).",
    "ale_ahc015": "Tilt forward (F) after every candy.",
    "ale_ahc024": "Output the given map unchanged.",
    "ale_ahc026": "For v = 1..n: if boxes lie on box v, move them (from the box directly above v) to the next stack, "
                  "then carry out v.",
    "ale_ahc039": "The best single cell of a 10 x 10 grid over the square (the axis-parallel cell with the largest "
                  "mackerels minus sardines).",
    "ale_ahc046": "Visit the targets in order with Move actions only: first vertically, then horizontally.",
}

BASELINE_CODE = {
    "ale_ahc008": r'''#include <bits/stdc++.h>
using namespace std;
int main() {
    int n; cin >> n;
    for (int i = 0; i < n; i++) { int x, y, t; cin >> x >> y >> t; }
    int m; cin >> m;
    for (int i = 0; i < m; i++) { int x, y; cin >> x >> y; }
    for (int turn = 0; turn < 300; turn++) {
        cout << string(m, '.') << endl;
        for (int i = 0; i < n; i++) { string s; cin >> s; }
    }
}
''',
    "ale_ahc011": r'''#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, t; cin >> n >> t;
    cout << "" << endl;
}
''',
    "ale_ahc015": r'''#include <bits/stdc++.h>
using namespace std;
int main() {
    for (int i = 0; i < 100; i++) { int f; cin >> f; }
    for (int t = 0; t < 100; t++) { int p; cin >> p; cout << "F" << endl; }
}
''',
    "ale_ahc024": r'''#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m; cin >> n >> m;
    vector<vector<int>> c(n, vector<int>(n));
    for (auto& row : c) for (auto& v : row) cin >> v;
    for (auto& row : c) { for (int j = 0; j < n; j++) cout << row[j] << (j + 1 < n ? ' ' : '\n'); }
}
''',
    "ale_ahc026": r'''#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m; cin >> n >> m;
    vector<vector<int>> st(m);
    for (int i = 0; i < m; i++) for (int j = 0; j < n / m; j++) { int b; cin >> b; st[i].push_back(b); }
    for (int v = 1; v <= n; v++) {
        int s = 0, p = 0;
        for (int i = 0; i < m; i++) for (int j = 0; j < (int)st[i].size(); j++) if (st[i][j] == v) { s = i; p = j; }
        if (p + 1 < (int)st[s].size()) {
            int d = (s + 1) % m;
            cout << st[s][p + 1] << ' ' << d + 1 << '\n';
            st[d].insert(st[d].end(), st[s].begin() + p + 1, st[s].end());
            st[s].resize(p + 1);
        }
        cout << v << " 0\n";
        st[s].pop_back();
    }
}
''',
    "ale_ahc039": r'''#include <bits/stdc++.h>
using namespace std;
int main() {
    // The best cell of a 10 x 10 grid: a closed square [x0, x1] x [y0, y1] with side 10000.
    int n; cin >> n;
    const int K = 10, W = 100000 / K;
    vector<vector<int>> val(K, vector<int>(K, 0));
    for (int i = 0; i < 2 * n; i++) {
        int x, y; cin >> x >> y;
        for (int gx = 0; gx < K; gx++) for (int gy = 0; gy < K; gy++)
            if (gx * W <= x && x <= (gx + 1) * W && gy * W <= y && y <= (gy + 1) * W) val[gx][gy] += (i < n ? 1 : -1);
    }
    int bx = 0, by = 0;
    for (int gx = 0; gx < K; gx++) for (int gy = 0; gy < K; gy++) if (val[gx][gy] > val[bx][by]) { bx = gx; by = gy; }
    int x0 = bx * W, x1 = (bx + 1) * W, y0 = by * W, y1 = (by + 1) * W;
    cout << "4\n" << x0 << ' ' << y0 << '\n' << x1 << ' ' << y0 << '\n' << x1 << ' ' << y1 << '\n' << x0 << ' ' << y1 << '\n';
}
''',
    "ale_ahc046": r'''#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m; cin >> n >> m;
    vector<int> r(m), c(m);
    for (int k = 0; k < m; k++) cin >> r[k] >> c[k];
    for (int k = 1; k < m; k++) {
        for (int i = r[k - 1]; i < r[k]; i++) cout << "M D\n";
        for (int i = r[k - 1]; i > r[k]; i--) cout << "M U\n";
        for (int j = c[k - 1]; j < c[k]; j++) cout << "M R\n";
        for (int j = c[k - 1]; j > c[k]; j--) cout << "M L\n";
    }
}
''',
}


def _dir(task: str) -> Path:
    return DATA / PROBLEMS[task]


def data(task: str) -> dict:
    return json.loads((_dir(task) / "data.json").read_text())


def reactive(task: str) -> bool:
    return data(task)["metadata"]["problem_type"] == "reactive"


def time_limit(task: str) -> float:
    return float(data(task)["constraints"]["time_limit"])


def public_seeds(task: str) -> list[int]:
    return [int(s) for s in data(task)["seeds"]["public"]]


def winner_mean(task: str) -> float:
    """The contest winner's mean per-case score on the private Lite seeds (rank 1 of standings_scores_lite.csv)."""
    rows = [line.split(",") for line in (_dir(task) / "standings_scores_lite.csv").read_text().splitlines()[1:]]
    top = min(rows, key=lambda r: int(r[0]))
    return float(top[1]) / len(data(task)["seeds"]["private_lite"])


def inputs(task: str) -> list[Path]:
    """The public cases, generated once with the problem's gen from the public seeds (gen writes in/0000.txt, ...)."""
    folder = _dir(task) / "public"
    seeds = public_seeds(task)
    files = [folder / "in" / f"{i:04d}.txt" for i in range(len(seeds))]
    if not all(f.is_file() for f in files):
        folder.mkdir(exist_ok=True)
        (folder / "seeds.txt").write_text("".join(f"{s}\n" for s in seeds))
        subprocess.run([str(_dir(task) / "tools" / "target" / "release" / "gen"), "seeds.txt"], cwd=folder, check=True,
                       capture_output=True, timeout=600)
    return files


def spec(task: str) -> dict:
    meta = data(task)["metadata"]
    n = len(public_seeds(task))
    return {"dir": PROBLEMS[task], "title": f"AtCoder Heuristic Contest {PROBLEMS[task][3:]}",
            "instances": [{"case": i, "seed": s} for i, s in enumerate(public_seeds(task))],
            "direction": "max" if meta["score_type"] == "maximize" else "min", "metric": "score",
            "baseline": BASELINE_TEXT[task], "reference": [winner_mean(task)] * n, "suite": "ale_bench"}


def instances(task: str) -> list[dict]:
    return [{**inst, "time_limit_s": time_limit(task)} for inst in spec(task)["instances"]]


def compile_program(code: str, folder: Path) -> tuple[Path | None, str]:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Main.cpp").write_text(code)
    binary = folder / "a.out"
    try:
        out = subprocess.run([CXX, *COMPILE, f"-I{ACL}", "-o", str(binary), "Main.cpp"], cwd=folder, capture_output=True,
                             text=True, timeout=COMPILE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, "compilation timed out"
    return (binary, "") if out.returncode == 0 and binary.is_file() else (None, out.stderr)


def _limits(cpu_s: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
        os.setsid()
    return apply


def _score(stderr: str) -> tuple[int | None, str]:
    """ALE-Bench's judgment of the tester's stderr: (score, '') or (None, wrong-answer message)."""
    stderr = (stderr or "").strip()
    if "wrong answer: " in stderr:
        return None, "wrong answer: " + stderr.split("wrong answer: ")[1][:250]
    match = re.match(r"Score = (\d+)", stderr.splitlines()[-1]) if stderr else None
    return (int(match.group(1)), "") if match else (None, "wrong answer: " + stderr[-250:])


def _run_case(task: str, binary: Path, problem: dict, source: Path, work: Path) -> dict:
    limit = problem["time_limit_s"]
    ceil = math.ceil(limit + 0.1)
    tester = str(_dir(task) / "tools" / "target" / "release" / "tester")
    out_file = work / f"{problem['case']:04d}.out"
    case = {"problem": problem, "valid": False, "score": None, "traceback_tail": "", "error_trace": ""}
    start = time.time()
    exit_file = work / f"{problem['case']:04d}.exit"     # interactive: the program's own exit code (crash = bug)
    exit_file.unlink(missing_ok=True)
    with source.open() as stdin, out_file.open("w") as stdout:
        wrapped = ["/bin/sh", "-c", '"$0"; echo $? > "$1"', str(binary), str(exit_file)]
        command = [tester, *wrapped] if reactive(task) else [str(binary)]
        child = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE, text=True, cwd=work,
                                 preexec_fn=_limits(ceil + 1))
        try:
            _, stderr = child.communicate(timeout=ceil + 0.2)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, 9)
            child.communicate()
            return {**case, "status": "timeout", "runtime_s": round(time.time() - start, 3),
                    "wall_s": round(time.time() - start, 3), "reason": f"Exceeded the time limit of {limit:g} s"}
    runtime = round(time.time() - start, 3)
    case.update(runtime_s=runtime, wall_s=runtime)
    if runtime > limit:
        return {**case, "status": "timeout", "reason": f"Exceeded the time limit of {limit:g} s"}
    if not reactive(task):
        if child.returncode != 0:
            return {**case, "status": "error", "reason": f"RuntimeError: exit code {child.returncode}",
                    "error_trace": (stderr or "")[-3000:], "traceback_tail": (stderr or "")[-600:]}
        judged = subprocess.run([tester, str(source), str(out_file)], capture_output=True, text=True, cwd=work,
                                timeout=120)
        if judged.returncode != 0:
            return {**case, "status": "ok", "reason": "wrong answer: " + (judged.stderr or "")[-250:]}
        stderr = judged.stderr
    else:
        code = exit_file.read_text().strip() if exit_file.is_file() else ""
        if code not in ("", "0"):
            return {**case, "status": "error", "reason": f"RuntimeError: exit code {code}",
                    "error_trace": (stderr or "")[-3000:], "traceback_tail": (stderr or "")[-600:]}
        if child.returncode != 0:
            return {**case, "status": "ok", "reason": "wrong answer: " + (stderr or "")[-250:]}
    score, reason = _score(stderr)
    if score is None:
        return {**case, "status": "ok", "reason": reason}
    return {**case, "status": "ok", "valid": True, "score": float(score), "reason": ""}


def run_program(task: str, code: str, folder: Path) -> dict:
    """Compile once, then run every public case (PARALLEL at a time) and judge it as ALE-Bench does."""
    binary, error = compile_program(code, folder)
    problems = instances(task)
    if binary is None:
        line = next((l for l in error.splitlines() if "error" in l), error.strip()[:200])
        cases = [{"problem": p, "valid": False, "score": None, "runtime_s": None, "wall_s": 0.0, "status": "error",
                  "reason": f"CompileError: {line}"[:300], "traceback_tail": error[-600:], "error_trace": error[-3000:]}
                 for p in problems]
    else:
        files = inputs(task)
        work = folder / "cases"
        work.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            cases = list(pool.map(lambda p: _run_case(task, binary, p, files[p["case"]], work), problems))
        crash = next((c for c in cases if c["status"] == "error"), None)
        if crash is not None:
            _diagnose(task, code, folder / "debug", crash, files[crash["problem"]["case"]])
    return {"task": task, "budget_s": None, "cases": cases, "all_valid": all(c["valid"] for c in cases),
            "scores": [c["score"] for c in cases]}


FRAME = re.compile(r"#(\d+) 0x[0-9a-f]+ in (.+?) \S*Main\.cpp:(\d+)")


def _diagnose(task: str, code: str, folder: Path, case: dict, source: Path) -> None:
    """A crash (non-zero exit, e.g. 139 = segmentation fault) gives no traceback in C++. Re-run this case once with an
    AddressSanitizer/UBSan build to report where it happened, like a Python traceback: the sanitizer's error, the stack
    frames in Main.cpp and their source lines. Only the case's reason and error_trace (read by the fix loop) change;
    the scores come from the normal build. If the sanitizer build does not reproduce the crash, the exit code stays."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Main.cpp").write_text(code)
    binary = folder / "a.out"
    try:
        built = subprocess.run([CXX, *COMPILE, *DEBUG_COMPILE, f"-I{ACL}", "-o", str(binary), "Main.cpp"], cwd=folder,
                               capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return
    if built.returncode != 0 or not binary.is_file():
        return
    limit = case["problem"]["time_limit_s"] * DEBUG_TIME_FACTOR + 5
    tester = str(_dir(task) / "tools" / "target" / "release" / "tester")
    command = [tester, str(binary)] if reactive(task) else [str(binary)]
    try:
        with source.open() as stdin:
            run = subprocess.run(command, stdin=stdin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                                 cwd=folder, timeout=limit, env={**os.environ, **SANITIZER_ENV},
                                 start_new_session=True)
        stderr = run.stderr or ""
    except subprocess.TimeoutExpired:
        return
    lines = stderr.splitlines()
    head = next((l.split("ERROR: ", 1)[1] for l in lines if "ERROR: AddressSanitizer" in l or "ERROR: UndefinedBehavior" in l),
                next((l.split("runtime error: ", 1)[1].join(["UndefinedBehaviorSanitizer: ", ""])
                      for l in lines if "runtime error: " in l), None))
    if head is None:
        case["reason"] = f"{case['reason']} (not reproduced under AddressSanitizer: no location available)"[:300]
        return
    head = re.sub(r" (?:\(pc |at pc ).*", "", head).strip()          # registers and addresses: noise
    crash_stack = re.split(r"\n(?:0x[0-9a-f]+ is located|allocated by|freed by|previously allocated)", stderr)[0]
    source_lines = code.splitlines()
    frames, previous, repeats = [], None, 0
    for match in FRAME.finditer(crash_stack):
        line = int(match.group(3))
        key = (line, match.group(2).split("(")[0])
        if key == previous:                     # recursion: one frame and a repeat count
            repeats += 1
            continue
        if repeats > 1:
            frames.append(f"  (the frame above repeats {repeats} more times)")
        previous, repeats = key, 0
        text = source_lines[line - 1].strip() if 0 < line <= len(source_lines) else ""
        frames.append(f"  Main.cpp, line {line}, in {key[1]}\n    {text}")
        if len(frames) >= 8:
            break
    if repeats > 1:
        frames.append(f"  (the frame above repeats {repeats} more times)")
    where = frames[0].split("\n")[0].strip() if frames else "location unknown"
    case["reason"] = f"{case['reason']} ({head} at {where})"[:300]
    case["error_trace"] = ("Crash reported by AddressSanitizer (debug build of the same program, same case):\n"
                           + head.strip() + "\nStack (innermost first):\n" + "\n".join(frames))[-3000:]
    case["traceback_tail"] = case["error_trace"][-600:]


def statement(task: str) -> str:
    """The English statement without images, SVG figures, links and the tools section."""
    text = (_dir(task) / "statement_en.md").read_text()
    text = re.sub(r"<svg.*?</svg>", "", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"<a [^>]*>(.*?)</a>", r"\1", text, flags=re.S)
    text = re.sub(r"<span[^>]*>(.*?)</span>", r"\1", text, flags=re.S)
    text = re.sub(r"<font[^>]*>(.*?)</font>", r"\1", text, flags=re.S)
    text = re.split(r"\nTools \(Input generator and visualizer\)\n", text)[0]
    text = text.replace("{sample example}", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def background(task: str, baseline_scores: list[float]) -> str:
    problems = instances(task)
    limit = time_limit(task)
    shown = ", ".join(f"{v:.6g}" for v in baseline_scores)
    return "\n".join([
        f"PROBLEM: {spec(task)['title']}.",
        statement(task),
        "",
        "PROGRAM CONTRACT. Your proposal must contain the complete program, as one ```cpp block with a complete C++ "
        "program (int main). It is compiled once (g++, -std=gnu++20 -O2, AtCoder Library available as "
        "<atcoder/all>; no Boost, GMP or Eigen) and run as is on every test case: it reads the input from standard "
        "input and writes the output to standard output in the format above"
        + (", flushing after every output line (interactive problem)" if reactive(task) else "")
        + f". Time limit: {limit:g} s per case (wall clock), one CPU thread, 1 GiB memory; measure time in the "
        "program and stop with a margin. Any wrong answer, crash or time-limit overrun on a case scores zero for "
        "the program.",
        f"Test cases: {len(problems)} (the public seeds, generated with the official generator).",
        "",
        "BASELINE (a simple, obvious method; an idea is accepted only if it scores above it on every case):",
        BASELINE_TEXT[task],
        f"Baseline score per case: {shown}.",
        "",
        "Every program you write is run on all cases during the search. Its validity, score and runtime become "
        "part of the idea's quality, and are reported back to you. The final evaluation runs the frozen program "
        "once more on the same cases.",
    ])


def source_url(task: str) -> str:
    return f"https://huggingface.co/datasets/{DATA_REPO}/blob/{DATA_PIN}/{PROBLEMS[task]}.zip"


def source_files(task: str) -> dict[str, str]:
    import hashlib
    folder = _dir(task)
    files = [folder / "data.json", folder / "statement_en.md", folder / "standings_scores_lite.csv",
             folder / "tools" / "Cargo.lock", *sorted((folder / "tools" / "src").rglob("*.rs"))]
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


CPP_BLOCK = re.compile(r"```(?:cpp|c\+\+|cc|cxx)[^\n]*\n(.*?)```", re.DOTALL)


def extract_program(text: str) -> str | None:
    """The last fenced C++ block that defines main."""
    blocks = [b for b in CPP_BLOCK.findall(text or "") if re.search(r"\bint\s+main\s*\(", b)]
    return blocks[-1] if blocks else None


def replace_program(text: str, code: str) -> str:
    matches = [m for m in CPP_BLOCK.finditer(text or "") if re.search(r"\bint\s+main\s*\(", m.group(1))]
    if not matches:
        return text
    last = matches[-1]
    return text[:last.start(1)] + (code if code.endswith("\n") else code + "\n") + text[last.end(1):]


PROMPT_EDITS = [
    # (text in the program-QD prompts, C++ wording); applied only to ALE-Bench tasks (prompts.adapt)
    ("```python block defining solve(problem)", "```cpp block with a complete C++ program (int main)"),
    ("```python\n", "```cpp\n"),
    ("Python raised the error shown", "it failed to compile or crashed with the error shown"),
    ("EXCEPTION (raised when solve was called with", "ERROR (when compiled, or when run on"),
    ("Read every input, including time_limit_s, from problem; keep a valid construction at all times; return the "
     "best valid one found before time_limit_s.",
     "Read the input from standard input and write the output to standard output; keep a valid solution at all "
     "times and output the best valid one found before the time limit."),
    ("Use only the standard library and numpy (networkx and sympy are available; scipy is not), one CPU thread.",
     "Use only the C++ standard library and the AtCoder Library, one CPU thread."),
    ("solve(problem)", "the program"),
    ("time_limit_s", "the time limit"),
]
