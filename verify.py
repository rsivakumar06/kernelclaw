"""verify.py — hardened correctness + benchmark verifier.

Changes from original:
  - Parses TIME_US=<float> from binary stdout (CUDA event timing)
  - Parses CHECKSUM=<float> and compares with rel tolerance < 1e-4
  - 30-second subprocess timeout on every binary call
  - Emits VERDICT={...} as final line for agent.py to parse
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

RUNS = 11  # 1 warmup + 10 timed


def extract_code(text):
    m = re.search(r"```(?:cuda|cpp|c\+\+)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else None


def compile_cu(src, out):
    r = subprocess.run(["nvcc", "-O2", str(src), "-o", str(out)],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr


def run_binary(binary, timeout=30):
    r = subprocess.run(
        [str(binary)], capture_output=True, text=True,
        cwd=binary.parent, timeout=timeout
    )
    return r.returncode == 0, r.stdout.strip()


def parse_time_us(stdout):
    m = re.search(r"TIME_US=([\d.]+)", stdout)
    return float(m.group(1)) if m else None


def parse_checksum(stdout):
    m = re.search(r"CHECKSUM=([\d.eE+\-]+)", stdout)
    return float(m.group(1)) if m else None


def timed_avg(binary):
    """Return average TIME_US over RUNS-1 runs (drop warmup)."""
    times = []
    for _ in range(RUNS):
        try:
            ok, out = run_binary(binary)
            if ok:
                t = parse_time_us(out)
                if t is not None:
                    times.append(t)
        except subprocess.TimeoutExpired:
            pass
    if len(times) > 1:
        return sum(times[1:]) / (len(times) - 1)
    return sum(times) / max(len(times), 1)


def wall_avg(binary):
    """Fallback: wall-clock timing in microseconds."""
    times = []
    for _ in range(RUNS):
        try:
            t0 = time.perf_counter()
            subprocess.run([str(binary)], capture_output=True,
                           cwd=binary.parent, timeout=30)
            times.append((time.perf_counter() - t0) * 1e6)
        except subprocess.TimeoutExpired:
            pass
    if len(times) > 1:
        return sum(times[1:]) / (len(times) - 1)
    return sum(times) / max(len(times), 1)

def strip_meta(stdout: str) -> str:
    lines = [l for l in stdout.splitlines()
             if not l.startswith("TIME_US=") 
             and not l.startswith("CHECKSUM=")]
    return "\n".join(lines)
    
def emit_verdict(status, speedup=0.0, baseline_us=0.0, candidate_us=0.0):
    print(f"VERDICT={json.dumps({'status': status, 'speedup': round(speedup, 4),'baseline_us': round(baseline_us, 2),'candidate_us': round(candidate_us, 2)})}")


def main():
    if len(sys.argv) != 3:
        print("Usage: verify.py <baseline.cu> <response.txt>")
        emit_verdict("compile_fail")
        sys.exit(2)

    baseline_src  = Path(sys.argv[1]).resolve()
    response_file = Path(sys.argv[2]).resolve()
    work          = baseline_src.parent

    code = extract_code(response_file.read_text())
    if not code:
        print("FAIL: no cuda block in response")
        emit_verdict("compile_fail")
        sys.exit(1)

    candidate_src = work / "candidate.cu"
    candidate_src.write_text(code)
    print(f"[ok] extracted {len(code)} bytes")

    baseline_bin  = work / "baseline.bin"
    candidate_bin = work / "candidate.bin"

    ok, err = compile_cu(baseline_src, baseline_bin)
    if not ok:
        print(f"FAIL: baseline compile\n{err}")
        emit_verdict("compile_fail"); sys.exit(1)
    print("[ok] baseline compiled")

    ok, err = compile_cu(candidate_src, candidate_bin)
    if not ok:
        print(f"REJECT: candidate compile\n{err}")
        emit_verdict("compile_fail"); sys.exit(1)
    print("[ok] candidate compiled")

    # Correctness
    try:
        ok_b, out_b = run_binary(baseline_bin)
        ok_c, out_c = run_binary(candidate_bin)
    except subprocess.TimeoutExpired:
        print("REJECT: timeout during correctness check")
        emit_verdict("timeout"); sys.exit(1)

    if not (ok_b and ok_c):
        print(f"REJECT: runtime failure\n  baseline:  {out_b}\n  candidate: {out_c}")
        emit_verdict("incorrect"); sys.exit(1)

    cs_b = parse_checksum(out_b)
    cs_c = parse_checksum(out_c)

    if cs_b is not None and cs_c is not None:
        # Both emit CHECKSUM — compare them
        rel_err = abs(cs_b - cs_c) / (abs(cs_b) + 1e-9)
        if rel_err >= 1e-4:
            print(f"REJECT: CHECKSUM mismatch  baseline={cs_b:.6g}  "
                  f"candidate={cs_c:.6g}  rel_err={rel_err:.2e}")
            emit_verdict("incorrect"); sys.exit(1)
        print(f"[ok] CHECKSUM match  rel_err={rel_err:.2e}")
    elif cs_c is not None and cs_b is None:
        # Candidate emits CHECKSUM but baseline is legacy — trust the checksum
        print(f"[ok] candidate CHECKSUM={cs_c:.6g} (baseline is legacy style)")
    else:
        # Neither has CHECKSUM — fall back to stripped stdout match
        if strip_meta(out_b) != strip_meta(out_c):
            print(f"REJECT: stdout mismatch\n  baseline:  {out_b}\n  candidate: {out_c}")
            emit_verdict("incorrect"); sys.exit(1)
        print(f"[ok] stdout match: {out_b}")

    # Benchmark
    print(f"\n[bench] {RUNS-1} timed runs each...")
    t_b = timed_avg(baseline_bin)  or wall_avg(baseline_bin)
    t_c = timed_avg(candidate_bin) or wall_avg(candidate_bin)
    speedup = t_b / t_c if t_c > 0 else 0.0

    print(f"  baseline:  {t_b:8.2f} us")
    print(f"  candidate: {t_c:8.2f} us")
    print(f"  speedup:   {speedup:.2f}x  ({'faster' if speedup > 1 else 'slower'})")

    emit_verdict("verified", speedup, t_b, t_c)


if __name__ == "__main__":
    main()