#!/usr/bin/env python3

#####################################################################################
#
# The MIT License (MIT)
#
# Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#####################################################################################

import argparse
import re
import subprocess
import sys
import tempfile
import math
import matplotlib.pyplot as plt
from pathlib import Path

# parse lines like: L2 errors (t=3.0): P=1.23e-5, D=9.87e-6
L2_LINE_RE = re.compile(
    r"L2\s*errors\s*\(t\s*=\s*([^)]+)\)\s*:\s*P\s*=\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*D\s*=\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)


def slope(x, y):
    lx = [math.log(v) for v in x]
    ly = [math.log(v) for v in y]
    mx = sum(lx)/len(lx)
    my = sum(ly)/len(ly)
    num = sum((a-mx)*(b-my) for a, b in zip(lx, ly))
    den = sum((a-mx)**2 for a in lx)
    return num/den if den else float('nan')


def rewrite_time_step(rc_text: str, dt: float, header_regex=r'^\s*\[\s*TIME\s*STEP\s*\]\s*$') -> str:
    lines = rc_text.splitlines()
    hdr_re = re.compile(header_regex, re.IGNORECASE)
    header_idx = None
    for i, line in enumerate(lines):
        if hdr_re.match(line):
            header_idx = i
            break
    if header_idx is None:
        # help debug: show nearby lines that contain 'TIME' or 'STEP'
        sample = '\n'.join([ln for ln in lines if (
            'TIME' in ln.upper() or 'STEP' in ln.upper())][:10])
        raise RuntimeError("Could not find [TIME STEP] header in the .rc file.\n"
                           "Saw lines containing 'TIME' or 'STEP':\n" + sample)

    # find the line to replace, skip blank lines
    j = header_idx + 1
    while j < len(lines) and lines[j].strip() == '':
        j += 1

    # if next line is another header (starts with '[') or doesn't exist, insert
    if j >= len(lines) or lines[j].lstrip().startswith('['):
        lines.insert(header_idx + 1, str(dt))
    else:
        lines[j] = str(dt)

    return '\n'.join(lines) + ('\n' if rc_text.endswith('\n') else '')


def run_case(binary: Path, rc_path: Path, mpiexec: str):
    cmd = []
    if mpiexec:
        cmd.extend(mpiexec.split())
    cmd += [str(binary), str(rc_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    m = L2_LINE_RE.search(proc.stdout)
    if not m:
        raise RuntimeError("Could not parse 'L2 errors (t=...): P=..., D=...' from output.\n"
                           "--- STDOUT ---\n" + proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    t = float(m.group(1))
    p = float(m.group(2))
    d = float(m.group(3))
    return t, p, d


def main():
    ap = argparse.ArgumentParser(
        description="Sweep TIME STEP and plot L2 errors vs dt.")
    ap.add_argument("--binary", default="./waveMain", help="Path to waveMain")
    ap.add_argument("--rc", required=True,
                    help="Base .rc file (won't be modified)")
    ap.add_argument("--dt", nargs="+", type=float,
                    default=[0.4, 0.2, 0.1, 0.05, 0.025],
                    help="TIME STEP values to test")
    ap.add_argument("--mpiexec", default="",
                    help='MPI launcher, e.g. "mpirun -n 1"')
    ap.add_argument("--out-png", default="l2error_dt.png")
    args = ap.parse_args()

    binary = Path(args.binary).resolve()
    base_rc = Path(args.rc).resolve()
    rc_text = base_rc.read_text(encoding="utf-8")

    results = []  # (dt, t_final, L2P, L2D)

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for dt in args.dt:
            rc_mod = rewrite_time_step(rc_text, dt)
            rc_tmp = tmpdir / f"{base_rc.stem}_dt_{dt}.rc"
            rc_tmp.write_text(rc_mod, encoding="utf-8")
            print(f"[run] dt={dt} -> {rc_tmp.name}")
            t_final, L2P, L2D = run_case(binary, rc_tmp, args.mpiexec)
            print(f"      L2 errors (t={t_final}): P={L2P:.6e}, D={L2D:.6e}")
            results.append((dt, t_final, L2P, L2D))

    results.sort(key=lambda x: x[0])
    dts = [r[0] for r in results]
    errsP = [r[2] for r in results]
    errsD = [r[3] for r in results]
    print(f"[info] observed order (P): {slope(dts, errsP):.3f}")
    print(f"[info] observed order (D): {slope(dts, errsD):.3f}")

    # final plot
    plt.figure()
    plt.loglog(dts, errsP, marker="o", label=r"$\|e_p\|_{L^2}$")
    plt.loglog(dts, errsD, marker="s", label=r"$\|e_d\|_{L^2}$")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"$L^2$ error")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()

    ax = plt.gca()
    ax.set_xticks(dts)
    ax.set_xticklabels([f"{dt:g}" for dt in dts])
    ax.tick_params(axis="x", labelrotation=30)
    plt.margins(x=0.05)

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"[write] {args.out_png}")


if __name__ == "__main__":
    main()
