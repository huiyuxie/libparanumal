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


def _find_header_index(lines, header_regex):
    hdr_re = re.compile(header_regex, re.IGNORECASE)
    for i, line in enumerate(lines):
        if hdr_re.match(line):
            return i
    return None


def _get_scalar_after_header(rc_text: str, header_regex: str):
    lines = rc_text.splitlines()
    idx = _find_header_index(lines, header_regex)
    if idx is None:
        return None, None
    j = idx + 1
    while j < len(lines) and lines[j].strip() == '':
        j += 1
    if j >= len(lines) or lines[j].lstrip().startswith('['):
        return None, None
    return lines[j].strip(), j


def _rewrite_scalar_after_header(rc_text: str, header_regex: str, value: str) -> str:
    lines = rc_text.splitlines()
    idx = _find_header_index(lines, header_regex)
    if idx is None:
        key = re.sub(r"[\\^$.*+?{}\[\]|()]", "", header_regex).upper()
        sample = '\n'.join([ln for ln in lines if any(
            tok in ln.upper() for tok in key.split())][:10])
        raise RuntimeError(f"could not find header matching {header_regex!r} in the .rc file.\n"
                           f"sample of lines with similar tokens:\n{sample}")
    j = idx + 1
    while j < len(lines) and lines[j].strip() == '':
        j += 1
    if j >= len(lines) or lines[j].lstrip().startswith('['):
        lines.insert(idx + 1, str(value))
    else:
        lines[j] = str(value)
    return '\n'.join(lines) + ('\n' if rc_text.endswith('\n') else '')


def rewrite_time_step(rc_text: str, dt: float) -> str:
    return _rewrite_scalar_after_header(rc_text, r'^\s*\[\s*TIME\s*STEP\s*\]\s*$', str(dt))


def rewrite_box_nx_ny(rc_text: str, nx: int, ny: int) -> str:
    out = _rewrite_scalar_after_header(
        rc_text, r'^\s*\[\s*BOX\s*NX\s*\]\s*$', str(int(nx)))
    out = _rewrite_scalar_after_header(
        out,     r'^\s*\[\s*BOX\s*NY\s*\]\s*$', str(int(ny)))
    return out


def read_time_step(rc_text: str):
    val, _ = _get_scalar_after_header(
        rc_text, r'^\s*\[\s*TIME\s*STEP\s*\]\s*$')
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def run_case(binary: Path, rc_path: Path, mpiexec: str):
    cmd = []
    if mpiexec:
        cmd.extend(mpiexec.split())
    cmd += [str(binary), str(rc_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    m = L2_LINE_RE.search(proc.stdout)
    if not m:
        raise RuntimeError("could not parse 'L2 errors (t=...): P=..., D=...' from output.\n"
                           "--- stdout ---\n" + proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    t = float(m.group(1))
    p = float(m.group(2))
    d = float(m.group(3))
    return t, p, d


def main():
    ap = argparse.ArgumentParser(
        description="sweep time step (dt) and/or grid resolution n (nx=ny=h) and plot l2 errors."
    )
    ap.add_argument("--binary", default="./waveMain", help="path to waveMain")
    ap.add_argument("--rc", required=True,
                    help="base .rc file (won't be modified)")
    # dt-sweep: list of dt values; if omitted, dt-sweep is skipped
    ap.add_argument("--dt", nargs="+", type=float, default=None,
                    help="time step values to test (dt-sweep), e.g., 0.4 0.2 0.1 0.05.")
    # n-sweep: absolute target grid sizes; each value sets nx=ny=n
    ap.add_argument("--h", nargs="+", type=int, default=None,
                    help="absolute grid sizes n (sets [BOX NX]=[BOX NY]=h), e.g., 2 4 8 16.")
    # fixed dt to use during n-sweep
    ap.add_argument("--dt-fixed", type=float, default=None,
                    help="fixed time step to use for n-sweep (defaults to first --dt value, else value in .rc).")
    # fixed n to use during dt-sweep (sets nx=ny=n for every dt)
    ap.add_argument("--h-fixed", type=int, default=None,
                    help="absolute grid size h to use during dt-sweep (sets nx=ny=h).")
    ap.add_argument("--mpiexec", default="",
                    help='mpi launcher, e.g. "mpirun -n 4"')
    ap.add_argument("--out-png-dt", default="l2error_dt.png",
                    help="output png for dt-sweep")
    ap.add_argument("--out-png-h",  default="l2error_h.png",
                    help="output png for n-sweep (nx=ny=h)")
    args = ap.parse_args()

    binary = Path(args.binary).resolve()
    base_rc = Path(args.rc).resolve()
    rc_text = base_rc.read_text(encoding="utf-8")

    did_anything = False

    # -------------------- dt-sweep -------------------- #
    if args.dt:
        did_anything = True

        # optionally set absolute nx=ny for all dt runs if --h-fixed is given
        rc_base_for_dt = rc_text
        if args.h_fixed is not None:
            h = max(1, int(args.h_fixed))
            rc_base_for_dt = rewrite_box_nx_ny(rc_base_for_dt, h, h)
            print(
                f"[info:dt] using fixed grid NX=NY={h} (denoted h) for all dt values")

        results = []  # (dt, t_final, L2P, L2D)
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            for dt in args.dt:
                rc_mod = rewrite_time_step(rc_base_for_dt, dt)
                rc_tmp = tmpdir / f"{base_rc.stem}_dt_{dt}.rc"
                rc_tmp.write_text(rc_mod, encoding="utf-8")
                print(f"[run:dt] dt={dt:g} -> {rc_tmp.name}")
                t_final, L2P, L2D = run_case(binary, rc_tmp, args.mpiexec)
                print(
                    f"          l2 errors (t={t_final}): P={L2P:.6e}, D={L2D:.6e}")
                results.append((dt, t_final, L2P, L2D))

        # sort by dt
        results.sort(key=lambda x: x[0])
        dts = [r[0] for r in results]
        errsP = [r[2] for r in results]
        errsD = [r[3] for r in results]
        print(f"[info:dt] observed order (p): {slope(dts, errsP):.3f}")
        print(f"[info:dt] observed order (d): {slope(dts, errsD):.3f}")

        # plot dt-sweep
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
        plt.title(rf"{base_rc.name} — $L^2$ error vs $\Delta t$")
        plt.tight_layout()
        plt.savefig(args.out_png_dt, dpi=200)
        print(f"[write] {args.out_png_dt}")

    # -------------------- h-sweep -------------------- #
    if args.h:
        did_anything = True

        # choose fixed dt for h-sweep
        dt_fixed = (args.dt_fixed if args.dt_fixed is not None
                    else (args.dt[0] if args.dt else read_time_step(rc_text)))
        if dt_fixed is None:
            raise RuntimeError(
                "no time step found; specify --dt-fixed for h-sweep.")

        print(f"[info:h] using fixed dt = {dt_fixed:g} for all h values")

        results_h = []  # (h, nx, ny, t_final, L2P, L2D)

        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            for h in args.h:
                h = max(1, int(h))
                nx = ny = h
                # start from base text each time
                rc_mod = rewrite_time_step(rc_text, dt_fixed)
                rc_mod = rewrite_box_nx_ny(rc_mod, nx, ny)
                rc_tmp = tmpdir / f"{base_rc.stem}_h_{h}.rc"
                rc_tmp.write_text(rc_mod, encoding="utf-8")
                print(f"[run:h] h={h} -> NX=NY={nx} ({rc_tmp.name})")
                t_final, L2P, L2D = run_case(binary, rc_tmp, args.mpiexec)
                print(
                    f"       l2 errors (t={t_final}): P={L2P:.6e}, D={L2D:.6e}")
                results_h.append((float(h), nx, ny, t_final, L2P, L2D))

        # sort by h
        results_h.sort(key=lambda r: r[0])
        hs = [r[0] for r in results_h]
        errsP = [r[4] for r in results_h]
        errsD = [r[5] for r in results_h]

        print(f"[info:h] observed slope vs h (p): {slope(hs, errsP):.3f}")
        print(f"[info:h] observed slope vs h (d): {slope(hs, errsD):.3f}")

        # plot error vs h
        plt.figure()
        plt.loglog(hs, errsP, marker="o", label=r"$\|e_p\|_{L^2}$")
        plt.loglog(hs, errsD, marker="s", label=r"$\|e_d\|_{L^2}$")
        plt.xlabel(r"$h$")
        plt.ylabel(r"$L^2$ error")
        plt.grid(True, which="both", linestyle=":")
        plt.legend()
        ax = plt.gca()
        ax.set_xticks(hs)
        ax.set_xticklabels([f"{int(h)}" for h in hs])
        ax.tick_params(axis="x", labelrotation=30)
        plt.margins(x=0.05)
        plt.title(rf"{base_rc.name} — $L^2$ error vs $h$")
        plt.tight_layout()
        plt.savefig(args.out_png_h, dpi=200)
        print(f"[write] {args.out_png_h}")

    if not did_anything:
        print("nothing to do: provide --dt (for dt-sweep) and/or --h (for n-sweep).", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
