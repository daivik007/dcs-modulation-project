"""
Run the non-coherent simulations and save CSVs + the BER plot.

    python noncoherent/run_noncoherent.py            # from the repo root
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel.awgn import get_rng
from noncoherent.noncoherent_schemes import SCHEMES, simulate_curve

EBN0_DB = np.arange(0, 13, 1.0)      # 0 to 12 dB, as in the project plan
MIN_ERRORS_TO_PLOT = 20              # drop points with too few errors to trust
DATA = ROOT / "plots" / "data"
PLOTS = ROOT / "plots"
DATA.mkdir(parents=True, exist_ok=True)

NAMES = {"BASK": "Non-coherent BASK (envelope)",
         "BFSK": "Non-coherent BFSK (energy)",
         "DPSK": "DPSK (differential)"}
STYLE = {"BASK": ("tab:blue", "o"), "BFSK": ("tab:green", "s"), "DPSK": ("tab:red", "^")}


def main():
    rng = get_rng(2026)
    fine = np.linspace(0, 12, 121)
    fig, ax = plt.subplots(figsize=(7, 5))

    for name in ["BASK", "BFSK", "DPSK"]:
        rows = simulate_curve(name, EBN0_DB, rng)
        theory = SCHEMES[name][1]
        with open(DATA / f"noncoherent_{name.lower()}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ebn0_db", "ber_sim", "ber_theory", "n_errors", "n_bits"])
            for e, ber, err, n in rows:
                w.writerow([e, f"{ber:.6e}", f"{float(theory(e)):.6e}", err, n])
        color, marker = STYLE[name]
        ax.semilogy(fine, theory(fine), "-", color=color, label=f"{NAMES[name]}, theory")
        ok = [(e, b) for e, b, err, _ in rows if err >= MIN_ERRORS_TO_PLOT]
        ax.semilogy(*zip(*ok), marker, color=color, mfc="none", label=f"{NAMES[name]}, simulated")
        print(name, "done")

    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit error rate")
    ax.set_title("Non-coherent schemes over AWGN: simulated vs theoretical BER")
    ax.set_ylim(1e-6, 1)
    ax.set_xlim(0, 12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "noncoherent_ber.png", dpi=200)


if __name__ == "__main__":
    main()
