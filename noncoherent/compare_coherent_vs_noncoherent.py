"""
Final coherent vs non-coherent comparison (owner: Himanshu Singh).

    python noncoherent/compare_coherent_vs_noncoherent.py     # from the repo root

Inputs  (plots/data/, columns: ebn0_db, ber_sim)
    noncoherent_bask.csv, noncoherent_bfsk.csv, noncoherent_dpsk.csv   (this folder's run script)
    coherent_bask.csv, coherent_bfsk.csv, coherent_bpsk.csv            (Tanushree and Daivik)
If a coherent CSV is missing, only the theoretical coherent curve is drawn,
so this script runs before the other two members have finished.

Outputs
    plots/comparison_coherent_vs_noncoherent.png   BER overlay
    plots/snr_penalty.png                          dB gap vs target BER
    plots/data/snr_penalty.csv                     Eb/N0 needed and gap at fixed BER
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
from scipy.optimize import brentq

from channel.awgn import db_to_lin, qfunc
from noncoherent.noncoherent_schemes import (bask_nc_ber_theory, bfsk_nc_ber_theory,
                                             dpsk_ber_theory)

DATA = ROOT / "plots" / "data"
PLOTS = ROOT / "plots"


# Coherent theory (Eb is the average bit energy in every case)
def bask_coh_theory(e):
    return qfunc(np.sqrt(db_to_lin(e)))          # OOK, Q(sqrt(Eb/N0))


def bfsk_coh_theory(e):
    return qfunc(np.sqrt(db_to_lin(e)))          # orthogonal FSK, Q(sqrt(Eb/N0))


def bpsk_coh_theory(e):
    return qfunc(np.sqrt(2 * db_to_lin(e)))      # Q(sqrt(2 Eb/N0))


# (label, coherent theory, non-coherent theory, coherent csv, non-coherent csv, colour)
PAIRS = [
    ("BASK", bask_coh_theory, bask_nc_ber_theory, "coherent_bask.csv", "noncoherent_bask.csv", "tab:blue"),
    ("BFSK", bfsk_coh_theory, bfsk_nc_ber_theory, "coherent_bfsk.csv", "noncoherent_bfsk.csv", "tab:green"),
    ("BPSK vs DPSK", bpsk_coh_theory, dpsk_ber_theory, "coherent_bpsk.csv", "noncoherent_dpsk.csv", "tab:red"),
]


def load(name):
    path = DATA / name
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["ebn0_db"]) for r in rows]), np.array([float(r["ber_sim"]) for r in rows])


def required_ebn0_db(theory, target):
    """Eb/N0 (dB) at which theory(Eb/N0) = target."""
    f = lambda e: np.log10(float(theory(e))) - np.log10(target)
    return brentq(f, -5.0, 20.0, xtol=1e-6)


def main():
    fine = np.linspace(0, 12, 121)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, coh, nc, coh_csv, nc_csv, color in PAIRS:
        # coherent BASK and coherent BFSK have the same formula, so BASK is drawn thick underneath
        ax.semilogy(fine, coh(fine), "-", color=color, lw=5 if label == "BASK" else 1.5,
                    alpha=0.45 if label == "BASK" else 1.0,
                    label=f"{label.split(' vs ')[0]} coherent (theory)")
        ax.semilogy(fine, nc(fine), "--", color=color,
                    label=f"{label.split(' vs ')[-1] if 'vs' in label else label} non-coherent (theory)")
        for csv_name, marker, tag in [(coh_csv, "o", "coherent sim"), (nc_csv, "x", "non-coherent sim")]:
            d = load(csv_name)
            if d is not None:
                e, b = d
                m = b > 1e-6
                ax.semilogy(e[m], b[m], marker, color=color, ms=4, alpha=0.8)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit error rate")
    ax.set_title("Coherent (solid) vs non-coherent (dashed) over AWGN")
    ax.set_xlim(0, 12)
    ax.set_ylim(1e-6, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(PLOTS / "comparison_coherent_vs_noncoherent.png", dpi=200)

    # SNR penalty at fixed BER
    targets = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    rows = []
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
    for label, coh, nc, _, _, color in PAIRS:
        gaps = []
        for t in targets:
            ec, en = required_ebn0_db(coh, t), required_ebn0_db(nc, t)
            gaps.append(en - ec)
            rows.append((label, t, ec, en, en - ec))
        ax2.semilogx(targets, gaps, "o-", color=color, label=label)
    ax2.set_xlabel("Target BER")
    ax2.set_ylabel("Non-coherent penalty (dB)")
    ax2.set_title("Eb/N0 penalty of non-coherent detection")
    ax2.invert_xaxis()
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(PLOTS / "snr_penalty.png", dpi=200)

    with open(DATA / "snr_penalty.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "target_ber", "ebn0_coherent_db", "ebn0_noncoherent_db", "penalty_db"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.0e}", f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.2f}"])
    print(f"{'pair':14s} {'BER':>6s} {'coh dB':>8s} {'nc dB':>8s} {'gap dB':>7s}")
    for r in rows:
        print(f"{r[0]:14s} {r[1]:6.0e} {r[2]:8.2f} {r[3]:8.2f} {r[4]:7.2f}")


if __name__ == "__main__":
    main()
