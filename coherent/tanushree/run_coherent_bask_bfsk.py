"""
Run the coherent BASK / BFSK simulations, save CSVs and the BER plot.

    python coherent/run_coherent_bask_bfsk.py        # from the repo root

Outputs
    plots/data/coherent_bask.csv, coherent_bfsk.csv   (read by compare_coherent_vs_noncoherent.py)
    plots/coherent_bask_bfsk_ber.png
    plots/constellations_coherent.png, plots/gram_schmidt_basis.png
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
from coherent import gs_signal_space
from coherent.coherent_bask_bfsk import (SCHEMES, coherent_trial, simulate_curve,
                                         simulate_point, waveform_trial)

EBN0_DB = np.arange(0, 13, 1.0)      # 0 to 12 dB, as in the project plan
MIN_ERRORS_TO_PLOT = 20
DATA = ROOT / "plots" / "data"
PLOTS = ROOT / "plots"
DATA.mkdir(parents=True, exist_ok=True)

NAMES = {"BASK": "Coherent BASK (correlator)", "BFSK": "Coherent BFSK (correlator)"}
STYLE = {"BASK": ("tab:blue", "o"), "BFSK": ("tab:green", "s")}


def verify_waveform_receiver(rng):
    """Waveform-level correlator vs signal-space shortcut at a few Eb/N0 values."""
    print("\nCorrelator check (waveform level vs signal-space level vs theory)")
    for name in ("BASK", "BFSK"):
        theory = SCHEMES[name][1]
        for e in (4.0, 8.0):
            bw, _, _ = simulate_point(lambda b, x, r: waveform_trial(name, b, x, r), e, rng,
                                      target_errors=300, batch=50_000, max_bits=2_000_000)
            bs, _, _ = simulate_point(lambda b, x, r: coherent_trial(name, b, x, r), e, rng,
                                      target_errors=300, batch=50_000, max_bits=2_000_000)
            print(f"  {name} {e:4.1f} dB: waveform {bw:.3e}  signal-space {bs:.3e}  "
                  f"theory {float(theory(e)):.3e}")


def main():
    gs_signal_space.main()           # Gram-Schmidt table + constellation + basis plots

    rng = get_rng(2027)
    fine = np.linspace(0, 12, 121)
    fig, ax = plt.subplots(figsize=(7, 5))

    for name in ["BASK", "BFSK"]:
        rows = simulate_curve(name, EBN0_DB, rng)
        theory = SCHEMES[name][1]
        with open(DATA / f"coherent_{name.lower()}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ebn0_db", "ber_sim", "ber_theory", "n_errors", "n_bits"])
            for e, ber, err, n in rows:
                w.writerow([e, f"{ber:.6e}", f"{float(theory(e)):.6e}", err, n])
        color, marker = STYLE[name]
        # BASK and BFSK have identical theory, so BASK is drawn thick underneath
        ax.semilogy(fine, theory(fine), "-", color=color, lw=5 if name == "BASK" else 1.5,
                    alpha=0.45 if name == "BASK" else 1.0, label=f"{NAMES[name]}, theory")
        ok = [(e, b) for e, b, err, _ in rows if err >= MIN_ERRORS_TO_PLOT]
        ax.semilogy(*zip(*ok), marker, color=color, mfc="none", label=f"{NAMES[name]}, simulated")
        print(name, "done")

    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit error rate")
    ax.set_title("Coherent BASK and BFSK over AWGN: simulated vs theoretical BER")
    ax.set_ylim(1e-6, 1)
    ax.set_xlim(0, 12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "coherent_bask_bfsk_ber.png", dpi=200)

    verify_waveform_receiver(rng)


if __name__ == "__main__":
    main()
