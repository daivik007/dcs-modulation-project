"""
Gram-Schmidt orthogonalization and signal-space diagrams for BASK, BFSK, BPSK, QPSK
(owner: Tanushree Paidi).

    python coherent/tanushree/gs_signal_space.py        # from the repo root

Everything is computed numerically on sampled waveforms, so the basis functions
plotted here are exactly the ones the correlator receiver projects onto.
All signal sets are scaled so the AVERAGE energy per bit is Eb.

Outputs
    plots/constellations_coherent.png   signal-space diagrams
    plots/gram_schmidt_basis.png        the orthonormal basis functions phi_j(t)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLOTS = ROOT / "plots"

# ---- waveform parameters (normalised: Rb = 1 bit/s) ---------------------------
Tb = 1.0             # bit duration
SPS = 20             # samples per bit
FC = 4.0             # BASK / BPSK / QPSK carrier (integer cycles per Tb)
F1, F2 = 4.0, 5.0    # BFSK tones; spacing 1/Tb makes them orthogonal over Tb


def time_axis(T):
    """Sample times for a symbol of length T (endpoint excluded)."""
    n = int(round(SPS * T / Tb))
    dt = Tb / SPS
    return np.arange(n) * dt, dt


def inner(a, b, dt):
    """<a, b> = integral of a(t) b(t) dt (rectangle rule)."""
    return np.sum(a * b) * dt


# ---- signal sets --------------------------------------------------------------
def signal_set(name, Eb=1.0):
    """Return (t, dt, [waveforms], [bit labels]) with average bit energy Eb."""
    name = name.upper()
    if name == "BASK":                       # on-off keying: s0 = 0, s1 = A cos
        t, dt = time_axis(Tb)
        E1 = 2 * Eb                          # (0 + E1) / 2 = Eb
        s1 = np.sqrt(2 * E1 / Tb) * np.cos(2 * np.pi * FC * t)
        return t, dt, [np.zeros_like(t), s1], ["0", "1"]
    if name == "BFSK":
        t, dt = time_axis(Tb)
        A = np.sqrt(2 * Eb / Tb)
        return t, dt, [A * np.cos(2 * np.pi * F1 * t), A * np.cos(2 * np.pi * F2 * t)], ["0", "1"]
    if name == "BPSK":
        t, dt = time_axis(Tb)
        A = np.sqrt(2 * Eb / Tb)
        c = np.cos(2 * np.pi * FC * t)
        return t, dt, [A * c, -A * c], ["0", "1"]
    if name == "QPSK":                       # Ts = 2 Tb, Es = 2 Eb, Gray mapping
        Ts = 2 * Tb
        t, dt = time_axis(Ts)
        A = np.sqrt(2 * (2 * Eb) / Ts)
        phases = {"00": 45, "01": 135, "11": 225, "10": 315}
        sigs = [A * np.cos(2 * np.pi * FC * t + np.deg2rad(p)) for p in phases.values()]
        return t, dt, sigs, list(phases)
    raise ValueError(name)


# ---- Gram-Schmidt -------------------------------------------------------------
def gram_schmidt(signals, dt, tol=1e-9, seed=None):
    """
    Gram-Schmidt on a list of sampled signals.

    Returns
        Phi    (N, L)  orthonormal basis functions
        coords (M, N)  coordinates, s_i(t) = sum_j coords[i, j] * phi_j(t)

    A zero signal, or one that is a combination of earlier ones, adds no basis function.
    seed: optional signals orthogonalised first, only to fix the orientation of the
          axes (QPSK uses cos / -sin so the constellation is the textbook one).
    """
    Emax = max(inner(s, s, dt) for s in signals)
    basis = []
    for s in list(seed or []) + list(signals):
        g = s.astype(float).copy()
        for phi in basis:
            g -= inner(s, phi, dt) * phi
        e = inner(g, g, dt)
        if e > tol * Emax:
            basis.append(g / np.sqrt(e))
    Phi = np.array(basis)
    coords = np.array([[inner(s, phi, dt) for phi in Phi] for s in signals])
    return Phi, coords


# ---- plotting -----------------------------------------------------------------
def plot_constellation(ax, name, coords, labels):
    N = coords.shape[1]
    pts = np.hstack([coords, np.zeros((len(coords), 1))]) if N == 1 else coords
    ax.scatter(pts[:, 0], pts[:, 1], s=80, zorder=3)
    for p, lab in zip(pts, labels):
        ax.annotate(lab, p, textcoords="offset points", xytext=(8, 8))
    lim = 1.4 * np.abs(pts).max()
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$" if N > 1 else "")
    ax.set_title(f"{name}  (N={N})")
    ax.grid(alpha=0.3)


def main():
    names = ["BASK", "BFSK", "BPSK", "QPSK"]
    fig_c, axc = plt.subplots(1, 4, figsize=(16, 4))
    fig_b, axb = plt.subplots(4, 2, figsize=(11, 9))

    for i, name in enumerate(names):
        t, dt, sigs, labels = signal_set(name, Eb=1.0)
        seed = None
        if name == "QPSK":
            seed = [np.cos(2 * np.pi * FC * t), -np.sin(2 * np.pi * FC * t)]
        Phi, coords = gram_schmidt(sigs, dt, seed=seed)

        # console derivation table, copy into the report
        energies = [inner(s, s, dt) for s in sigs]
        bits_per_sym = 2 if name == "QPSK" else 1
        print(f"\n=== {name}: N = {Phi.shape[0]} basis function(s) ===")
        for lab, c, e in zip(labels, coords, energies):
            print(f"  symbol {lab:>2}: coords = {np.round(c, 3)}, energy = {e:.3f}")
        print(f"  average Eb = {np.mean(energies) / bits_per_sym:.3f} (should be 1.0)")
        G = Phi @ Phi.T * dt
        print(f"  orthonormality error = {np.abs(G - np.eye(len(G))).max():.1e}")

        plot_constellation(axc[i], name, coords, labels)
        for j in range(2):
            ax = axb[i, j]
            if j < Phi.shape[0]:
                ax.plot(t, Phi[j])
                ax.set_title(f"{name}: $\\phi_{j + 1}(t)$")
                ax.grid(alpha=0.3)
            else:
                ax.axis("off")

    fig_c.tight_layout()
    fig_b.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    fig_c.savefig(PLOTS / "constellations_coherent.png", dpi=200)
    fig_b.savefig(PLOTS / "gram_schmidt_basis.png", dpi=200)
    print("\nSaved plots/constellations_coherent.png and plots/gram_schmidt_basis.png")


if __name__ == "__main__":
    main()
