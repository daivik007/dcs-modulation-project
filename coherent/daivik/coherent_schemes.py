"""
Coherent detection over AWGN: BPSK / QPSK / 8-PSK / 16-QAM / MSK

Daivik's slice: Gray-coded mappers, correlator (matched-filter) receivers,
theoretical BER, Monte Carlo BER comparison, bandwidth efficiency.

Run `python coherent_schemes.py` to regenerate every plot into ../../plots/.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

from awgn_stub import awgn  # swap for Himanshu's shared module here

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "plots")

# ---------------------------------------------------------------
# 1. Bit-to-symbol mapping (Gray coding) -> constellation points
# ---------------------------------------------------------------

def bpsk_map(bits):
    # 0 -> -1, 1 -> +1
    return 2 * np.asarray(bits) - 1 + 0j

def qpsk_map(bits):
    # bits grouped in pairs, Gray-coded, normalized to unit energy
    bits = np.asarray(bits).reshape(-1, 2)
    i = 2 * bits[:, 0] - 1
    q = 2 * bits[:, 1] - 1
    return (i + 1j * q) / np.sqrt(2)

def psk8_map(bits):
    # bits grouped in triples, Gray-coded around the circle, unit energy.
    # Circular Gray labelling: the label placed at angle index i is
    # i ^ (i>>1), so bits -> angle is the inverse, i = v^(v>>1)^(v>>2).
    # Plain v ^ (v>>1) is NOT circular Gray (two adjacent pairs then
    # differ in 2 bits), which costs real BER.
    bits = np.asarray(bits).reshape(-1, 3)
    v = (bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]).astype(int)
    idx = v ^ (v >> 1) ^ (v >> 2)             # bits -> Gray angle index
    return np.exp(1j * idx * np.pi / 4)

def qam16_map(bits):
    # bits grouped in 4s: 2 for I, 2 for Q, Gray-coded amplitude levels
    # lookup keyed on the 2-bit index 0..3 instead of a per-row dict
    level = np.array([-3, -1, 3, 1])        # index 2b0+b1 -> level
    bits = np.asarray(bits).reshape(-1, 4)
    i = level[2 * bits[:, 0] + bits[:, 1]]
    q = level[2 * bits[:, 2] + bits[:, 3]]
    norm = np.sqrt(10)                       # average energy normalization
    return (i + 1j * q) / norm

def msk_map(bits, Tb=1.0, fs=100):
    """
    MSK as continuous-phase FSK, h=0.5. Returns the baseband complex
    envelope samples (not just symbol points -- MSK doesn't have a
    fixed constellation like the others, it's a phase trajectory).
    """
    bits = 2 * np.asarray(bits) - 1          # +/-1
    n = len(bits)
    t = np.arange(0, n * Tb, 1 / fs)
    # phase at the start of each bit interval, unwrapped (continuous phase)
    th_k = np.concatenate([[0.0], np.cumsum(bits * np.pi / 2)])
    k = np.minimum((t / Tb).astype(int), n - 1)
    phase = th_k[k] + bits[k] * (np.pi / (2 * Tb)) * (t - k * Tb)
    return np.exp(1j * phase), t, th_k

def msk_q_basis(Tb=1.0, sps=32):
    """
    Quadrature correlator basis for coherent MSK detection, sampled at
    sps samples/Tb over a 2Tb window and normalized to unit energy.

    Basis: cos(pi*t/2Tb) (I) and sin(pi*t/2Tb) (Q) over the 2Tb interval.
    """
    t = np.arange(2 * sps) * (Tb / sps)
    bI = np.cos(np.pi * t / (2 * Tb))
    bQ = np.sin(np.pi * t / (2 * Tb))
    nrm = np.sqrt(np.sum(bQ ** 2) * (Tb / sps))
    return bI / nrm, bQ / nrm, t

def msk_windows(a, sps=32):
    """
    Build n overlapping 2Tb MSK waveform windows at once (vectorised).

    Window j starts at bit j with the continuous phase accumulated from all
    preceding bits. Returns (windows, phase_at_window_start).
    """
    a = np.asarray(a, float)
    n_win = len(a)
    th_k = np.concatenate([[0.0], np.cumsum(a * np.pi / 2)])
    theta_start = th_k[:n_win]                       # phase at each window start
    w = np.arange(n_win)[:, None]
    p = np.arange(2 * sps)[None, :]
    second = p >= sps
    bit = np.minimum(w + second, n_win - 1)
    local = (p - second * sps) * (1.0 / sps)
    phase = th_k[bit] + a[bit] * (np.pi / 2) * local
    return np.exp(1j * phase), theta_start


# ---------------------------------------------------------------
# 2. Demappers (hard decision) -> bits
# ---------------------------------------------------------------

def bpsk_demap(sym):
    return (np.real(sym) > 0).astype(int).reshape(-1)

def qpsk_demap(sym):
    b = np.empty((len(sym), 2), int)
    b[:, 0] = np.real(sym) > 0
    b[:, 1] = np.imag(sym) > 0
    return b.reshape(-1)

def _gray_to_bits(idx):
    """Gray angle index -> MSB-first bits (label at angle i is i^(i>>1))."""
    g = np.asarray(idx, int)
    v = g ^ (g >> 1)
    return np.column_stack([(v & 4) > 0, (v & 2) > 0, (v & 1) > 0])

def psk8_demap(sym):
    # nearest constellation point in angle, then Gray -> binary
    gray = np.mod(np.round(np.angle(sym) / (np.pi / 4)).astype(int), 8)
    return _gray_to_bits(gray).reshape(-1)

def qam16_demap(sym):
    norm = np.sqrt(10)
    levels = np.array([-3, -1, 1, 3])
    # nearest of {-3,-1,1,3} per dimension, then level -> Gray bits
    li = np.abs(np.real(sym)[:, None] * norm - levels).argmin(axis=1)
    lq = np.abs(np.imag(sym)[:, None] * norm - levels).argmin(axis=1)
    gray2bits = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])  # levels -3,-1,+1,+3
    return np.column_stack([gray2bits[li], gray2bits[lq]]).reshape(-1)


# ---------------------------------------------------------------
# 3. Correlator (matched-filter) receivers, symbol rate, ideal sync
# ---------------------------------------------------------------

# For the linear/PSK schemes the symbol interval is Tb, the matched filter
# is the unit-energy constellation template, so the correlator output is
# just the received symbol. We keep the explicit correlator step so all
# schemes share one receiver interface.

def _mf_scores(rx, constell):
    """
    Correlator bank over the constellation: for each candidate template c
    the matched-filter output is <rx, c> = rx * conj(c) at symbol rate
    (the template is a unit-energy pulse over one symbol). Decision is the
    real part, which is the projection onto c.
    """
    return np.real(rx[:, None] * np.conj(constell)[None, :])

def detect_bpsk(rx):
    return bpsk_demap(rx)

def detect_qpsk(rx):
    return qpsk_demap(rx)

def detect_psk8(rx):
    constell = np.exp(1j * np.arange(8) * np.pi / 4)
    idx = _mf_scores(rx, constell).argmax(axis=1)
    return _gray_to_bits(idx).reshape(-1)

def detect_qam16(rx):
    return qam16_demap(rx)


def msk_correlator_rx(rx_wave, theta_start, Tb=1.0, sps=32):
    """
    Coherent MSK detection with I/Q correlators over 2Tb intervals.

    Approach: two correlators per 2Tb window, basis cos(pi*t/2Tb) (I) and
    sin(pi*t/2Tb) (Q) normalized to unit energy. With ideal carrier sync
    (each window derotated by its known start phase) the Q correlator's
    imaginary part equals a_k exactly (unit amplitude), so bit k is
    recovered with BPSK SNR from window k. Bit k+1 is a weak statistic in
    the same window, so we use overlapping windows: window k+1 supplies
    bit k+1 at the same full SNR. One extra bit is appended so the last
    window exists.

    Note the max-|corr| ML bank over the four 2-bit patterns is NOT usable
    here: those four waveforms are not orthogonal (pairwise correlation
    ~0.5-0.64 and two pairs are exact conjugates), so the bank loses ~3 dB
    and does not achieve the BPSK BER. The I/Q correlator above does.
    """
    bI, bQ, _ = msk_q_basis(Tb, sps)
    seg = rx_wave.reshape(-1, 2 * sps) * np.exp(-1j * np.asarray(theta_start))[:, None]
    I = np.einsum("ij,j->i", seg, bI)
    Q = np.einsum("ij,j->i", seg, bQ)
    return (Q.imag > 0).astype(int)


# ---------------------------------------------------------------
# 4. Theoretical BER formulas (function of Eb/N0 in dB)
# ---------------------------------------------------------------

def q_func(x):
    return 0.5 * erfc(x / np.sqrt(2))

def ber_bpsk_theory(ebn0_db):
    ebn0 = 10 ** (ebn0_db / 10)
    return q_func(np.sqrt(2 * ebn0))

def ber_qpsk_theory(ebn0_db):
    # same as BPSK per bit for Gray-coded QPSK
    return ber_bpsk_theory(ebn0_db)

def ber_msk_theory(ebn0_db):
    # coherent MSK has same theoretical BER as BPSK
    return ber_bpsk_theory(ebn0_db)

def ber_psk8_theory(ebn0_db):
    # High-SNR approximation for Gray-coded M-PSK: Pb ~ (2/k) Q(sqrt(2k Eb/N0)
    # sin(pi/M)), with k = log2(M) = 3. Valid away from the low-SNR region
    # (it overestimates the true Pb below roughly 0 dB).
    k = 3
    ebn0 = 10 ** (ebn0_db / 10)
    return (2 / k) * q_func(np.sqrt(2 * k * ebn0) * np.sin(np.pi / 8))

def ber_16qam_theory(ebn0_db):
    M = 16
    k = np.log2(M)
    ebn0 = 10 ** (ebn0_db / 10)
    esn0 = k * ebn0
    term = (1 - 1 / np.sqrt(M)) * q_func(np.sqrt(3 * esn0 / (M - 1)))
    return (4 / k) * term


# ---------------------------------------------------------------
# 5. Monte Carlo BER
# ---------------------------------------------------------------

SCHEMES = {
    "BPSK":   dict(k=1, map=bpsk_map,  det=detect_bpsk,  th=ber_bpsk_theory),
    "QPSK":   dict(k=2, map=qpsk_map,  det=detect_qpsk,  th=ber_qpsk_theory),
    "8-PSK":  dict(k=3, map=psk8_map,  det=detect_psk8,  th=ber_psk8_theory),
    "16-QAM": dict(k=4, map=qam16_map, det=detect_qam16, th=ber_16qam_theory),
    "MSK":    dict(k=1, map=None,      det=None,         th=ber_msk_theory),
}

def mc_ber(name, ebn0_db, rng, min_errors=100, max_bits=20_000_000):
    """
    Adaptive: keep sending blocks until >= min_errors or max_bits.
    Returns (ber, errors, bits_simulated).
    """
    s = SCHEMES[name]
    k = s["k"]
    errors = 0
    total = 0
    if name == "MSK":
        Tb, sps = 1.0, 32
        fs = sps / Tb                             # samples per unit time
        while errors < min_errors and total < max_bits:
            n_bit = max(1, min(1 << 14, max_bits - total))
            # one extra bit so every bit has its own 2Tb window
            bits = rng.integers(0, 2, n_bit + 1)
            wave, theta = msk_windows(2 * bits - 1, sps)
            # waveform is sampled, so pass the sample rate per bit
            rx = awgn(wave, ebn0_db, bits_per_symbol=k,
                      samples_per_bit=fs * Tb, rng=rng)
            hard = msk_correlator_rx(rx, theta, Tb, sps)[:n_bit]
            errors += int(np.count_nonzero(hard != bits[:n_bit]))
            total += n_bit
        return errors / max(total, 1), errors, total

    while errors < min_errors and total < max_bits:
        # round down to a whole number of symbols, and never to zero
        n_bit = max(k, min(1 << 19, max_bits - total))
        n_bit -= n_bit % k
        if n_bit == 0:
            break
        bits = rng.integers(0, 2, n_bit)
        rx = awgn(s["map"](bits), ebn0_db, k, rng=rng)
        hard = s["det"](rx)
        errors += int(np.count_nonzero(hard != bits))
        total += n_bit
    return errors / max(total, 1), errors, total




def run_ber_sweep(seed=1234, min_errors=100, max_bits=20_000_000):
    ebn0_range = np.arange(0, 15, 1.0)
    out = {}
    for name in SCHEMES:
        rng = np.random.default_rng(seed)
        bers, errs, bits = [], [], []
        for e in ebn0_range:
            b, ne, nt = mc_ber(name, e, rng, min_errors, max_bits)
            bers.append(b)
            errs.append(ne)
            bits.append(nt)
        out[name] = dict(ber=np.array(bers), errors=errs, bits=bits)
        print(f"  {name} done")
    return ebn0_range, out


# ---------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------

def plot_constellation(symbols, title):
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.scatter(symbols.real, symbols.imag, s=18, alpha=0.7)
    plt.axhline(0, color="gray", lw=0.5)
    plt.axvline(0, color="gray", lw=0.5)
    plt.title(title)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, title.replace(" ", "_") + ".png"), dpi=150)
    plt.close()

def plot_msk_phase(bits, title="MSK Phase Trajectory"):
    os.makedirs(PLOT_DIR, exist_ok=True)
    Tb, fs = 1.0, 200
    _, t, _ = msk_map(bits, Tb, fs)
    a = 2 * np.asarray(bits) - 1
    th_k = np.concatenate([[0.0], np.cumsum(a * np.pi / 2)])
    k = np.minimum((t / Tb).astype(int), len(a) - 1)
    phase = th_k[k] + a[k] * (np.pi / (2 * Tb)) * (t - k * Tb)

    plt.figure(figsize=(8, 3.5))
    plt.plot(t / Tb, phase, lw=2)
    for j in range(1, len(bits)):
        plt.axvline(j, color="gray", lw=0.5, ls=":")
    # use half-pi ticks with one decimal (a 0.5 grid needs 0.1 not 0.01)
    step = 0.5
    lo = step * np.floor(phase.min() / (np.pi * step))
    hi = step * np.ceil(phase.max() / (np.pi * step))
    ticks = np.arange(lo, hi + 0.5 * step, step)
    plt.yticks(ticks * np.pi, [f"{v:+.1f}pi" for v in ticks])
    plt.xticks(range(len(bits) + 1))
    plt.xlabel("t / Tb")
    plt.ylabel("phase")
    plt.title(f"{title}\nbits: {''.join(map(str, bits))}  (each bit = $\\pm\\pi/2$)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, title.replace(" ", "_") + ".png"), dpi=150)
    plt.close()

PLOT_FLOOR = 1e-6      # theoretical BER below this is not plotted

def _plot_mask(th):
    # drop points where theory is below the plot floor: the sim cannot
    # resolve them within the bit cap and would show misleading zeros
    return np.asarray(th) >= PLOT_FLOOR

def plot_ber_per_scheme(name, ebn0_range, data):
    os.makedirs(PLOT_DIR, exist_ok=True)
    th = SCHEMES[name]["th"](ebn0_range)
    m = _plot_mask(th)
    plt.figure(figsize=(6, 4.5))
    plt.semilogy(np.asarray(ebn0_range)[m], np.asarray(th)[m],
                 label="theoretical", lw=1.5)
    plt.semilogy(np.asarray(ebn0_range)[m],
                 np.maximum(np.asarray(data["ber"])[m], 1e-7),
                 "o", ms=4, label="simulated")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title(f"{name}: simulated vs theoretical")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.ylim(PLOT_FLOOR / 3, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"ber_{name.replace(' ', '_')}.png"), dpi=150)
    plt.close()

def plot_ber_combined(ebn0_range, out):
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(7, 5))
    ebn0_range = np.asarray(ebn0_range)
    for name in SCHEMES:
        th = SCHEMES[name]["th"](ebn0_range)
        m = _plot_mask(th)
        plt.semilogy(ebn0_range[m], np.asarray(th)[m], lw=1.5,
                     label=f"{name} theory")
        plt.semilogy(ebn0_range[m], np.maximum(np.asarray(out[name]["ber"])[m], 1e-7),
                     "o", ms=4, label=f"{name} sim")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title("Coherent BER over AWGN")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(fontsize=7, ncol=2)
    plt.ylim(PLOT_FLOOR / 3, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "ber_combined.png"), dpi=150)
    plt.close()

def plot_ber_vs_order(ebn0_db=8.0):
    os.makedirs(PLOT_DIR, exist_ok=True)
    order = ["BPSK", "QPSK", "8-PSK", "16-QAM"]
    th = [SCHEMES[n]["th"](ebn0_db) for n in order]
    plt.figure(figsize=(6, 4.5))
    plt.semilogy(range(len(order)), th, "o-", lw=1.5)
    plt.xticks(range(len(order)), order)
    plt.xlabel("modulation order")
    plt.ylabel("BER")
    plt.title(f"Theoretical BER vs modulation order at Eb/N0 = {ebn0_db} dB")
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "ber_vs_modulation_order.png"), dpi=150)
    plt.close()

def plot_bandwidth_efficiency(table):
    os.makedirs(PLOT_DIR, exist_ok=True)
    names = list(table.keys())
    vals = [table[n]["efficiency"] for n in names]
    plt.figure(figsize=(6, 4.5))
    plt.bar(names, vals, color="steelblue")
    for i, v in enumerate(vals):
        plt.text(i, v, f" {v:.2f}", ha="center", va="bottom")
    plt.ylabel("bandwidth efficiency (bits/s/Hz)")
    plt.title("Bandwidth efficiency (null-to-null bandwidth)")
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "bandwidth_efficiency.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------
# 7. Bandwidth efficiency (bits/s/Hz)
# ---------------------------------------------------------------

def bandwidth_efficiency():
    """
    Bandwidth efficiency eta = Rb / B, in bits/s/Hz, using null-to-null
    (main-lobe-to-main-lobe) bandwidth B:
      - linear PSK/QAM with rectangular symbol pulses: B = 2*Rs,
        so eta = Rb/(2*Rs) = k/2
      - MSK: continuous-phase FSK occupies B = 1.5*Rb null-to-null
        (vs 2*Rb for BFSK), so eta = 1/1.5
    """
    schemes = {
        "BPSK":   dict(k=1, b_over_rb=2.0),
        "QPSK":   dict(k=2, b_over_rb=2.0),
        "MSK":    dict(k=1, b_over_rb=1.5),
        "8-PSK":  dict(k=3, b_over_rb=2.0),
        "16-QAM": dict(k=4, b_over_rb=2.0),
    }
    table = {}
    for name, d in schemes.items():
        eta = d["k"] / d["b_over_rb"]
        table[name] = dict(bits_per_symbol=d["k"],
                           bandwidth_over_rb=d["b_over_rb"],
                           efficiency=eta)
    return table

def save_bandwidth_table(table, path):
    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("scheme,bits_per_symbol,bandwidth_over_Rb,bandwidth_efficiency_bits_per_Hz\n")
        for name, d in table.items():
            f.write(f"{name},{d['bits_per_symbol']},{d['bandwidth_over_rb']},"
                    f"{d['efficiency']:.4f}\n")


# ---------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------

def verify_msk_phase():
    bits = np.array([1, 1, 0, 1, 0, 0])
    _, _, th_k = msk_map(bits, fs=10)
    expected = np.array([0.5, 1.0, 0.5, 1.0, 0.5, 0.0]) * np.pi
    ok = np.allclose(th_k[1:], expected)
    print(f"MSK end-of-bit phases (unwrapped, /pi): {np.round(th_k[1:]/np.pi, 6)}")
    print(f"  expected: {np.round(expected/np.pi, 6)}  -> {'OK' if ok else 'MISMATCH'}")
    assert ok
    return ok

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("Verifying MSK phase continuity...")
    verify_msk_phase()

    rng = np.random.default_rng(42)

    print("Signal-space diagrams...")
    bits = rng.integers(0, 2, 200)
    plot_constellation(bpsk_map(bits[:200]), "BPSK Constellation")
    b2 = rng.integers(0, 2, 400)
    plot_constellation(qpsk_map(b2), "QPSK Constellation")
    plot_constellation(psk8_map(rng.integers(0, 2, 600)), "8PSK Constellation")
    plot_constellation(qam16_map(rng.integers(0, 2, 800)), "16QAM Constellation")
    plot_msk_phase([1, 1, 0, 1, 0, 1, 1, 0, 0, 1])

    print("Monte Carlo BER sweep (this takes a while)...")
    ebn0_range, out = run_ber_sweep()

    print("BER plots...")
    for name in SCHEMES:
        plot_ber_per_scheme(name, ebn0_range, out[name])
    plot_ber_combined(ebn0_range, out)
    plot_ber_vs_order(8.0)

    table = bandwidth_efficiency()
    plot_bandwidth_efficiency(table)
    print("\nBandwidth efficiency (bits/s/Hz, null-to-null bandwidth):")
    print(f"  {'scheme':<8} {'bits/sym':>9} {'B/Rb':>6} {'eta':>7}")
    for name, d in table.items():
        print(f"  {name:<8} {d['bits_per_symbol']:>9} {d['bandwidth_over_rb']:>6} "
              f"{d['efficiency']:>7.3f}")
    save_bandwidth_table(table, os.path.join(PLOT_DIR, "bandwidth_efficiency.csv"))

    # A point is only usable if the sim accumulated enough errors for the
    # BER estimate to be meaningful; below that the run just hit the bit
    # cap, and sim/theory is dominated by counting noise.
    min_err = 100
    print(f"\nSimulated vs theoretical BER (points with >= {min_err} errors):")
    for name in SCHEMES:
        th = SCHEMES[name]["th"](ebn0_range)
        ber = np.asarray(out[name]["ber"])
        errs = np.array(out[name]["errors"])
        ok = errs >= min_err
        excluded = [(int(e), int(n)) for e, n in zip(ebn0_range, errs) if n < min_err]
        rel = np.abs(ber[ok] - th[ok]) / np.maximum(th[ok], 1e-12)
        print(f"  {name:<8} max rel dev = {rel.max():.3f}   "
              f"({int(ok.sum())} pts used)")
        print(f"           excluded (Eb/N0 dB, errors): "
              + (", ".join(f"({a}, {b})" for a, b in excluded) if excluded else "none"))

if __name__ == "__main__":
    main()