"""
Coherent BPSK / QPSK / MSK / 16-QAM
Daivik's slice: signal-space diagrams, transmitter, theoretical BER,
bandwidth efficiency. NO noise/channel/receiver here yet -- that plugs
in once Himanshu's shared AWGN module is ready (see bottom TODO).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. Bit-to-symbol mapping (Gray coding) -> constellation points
# ---------------------------------------------------------------

def bpsk_map(bits):
    # 0 -> -1, 1 -> +1
    return 2 * bits - 1 + 0j

def qpsk_map(bits):
    # bits grouped in pairs, Gray-coded, normalized to unit energy
    bits = bits.reshape(-1, 2)
    i = 2 * bits[:, 0] - 1
    q = 2 * bits[:, 1] - 1
    return (i + 1j * q) / np.sqrt(2)

def qam16_map(bits):
    # bits grouped in 4s: 2 for I, 2 for Q, Gray-coded amplitude levels
    gray2level = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
    bits = bits.reshape(-1, 4)
    i = np.array([gray2level[(b[0], b[1])] for b in bits])
    q = np.array([gray2level[(b[2], b[3])] for b in bits])
    norm = np.sqrt(10)  # average energy normalization for 16-QAM
    return (i + 1j * q) / norm

def msk_map(bits, Tb=1.0, fs=100):
    """
    MSK as continuous-phase FSK, h=0.5. Returns the baseband complex
    envelope samples (not just symbol points -- MSK doesn't have a
    fixed constellation like the others, it's a phase trajectory).
    """
    bits = 2 * bits - 1  # +/-1
    t = np.arange(0, len(bits) * Tb, 1 / fs)
    phase = np.zeros_like(t)
    for k, b in enumerate(bits):
        idx = (t >= k * Tb) & (t < (k + 1) * Tb)
        phase[idx] = phase[idx - 1][0] if k > 0 else 0
        phase[idx] += b * (np.pi / (2 * Tb)) * (t[idx] - k * Tb)
    envelope = np.exp(1j * phase)
    return envelope, t


# ---------------------------------------------------------------
# 2. Signal-space / constellation plots
# ---------------------------------------------------------------

def plot_constellation(symbols, title):
    plt.figure(figsize=(4, 4))
    plt.scatter(symbols.real, symbols.imag)
    plt.axhline(0, color="gray", lw=0.5)
    plt.axvline(0, color="gray", lw=0.5)
    plt.title(title)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------
# 3. Theoretical BER formulas (function of Eb/N0 in dB)
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

def ber_16qam_theory(ebn0_db):
    M = 16
    k = np.log2(M)
    ebn0 = 10 ** (ebn0_db / 10)
    esn0 = k * ebn0
    # standard approx for rectangular M-QAM, Gray-coded
    term = (1 - 1 / np.sqrt(M)) * q_func(np.sqrt(3 * esn0 / (M - 1)))
    return (4 / k) * term


# ---------------------------------------------------------------
# 4. Bandwidth efficiency (bits/sec/Hz)
# ---------------------------------------------------------------

def bandwidth_efficiency():
    # bits/symbol / (bandwidth in symbols/sec terms), using Nyquist
    # minimum bandwidth Rs for each scheme -- fill in your report's
    # exact bandwidth definition (e.g. null-to-null vs Nyquist) here
    schemes = {
        "BPSK": 1,          # 1 bit/symbol, B = Rs
        "QPSK": 2,          # 2 bits/symbol, B = Rs
        "MSK": 1,           # 1 bit/symbol, B = 1.5*Rs (approx, tighter than BFSK)
        "16-QAM": 4,        # 4 bits/symbol, B = Rs
    }
    return schemes


# ---------------------------------------------------------------
# 5. Quick sanity check / demo
# ---------------------------------------------------------------

if __name__ == "__main__":
    n_bits = 10000

    bits = rng.integers(0, 2, n_bits)
    plot_constellation(bpsk_map(bits[:200]), "BPSK Constellation")

    bits = rng.integers(0, 2, n_bits - n_bits % 2)
    plot_constellation(qpsk_map(bits[:200]), "QPSK Constellation")

    bits = rng.integers(0, 2, n_bits - n_bits % 4)
    plot_constellation(qam16_map(bits[:400]), "16QAM Constellation")

    ebn0_range = np.arange(0, 14, 1)
    plt.figure()
    plt.semilogy(ebn0_range, ber_bpsk_theory(ebn0_range), label="BPSK")
    plt.semilogy(ebn0_range, ber_qpsk_theory(ebn0_range), label="QPSK")
    plt.semilogy(ebn0_range, ber_msk_theory(ebn0_range), label="MSK")
    plt.semilogy(ebn0_range, ber_16qam_theory(ebn0_range), label="16-QAM")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER (theoretical)")
    plt.legend()
    plt.grid(True, which="both")
    plt.title("Theoretical BER curves")
    plt.savefig("theoretical_BER_curves.png", dpi=150)
    plt.close()

    print("Bandwidth efficiency (bits/sec/Hz, ideal Nyquist):")
    for scheme, bits_per_sym in bandwidth_efficiency().items():
        print(f"  {scheme}: {bits_per_sym} bits/symbol")

# ---------------------------------------------------------------
# TODO once Himanshu's AWGN channel module is ready:
#   - import his awgn_channel(symbols, ebn0_db) function
#   - pass bpsk_map()/qpsk_map()/qam16_map()/msk_map() output through it
#   - implement correlator/matched-filter receivers to recover bits
#   - compute simulated BER, overlay on the theoretical curves above
# ---------------------------------------------------------------
