"""
Non-coherent BASK, BFSK and DPSK over AWGN (owner: Himanshu Singh).

All schemes work in signal space with Eb = 1. Each carrier frequency has two
unit-energy basis functions (cosine and sine), so a received tone is a complex
number r = x * exp(j*theta) + n. The carrier phase theta is unknown to the
receiver and is never estimated. That is what "non-coherent" means here.

Noise comes from the shared module channel/awgn.py.
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import ncx2

from channel.awgn import add_awgn, db_to_lin, n0_from_ebn0, qfunc  # noqa: F401

EB = 1.0  # bit energy used by every scheme in this file


# --------------------------------------------------------------------------
# Non-coherent BASK (on-off keying, envelope detection)
# --------------------------------------------------------------------------
# bit 1 -> amplitude sqrt(2*Eb) (energy 2*Eb), bit 0 -> 0. Average Eb = Eb.
@lru_cache(maxsize=None)
def bask_nc_threshold(ebn0_db):
    """Envelope threshold that minimises the BER at this Eb/N0."""
    nu = np.sqrt(2 * EB)

    def log_ber(t):
        return np.log(bask_nc_ber_theory(ebn0_db, t) + 1e-300)

    res = minimize_scalar(log_ber, bounds=(1e-6, nu), method="bounded",
                          options={"xatol": 1e-9})
    return float(res.x)


def bask_nc_ber_theory(ebn0_db, threshold=None):
    """Exact BER of envelope-detected OOK.

    Pb = 0.5 * [ P(R > T | 0) + P(R < T | 1) ]
    R | 0 is Rayleigh : P(R > T) = exp(-T^2 / N0)
    R | 1 is Rician   : P(R < T) = noncentral chi-square CDF, 2 dof
    """
    ebn0_db = np.atleast_1d(np.asarray(ebn0_db, dtype=float))
    out = np.empty_like(ebn0_db)
    for i, e in enumerate(ebn0_db):
        n0 = n0_from_ebn0(e, EB)
        sigma2 = n0 / 2.0
        nu = np.sqrt(2 * EB)
        t = bask_nc_threshold(float(e)) if threshold is None else threshold
        p_fa = np.exp(-t ** 2 / n0)
        p_miss = ncx2.cdf(t ** 2 / sigma2, 2, nu ** 2 / sigma2)
        out[i] = 0.5 * (p_fa + p_miss)
    return out if out.size > 1 else out[0]


def bask_nc_ber_highsnr(ebn0_db):
    """High-SNR approximation: Pb ~ 0.5 * exp(-Eb / (2 N0))."""
    return 0.5 * np.exp(-db_to_lin(ebn0_db) / 2.0)


def bask_nc_tx(bits):
    """Bit -> amplitude on the cosine basis function, shape (N,)."""
    return np.sqrt(2 * EB) * bits.astype(float)


def bask_nc_rx(y, ebn0_db):
    """Envelope detector: |y| compared with the optimal threshold."""
    return (np.abs(y) > bask_nc_threshold(float(ebn0_db))).astype(int)


def bask_nc_trial(bits, ebn0_db, rng):
    x = bask_nc_tx(bits)
    theta = rng.uniform(0, 2 * np.pi, bits.size)       # unknown carrier phase
    x = x * np.exp(1j * theta)
    y = add_awgn(x, ebn0_db, bits_per_symbol=1, es=EB, rng=rng)
    return bask_nc_rx(y, ebn0_db)


# --------------------------------------------------------------------------
# Non-coherent BFSK (two orthogonal tones, envelope / energy comparison)
# --------------------------------------------------------------------------
def bfsk_nc_ber_theory(ebn0_db):
    """Pb = 0.5 * exp(-Eb / (2 N0))  (exact for orthogonal tones)."""
    return 0.5 * np.exp(-db_to_lin(ebn0_db) / 2.0)


def bfsk_nc_tx(bits):
    """Bit -> (N, 2) complex array, one column per tone. Energy Eb on the sent tone."""
    x = np.zeros((bits.size, 2), dtype=complex)
    x[np.arange(bits.size), bits] = np.sqrt(EB)
    return x


def bfsk_nc_rx(y):
    """Pick the tone with the larger envelope (equivalent to larger energy)."""
    return (np.abs(y[:, 1]) ** 2 > np.abs(y[:, 0]) ** 2).astype(int)


def bfsk_nc_trial(bits, ebn0_db, rng):
    x = bfsk_nc_tx(bits)
    theta = rng.uniform(0, 2 * np.pi, (bits.size, 1))
    x = x * np.exp(1j * theta)
    y = add_awgn(x, ebn0_db, bits_per_symbol=1, es=EB, rng=rng)
    return bfsk_nc_rx(y)


# --------------------------------------------------------------------------
# DPSK (differential detection, no carrier recovery)
# --------------------------------------------------------------------------
def dpsk_ber_theory(ebn0_db):
    """Pb = 0.5 * exp(-Eb / N0)."""
    return 0.5 * np.exp(-db_to_lin(ebn0_db))


def dpsk_tx(bits):
    """Differential encoding d[k] = d[k-1] XOR b[k], then BPSK mapping.
    A reference symbol (d[0] = 0) is sent first, so N bits give N+1 symbols."""
    d = np.zeros(bits.size + 1, dtype=int)
    d[1:] = np.bitwise_xor.accumulate(bits)
    return np.sqrt(EB) * (1 - 2 * d).astype(float)


def dpsk_rx(y):
    """z[k] = y[k] * conj(y[k-1]). Re(z) > 0 means no phase change, so bit 0."""
    z = y[1:] * np.conj(y[:-1])
    return (z.real < 0).astype(int)


def dpsk_trial(bits, ebn0_db, rng):
    x = dpsk_tx(bits).astype(complex)
    theta = rng.uniform(0, 2 * np.pi)                   # one unknown phase for the block
    x = x * np.exp(1j * theta)
    y = add_awgn(x, ebn0_db, bits_per_symbol=1, es=EB, rng=rng)
    return dpsk_rx(y)


# --------------------------------------------------------------------------
# Monte Carlo engine
# --------------------------------------------------------------------------
SCHEMES = {
    "BASK": (bask_nc_trial, bask_nc_ber_theory),
    "BFSK": (bfsk_nc_trial, bfsk_nc_ber_theory),
    "DPSK": (dpsk_trial, dpsk_ber_theory),
}


def simulate_point(trial, ebn0_db, rng, target_errors=200, batch=200_000, max_bits=20_000_000):
    """Run batches until target_errors is reached or max_bits is used up."""
    errors = 0
    total = 0
    while errors < target_errors and total < max_bits:
        bits = rng.integers(0, 2, batch)
        bits_hat = trial(bits, ebn0_db, rng)
        errors += int(np.count_nonzero(bits != bits_hat))
        total += batch
    return errors / total, errors, total


def simulate_curve(name, ebn0_db_list, rng, **kw):
    trial, _ = SCHEMES[name]
    rows = []
    for e in ebn0_db_list:
        ber, err, n = simulate_point(trial, e, rng, **kw)
        rows.append((e, ber, err, n))
    return rows
