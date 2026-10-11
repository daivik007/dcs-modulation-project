"""
Coherent BASK (OOK) and coherent BFSK over AWGN (owner: Tanushree Paidi).

The receiver is a correlator (matched filter): project the received signal onto
the Gram-Schmidt basis functions, then choose the nearest constellation point
(minimum distance = maximum likelihood for AWGN). "Coherent" means the carrier
phase is known exactly, so unlike noncoherent_schemes.py there is no random theta.

Two levels of simulation:
  * coherent_trial()   signal-space level, noise from the shared channel/awgn.py.
                       This is the main simulation and matches Himanshu's convention.
  * waveform_trial()   full sampled waveforms with explicit correlators. Slower; used
                       only to prove that the signal-space shortcut is equivalent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from channel.awgn import add_awgn, db_to_lin, qfunc
from coherent.tanushree.gs_signal_space import SPS, gram_schmidt, signal_set

EB = 1.0  # average bit energy, same as noncoherent_schemes.py


def _build(name):
    t, dt, sigs, _ = signal_set(name, Eb=EB)
    Phi, coords = gram_schmidt(sigs, dt)
    return dt, np.array(sigs), Phi, coords


_SETS = {n: _build(n) for n in ("BASK", "BFSK")}


# --------------------------------------------------------------------------
# Theory (Eb is the average bit energy)
# --------------------------------------------------------------------------
def bask_coh_ber_theory(ebn0_db):
    """Coherent OOK: d^2 = 2 Eb, so Pb = Q(sqrt(Eb/N0))."""
    return qfunc(np.sqrt(db_to_lin(ebn0_db)))


def bfsk_coh_ber_theory(ebn0_db):
    """Coherent orthogonal FSK: d^2 = 2 Eb, so Pb = Q(sqrt(Eb/N0))."""
    return qfunc(np.sqrt(db_to_lin(ebn0_db)))


# --------------------------------------------------------------------------
# Signal-space simulation (uses the shared AWGN module)
# --------------------------------------------------------------------------
def coherent_tx(name, bits):
    """Bit -> Gram-Schmidt coordinates, shape (N, dim). BASK dim=1, BFSK dim=2."""
    coords = _SETS[name][3]
    return coords[bits]


def coherent_rx(name, y):
    """Minimum-distance decision on the correlator outputs r_j."""
    coords = _SETS[name][3]
    r = np.real(y)                                   # coherent: only the in-phase part matters
    d = ((r[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def coherent_trial(name, bits, ebn0_db, rng):
    x = coherent_tx(name, bits).astype(complex)      # complex only to match add_awgn's usage
    y = add_awgn(x, ebn0_db, bits_per_symbol=1, es=EB, rng=rng)
    return coherent_rx(name, y)


def bask_coh_trial(bits, ebn0_db, rng):
    return coherent_trial("BASK", bits, ebn0_db, rng)


def bfsk_coh_trial(bits, ebn0_db, rng):
    return coherent_trial("BFSK", bits, ebn0_db, rng)


# --------------------------------------------------------------------------
# Waveform-level correlator receiver (verification only)
# --------------------------------------------------------------------------
def waveform_trial(name, bits, ebn0_db, rng):
    """Transmit sampled waveforms, add white noise, correlate with phi_j(t)."""
    dt, wave, Phi, coords = _SETS[name]
    n0 = EB / db_to_lin(ebn0_db)
    tx = wave[bits].reshape(-1)                            # modulator
    rx = tx + np.sqrt(n0 / (2 * dt)) * rng.standard_normal(tx.size)   # PSD N0/2
    r = rx.reshape(bits.size, SPS)
    proj = (r @ Phi.T) * dt                                # r_j = integral r(t) phi_j(t) dt
    d = ((proj[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


# --------------------------------------------------------------------------
# Monte Carlo engine (same stopping rule as noncoherent_schemes.simulate_point)
# --------------------------------------------------------------------------
SCHEMES = {
    "BASK": (bask_coh_trial, bask_coh_ber_theory),
    "BFSK": (bfsk_coh_trial, bfsk_coh_ber_theory),
}


def simulate_point(trial, ebn0_db, rng, target_errors=200, batch=200_000, max_bits=20_000_000):
    errors = 0
    total = 0
    while errors < target_errors and total < max_bits:
        bits = rng.integers(0, 2, batch)
        errors += int(np.count_nonzero(bits != trial(bits, ebn0_db, rng)))
        total += batch
    return errors / total, errors, total


def simulate_curve(name, ebn0_db_list, rng, **kw):
    trial, _ = SCHEMES[name]
    rows = []
    for e in ebn0_db_list:
        ber, err, n = simulate_point(trial, e, rng, **kw)
        rows.append((e, ber, err, n))
    return rows
