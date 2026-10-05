"""
Shared AWGN channel for the DCS project (owner: Himanshu Singh).

Everyone imports from here so that every BER curve uses the same noise
convention.

Convention
----------
* Signals are handled in signal space, i.e. as coordinates on the
  orthonormal basis functions from Gram-Schmidt (each basis function has
  unit energy).
* A complex coordinate x = a + jb means a on the cosine basis function and
  b on the sine basis function, so its energy is |x|^2 = a^2 + b^2.
* Noise has two-sided PSD N0/2, so every real coordinate gets noise with
  variance N0/2. A complex coordinate gets N0/2 on the real part and N0/2 on
  the imaginary part (total variance N0).
* Eb/N0 is set through the average symbol energy Es:
      Eb = Es / bits_per_symbol,   N0 = Eb / 10^(EbN0_dB / 10)

Typical use (from the repo root)
--------------------------------
    from channel.awgn import add_awgn, qfunc

    # BPSK, Eb = 1
    bits = rng.integers(0, 2, N)
    x = (1 - 2 * bits).astype(float)
    y = add_awgn(x, ebn0_db=6, bits_per_symbol=1, es=1.0, rng=rng)
    bits_hat = (y < 0).astype(int)

    # QPSK with unit-energy symbols (Es = 1, 2 bits per symbol), complex form
    y = add_awgn(sym, ebn0_db=6, bits_per_symbol=2, es=1.0, rng=rng)
"""
import numpy as np
from scipy.special import erfc


def qfunc(x):
    """Gaussian tail probability Q(x) = 0.5 * erfc(x / sqrt(2))."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def get_rng(seed=None):
    """Seeded NumPy Generator. Use one seed per script so results repeat."""
    return np.random.default_rng(seed)


def db_to_lin(db):
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)


def n0_from_ebn0(ebn0_db, eb=1.0):
    """Noise PSD N0 for a given Eb/N0 (dB) and bit energy Eb."""
    return eb / db_to_lin(ebn0_db)


def average_symbol_energy(x):
    """Measured average symbol energy of an array of signal-space points.

    x of shape (N,)   : one real or complex coordinate per symbol.
    x of shape (N, D) : D coordinates per symbol (energy summed over D).
    """
    x = np.asarray(x)
    energy = np.abs(x) ** 2
    if energy.ndim > 1:
        energy = energy.reshape(energy.shape[0], -1).sum(axis=1)
    return float(np.mean(energy))


def add_awgn(x, ebn0_db, bits_per_symbol=1, es=None, rng=None):
    """Add white Gaussian noise to signal-space points at a given Eb/N0.

    Parameters
    ----------
    x : ndarray, real or complex, shape (N,) or (N, D)
    ebn0_db : float, Eb/N0 in dB
    bits_per_symbol : int, k = log2(M)
    es : float or None. Average symbol energy. Pass it when you know it
         (exact N0). If None it is measured from x, which has small random
         error for short sequences.
    rng : numpy Generator (see get_rng). A fresh unseeded one is used if None.

    Returns
    -------
    y : ndarray, same shape and type as x, with noise added.
    """
    rng = get_rng() if rng is None else rng
    x = np.asarray(x)
    if es is None:
        es = average_symbol_energy(x)
    eb = es / bits_per_symbol
    n0 = n0_from_ebn0(ebn0_db, eb)
    sigma = np.sqrt(n0 / 2.0)  # std dev per real dimension

    if np.iscomplexobj(x):
        noise = sigma * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    else:
        noise = sigma * rng.standard_normal(x.shape)
    return x + noise
