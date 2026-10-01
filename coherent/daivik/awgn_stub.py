"""
Local AWGN stub -- placeholder until Himanshu's shared channel module lands.

To swap in his version, change the import in coherent_schemes.py and delete
this file. Signature:
    awgn(signal, ebn0_db, bits_per_symbol=1, samples_per_bit=1, rng=None)
"""

import numpy as np


def awgn(signal, ebn0_db, bits_per_symbol=1, samples_per_bit=1, rng=None):
    """
    Add AWGN to `signal` at the given Eb/N0 (dB). Returns an array with the
    same shape as the input.

    Es = 1 normalization:
        Eb   = Es / bits_per_symbol
        N0   = Es / (bits_per_symbol * Eb/N0)
    Noise variance per dimension is N0/2, multiplied by samples_per_bit
    (a unit-amplitude waveform sampled at samples_per_bit samples per Tb
    carries N0/samples_per_bit per sample, so sampling by sps scales the
    per-sample variance by sps).

    Real input gets real noise only; complex input gets complex noise.
    """
    signal = np.asarray(signal)
    rng = np.random.default_rng() if rng is None else rng

    es = 1.0
    ebn0 = 10 ** (ebn0_db / 10.0)              # Eb/N0, linear
    n0 = es / (bits_per_symbol * ebn0)         # N0

    sigma = np.sqrt(n0 / 2.0 * samples_per_bit)

    if np.iscomplexobj(signal):
        noise = sigma * (rng.standard_normal(signal.shape)
                         + 1j * rng.standard_normal(signal.shape))
    else:
        noise = sigma * rng.standard_normal(signal.shape)
    return signal + noise