# Member 3: Non-Coherent Schemes, Channel Model and Final Comparison

Himanshu Singh (24BEC0304). BECE306P Digital Communication System Lab, Review 1 project.

## 1. Theory

### 1.1 Why non-coherent detection exists

A coherent receiver needs a local carrier that matches the received carrier in frequency and phase. It gets it from a phase-locked loop or a Costas loop. That adds hardware, needs time to lock at the start of every packet, and fails when phase noise or fast fading moves the carrier phase faster than the loop can follow.

A non-coherent receiver treats the carrier phase theta as an unknown constant and uses a statistic that does not depend on it. Each carrier frequency is correlated with two basis functions, a cosine and a sine, both with unit energy:

- r_c = x cos(theta) + n_c
- r_s = x sin(theta) + n_s

Written as a complex number, r = x e^(j theta) + n, where n_c and n_s are independent Gaussian with variance N0/2 each. The magnitude |r| and the energy |r|^2 contain no theta in the signal term, so the phase never has to be estimated.

### 1.2 The three detection principles

| Principle | Statistic | Used by |
|---|---|---|
| Envelope detection | R = sqrt(r_c^2 + r_s^2), compared with a threshold or with another branch | BASK, BFSK |
| Energy detection | R^2 = r_c^2 + r_s^2, compared with a threshold or with another branch | BASK, BFSK |
| Differential detection | Re( r_k r_(k-1)* ), the phase change between two symbols | DPSK |

Envelope and energy detection give the same decision because squaring is monotonic for positive values. In this project both are done on the correlator outputs over one symbol, so they have the same BER. A bandpass-filter, square-law, integrator detector sums more noise terms when the time-bandwidth product is above 1, and it loses extra SNR compared with this correlator form.

Differential detection uses the previous symbol as the phase reference. It only works if theta stays almost constant over two symbols, which is true for any realistic oscillator drift at normal symbol rates. The information is carried in the phase change between symbols, so the transmitter differentially encodes the bits: d_k = d_(k-1) XOR b_k.

### 1.3 Statistics used in the derivations

With noise variance sigma^2 = N0/2 per real dimension:

- Noise only: R is Rayleigh. P(R > T) = exp(-T^2 / (2 sigma^2)) = exp(-T^2 / N0).
- Signal of amplitude A plus noise: R is Rician. P(R < T) is a noncentral chi-square CDF with 2 degrees of freedom and noncentrality A^2 / sigma^2.

## 2. Theoretical BER

### 2.1 Non-coherent BFSK

Two orthogonal tones, energy Eb on the tone that is sent. Four basis functions (cosine and sine for each tone). If bit 0 is sent, R0 is Rician with signal energy Eb, R1 is Rayleigh, and an error occurs when R1 > R0:

P(error | 0) = integral over r0 of f_Rice(r0) exp(-r0^2 / N0) dr0

Combining exp(-r0^2 / N0) with the Rician pdf gives another Rician-shaped integrand with half the noise variance and a signal amplitude of half, so the integral has a closed form:

**Pb = (1/2) exp(-Eb / (2 N0))**

### 2.2 DPSK

The decision variable is Re(r_k r_(k-1)*). Using the identity

Re(r_k r_(k-1)*) = ( |r_k + r_(k-1)|^2 - |r_k - r_(k-1)|^2 ) / 4

and assuming no phase change was sent, r_k + r_(k-1) = 2 sqrt(Eb) e^(j theta) + noise, and r_k - r_(k-1) is noise only. The two noise terms are independent, each with variance N0 per real dimension. This is the same problem as non-coherent orthogonal detection with signal energy 4 Eb and noise variance N0, so the result of 2.1 applies with Eb/(2 N0) replaced by Eb/N0:

**Pb = (1/2) exp(-Eb / N0)**

### 2.3 Non-coherent BASK (on-off keying)

Bit 1 is sent with energy E1 = 2 Eb and bit 0 with zero energy, so the average bit energy is Eb. With threshold T on the envelope:

- False alarm, P(R > T | 0) = exp(-T^2 / N0)
- Miss, P(R < T | 1) = F_ncx2( T^2 / sigma^2 ; 2, E1 / sigma^2 )

**Pb = (1/2) [ exp(-T^2 / N0) + F_ncx2( 2 T^2 / N0 ; 2, 4 Eb / N0 ) ]**

The best T depends on Eb/N0 and must be found numerically. In the code it is found with a bounded 1-D search at each Eb/N0. It tends to sqrt(E1)/2 at high SNR (0.97 at 4 dB, 0.77 at 12 dB, with sqrt(E1)/2 = 0.707). With T = sqrt(E1)/2 the false-alarm term gives the usual approximation Pb ~ (1/2) exp(-Eb / (2 N0)). It is an upper estimate. At 12 dB it gives 1.8e-4 against the exact 9.3e-5.

Note on the receiver: this detector needs N0 to set T. A fixed threshold at sqrt(E1)/2 removes that requirement but about doubles the BER over this range (3.6e-3 against 1.9e-3 at 10 dB).

## 3. Simulation

- Channel: `channel/awgn.py`. Noise is added to signal-space coordinates, N0/2 per real dimension, N0 = Eb / 10^(Eb/N0 in dB / 10), Eb = 1. All three members call `add_awgn`.
- Unknown phase: BASK and BFSK use an independent uniform phase on every symbol. DPSK uses one uniform phase per block, constant across consecutive symbols.
- BASK receiver: |y| against the optimal threshold. BFSK receiver: larger of |y_0|^2 and |y_1|^2. DPSK receiver: sign of Re(y_k y_(k-1)*).
- Monte Carlo: Eb/N0 from 0 to 12 dB in 1 dB steps. Each point runs in batches of 200,000 bits until 200 errors are counted or 20 million bits are used. With 200 errors the relative standard error is about 7%. Points with fewer than 20 errors are not plotted. Seed is fixed at 2026.

### Simulated vs theoretical BER

| Eb/N0 | BASK sim | BASK theory | BFSK sim | BFSK theory | DPSK sim | DPSK theory |
|---|---|---|---|---|---|---|
| 6 dB | 4.66e-2 | 4.67e-2 | 6.75e-2 | 6.83e-2 | 9.76e-3 | 9.33e-3 |
| 8 dB | 1.35e-2 | 1.33e-2 | 2.12e-2 | 2.13e-2 | 8.83e-4 | 9.09e-4 |
| 10 dB | 1.80e-3 | 1.91e-3 | 3.17e-3 | 3.37e-3 | 2.31e-5 | 2.27e-5 |

All simulated values sit within the statistical error of the formulas (see `plots/noncoherent_ber.png`).

## 4. Coherent vs non-coherent comparison

Coherent references used: BASK Q(sqrt(Eb/N0)), BFSK Q(sqrt(Eb/N0)), BPSK Q(sqrt(2 Eb/N0)). Coherent BASK and coherent BFSK have the same BER curve when BASK is measured with average Eb. The paired comparison is BASK with non-coherent BASK, BFSK with non-coherent BFSK, and BPSK with DPSK.

### Eb/N0 needed (dB) and non-coherent penalty

| Target BER | BASK coh | BASK non-coh | Penalty | BFSK coh | BFSK non-coh | Penalty | BPSK | DPSK | Penalty |
|---|---|---|---|---|---|---|---|---|---|
| 1e-2 | 7.33 | 8.36 | 1.02 | 7.33 | 8.93 | 1.60 | 4.32 | 5.92 | 1.60 |
| 1e-3 | 9.80 | 10.51 | 0.71 | 9.80 | 10.94 | 1.14 | 6.79 | 7.93 | 1.14 |
| 1e-4 | 11.41 | 11.96 | 0.55 | 11.41 | 12.31 | 0.90 | 8.40 | 9.30 | 0.90 |
| 1e-5 | 12.60 | 13.05 | 0.45 | 12.60 | 13.35 | 0.75 | 9.59 | 10.34 | 0.75 |
| 1e-6 | 13.54 | 13.93 | 0.39 | 13.54 | 14.19 | 0.65 | 10.53 | 11.18 | 0.65 |

Figures: `plots/comparison_coherent_vs_noncoherent.png` and `plots/snr_penalty.png`.

## 5. Analysis and conclusion

**The penalty is small and shrinks as BER falls.** The commonly quoted "1 to 3 dB" gap holds only at high error rates. At BER 1e-1 the gap is 2.0 to 2.9 dB, and at 1e-2 it is 1.0 to 1.6 dB. At the BER values used in practice (1e-3 to 1e-6) it is 0.4 to 1.1 dB. At high SNR the coherent and non-coherent formulas have the same exponent. BPSK is Q(sqrt(2 Eb/N0)), about exp(-Eb/N0) / (2 sqrt(pi Eb/N0)), against (1/2) exp(-Eb/N0) for DPSK. The coherent expression has an extra factor of order 1/sqrt(Eb/N0), which is a fixed number of dB at a given BER and shrinks in dB terms as the BER target gets lower.

**DPSK is the best non-coherent scheme here.** At every BER DPSK needs about 3 dB less Eb/N0 than non-coherent BFSK (9.30 dB against 12.31 dB at BER 1e-4), because its exponent is Eb/N0 and not Eb/(2 N0). It is within 0.9 dB of coherent BPSK at 1e-4 and 0.65 dB at 1e-6.

**Non-coherent BASK looks better than non-coherent BFSK only because of how Eb is defined.** It needs 11.96 dB against 12.31 dB at 1e-4 when energy is averaged over equiprobable bits. At equal peak power, BASK has half the average bit energy of BFSK, so it needs about 2.7 dB more (11.96 + 3.01 - 12.31) at 1e-4. BASK also needs an SNR-dependent threshold, and it is the scheme most affected by amplitude fading.

**Complexity.** Coherent receivers need carrier recovery (PLL or Costas loop) and, for PSK, a way to resolve phase ambiguity. Non-coherent BASK and BFSK need only envelope or energy detectors. DPSK needs one symbol delay and a multiplier. The cost of DPSK is that the reference symbol carries noise too, so bit errors tend to occur in pairs. The formula already includes this.

**Conclusion.** Dropping carrier recovery costs between 0.4 and 1.6 dB of Eb/N0 over the 1e-2 to 1e-6 BER range. For short packets, noisy oscillators, or low-power nodes (RFID, sensor nodes, legacy modems), that is a cheap price for a simpler receiver. Where the link budget is tight and the carrier is stable (satellite links, 5G, Wi-Fi), the coherent receiver is worth its complexity because it also supports higher-order modulation such as 16-QAM, which non-coherent detection handles poorly.
