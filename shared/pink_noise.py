"""
shared/pink_noise.py — FFT spectral-shaping colored noise generators.
=====================================================================
Extracted from simulation/ranger_mini_v3/exploration_policies.py.
Pure numpy/scipy — no MuJoCo dependency.

Implements FFT spectral-shaping colored noise as described in:

  MINav: "120 Minutes and a Laptop: Minimalist Image-goal Navigation
  via Unsupervised Exploration and Offline RL"

  The paper specifies: generate a randomly initialized spectrum in the
  frequency domain, shape it according to PSD(f) ∝ 1/f^β with β=1,
  then transform back to the time domain via inverse FFT.
"""

import numpy as np
import scipy.stats


# ============================================================
# FFT Colored Noise Generator
# ============================================================

class FFTColoredNoise:
    """FFT spectral-shaping colored noise generator.

    Generates noise sequences whose power spectral density follows:

        PSD(f) ∝ 1 / f^β

    This is done entirely in the frequency domain — no FFT of white
    noise. Instead, a complex-valued spectrum is constructed directly
    from independent Gaussian draws, shaped by the 1/f^(β/2) filter
    (amplitude filter, because PSD = |spectrum|^2), and then brought
    back to the time domain via irfft.

    Parameters
    ----------
    beta : float
        Spectral exponent.
        - 0 → White noise  (flat PSD)
        - 1 → Pink noise   (PSD ∝ 1/f)  — MINav default
        - 2 → Brown noise  (PSD ∝ 1/f²)
    buffer_size : int
        Number of samples to generate per FFT call. Must be even.
        Larger values reduce FFT overhead and improve spectral accuracy.
        Default: 8192.
    seed : int or None
        Optional random seed for reproducibility.

    Notes
    -----
    Why β/2 in the amplitude filter?
        The power spectral density is the squared magnitude of the
        spectrum. To obtain PSD ∝ 1/f^β we need:

            |H(f)|^2 ∝ 1/f^β  →  |H(f)| ∝ 1/f^(β/2)

        So the amplitude filter uses β/2, not β.

    Why normalize before the Gaussian CDF?
        UniformColoredNoise applies norm.cdf(raw) to transform samples
        into [0, 1] while preserving temporal correlation (probability
        integral transform). This only works correctly when the marginal
        distribution of `raw` is close to N(0,1). The normalization
        step (zero mean, unit variance) ensures that assumption holds.
    """

    def __init__(self, beta: float = 1.0, buffer_size: int = 8192, seed: int = None):
        if buffer_size % 2 != 0:
            buffer_size += 1  # irfft requires even N for clean reconstruction
        self.beta = beta
        self.buffer_size = buffer_size
        self._rng = np.random.default_rng(seed)
        self._buffer: np.ndarray = np.empty(0)
        self._index: int = 0
        self._generate_buffer()

    def _generate_buffer(self) -> None:
        """Generate one block of FFT-shaped colored noise.

        Algorithm (MINav paper):
        1. Build a random complex-valued spectrum directly.
        2. Compute the one-sided frequency axis via rfftfreq.
        3. Construct the 1/f^(β/2) amplitude filter; set DC bin to 0.
        4. Multiply spectrum by the filter.
        5. Reconstruct the real-valued signal via irfft.
        6. Normalize: subtract mean, divide by std → N(0,1) marginal.
        """
        N = self.buffer_size
        n_freq = N // 2 + 1

        # Step 1: Random complex spectrum (white before filtering)
        real = self._rng.standard_normal(n_freq)
        imag = self._rng.standard_normal(n_freq)
        spectrum = real + 1j * imag

        # Step 2: One-sided frequency axis
        freqs = np.fft.rfftfreq(N)

        # Step 3: Amplitude filter  H(f) = 1 / f^(β/2)
        filt = np.empty(n_freq)
        filt[0] = 0.0
        filt[1:] = 1.0 / (freqs[1:] ** (self.beta / 2.0))

        # Step 4: Shape the spectrum
        spectrum *= filt

        # Step 5: Inverse FFT → real-valued time series
        pink = np.fft.irfft(spectrum, n=N)

        # Step 6: Normalize to zero mean, unit variance
        mean = np.mean(pink)
        std = np.std(pink)
        pink -= mean
        if std > 1e-12:
            pink /= std
        else:
            pink = self._rng.standard_normal(N)

        self._buffer = pink
        self._index = 0

    def sample(self) -> float:
        """Return the next scalar sample.

        Automatically regenerates the internal buffer when exhausted.
        """
        if self._index >= self.buffer_size:
            self._generate_buffer()
        val = float(self._buffer[self._index])
        self._index += 1
        return val


# ============================================================
# Uniform Colored Noise  (probability integral transform)
# ============================================================

class UniformColoredNoise:
    """Transforms FFT colored noise into a Uniform[min, max] marginal.

    The raw FFT pink noise has a near-Gaussian marginal N(0,1) after
    normalization. Applying the Gaussian CDF (probability integral
    transform) maps it to Uniform[0,1] while preserving the temporal
    autocorrelation structure (1/f memory).

    This matches the MINav paper's "pink-uniform" exploration noise.

    Parameters
    ----------
    beta : float
        Spectral exponent forwarded to FFTColoredNoise.
    range_val : tuple[float, float]
        Target output interval (min, max) for the action dimension.
    buffer_size : int
        Buffer size forwarded to FFTColoredNoise. Default: 8192.
    seed : int or None
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        beta: float = 1.0,
        range_val: tuple = (-1.0, 1.0),
        buffer_size: int = 8192,
        seed: int = None,
    ):
        self.noise = FFTColoredNoise(beta=beta, buffer_size=buffer_size, seed=seed)
        self.range = range_val

    def sample(self) -> float:
        """Return one sample in [range_val[0], range_val[1]].

        Pipeline:
            raw   ~ FFT pink noise  (N(0,1) marginal)
            u     = norm.cdf(raw)   (Uniform[0,1] via PIT)
            out   = min + (max - min) * u   (linear rescale)
        """
        raw = self.noise.sample()
        u = scipy.stats.norm.cdf(raw)
        return self.range[0] + (self.range[1] - self.range[0]) * u
