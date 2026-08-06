# Pink Noise for Unsupervised Exploration

This document explains the implementation of FFT spectral-shaping pink noise used in our exploration framework, which matches the methodology described in the MINav paper ("120 Minutes and a Laptop: Minimalist Image-goal Navigation via Unsupervised Exploration and Offline RL").

## Background

In reinforcement learning and robotics exploration, temporally correlated noise (like pink noise or Ornstein-Uhlenbeck noise) often yields better state-space coverage than uncorrelated white noise. Pink noise, characterized by a power spectral density (PSD) inversely proportional to frequency ($1/f$), provides a balance: it maintains short-term momentum (low frequencies) while still exhibiting some randomness (high frequencies).

## MINav Paper Methodology

The MINav paper specifies the following procedure for generating "pink-uniform" noise:
1. Generate a randomly initialized spectrum in the frequency domain.
2. Shape it according to $PSD(f) \propto 1/f^\beta$ with $\beta=1$.
3. Transform it back to the time domain via inverse Fast Fourier Transform (iFFT).
4. Apply the probability integral transform (using the normal CDF) to map the near-Gaussian marginal to a uniform distribution, preserving temporal correlations.
5. Scale to the desired action bounds.

## Our Implementation (`FFTColoredNoise`)

We replaced the previous Voss-McCartney approximation with a direct frequency-domain generation algorithm to exactly match the paper's specification.

### 1. Complex Spectrum Generation
Instead of computing the FFT of white noise, we directly construct a random complex-valued spectrum. The real and imaginary components of each frequency bin are drawn independently from a standard normal distribution $\mathcal{N}(0, 1)$.

### 2. Amplitude Filtering (Why $\beta/2$?)
The target is a power spectral density $PSD(f) \propto 1/f^\beta$. Since $PSD(f) = |H(f)|^2$, the amplitude filter applied to the complex spectrum must be proportional to $1/f^{\beta/2}$.
- The DC component (0 Hz) is set to 0 to remove any mean offset.

### 3. Inverse FFT
The shaped spectrum is transformed back to the time domain using `np.fft.irfft`, producing a real-valued sequence.

### 4. Normalization
The resulting time-domain signal is normalized to have zero mean and unit variance. This ensures that the marginal distribution is approximately standard normal, which is a strict requirement for the next step.

## Uniform Mapping (`UniformColoredNoise`)

The `UniformColoredNoise` class takes the standard normal pink noise samples and applies the probability integral transform:

```python
u = scipy.stats.norm.cdf(raw)
```

Because `norm.cdf` is a strictly monotonically increasing function, it acts as a rank-preserving transformation. This maps the $\mathcal{N}(0, 1)$ marginal distribution perfectly into a Uniform(0, 1) distribution **without destroying the $1/f$ temporal autocorrelation structure**. 

Finally, the uniform samples are linearly scaled to the minimum and maximum ranges configured for each action dimension (e.g., speed, steering).

## Streaming via Internal Buffer
To avoid computing the FFT at every timestep, the implementation uses an internal buffer (default size: 8192). The sequence is pre-generated in blocks, and `sample()` yields one value at a time. Once the buffer is exhausted, the next sequence of 8192 samples is automatically generated, providing seamless, infinite streaming for long exploration sessions.
