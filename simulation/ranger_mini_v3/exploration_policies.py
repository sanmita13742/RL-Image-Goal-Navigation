"""
ranger_mujoco/exploration_policies.py
============================================================
Exploration framework based on Motion Primitives, Colored Noise,
Adaptive Scheduling, and Behavioral Loop Detection.

Noise Generation
----------------
Implements FFT spectral-shaping colored noise as described in:

  MINav: "120 Minutes and a Laptop: Minimalist Image-goal Navigation
  via Unsupervised Exploration and Offline RL"

  The paper specifies: generate a randomly initialized spectrum in the
  frequency domain, shape it according to PSD(f) ∝ 1/f^β with β=1,
  then transform back to the time domain via inverse FFT.
"""

import abc
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.stats
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))
from robot_base import DriveCommand


class Primitive(Enum):
    ACKERMANN = 1
    SPIN = 2
    TRAVERSE = 3
    DIAGONAL = 4
    REVERSE = 5


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
        step (zero mean, unit variance) ensures that assumption holds,
        so that norm.cdf maps the marginal to Uniform[0,1] and the
        linear rescaling then hits the target action range.

    Why does the PIT preserve temporal correlation?
        norm.cdf is a monotone transformation. Monotone transforms
        preserve rank ordering of samples, which preserves the
        autocorrelation structure (the 1/f temporal memory) that
        makes pink noise useful for smooth exploration.
    """

    def __init__(self, beta: float = 1.0, buffer_size: int = 8192):
        if buffer_size % 2 != 0:
            buffer_size += 1  # irfft requires even N for clean reconstruction
        self.beta = beta
        self.buffer_size = buffer_size
        self._buffer: np.ndarray = np.empty(0)
        self._index: int = 0
        self._generate_buffer()

    def _generate_buffer(self) -> None:
        """Generate one block of FFT-shaped colored noise.

        Algorithm (MINav paper):
        1. Build a random complex-valued spectrum directly.
           real and imag are drawn independently from N(0,1), so the
           initial spectrum has flat (white) PSD before filtering.
        2. Compute the one-sided frequency axis via rfftfreq.
        3. Construct the 1/f^(β/2) amplitude filter; set DC bin to 0.
        4. Multiply spectrum by the filter.
        5. Reconstruct the real-valued signal via irfft.
        6. Normalize: subtract mean, divide by std → N(0,1) marginal.
        """
        N = self.buffer_size
        n_freq = N // 2 + 1  # number of rfft bins

        # Step 1: Random complex spectrum (white before filtering)
        real = np.random.randn(n_freq)
        imag = np.random.randn(n_freq)
        spectrum = real + 1j * imag

        # Step 2: One-sided frequency axis  (0, 1/N, 2/N, ..., 1/2)
        freqs = np.fft.rfftfreq(N)  # d=1 → normalized [0, 0.5]

        # Step 3: Amplitude filter  H(f) = 1 / f^(β/2)
        #   DC bin set to 0 to remove any mean offset from the filter
        #   and avoid division-by-zero.
        filt = np.empty(n_freq)
        filt[0] = 0.0
        filt[1:] = 1.0 / (freqs[1:] ** (self.beta / 2.0))

        # Step 4: Shape the spectrum
        spectrum *= filt

        # Step 5: Inverse FFT → real-valued time series
        pink = np.fft.irfft(spectrum, n=N)

        # Step 6: Normalize to zero mean, unit variance
        #   This is required so that norm.cdf(sample) maps correctly
        #   to Uniform[0,1] in UniformColoredNoise.
        mean = np.mean(pink)
        std = np.std(pink)
        pink -= mean
        if std > 1e-12:
            pink /= std
        else:
            # Degenerate case (should not happen) — fall back to white
            pink = np.random.randn(N)

        self._buffer = pink
        self._index = 0

    def sample(self) -> float:
        """Return the next scalar sample.

        Automatically regenerates the internal buffer when exhausted.
        Streaming is seamless — callers see an infinite sequence.
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
    autocorrelation structure (1/f memory). A linear rescaling then
    maps to the desired action range [a, b].

    This matches the MINav paper's "pink-uniform" exploration noise.

    Parameters
    ----------
    beta : float
        Spectral exponent forwarded to FFTColoredNoise.
    range_val : tuple[float, float]
        Target output interval (min, max) for the action dimension.
    buffer_size : int
        Buffer size forwarded to FFTColoredNoise. Default: 8192.
    """

    def __init__(
        self,
        beta: float = 1.0,
        range_val: tuple = (-1.0, 1.0),
        buffer_size: int = 8192,
    ):
        self.noise = FFTColoredNoise(beta=beta, buffer_size=buffer_size)
        self.range = range_val

    def sample(self) -> float:
        """Return one sample in [range_val[0], range_val[1]].

        Pipeline:
            raw   ~ FFT pink noise  (N(0,1) marginal)
            u     = norm.cdf(raw)   (Uniform[0,1] via PIT)
            out   = min + (max - min) * u   (linear rescale)
        """
        raw = self.noise.sample()
        # Probability integral transform: Gaussian → Uniform[0,1]
        u = scipy.stats.norm.cdf(raw)
        # Linear mapping to configured action range
        return self.range[0] + (self.range[1] - self.range[0]) * u


# ============================================================
# Base Exploration Policy
# ============================================================

class BaseExploration(abc.ABC):
    """Abstract base class with stateful, adaptive holonomic collision recovery."""

    def __init__(self, control_freq: float):
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

        self.current_cmd = DriveCommand()
        self.timer = 0

        # State Machine
        self.recovery_state = "NORMAL"
        self.recovery_timer = 0
        self.recovery_cmd = DriveCommand()

        # Metrics & Tracking
        self.primitive_counts = {p: 0 for p in Primitive}
        self.primitive_duration_sum = {p: 0.0 for p in Primitive}
        self.primitive_transitions = {p: {p2: 0 for p2 in Primitive} for p in Primitive}
        self.current_primitive = Primitive.ACKERMANN
        self.active_primitive_timer = 0.0

        # Adaptive Meta-Scheduler & Behavioral Loop Detection
        self.visit_grid = defaultdict(int)
        self.grid_res = 1.0
        self.visited_history = []  # Track new cells over window
        self.window_size = int(control_freq * 30)  # 30 second window

        self.x, self.y, self.yaw = 0.0, 0.0, 0.0

    def _switch_primitive(self, new_prim: Primitive):
        self.primitive_transitions[self.current_primitive][new_prim] += 1
        self.primitive_counts[self.current_primitive] += 1
        self.primitive_duration_sum[self.current_primitive] += self.active_primitive_timer

        self.current_primitive = new_prim
        self.active_primitive_timer = 0.0

    def _update_visit_grid(self, x: float, y: float):
        cell = (int(math.floor(x / self.grid_res)), int(math.floor(y / self.grid_res)))
        if self.visit_grid[cell] == 0:
            self.visited_history.append(1)
        else:
            self.visited_history.append(0)

        self.visit_grid[cell] += 1

        if len(self.visited_history) > self.window_size:
            self.visited_history.pop(0)

    def get_action(self, depth_img: np.ndarray, x: float, y: float, yaw: float) -> tuple[DriveCommand, Primitive]:
        self.x, self.y, self.yaw = x, y, yaw
        self._update_visit_grid(x, y)

        h, w = depth_img.shape
        left_third = depth_img[:, :w//3]
        center_third = depth_img[:, w//3:2*w//3]
        right_third = depth_img[:, 2*w//3:]

        min_c = np.min(center_third)
        min_l = np.min(left_third)
        min_r = np.min(right_third)
        min_depth = min(min_c, min_l, min_r)

        # ─── Collision Recovery State Machine ───

        if self.recovery_timer > 0:
            self.recovery_timer -= 1
            self.active_primitive_timer += self.dt
            if self.recovery_timer == 0:
                self.recovery_state = "NORMAL"
            return self.recovery_cmd, self.current_primitive

        # 1. Reactive Escape (Obstacles)
        if min_depth < 0.6:
            if min_l > 1.0:
                self.recovery_state = "REACTIVE_ESCAPE_TRAVERSE"
                self.recovery_cmd = DriveCommand(v_linear=0.0, v_lateral=0.8, v_angular=0.0)
                self.recovery_timer = int(self.control_freq * 1.5)
                self._switch_primitive(Primitive.TRAVERSE)
            elif min_r > 1.0:
                self.recovery_state = "REACTIVE_ESCAPE_TRAVERSE"
                self.recovery_cmd = DriveCommand(v_linear=0.0, v_lateral=-0.8, v_angular=0.0)
                self.recovery_timer = int(self.control_freq * 1.5)
                self._switch_primitive(Primitive.TRAVERSE)
            elif min_depth < 0.3:
                self.recovery_state = "REACTIVE_ESCAPE_REVERSE"
                steer = -1.0 if min_l > min_r else 1.0
                self.recovery_cmd = DriveCommand(v_linear=-0.8, v_lateral=0.0, v_angular=steer)
                self.recovery_timer = int(self.control_freq * 1.5)
                self._switch_primitive(Primitive.REVERSE)
            else:
                self.recovery_state = "REACTIVE_ESCAPE_SPIN"
                spin = 1.5 if min_l > min_r else -1.5
                self.recovery_cmd = DriveCommand(v_linear=0.0, v_lateral=0.0, v_angular=spin)
                self.recovery_timer = int(self.control_freq * 1.0)
                self._switch_primitive(Primitive.SPIN)

            self.active_primitive_timer += self.dt
            return self.recovery_cmd, self.current_primitive

        # 2. Proactive Escape (Behavioral Loop Detection)
        # If we have collected enough history and efficiency is near zero
        efficiency = sum(self.visited_history) / len(self.visited_history) if len(self.visited_history) > 0 else 1.0

        if len(self.visited_history) == self.window_size and efficiency < 0.005:
            # We are trapped in a behavioral loop (e.g. corridor sweeping, huge revisits)
            self.recovery_state = "PROACTIVE_ESCAPE"
            # Force crab walk away or reverse
            if random.random() < 0.5:
                lat = random.choice([-1.0, 1.0])
                self.recovery_cmd = DriveCommand(v_linear=0.0, v_lateral=lat, v_angular=0.0)
                self.recovery_timer = int(self.control_freq * 3.0)  # Escape for longer
                self._switch_primitive(Primitive.TRAVERSE)
            else:
                self.recovery_cmd = DriveCommand(v_linear=-0.8, v_lateral=0.0, v_angular=random.uniform(-1, 1))
                self.recovery_timer = int(self.control_freq * 3.0)
                self._switch_primitive(Primitive.REVERSE)

            self.visited_history = []  # Reset history to avoid cascading proactive escapes
            self.active_primitive_timer += self.dt
            return self.recovery_cmd, self.current_primitive

        # ─── Normal Exploration ───

        if self.timer <= 0:
            new_cmd, new_prim, hold_time = self._sample_primitive(efficiency)

            if new_prim != self.current_primitive:
                self._switch_primitive(new_prim)

            self.current_cmd = new_cmd
            self.timer = int(self.control_freq * hold_time)

        else:
            self.current_cmd = self._modulate_primitive(self.current_cmd, self.current_primitive)

        self.timer -= 1
        self.active_primitive_timer += self.dt
        return self.current_cmd, self.current_primitive

    @abc.abstractmethod
    def _sample_primitive(self, efficiency: float) -> tuple[DriveCommand, Primitive, float]:
        pass

    @abc.abstractmethod
    def _modulate_primitive(self, cmd: DriveCommand, prim: Primitive) -> DriveCommand:
        pass


# ============================================================
# Primitive Exploration Policy
# ============================================================

class PrimitiveExplorationPolicy(BaseExploration):
    def __init__(self, control_freq: float, beta: int = 1):
        super().__init__(control_freq)
        self.beta = beta

        self.speed_noise = UniformColoredNoise(beta, range_val=(0.3, 1.8))
        self.steer_noise = UniformColoredNoise(beta, range_val=(-1.0, 1.0))
        self.lat_noise   = UniformColoredNoise(beta, range_val=(-1.0, 1.0))

        self.base_probs = {
            Primitive.ACKERMANN: 0.50,
            Primitive.DIAGONAL:  0.20,
            Primitive.TRAVERSE:  0.10,
            Primitive.SPIN:      0.10,
            Primitive.REVERSE:   0.10,
        }

    def _predict_cell(self, prim: Primitive, dist: float = 1.5) -> tuple[int, int]:
        """Roughly predict the future cell if this primitive is chosen."""
        px, py = self.x, self.y
        if prim == Primitive.ACKERMANN:
            px += math.cos(self.yaw) * dist
            py += math.sin(self.yaw) * dist
        elif prim == Primitive.REVERSE:
            px -= math.cos(self.yaw) * dist
            py -= math.sin(self.yaw) * dist
        elif prim == Primitive.TRAVERSE:
            # Assuming lateral right is -90 deg from yaw
            px += math.cos(self.yaw - math.pi/2) * dist
            py += math.sin(self.yaw - math.pi/2) * dist
        elif prim == Primitive.DIAGONAL:
            px += math.cos(self.yaw - math.pi/4) * dist
            py += math.sin(self.yaw - math.pi/4) * dist

        return (int(math.floor(px / self.grid_res)), int(math.floor(py / self.grid_res)))

    def _sample_primitive(self, efficiency: float) -> tuple[DriveCommand, Primitive, float]:
        probs = self.base_probs.copy()

        # 1. Adaptive Meta-Scheduler: Boost escape primitives if coverage stagnates
        if len(self.visited_history) == self.window_size and efficiency < 0.05:
            probs[Primitive.ACKERMANN] *= 0.5
            probs[Primitive.TRAVERSE]  *= 2.0
            probs[Primitive.DIAGONAL]  *= 1.5
            probs[Primitive.REVERSE]   *= 1.5

        # 2. Soft Revisitation Penalty: Repel from hotspots
        for p in Primitive:
            projected = self._predict_cell(p)
            visits = self.visit_grid[projected]
            if visits > 0:
                probs[p] *= math.exp(-0.1 * visits)

        # Normalize
        total_p = sum(probs.values())
        if total_p == 0:
            probs = self.base_probs.copy()  # fallback
            total_p = sum(probs.values())

        keys = list(probs.keys())
        weights = [probs[k]/total_p for k in keys]

        prim = random.choices(keys, weights=weights, k=1)[0]
        hold_time = random.uniform(1.0, 3.0)

        return self._modulate_primitive(DriveCommand(), prim), prim, hold_time

    def _modulate_primitive(self, cmd: DriveCommand, prim: Primitive) -> DriveCommand:
        v = self.speed_noise.sample()
        w = self.steer_noise.sample()
        lat = self.lat_noise.sample()

        if prim == Primitive.ACKERMANN:
            return DriveCommand(v_linear=v, v_lateral=0.0, v_angular=w)
        elif prim == Primitive.SPIN:
            return DriveCommand(v_linear=0.0, v_lateral=0.0, v_angular=w * 1.5)
        elif prim == Primitive.TRAVERSE:
            return DriveCommand(v_linear=0.0, v_lateral=lat, v_angular=0.0)
        elif prim == Primitive.DIAGONAL:
            return DriveCommand(v_linear=v, v_lateral=lat, v_angular=0.0)
        elif prim == Primitive.REVERSE:
            return DriveCommand(v_linear=-v * 0.5, v_lateral=0.0, v_angular=w)

        return DriveCommand()
