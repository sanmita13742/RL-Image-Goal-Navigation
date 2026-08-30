Ran command: `python ranger_mujoco\exploration_policies.py`
Viewed exploration_policies.py:99-108
Edited exploration_policies.py

I have fully refactored `exploration_policies.py` to follow strict DRY (Don't Repeat Yourself) principles!

### The New Architecture
I moved all the heavy lifting (collision avoidance logic, timers, and holding actions) into the abstract base class `ExplorationPolicy`. Now, subclasses only need to implement a single `_sample_noise()` method to define what their specific noise distribution looks like. 

### How to Tune the Policies
The main entry point for tuning your exploration is in **`ranger_mujoco/random_explore.py`**. Look for this line right before the `while frame_idx < MAX_EPISODE_STEPS:` loop (around line 134):

```python
policy = LinearExploration(CONTROL_FREQ, hold_time=1.0)
```

Here are the specific tuners you have access to:

#### 1. Tuning the Time Period (`hold_time`)
You can control how long an action is held by passing `hold_time` (in seconds) to the policy constructor. 
- **`hold_time=0.0`**: Samples a new action at every single simulation control step (this is what `WhiteNoiseExploration` uses by default).
- **`hold_time=1.0`**: Holds the same command for 1.0 second, resulting in long, smooth, sweeping movements (this is what `LinearExploration` uses by default).
- You can freely pass `hold_time=0.5` or `hold_time=2.0` into `LinearExploration` to get different scales of piecewise-linear movement.

#### 2. Tuning the Noise Boundaries (`v_range` and `w_range`)
Both policies accept keyword arguments to limit the noise scales. You can pass them in `random_explore.py` like this:
```python
policy = WhiteNoiseExploration(
    CONTROL_FREQ, 
    v_range=(0.0, 1.0),   # Only drive forward between 0 and 1 m/s
    w_range=(-0.5, 0.5)   # Gentle turns only
)
```

#### 3. Adding New Noises (e.g., Pink Noise or OU Noise)
When you're ready to add the advanced noises from the MINav paper (like Ornstein-Uhlenbeck or Pink Noise), simply open `exploration_policies.py`, create a new class `class OUNoiseExploration(ExplorationPolicy):`, and define `def _sample_noise(self):`. The base class will automatically handle applying that noise to the robot while simultaneously protecting it from LiDAR-detected collisions!