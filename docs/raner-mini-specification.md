Set Wheel Geometry: When creating the four wheels, define their <geom> tags as cylinders with a radius of approximately 160–175 mm (to achieve the overall height of 346 mm given a 117 mm axle clearance) and a width/thickness of exactly 100 mm:

XML
<geom type="cylinder" size="0.165 0.05" rgba="0.2 0.2 0.2 1" friction="1.2 0.005 0.0001"/>
(Note: MuJoCo uses half-lengths for cylinder thickness, so 0.05 equals a 100 mm total width).

Bounding Box Adjustments: Keep your main chassis <geom> length at 720 mm, but ensure your wheel joint origins are spaced out so the front-to-back wheel footprint extends to 751 mm.

Top Deck Mounting: If you plan to mount a robotic arm or sensors, center your attachment points along the two mounting rails spaced 230 mm apart.


To replicate the **AgileX Ranger Mini 3.0 (V3)** in MuJoCo, you need to model its unique 8-degree-of-freedom (8-DOF) actuation system. Unlike standard differential-drive or skid-steer robots, the Ranger Mini uses **four-wheel independent steering and four-wheel independent driving (4WDS/4WD)**, combined with an independent swing-arm suspension.

Here are the physical, kinematic, and performance specifications needed to build an accurate `.xml` MJCF (MuJoCo XML) model.

## Physical Dimensions & Mass Properties

When defining the root chassis `<body/>` and geometry bounding boxes in MuJoCo, use the following structural measurements:

| Property | Value | MuJoCo Modeling Notes |
| --- | --- | --- |
| **Chassis Dimensions (L × W × H)** | $720 \times 500 \times 345\text{ mm}$ | Use for primary box collision geometry |
| **Platform Weight** | $75\text{ kg}$ | Set as the inertial mass of the base link |
| **Rated Payload Capacity** | $100\text{ kg}$ (up to $120\text{ kg}$ max) | Add as an external test mass on the upper T-slot deck |
| **Ground Clearance** | $105\text{ mm}$ | Distance from ground plane to underside of lower chassis |
| **Obstacle Clearance** | $75\text{ mm}$ | Max step height without high-centering |
| **Max Climbing Grade** | $15^\circ$ under full payload | Useful for setting up test incline ramps in your environment |

---

## Kinematic Architecture & Actuation

The robot requires **8 active joints** in your MuJoCo model: 4 vertical steering hinges (yaw) and 4 horizontal driving hinges (pitch).

### Motor Specs & Actuator Limits

To configure your `<actuator>` tags correctly:

* **Drive Motors (4x):** Powered by four $350\text{ W}$ brushless motor control systems. In MuJoCo, model these as `<velocity>` or `<motor>` actuators capable of generating a maximum linear velocity of **$2.0\text{ m/s}$ ($7.2\text{ km/h}$)**.
* **Steering Motors (4x):** Driven by four $100\text{ W}$ harmonic drives. These provide high-torque, precision yaw control. Model them as `<position>` actuators with a continuous range of $[-\pi, \pi]$ radians.
* **Power System:** $48\text{ V}, 24\text{ Ah}$ lithium battery providing $7\text{--}8\text{ hours}$ of continuous runtime ($45\text{ km}$ total travel distance).

---

## Steering Modes & Control Logic

Because each wheel can steer independently, your MuJoCo controller must coordinate wheel angles ($\theta_i$) and wheel angular velocities ($\omega_i$) across four operating modes:

1. **Spin Mode (In-Situ Rotation):**
* **Behavior:** The robot rotates $360^\circ$ in place with a **$0^\circ$ turning radius**.
* **Kinematics:** Steer all four wheels so their rolling axes are tangent to a circle centered on the robot's center of mass. Left and right wheels drive in opposing directions.


2. **Traverse (Crab) Mode:**
* **Behavior:** Pure lateral (sideways) movement without rotating the chassis.
* **Kinematics:** All four steering joints are locked at $+90^\circ$ or $-90^\circ$; drive velocities are identical across all wheels.


3. **Diagonal Mode:**
* **Behavior:** The robot translates at an oblique angle while keeping its forward heading constant.
* **Kinematics:** All four steering joints are synchronized to a target angle $\alpha$ where $0^\circ < \vert{}\alpha\vert{} < 90^\circ$.


4. **Ackermann & Double Ackermann Mode:**
* **Behavior:** Smooth, high-speed cornering similar to a passenger vehicle.
* **Kinematics:** In Double Ackermann, both front and rear axles steer in opposite directions. The inner wheels turn at a sharper angle than the outer wheels to intersect at a common instantaneous center of rotation (ICR), preventing lateral wheel scrub.



---


---

## MuJoCo Implementation Hierarchy

When structuring your `.xml` file, build the kinematic tree in this order:

1. **`worldbody`** $\rightarrow$ **`base_link`** (Free joint, mass = $75\text{ kg}$, box geometry $720\times500\times345\text{ mm}$)
2. **`suspension_link_[fl,fr,rl,rr]`** $\rightarrow$ Attach via a single-axis hinge joint with `<stiffness>` and `<damping>` attributes to simulate the independent swing arms.
3. **`steering_link_[fl,fr,rl,rr]`** $\rightarrow$ Attach to the suspension link via a Z-axis (yaw) hinge. Controlled by position actuators.
4. **`wheel_link_[fl,fr,rl,rr]`** $\rightarrow$ Attach to the steering link via a Y-axis (pitch) hinge. Add cylinder geometries with high friction coefficients (`friction="1.2 0.005 0.0001"`) and control via velocity actuators.

Specification Breakdown
Parameter
In Previous Response?
Spec Sheet / Diagram Value
Why It Matters for MuJoCo
Dimensions (L × W × H)
Yes
720 × 500 × 345 mm
Primary chassis collision box geometry.
Platform Weight
Yes
75 kg
Root body inertial mass (<body mass="75">).
Max Payload
Yes
100 kg
Maximum external mass to attach to the top deck during testing.
Climbing Ability
Yes
≤15° (with load)
Setting the tilt angle of incline test ramps.
Max Speed
Yes
2 m/s
Velocity limit for wheel drive actuators.
Max Travel Distance
Yes
45 km
Useful for long-duration battery discharge simulations.
Obstacle Clearance
Yes
75 mm
Maximum step height for terrain obstacle courses.
Wheel Width
No
100 mm
Critical for MuJoCo: Use this as the height/thickness of your wheel <cylinder> geometries.
Top Rail Spacing
No
230 mm
Essential for accurately placing sensors (LiDAR, cameras) or payloads on the top T-slots.
Overall Length (with wheels/bumpers)
No
751 mm
The 720 mm value is the chassis box; 751 mm is the total bounding box length from wheel-edge to wheel-edge.
Underbody Clearance
Nuance
117 mm
My previous text noted 105 mm general clearance; the drawing shows 117 mm specifically at the inner chassis arch.
IP Rating
No
IP54
Irrelevant for physics simulation (dust/water resistance).
Communication
No
Standard CAN, 232 Serial
Irrelevant for physics, but useful if building a ROS2/CAN-bus hardware-in-the-loop bridge.



Proposed Changes
Kinematics Engine
Update the base controller to support crab walking (lateral movement).

[MODIFY] robot_base.py
Add v_lateral: float = 0.0 to the DriveCommand dataclass.
Update _get_wheel_ik(v_forward, v_lateral, v_angular, x, y) to calculate the exact Ackermann vectors for omni-directional movement:
vx = v_forward - v_angular * y
vy = v_lateral + v_angular * x
Update ackermann_angles and ackermann_wheel_speeds to accept and pass v_lateral.
[MODIFY] robot.py
Update constants to match the spec:
WHEEL_RADIUS = 0.165
TRACK_WIDTH = 0.60 (to clear the 500mm wide chassis with 100mm wheels)
WHEELBASE = 0.421 (calculated from 751mm overall footprint minus two 165mm wheel radii)
Ensure apply_command passes cmd.v_lateral to the kinematics functions.
Update mass_total to 111.0 kg (75kg base + 36kg wheels/steer).
Physics & Geometry (MuJoCo XML)
Update the model dimensions to match the official AgileX spec sheet.

[MODIFY] ranger_mini_v3.xml
Chassis: Increase box size to 0.36 0.25 0.1725 (720x500x345 mm). Set inertial mass to exactly 75 kg.
Ground Clearance: Set base_link height to Z=0.2775 so the bottom face rests at exactly 105mm ground clearance.
Wheels: Update wheel cylinders to size="0.165 0.05" (Radius 165mm, Width 100mm).
Steering Joints: Reposition joints to match the new 421mm wheelbase and 600mm track width, ensuring the wheels tuck perfectly beside the chassis without collision.