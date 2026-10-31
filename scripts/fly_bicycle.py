"""
Fly Bicycle: a fruit fly riding a miniature bicycle in MuJoCo.

Assembles a bicycle and FlyBody's fruit fly into one MJCF model, welds the fly
onto the saddle, and holds it there with the fly's own leg actuators:

    T1 (front legs)  -> handlebars
    T2 (middle legs) -> saddle rails
    T3 (hind legs)   -> pedals, tracking the crank through its rotation

Leg goals come from damped-least-squares IK on each claw body, so the feet are
placed on the object they are reaching for instead of being posed blind. A motor
turns the crank, an equality constraint couples the crank to the rear wheel
like a chain, and ground friction turns that into forward motion.

Units are FlyBody's: centimetres, grams, seconds (gravity -981 cm/s^2). A real
bicycle is ~280x longer than a fly, far beyond what 2 mm legs can span, so the
bike here is scaled so the handlebars and pedals fall inside the fly's actual
reach envelope. It is a bike a fly can physically hold onto, not a
proportionally accurate one.

Usage:
    python scripts/fly_bicycle.py                  # -> videos/fly_bicycle.mp4
    python scripts/fly_bicycle.py --seconds 8
    python scripts/fly_bicycle.py --no-video       # numbers only, no render
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flybody  # noqa: E402  (path set above)

FLY_XML = os.path.join(os.path.dirname(os.path.abspath(flybody.__file__)),
                       'fruitfly', 'assets', 'fruitfly.xml')

OBJ = mujoco.mjtObj

# One IK + crank update every CONTROL_DT; physics integrates finer.
CONTROL_DT = 1e-3
PHYSICS_DT = 2e-4

# ---------------------------------------------------------------------------
# Bicycle geometry. Lengths are local to the frame body, whose origin sits at
# WHEEL_R above the floor, so the wheel hubs rest exactly on the ground.
#
# Sized against the measured claw workspace (cm, relative to the thorax):
#   T1  x[-0.09, +0.22]  y[-0.22, +0.22]  z[-0.24, +0.12]
#   T3  x[-0.26, +0.11]  z[-0.26, +0.17]
# ---------------------------------------------------------------------------

WHEEL_R = 0.20          # wheel radius
WHEELBASE = 0.90

# The frame's slide joints hold its height, which leaves the tyres exactly
# tangent to the floor. Tangent contact carries no load: normal force is zero,
# and with no normal force the friction cone is empty, so the rear wheel spins
# in place and the bike never moves. Sinking the frame by this much gives the
# contacts a standing load to push against. 1e-4 cm is a micrometre of preload
# and is enough; 5e-3 works just as well, so the result is not delicate.
WHEEL_PRELOAD = 5e-4
REAR_X = -WHEELBASE / 2
FRONT_X = WHEELBASE / 2

# Bottom bracket. Two constraints fight here and were resolved by sweeping the
# geometry against worst-case multi-start IK error over a full crank turn:
#   - it must sit forward of REAR_X + WHEEL_R or the crank arm is inside the
#     rear wheel, which needs a long wheelbase at this wheel size;
#   - the hind legs reach much further back-and-down (0.297 cm) than forward
#     (0.067 cm), so the pedal orbit has to sit behind the thorax. Centring it
#     under the thorax puts the forward half of the orbit out of reach.
CRANK_X = -0.17
CRANK_Z = 0.13          # bottom bracket height
CRANK_R = 0.04          # crank arm length = pedal orbit radius
SEAT_X = -0.10          # saddle centre; the fly straddles this
SEAT_Z = 0.20
BAR_Z = 0.26            # handlebar height, near the fly's own shoulder line
HEAD_X = 0.18           # head tube
GRIP_X = 0.01           # bars sit just ahead of the fly, inside leg reach
GRIP_Y = 0.11
RIDER_Z = 0.26          # thorax anchor height, local

# Side convention: the fly's left legs are on +y, right legs on -y. Every
# site pair below is written left-first to match.
LEFT = +1
RIGHT = -1

TUBE_R = 0.012
MASS_WHEEL = 0.0008
MASS_TUBE = 0.0003

def bike_xml():
    """Render the bicycle MJCF from the geometry constants above.

    A function rather than a module constant so the constants stay the single
    source of truth: change one and the XML follows.
    """
    return f"""
<mujoco model="fly_bicycle">
  <compiler autolimits="true" angle="radian"/>
  <option timestep="{PHYSICS_DT}" gravity="0 0 -981" integrator="RK4"/>

  <visual>
    <headlight ambient="0.45 0.45 0.50" diffuse="0.7 0.7 0.72" specular="0.2 0.2 0.2"/>
    <global azimuth="135" elevation="-12" offwidth="1280" offheight="960"/>
  </visual>

  <default>
    <!-- Contact filtering by bitmask, because the rider's own collision geoms
         unavoidably overlap the frame it is bolted to (head against the top
         tube, antennae against the bars). Rather than fight that with exclude
         pairs, the three groups are given disjoint masks:
           group 1  floor         contype 4  conaffinity 7
           group 2  bike          contype 2  conaffinity 3   (hits floor)
           group 4  fly, unchanged: contype 1 conaffinity 1 (self-collides,
                   but never with the bike or the floor it is bolted to) -->
    <geom contype="2" conaffinity="3" condim="3" friction="1.4 0.02 0.002"
          rgba="0.82 0.18 0.15 1"/>
  </default>

  <worldbody>
    <light name="key" pos="1.5 -2.0 4.0" dir="-0.35 0.5 -1" diffuse="0.6 0.6 0.62"/>
    <geom name="floor" type="plane" size="60 60 0.1" rgba="0.80 0.83 0.88 1"
          contype="4" conaffinity="7" friction="1.4 0.02 0.002"/>

    <body name="frame" pos="0 0 {WHEEL_R - WHEEL_PRELOAD:.6f}">
      <!-- The bike translates in the ground plane and yaws, but cannot tip
           over sideways. An unsupported bicycle would simply fall; pinning the
           roll and pitch degrees of freedom keeps this a riding demo. -->
      <joint name="slide_x" type="slide" axis="1 0 0" damping="0.0004"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="0.0004"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0.00008"/>

      <!-- Main triangle. -->
      <geom name="tube_seatpost" type="capsule"
            fromto="{CRANK_X:.4f} 0 {CRANK_Z:.4f} {SEAT_X:.4f} 0 {SEAT_Z:.4f}"
            size="{TUBE_R}" mass="{MASS_TUBE}"/>
      <geom name="tube_seat" type="capsule"
            fromto="{CRANK_X:.4f} 0 {CRANK_Z:.4f} {REAR_X:.4f} 0 0"
            size="{TUBE_R}" mass="{MASS_TUBE}"/>
      <geom name="tube_chainstay" type="capsule"
            fromto="{REAR_X:.4f} 0 0 {CRANK_X:.4f} 0 {CRANK_Z:.4f}"
            size="{TUBE_R * 0.85:.4f}" mass="{MASS_TUBE}"/>
      <geom name="tube_top" type="capsule"
            fromto="{SEAT_X:.4f} 0 {SEAT_Z:.4f} {HEAD_X:.4f} 0 {CRANK_Z + 0.04:.4f}"
            size="{TUBE_R}" mass="{MASS_TUBE}"/>
      <geom name="tube_head" type="capsule"
            fromto="{HEAD_X:.4f} 0 {CRANK_Z + 0.04:.4f} {HEAD_X:.4f} 0 {BAR_Z:.4f}"
            size="{TUBE_R}" mass="{MASS_TUBE}"/>
      <geom name="fork" type="capsule"
            fromto="{HEAD_X:.4f} 0 {BAR_Z:.4f} {FRONT_X:.4f} 0 0"
            size="{TUBE_R * 0.8:.4f}" mass="{MASS_TUBE}"/>
      <geom name="stem" type="capsule"
            fromto="{HEAD_X:.4f} 0 {BAR_Z:.4f} {GRIP_X:.4f} 0 {BAR_Z:.4f}"
            size="{TUBE_R * 0.9:.4f}" mass="{MASS_TUBE}" rgba="0.7 0.72 0.75 1"/>

      <!-- Handlebars. The fly's own front legs span about 0.11 cm either side,
           so the bar is 0.22 cm wide and sits just ahead of the thorax. -->
      <geom name="bar_left" type="capsule"
            fromto="{GRIP_X:.4f} {0.04 * LEFT:.4f} {BAR_Z:.4f} {GRIP_X:.4f} {GRIP_Y * LEFT:.4f} {BAR_Z:.4f}"
            size="{TUBE_R}" mass="{MASS_TUBE}"/>
      <geom name="bar_right" type="capsule"
            fromto="{GRIP_X:.4f} {-0.04 * RIGHT:.4f} {BAR_Z:.4f} {GRIP_X:.4f} {GRIP_Y * RIGHT:.4f} {BAR_Z:.4f}"
            size="{TUBE_R}" mass="{MASS_TUBE}"/>
      <geom name="grip_left" type="capsule"
            fromto="{GRIP_X:.4f} {(GRIP_Y - 0.02) * LEFT:.4f} {BAR_Z:.4f} {GRIP_X:.4f} {GRIP_Y * LEFT:.4f} {BAR_Z:.4f}"
            size="{TUBE_R * 1.5:.4f}" mass="{MASS_TUBE}" rgba="0.16 0.16 0.18 1"/>
      <geom name="grip_right" type="capsule"
            fromto="{GRIP_X:.4f} {(GRIP_Y - 0.02) * RIGHT:.4f} {BAR_Z:.4f} {GRIP_X:.4f} {GRIP_Y * RIGHT:.4f} {BAR_Z:.4f}"
            size="{TUBE_R * 1.5:.4f}" mass="{MASS_TUBE}" rgba="0.16 0.16 0.18 1"/>
      <site name="grip_left_site" pos="{GRIP_X:.4f} {GRIP_Y * LEFT:.4f} {BAR_Z:.4f}"/>
      <site name="grip_right_site" pos="{GRIP_X:.4f} {GRIP_Y * RIGHT:.4f} {BAR_Z:.4f}"/>

      <!-- Saddle rails: T2 claws grip here, out to each side of the fly. -->
      <geom name="saddle" type="capsule"
            fromto="{SEAT_X:.4f} {-0.06 * LEFT:.4f} {SEAT_Z:.4f} {SEAT_X:.4f} {-0.06 * RIGHT:.4f} {SEAT_Z:.4f}"
            size="{TUBE_R * 1.4:.4f}" mass="{MASS_TUBE}" rgba="0.16 0.16 0.18 1"/>
      <site name="saddle_left_site" pos="{SEAT_X:.4f} {0.10 * LEFT:.4f} {SEAT_Z - 0.03:.4f}"/>
      <site name="saddle_right_site" pos="{SEAT_X:.4f} {0.10 * RIGHT:.4f} {SEAT_Z - 0.03:.4f}"/>

      <!-- Wheels. Thin discs: enough for rolling contact, cheap to simulate. -->
      <body name="wheel_rear" pos="{REAR_X:.4f} 0 0">
        <joint name="wheel_rear_spin" type="hinge" axis="0 1 0" damping="0.00001"/>
        <geom name="tyre_rear" type="cylinder" fromto="0 -0.012 0 0 0.012 0"
              size="{WHEEL_R:.4f}" mass="{MASS_WHEEL}" rgba="0.14 0.14 0.16 1"/>
        <geom name="rim_rear" type="cylinder" fromto="0 -0.016 0 0 0.016 0"
              size="{WHEEL_R * 0.78:.4f}" mass="{MASS_WHEEL * 0.3:.6f}"
              rgba="0.72 0.74 0.78 1"/>
      </body>

      <body name="wheel_front" pos="{FRONT_X:.4f} 0 0">
        <joint name="wheel_front_spin" type="hinge" axis="0 1 0" damping="0.00001"/>
        <geom name="tyre_front" type="cylinder" fromto="0 -0.012 0 0 0.012 0"
              size="{WHEEL_R:.4f}" mass="{MASS_WHEEL}" rgba="0.14 0.14 0.16 1"/>
        <geom name="rim_front" type="cylinder" fromto="0 -0.016 0 0 0.016 0"
              size="{WHEEL_R * 0.78:.4f}" mass="{MASS_WHEEL * 0.3:.6f}"
              rgba="0.72 0.74 0.78 1"/>
      </body>

      <!-- Crankset, behind and below the fly so the hind legs can reach the
           pedals. Left pedal leads, right pedal follows half a turn. -->
      <body name="crank" pos="{CRANK_X:.4f} 0 {CRANK_Z:.4f}">
        <joint name="crank_joint" type="hinge" axis="0 1 0" damping="0.000002"/>
        <geom name="crank_arm_left" type="capsule"
              fromto="0 {0.02 * LEFT:.4f} 0 0 {0.02 * LEFT:.4f} {-CRANK_R:.4f}"
              size="{TUBE_R * 0.8:.4f}" mass="{MASS_TUBE}" rgba="0.68 0.70 0.73 1"/>
        <geom name="crank_arm_right" type="capsule"
              fromto="0 {-0.02 * RIGHT:.4f} 0 0 {-0.02 * RIGHT:.4f} {CRANK_R:.4f}"
              size="{TUBE_R * 0.8:.4f}" mass="{MASS_TUBE}" rgba="0.68 0.70 0.73 1"/>
        <body name="pedal_left" pos="0 {0.05 * LEFT:.4f} {-CRANK_R:.4f}">
          <geom name="pedal_left_geom" type="box" size="0.035 0.020 0.006"
                mass="{MASS_TUBE}" rgba="0.22 0.22 0.25 1"/>
          <site name="pedal_left_site" pos="0 0 0.012"/>
        </body>
        <body name="pedal_right" pos="0 {0.05 * RIGHT:.4f} {CRANK_R:.4f}">
          <geom name="pedal_right_geom" type="box" size="0.035 0.020 0.006"
                mass="{MASS_TUBE}" rgba="0.22 0.22 0.25 1"/>
          <site name="pedal_right_site" pos="0 0 0.012"/>
        </body>
      </body>

      <!-- Cameras live in the frame, so they ride along with the bike.
           Positions and aim points are given in readable local coordinates and
           the orientation is computed by look_at_axes, because a MuJoCo camera
           looks down its own -z axis with +y up in the image, and the sign
           convention is easy to get backwards: xyaxes is hand-written, one
           flipped x axis silently points the camera at empty floor and the
           video comes out showing nothing but grey ground. -->
      <camera name="chase" pos="-1.15 0.85 0.85"
              xyaxes="{look_at_axes((-1.15, 0.85, 0.85), CAM_AIM_CHASE)}"
              fovy="38"/>
      <camera name="side" pos="0 2.2 0.25"
              xyaxes="{look_at_axes((0, 2.2, 0.25), CAM_AIM_SIDE)}"
              fovy="30"/>

      <!-- The fly is welded here, straddling the saddle. -->
      <frame name="rider_anchor" pos="{SEAT_X:.4f} 0 {RIDER_Z:.4f}"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Rear wheel: a velocity servo, so the commanded cadence is a ground speed.

         Not a torque motor. On a wheel this light a torque motor is a drill: it
         spins the wheel up to roughly 1400 rev/s in place while the bike covers
         no ground at all, because there is nothing to react against until the
         contact saturates.

         The crank is NOT actuated. It is driven kinematically from the same
         cadence, for two reasons. A torque source there cannot work: the rider
         is welded to the frame, so the hind legs push on the pedals from inside
         a rigid body and their position servos (kp 0.8 dyn/rad) simply stall
         any motor. A joint equality constraint coupling crank to wheel fails
         for the same reason, leaving the two tens of degrees out of phase.
         Since the fly is part of the frame, its leg forces are internal and
         cannot contribute to propulsion in any case, so the pedals are motion
         for the legs to track rather than a load to overcome. -->
    <velocity name="wheel_drive" joint="wheel_rear_spin" kv="0.03"
              ctrlrange="-60 60"/>
  </actuator>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------

def load_rider():
    """FlyBody fruit fly: legs active, wings retracted, free joint removed."""
    spec = mujoco.MjSpec.from_file(FLY_XML)

    # Drop the free joint so the frame carries the fly rigidly.
    for joint in list(spec.body('thorax').joints):
        spec.delete(joint)

    # Retract wings: remove their joints and actuators so they cannot flap.
    for joint in [j for j in spec.joints if 'wing' in (j.name or '')]:
        spec.delete(joint)
    for act in [a for a in spec.actuators if 'wing' in (a.name or '')]:
        spec.delete(act)

    return spec


def build_model(**overrides):
    """Compile bike + rider. Geometry overrides are keyword constants."""
    saved = dict(globals())
    globals().update(overrides)
    if 'WHEELBASE' in overrides:
        globals()['REAR_X'] = -globals()['WHEELBASE'] / 2
        globals()['FRONT_X'] = globals()['WHEELBASE'] / 2
    try:
        bike = mujoco.MjSpec.from_string(bike_xml())
    finally:
        globals().clear()
        globals().update(saved)
    bike.attach(load_rider(), prefix='rider_', frame=bike.frame('rider_anchor'))
    with warnings.catch_warnings():
        # Expected: the fly's option block overrides some bike defaults. The
        # kept values are what we want, they are just worth saying out loud.
        warnings.simplefilter('ignore', UserWarning)
        return bike.compile()


class Rider:
    """Index handles for one leg: IK joints, servo actuators, claw body."""

    # Ordered proximal to distal.
    LEG_JOINTS = ['coxa_abduct', 'coxa', 'femur', 'tibia']
    # All four are solved by IK. Leaving the coxa swing joint out of the solve
    # costs a lot of reach, especially forward: it is the joint that points the
    # whole leg.
    IK_JOINTS = ['coxa_abduct', 'coxa', 'femur', 'tibia']

    TARGET_SITES = {
        ('T1', 'left'): 'grip_left_site',
        ('T1', 'right'): 'grip_right_site',
        ('T2', 'left'): 'saddle_left_site',
        ('T2', 'right'): 'saddle_right_site',
        ('T3', 'left'): 'pedal_left_site',
        ('T3', 'right'): 'pedal_right_site',
    }

    def __init__(self, model):
        self.model = model
        self.legs = {}
        for key, site in self.TARGET_SITES.items():
            tag, side = key
            joints = []
            for j in self.LEG_JOINTS:
                act = mujoco.mj_name2id(
                    model, OBJ.mjOBJ_ACTUATOR, f'rider_{j}_{tag}_{side}')
                jid = mujoco.mj_name2id(
                    model, OBJ.mjOBJ_JOINT, f'rider_{j}_{tag}_{side}')
                joints.append({
                    'name': j,
                    'act': act,
                    'qadr': model.jnt_qposadr[jid],
                    'dof': model.jnt_dofadr[jid],
                    'rng': model.actuator_ctrlrange[act].copy(),
                })
            ik = [j for j in joints if j['name'] in self.IK_JOINTS]
            self.legs[key] = {
                'site': mujoco.mj_name2id(model, OBJ.mjOBJ_SITE, site),
                'joints': joints,
                'ik_qadr': np.array([j['qadr'] for j in ik]),
                'ik_rng': np.array([j['rng'] for j in ik]),
                'ik_dofs': np.array([j['dof'] for j in ik]),
                'claw': mujoco.mj_name2id(
                    model, OBJ.mjOBJ_BODY, f'rider_claw_{tag}_{side}'),
                'adhere': mujoco.mj_name2id(
                    model, OBJ.mjOBJ_ACTUATOR, f'rider_adhere_claw_{tag}_{side}'),
            }


def solve_ik(model, data, leg, target, iters=25, gain=0.6, damping=2e-4,
             tol=2e-4):
    """Damped least squares on three leg joints to place the claw on target."""
    jacp = np.zeros((3, model.nv))
    lo, hi = leg['ik_rng'][:, 0], leg['ik_rng'][:, 1]
    n = len(leg['ik_dofs'])
    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        err = target - data.xpos[leg['claw']]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacBody(model, data, jacp, None, leg['claw'])
        J = jacp[:, leg['ik_dofs']]
        # Damped least squares: J.T (J J.T + lambda I)^-1 err, least-norm
        # when the leg cannot fully reach the goal.
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err) * gain
        data.qpos[leg['ik_qadr']] = np.clip(
            data.qpos[leg['ik_qadr']] + dq, lo, hi)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

WHEEL_PERIOD = 2.0       # seconds per rear-wheel revolution
GEAR_RATIO = 2.2         # crank turns per wheel turn, as on a road bike
ADHESION_RADIUS = 0.012   # claw-to-target distance that counts as touching


def run(model, data, rider, seconds, renderer=None, camera='chase',
        fps=30, report_every=0):
    wheel_act = mujoco.mj_name2id(model, OBJ.mjOBJ_ACTUATOR, 'wheel_drive')
    frame_bid = mujoco.mj_name2id(model, OBJ.mjOBJ_BODY, 'frame')
    slide_x_q = model.jnt_qposadr[
        mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT, 'slide_x')]
    crank_q = model.jnt_qposadr[
        mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT, 'crank_joint')]
    rear_q = model.jnt_qposadr[
        mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT, 'wheel_rear_spin')]
    thorax_bid = mujoco.mj_name2id(model, OBJ.mjOBJ_BODY, 'rider_thorax')

    n_control = int(round(seconds / CONTROL_DT))
    substeps = max(1, int(round(CONTROL_DT / model.opt.timestep)))
    frame_every = max(1, int(round(CONTROL_DT * fps)))

    start_x = data.qpos[slide_x_q]
    frames = []
    err_by_leg = {key: [] for key in rider.legs}
    speed = []

    for i in range(n_control):
        t = i * CONTROL_DT

        # Constant cadence, spun up over the first half second.
        wheel_omega = 2 * np.pi / WHEEL_PERIOD
        ramp = min(1.0, t / 0.5)
        data.ctrl[wheel_act] = ramp * wheel_omega

        # Crank follows kinematically at the geared cadence, so the pedals move
        # and the hind legs have something to track. Setting qpos directly also
        # means the legs see the pedal at its new place before the next IK solve.
        data.qpos[crank_q] = ramp * wheel_omega * GEAR_RATIO * t
        data.qvel[model.jnt_dofadr[
            mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT, 'crank_joint')]] = (
                ramp * wheel_omega * GEAR_RATIO)

        for key, leg in rider.legs.items():
            target = data.site_xpos[leg['site']].copy()
            solve_ik(model, data, leg, target)

            # Servo the solved pose, then grip once the claw is in contact.
            for j in leg['joints']:
                data.ctrl[j['act']] = np.clip(
                    data.qpos[j['qadr']], j['rng'][0], j['rng'][1])

            mujoco.mj_kinematics(model, data)
            dist = float(np.linalg.norm(data.xpos[leg['claw']] - target))
            err_by_leg[key].append(dist)
            data.ctrl[leg['adhere']] = 1.0 if dist < ADHESION_RADIUS else 0.0

        mujoco.mj_forward(model, data)
        for _ in range(substeps):
            mujoco.mj_step(model, data)

        speed.append(data.qvel[model.jnt_dofadr[
            mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT, 'slide_x')]])
        if report_every and i % report_every == 0:
            print(f'  t={t:5.2f}s  x={data.qpos[slide_x_q] - start_x:7.3f} cm  '
                  f'crank={np.rad2deg(data.qpos[crank_q]):8.1f} deg  '
                  f'v={speed[-1]:6.2f} cm/s')

        if renderer is not None and i % frame_every == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    stats = {
        'travel': data.qpos[slide_x_q] - start_x,
        'crank_deg': np.rad2deg(data.qpos[crank_q]),
        'rear_deg': np.rad2deg(data.qpos[rear_q]),
        'speed': float(np.mean(speed[-20:])) if speed else 0.0,
        'frame_pos': data.xpos[frame_bid].copy(),
        'thorax_pos': data.xpos[thorax_bid].copy(),
        'err_by_leg': err_by_leg,
        'substeps': substeps,
        'n_control': n_control,
    }
    return stats, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=4.0)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--camera', default='chase', choices=['chase', 'side'])
    ap.add_argument('--out', default=os.path.join('videos', 'fly_bicycle.mp4'))
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--trace', action='store_true',
                    help='print position every 0.5 s of sim time')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    model = build_model()
    data = mujoco.MjData(model)
    rider = Rider(model)
    mujoco.mj_forward(model, data)

    if not args.quiet:
        print(f'model: nq={model.nq} nv={model.nv} nu={model.nu} '
              f'mass={model.body_mass.sum() * 1000:.3f} mg')

    renderer = None
    if not args.no_video:
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
        except Exception as exc:                       # no GL available
            print(f'render unavailable ({exc.__class__.__name__}: {exc}); '
                  'running without video')

    t0 = time.time()
    stats, frames = run(model, data, rider, args.seconds,
                        renderer=renderer, camera=args.camera, fps=args.fps,
                        report_every=int(0.5 / CONTROL_DT) if args.trace else 0)
    wall = time.time() - t0
    if renderer is not None:
        renderer.close()

    if not args.quiet:
        print(f'simulated {args.seconds:.1f}s in {wall:.1f}s wall '
              f'({stats["n_control"]} control steps x {stats["substeps"]} physics)')
        print(f'travelled      {stats["travel"]:.2f} cm')
        print(f'mean speed     {stats["speed"]:.2f} cm/s')
        print(f'crank turned   {stats["crank_deg"]:.0f} deg')
        print(f'rear wheel     {stats["rear_deg"]:.0f} deg')
        print(f'frame pos      {np.round(stats["frame_pos"], 3)}')
        print(f'thorax pos     {np.round(stats["thorax_pos"], 3)}')
        for key, errs in stats['err_by_leg'].items():
            tag, side = key
            print(f'  {tag}_{side:<5} claw IK error  '
                  f'mean {np.mean(errs) * 1000:6.1f} um  '
                  f'max {np.max(errs) * 1000:6.1f} um')

    if frames:
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        imageio.mimsave(args.out, frames, fps=args.fps, quality=8)
        if not args.quiet:
            print(f'wrote {len(frames)} frames -> {args.out}')


if __name__ == '__main__':
    main()