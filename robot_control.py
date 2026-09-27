# -*- coding: utf-8 -*-
"""
Robot and Sensor Control Module.
Handles DJI RoboMaster EP hardware interaction (Gimbal, Chassis Mecanum, ToF sensor),
gimbal-only 4-direction scanning, and fine-tuning cell auto-recentering.
"""

import math
import time
from collections import defaultdict

import numpy as np

from config import (
    DEADBAND_M,
    DELTA,
    DIRECTIONS,
    ENABLE_LONGITUDINAL_RECENTER,
    ENABLE_RECENTER,
    GIMBAL_SPEED,
    GRID_SIZE_M,
    MAX_SHIFT_LATERAL_M,
    MAX_SHIFT_LONGITUDINAL_M,
    RECENTER_SPEED,
    ROT_SPEED,
    SAFE_FRONT_DIST_MM,
    SAFE_MAX_MM,
    SAFE_MIN_MM,
    SPEED,
    TARGET_WALL_DIST_MM,
    TOF_ID,
    WALL_THRESHOLD_MM,
    adjacent,
    in_bounds,
)
from navigation import is_wall_distance, sim_has_wall_between
import state

try:
    from robomaster import robot
except ImportError:
    robot = None

# ─── Noise filtering state ───────────────────────────────────────────
# Per-direction EMA history for temporal smoothing across consecutive scans.
# Key: (cell, direction)  Value: smoothed distance (mm)
_ema_history: dict = defaultdict(lambda: None)
EMA_ALPHA = 0.4          # Weight for new reading (0→trust old, 1→trust new)
NUM_SAMPLES = 9          # More samples → better outlier rejection
SAMPLE_INTERVAL_S = 0.03 # 30ms between samples (sensor rate ~100Hz@freq=10)
LIGHT_NOISE_BAND_MM = 80 # Borderline zone around WALL_THRESHOLD for re-confirm


def sub_tof_handler(info):
    """Callback to receive distance measurement from RoboMaster ToF sensor."""
    if isinstance(info, (list, tuple)) and len(info) >= TOF_ID:
        state.current_distance = info[TOF_ID - 1]
    elif not isinstance(info, (list, tuple)):
        state.current_distance = info


def sub_attitude_handler(info):
    """Callback to receive chassis attitude (yaw, pitch, roll) in degrees."""
    if isinstance(info, (list, tuple)) and len(info) >= 1:
        yaw = float(info[0])
        state.current_yaw = yaw
        if state.initial_yaw is None:
            state.initial_yaw = yaw


def sub_position_handler(info):
    """Callback to receive chassis position (x, y, z in meters/deg)."""
    if isinstance(info, (list, tuple)) and len(info) >= 2:
        state.chassis_pos_x = float(info[0])
        state.chassis_pos_y = float(info[1])


def _normalize_angle(deg):
    """Normalize angle to [-180, 180] degrees."""
    while deg > 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return deg


def distance_in_simulation(cell, direction):
    """Simulate ToF distance to border foam wall (TARGET_WALL_DIST_MM) or open cell (900mm)."""
    has_wall = sim_has_wall_between(cell, direction)
    return TARGET_WALL_DIST_MM if has_wall else 900


def _iqr_filter(samples):
    """
    Remove outliers from a list of distance samples using the IQR method.
    Returns only the inliers. If too few remain, returns the full list.
    This is critical for filtering noise spikes caused by lighting changes.
    """
    if len(samples) < 4:
        return samples
    arr = np.array(samples, dtype=float)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    # Use 1.5× IQR as fence — standard Tukey outlier rule
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inliers = arr[(arr >= lower) & (arr <= upper)]
    return inliers.tolist() if len(inliers) >= 3 else samples


def _ema_smooth(key, raw_value):
    """
    Apply Exponential Moving Average to smooth out temporal noise.
    This filters gradual lighting changes that cause drifting ToF readings.
    """
    prev = _ema_history[key]
    if prev is None:
        _ema_history[key] = raw_value
        return raw_value
    smoothed = EMA_ALPHA * raw_value + (1 - EMA_ALPHA) * prev
    _ema_history[key] = smoothed
    return smoothed


def read_distance_with_gimbal(ep_gimbal, cell, current_heading, target_dir, sim_mode):
    """
    Rotate ONLY the gimbal to target_dir relative to current_heading,
    read the ToF distance, and return distance in mm.
    Chassis does NOT move!

    Noise-robust pipeline:
      1. Collect NUM_SAMPLES raw readings
      2. Discard readings ≤ 30mm (floor/sensor noise)
      3. IQR outlier rejection (removes lighting-induced spikes)
      4. Median of remaining inliers
      5. EMA temporal smoothing (handles gradual light changes)
    """
    if sim_mode:
        time.sleep(0.12)
        return distance_in_simulation(cell, target_dir)

    # Relative turn index: 0=front, 1=right (clockwise), 2=back, 3=left (counter-clockwise)
    diff = (DIRECTIONS.index(target_dir) - DIRECTIONS.index(current_heading)) % 4
    yaw_map = {0: 0, 1: 90, 2: 180, 3: -90}
    target_yaw = yaw_map[diff]

    # pitch=3 degrees: low enough for 30cm foam wall, high enough to avoid floor reflection
    if ep_gimbal:
        ep_gimbal.moveto(pitch=3, yaw=target_yaw, pitch_speed=GIMBAL_SPEED, yaw_speed=GIMBAL_SPEED).wait_for_completed()

    # Allow sensor to settle after gimbal rotation
    time.sleep(0.20)

    # Step 1: Collect raw samples
    samples = []
    for _ in range(NUM_SAMPLES):
        samples.append(state.current_distance)
        time.sleep(SAMPLE_INTERVAL_S)

    # Step 2: Discard invalid readings (floor reflection, sensor error)
    valid = [s for s in samples if s > 30]
    if not valid:
        return float(samples[-1])

    # Step 3: IQR outlier rejection (removes lighting-induced spikes)
    filtered = _iqr_filter(valid)

    # Step 4: Median of inliers
    median_val = float(np.median(filtered))

    # Step 5: EMA temporal smoothing
    ema_key = (cell, target_dir)
    smoothed = _ema_smooth(ema_key, median_val)

    return smoothed


def turn_to_direction(ep_chassis, ep_gimbal, current, target, sim_mode):
    """Turn the chassis from current direction to target direction and keep gimbal centered."""
    if current == target:
        return
    if sim_mode:
        time.sleep(0.12)
        return

    # Use IMU closed-loop yaw if initial yaw has been calibrated
    # Confirmed conventions (from debug):
    #   chassis.move(z>0) = CCW (left),  z<0 = CW (right)
    #   IMU yaw: CW = increases,  CCW = decreases  (opposite to chassis!)
    # heading_offsets use IMU convention; z = -yaw_err converts to chassis convention
    heading_offsets = {"NORTH": 0.0, "EAST": 90.0, "SOUTH": 180.0, "WEST": -90.0}
    if state.initial_yaw is not None:
        target_yaw = _normalize_angle(state.initial_yaw + heading_offsets[target])
        yaw_err = _normalize_angle(target_yaw - state.current_yaw)
        z_cmd = round(-yaw_err, 1)
        print(f"[TURN DEBUG IMU] {current}->{target} | initial_yaw={state.initial_yaw:.1f} current_yaw={state.current_yaw:.1f} target_yaw={target_yaw:.1f} yaw_err={yaw_err:.1f} z_cmd={z_cmd}")
        if abs(yaw_err) > 1.5:
            ep_chassis.move(x=0, y=0, z=z_cmd, z_speed=ROT_SPEED).wait_for_completed()
            time.sleep(0.08)
            # Fine-tuning pass
            yaw_err2 = _normalize_angle(target_yaw - state.current_yaw)
            print(f"[TURN DEBUG FT] after_yaw={state.current_yaw:.1f} yaw_err2={yaw_err2:.1f}")
            if abs(yaw_err2) > 5.0:
                z_cmd2 = round(-yaw_err2, 1)
                print(f"[TURN DEBUG FT] fine-tune z_cmd2={z_cmd2}")
                ep_chassis.move(x=0, y=0, z=z_cmd2, z_speed=ROT_SPEED).wait_for_completed()
    else:
        turn = (DIRECTIONS.index(target) - DIRECTIONS.index(current)) % 4
        print(f"[TURN DEBUG FALLBACK] {current}->{target} turn={turn}")
        if turn == 1:
            ep_chassis.move(x=0, y=0, z=-90, z_speed=ROT_SPEED).wait_for_completed()   # CW = เลี้ยวขวา
        elif turn == 2:
            ep_chassis.move(x=0, y=0, z=-90, z_speed=ROT_SPEED).wait_for_completed()
            ep_chassis.move(x=0, y=0, z=-90, z_speed=ROT_SPEED).wait_for_completed()
        else:
            ep_chassis.move(x=0, y=0, z=90, z_speed=ROT_SPEED).wait_for_completed()    # CCW = เลี้ยวซ้าย

    if ep_gimbal and not sim_mode:
        ep_gimbal.recenter().wait_for_completed()


def move_one_cell(ep_chassis, sim_mode):
    """
    Move forward exactly 1 grid cell (0.60m) using smooth velocity control (drive_speed).
    Uses tight time-distance coupling and front-wall proximity braking to completely
    prevent overshooting into walls.
    """
    if sim_mode:
        time.sleep(0.18)
        return True

    # Record starting odometry position
    start_x = state.chassis_pos_x
    start_y = state.chassis_pos_y

    # Target distance: brake at ~0.575m so remaining momentum glides smoothly to 0.60m
    target_dist = GRID_SIZE_M - 0.025
    # Strict duration limit: at SPEED=0.30 m/s, traveling 0.575m takes ~1.92s.
    # Never allow running up to 2.8s which caused the 82cm overshoot!
    max_duration = (target_dist / SPEED) + 0.05

    # Start smooth forward motion
    ep_chassis.drive_speed(x=SPEED, y=0, z=0)

    emergency_stop = False
    front_wall_reached = False
    start_time = time.time()
    traveled = 0.0
    elapsed = 0.0

    while not state.stop_requested:
        time.sleep(0.015)
        # Compute distance traveled from wheel encoders
        traveled = math.hypot(state.chassis_pos_x - start_x, state.chassis_pos_y - start_y)
        elapsed = time.time() - start_time

        # 1. Emergency collision stop threshold
        if 40 <= state.current_distance <= SAFE_FRONT_DIST_MM:
            emergency_stop = True
            break

        # 2. Front wall proximity brake: if moving into a cell facing a wall,
        # and ToF shows we have reached the cell center (~140-160mm from wall),
        # brake smoothly immediately to prevent crashing into foam.
        if elapsed > 0.45 and 45 <= state.current_distance <= 140:
            front_wall_reached = True
            break

        # 3. Stop driving when target distance reached or strict timeout reached
        if traveled >= target_dist or elapsed >= max_duration:
            break

    # Stop motors smoothly - velocity brake to 0, NEVER reverses!
    ep_chassis.drive_speed(x=0, y=0, z=0)
    time.sleep(0.12)

    stop_reason = "EMERGENCY" if emergency_stop else (
        "FRONT_WALL" if front_wall_reached else (
            "ODOMETRY" if traveled >= target_dist else "TIMEOUT"
        )
    )
    print(f"[MOVE DEBUG] dist={traveled*100:.1f}cm elapsed={elapsed:.2f}s reason={stop_reason} front_tof={state.current_distance:.0f}mm")

    if emergency_stop:
        print(f"⚠ หยุดฉุกเฉิน! กำแพงข้างหน้า ({state.current_distance:.0f}mm < {SAFE_FRONT_DIST_MM}mm)")
        return False

    return True


def scan_4_directions(ep_chassis, ep_gimbal, position, current_heading, sim_mode, dashboard, step):
    """
    Stop robot, rotate ONLY the gimbal to scan all 4 cardinal directions,
    detect foam walls on cell borders, record open passages,
    and recenter the gimbal back to 0 degrees.
    Chassis DOES NOT rotate!

    Includes borderline re-confirmation: if a reading lands near the
    wall/open threshold (±LIGHT_NOISE_BAND_MM), a second scan is performed
    and the more conservative (shorter) distance is used. This prevents
    lighting noise from causing the robot to drive into a wall.
    """
    state.discovered_cells.add(position)
    state.visited_cells.add(position)

    readings = {}
    x, y = position

    if dashboard:
        dashboard.log(f"-> หยุดที่ {position} กิมบอลหมุนสแกน 4 ทิศ (หุ่นนิ่ง)...")

    # Gimbal rotates relative: front -> right -> back -> left -> center
    diff_order = [0, 1, 2, 3]
    for diff in diff_order:
        if state.stop_requested:
            break

        target_dir = DIRECTIONS[(DIRECTIONS.index(current_heading) + diff) % 4]
        dist = read_distance_with_gimbal(ep_gimbal, position, current_heading, target_dir, sim_mode)

        # ── Borderline re-confirmation ──
        # If reading is in the ambiguous zone near WALL_THRESHOLD, re-read
        # and take the shorter (more conservative) value to avoid driving
        # through what might actually be a wall.
        lower_band = WALL_THRESHOLD_MM - LIGHT_NOISE_BAND_MM
        upper_band = WALL_THRESHOLD_MM + LIGHT_NOISE_BAND_MM
        if not sim_mode and lower_band <= dist <= upper_band:
            if dashboard:
                dashboard.log(f"   ⚠ ค่าใกล้ขอบ ({dist:.0f}mm) → อ่านซ้ำเพื่อ confirm...")
            time.sleep(0.10)
            dist2 = read_distance_with_gimbal(ep_gimbal, position, current_heading, target_dir, sim_mode)
            # Use the shorter reading (conservative — assume wall)
            dist = min(dist, dist2)
            if dashboard:
                dashboard.log(f"   ✓ confirm: {dist:.0f}mm (เลือกค่าน้อยกว่า)")

        readings[target_dir] = dist
        state.current_distance = dist

        nxt = adjacent(position, target_dir)
        # Any direction leading outside the grid bounds is guaranteed to be a boundary wall of the maze
        if not in_bounds(nxt):
            has_wall = True
        else:
            has_wall = is_wall_distance(dist)

        if has_wall:
            if target_dir == "NORTH":
                state.detected_h_walls.add((x, y))
            elif target_dir == "SOUTH":
                state.detected_h_walls.add((x, y - 1))
            elif target_dir == "EAST":
                state.detected_v_walls.add((x, y))
            elif target_dir == "WEST":
                state.detected_v_walls.add((x - 1, y))
        else:
            if in_bounds(nxt):
                state.discovered_cells.add(nxt)

        dist_str = f"{dist:.0f} mm" if has_wall else f"{dist:.0f} mm (ที่โล่ง/ไกล)"
        border_desc = "กำแพงโฟม (WALL)" if has_wall else "ช่องเปิด (OPEN)"
        print(f"[{position} Gimbal {target_dir}] ToF: {dist_str} -> {border_desc}")

        if dashboard:
            dashboard.update(position, current_heading, step, f"Gimbal สแกนทิศ {target_dir}", extra_readings=readings, gimbal_dir=target_dir)
            dashboard.log(f"   [Gimbal {target_dir}] ระยะ {dist_str} -> {border_desc}")

    # Recenter gimbal back to center (0 degrees)
    if ep_gimbal and not sim_mode:
        ep_gimbal.recenter().wait_for_completed()

    if dashboard:
        dashboard.update(position, current_heading, step, "Gimbal Recenter เรียบร้อย", extra_readings=readings, gimbal_dir=None)

    return readings


def recenter_in_cell(ep_chassis, ep_gimbal, readings, position, current_heading, sim_mode, dashboard):
    """
    Recenter robot inside the grid cell using Mecanum strafing.
    Adjusts both lateral (left/right) and longitudinal (forward/backward)
    position to keep the robot centered between walls.

    RoboMaster chassis coordinate conventions:
      - x > 0: forward,  x < 0: backward
      - y > 0: strafe left,  y < 0: strafe right
      - z > 0: CCW,  z < 0: CW
    """
    if not ENABLE_RECENTER or state.stop_requested or not readings:
        return 0.0, 0.0

    # Determine left, right, front, and back directions relative to current heading
    h_idx = DIRECTIONS.index(current_heading)
    front_dir = current_heading
    right_dir = DIRECTIONS[(h_idx + 1) % 4]
    back_dir = DIRECTIONS[(h_idx + 2) % 4]
    left_dir = DIRECTIONS[(h_idx - 1) % 4]

    dist_front = readings.get(front_dir, 9999)
    dist_back = readings.get(back_dir, 9999)
    dist_left = readings.get(left_dir, 9999)
    dist_right = readings.get(right_dir, 9999)

    wall_front = is_wall_distance(dist_front)
    wall_back = is_wall_distance(dist_back)
    wall_left = is_wall_distance(dist_left)
    wall_right = is_wall_distance(dist_right)

    # 1. Lateral adjustment (Chassis Y: +y = Right, -y = Left in RoboMaster SDK)
    chassis_y_mm = 0.0
    if wall_left and wall_right:
        # Balanced directly between left and right walls
        # If dist_left < dist_right: too close to left wall -> move right (+y)
        # If dist_left > dist_right: too close to right wall -> move left (-y)
        chassis_y_mm = (dist_right - dist_left) / 2.0
    elif wall_left:
        # Single left wall: only nudge away if closer than SAFE_MIN_MM (115mm) or too far (>225mm)
        if dist_left < SAFE_MIN_MM:
            chassis_y_mm = SAFE_MIN_MM - dist_left
        elif dist_left > SAFE_MAX_MM:
            chassis_y_mm = SAFE_MAX_MM - dist_left
    elif wall_right:
        # Single right wall: only nudge away if closer than SAFE_MIN_MM (115mm) or too far (>225mm)
        if dist_right < SAFE_MIN_MM:
            chassis_y_mm = -(SAFE_MIN_MM - dist_right)
        elif dist_right > SAFE_MAX_MM:
            chassis_y_mm = -(SAFE_MAX_MM - dist_right)

    # 2. Longitudinal adjustment (Chassis X: +x = Forward, -x = Backward)
    # ปิดการขยับหน้า-หลังตอน recenter ตามค่า ENABLE_LONGITUDINAL_RECENTER เพื่อไม่ให้หุ่นถอยหลังสะสม noise
    chassis_x_mm = 0.0
    if ENABLE_LONGITUDINAL_RECENTER:
        if wall_front and wall_back:
            # Balanced directly between front and back walls
            chassis_x_mm = (dist_front - dist_back) / 2.0
        elif wall_front:
            chassis_x_mm = dist_front - TARGET_WALL_DIST_MM
        elif wall_back:
            chassis_x_mm = -(dist_back - TARGET_WALL_DIST_MM)

    chassis_x = chassis_x_mm / 1000.0
    chassis_y = chassis_y_mm / 1000.0

    # Clamp shifts to max limits
    chassis_y = max(-MAX_SHIFT_LATERAL_M, min(MAX_SHIFT_LATERAL_M, chassis_y))
    chassis_x = max(-MAX_SHIFT_LONGITUDINAL_M, min(MAX_SHIFT_LONGITUDINAL_M, chassis_x))

    if abs(chassis_y) < DEADBAND_M:
        chassis_y = 0.0
    if abs(chassis_x) < DEADBAND_M:
        chassis_x = 0.0

    if chassis_x == 0.0 and chassis_y == 0.0:
        if dashboard:
            dashboard.log("-> Recenter: กึ่งกลางช่องได้ระดับแล้ว")
        print("-> Recenter: กึ่งกลางช่องได้ระดับแล้ว")
        return 0.0, 0.0

    # Convert chassis (x, y) back to world (shift_x, shift_y) for reporting:
    # World: +X = EAST, -X = WEST, +Y = NORTH, -Y = SOUTH
    # Chassis: +x = forward, +y = right
    if current_heading == "NORTH":
        shift_x_m, shift_y_m = chassis_y, chassis_x
    elif current_heading == "EAST":
        shift_x_m, shift_y_m = chassis_x, -chassis_y
    elif current_heading == "SOUTH":
        shift_x_m, shift_y_m = -chassis_y, -chassis_x
    else:  # WEST
        shift_x_m, shift_y_m = -chassis_x, chassis_y

    msg = f"-> Recenter: ปรับจุดกึ่งกลาง (dx={shift_x_m*100:+.1f}cm, dy={shift_y_m*100:+.1f}cm)"
    if dashboard:
        dashboard.log(msg)
    print(msg)

    if sim_mode:
        time.sleep(0.12)
    else:
        ep_chassis.move(x=round(chassis_x, 3), y=round(chassis_y, 3), z=0, xy_speed=RECENTER_SPEED).wait_for_completed()
        if ep_gimbal:
            ep_gimbal.recenter().wait_for_completed()
        time.sleep(0.15)

    return shift_x_m, shift_y_m
