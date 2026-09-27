# -*- coding: utf-8 -*-
"""
Configuration and Constants Module for RoboMaster SLAM.
Contains grid settings, robot speed parameters, sensor thresholds,
and directional maps.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths configuration (Outputs are saved outside the 'code' folder)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.dirname(CODE_DIR)

# Grid & robot exploration configuration
GRID_W, GRID_H = 4, 5
GRID_SIZE_M = 0.60
SPEED = 0.30            # Smooth forward velocity speed (m/s)
ROT_SPEED = 50          # Precise turning speed (deg/s) - retains 90° precision
RECENTER_SPEED = 0.18   # Smooth strafe recentering (m/s)
GIMBAL_SPEED = 240      # Fast & smooth gimbal yaw speed (deg/s)
TOF_ID = 1
WALL_THRESHOLD_MM = 550 # Distance <= 550mm indicates a foam wall on immediate cell border (open passages >= 700mm)
TARGET_WALL_DIST_MM = 170 # True center distance from sensor to border wall in 60cm cell (~165-175mm)
SAFE_MIN_MM = 115       # Minimum safe distance to a single side wall (mm)
SAFE_MAX_MM = 225       # Maximum safe distance to a single side wall (mm)
SAFE_FRONT_DIST_MM = 85 # Emergency collision stop threshold (front wall at cell center is ~170mm, bumper is ~65mm closer)
MAX_SHIFT_LATERAL_M = 0.03      # Max lateral shift (3cm) – gentle nudge, prevents wheel slip
MAX_SHIFT_LONGITUDINAL_M = 0.03 # Max longitudinal shift (3cm)
DEADBAND_M = 0.020              # Deadband tolerance (2.0cm) – ignores minor offsets to prevent jitter
ENABLE_RECENTER = True              # เปิดระบบ Recenter ปรับกึ่งกลางช่อง
ENABLE_LONGITUDINAL_RECENTER = False # ปิดการปรับหน้า-หลัง เพื่อไม่ให้สะสม noise ในทิศทางเดินหน้า (ใช้ wheel odometry เดินตรงเป๊ะอยู่แล้ว)



DIRECTIONS = ["NORTH", "EAST", "SOUTH", "WEST"]
DELTA = {"NORTH": (0, 1), "EAST": (1, 0), "SOUTH": (0, -1), "WEST": (-1, 0)}
SYMBOL = {"NORTH": "^", "EAST": ">", "SOUTH": "v", "WEST": "<"}

# Simulated foam walls on cell borders (used only with --sim)
# horizontal_walls: wall between (x, y) and (x, y+1), stored as (x, y)
# vertical_walls: wall between (x, y) and (x+1, y), stored as (x, y)
SIM_H_WALLS = {(x, 0) for x in range(1, GRID_W + 1)} | {(x, GRID_H) for x in range(1, GRID_W + 1)}
SIM_V_WALLS = {(0, y) for y in range(1, GRID_H + 1)} | {(GRID_W, y) for y in range(1, GRID_H + 1)}
# Sample interior foam walls in simulation
SIM_H_WALLS |= {(1, 2), (2, 4), (3, 3)}
SIM_V_WALLS |= {(2, 1), (2, 2), (3, 4)}

# Ground Truth Foam Walls for Map Accuracy comparison (Ajarn Sahapong's maze)
GROUND_TRUTH_H_WALLS = set(SIM_H_WALLS)
GROUND_TRUTH_V_WALLS = set(SIM_V_WALLS)


def in_bounds(cell):
    """Check if the cell coordinate (x, y) is within grid bounds."""
    return 1 <= cell[0] <= GRID_W and 1 <= cell[1] <= GRID_H


def adjacent(cell, direction):
    """Return the adjacent cell coordinate in the given cardinal direction."""
    dx, dy = DELTA[direction]
    return cell[0] + dx, cell[1] + dy
