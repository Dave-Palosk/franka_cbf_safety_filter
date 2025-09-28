#!/usr/bin/env python3

import os
import yaml
import random
import numpy as np
import pinocchio as pin
import shutil

# --- Configuration ---
N_OBSTACLE_VARIATIONS = 100
N_PARAMETERS_VARIATIONS = 1
NUM_SCENARIOS = N_OBSTACLE_VARIATIONS * N_PARAMETERS_VARIATIONS * N_PARAMETERS_VARIATIONS # Total number of scenarios to generate

OUTPUT_DIR = "config/generated_scenarios"

RELEVANCE_DISTANCE_THRESHOLD = 0.2  # meters
MAX_TRIES_PER_OBSTACLE = 200

# --- Pinocchio Model Setup for Collision Checking ---
URDF_FILENAME = "fr3_robot.urdf"
package_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH = os.path.join(package_directory, "include", "urdf", URDF_FILENAME)
try:
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    print("Pinocchio model loaded successfully for collision checking.")
except Exception as e:
    print(f"Error: Could not load Pinocchio model from {URDF_PATH}. Cannot perform collision checks.")
    print(f"Details: {e}")
    exit()

# Define the robot's fixed initial joint configuration for checking start collisions
Q_INITIAL= np.zeros(model.nq)
Q_INITIAL[:7] = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

# Define the robot's control links (must match HOCBF_v1.py)
LINKS_DEF = [
    {'name': 'link1', 'start_frame_name': 'fr3_link0', 'end_frame_name': 'fr3_link1', 'radius': 0.08},
    {'name': 'link2', 'start_frame_name': 'fr3_link2', 'end_frame_name': 'fr3_link3', 'radius': 0.06},
    {'name': 'joint4',   'start_frame_name': 'fr3_link4', 'end_frame_name': 'fr3_link5_offset1', 'radius': 0.065},
    {'name': 'forearm1',   'start_frame_name': 'fr3_link5_offset2', 'end_frame_name': 'fr3_link5_offset3', 'radius': 0.035},
    {'name': 'forearm2',   'start_frame_name': 'fr3_link5_offset3', 'end_frame_name': 'fr3_link5', 'radius': 0.05},
    {'name': 'wrist',     'start_frame_name': 'fr3_link7', 'end_frame_name': 'fr3_hand',  'radius': 0.055},
    {'name': 'hand',     'start_frame_name': 'fr3_hand_offset1', 'end_frame_name': 'fr3_hand_offset2',  'radius': 0.03},
    {'name': 'end_effector',      'start_frame_name': 'fr3_hand_tcp',  'end_frame_name': 'fr3_hand_tcp', 'radius': 0.03},
]

# Get frame IDs from the Pinocchio model for all unique frames
unique_frame_names = set()
for link_def in LINKS_DEF:
    unique_frame_names.add(link_def['start_frame_name'])
    unique_frame_names.add(link_def['end_frame_name'])

EE_FRAME_NAME = 'fr3_hand_tcp' # Make sure EE frame is included
unique_frame_names.add(EE_FRAME_NAME)

frame_ids = {}
for name in unique_frame_names:
    try:
        frame_ids[name] = model.getFrameId(name)
    except IndexError:
        print(f"FATAL: Frame '{name}' not found in URDF. Cannot proceed.")
        exit()

# Define the bounds for randomization
# Keep these within the robot's reasonable workspace
X_RANGE = [-0.5, 0.7]
Y_RANGE = [-0.7, 0.7]
Z_RANGE = [0.0, 0.8]
RADIUS_RANGE = [0.05, 0.15]
NUM_OBSTACLES_RANGE = [2, 5] # Generate scenarios with 1 to 5 obstacles
MAX_CAPSULE_LENGTH = 0.4  # Maximum length for capsule obstacles

GOAL_X_RANGE = [-0.5, 0.7]
GOAL_Y_RANGE = [-0.6, 0.6] # Example: bias goals to one side
GOAL_Z_RANGE = [0.1, 0.7]

GAMMA_JS_RANGE = [0.5, 2, 5, 10, 15] # Range for gamma_js
BETA_JS_RANGE = [0.5, 3, 7.5, 15, 17.5] # Range for beta_js
def get_closest_point_on_segment(p1, p2, o):
    """Calculates the point on the line segment (p1, p2) that is closest to point o."""
    link_vec = p2 - p1
    link_len_sq = np.dot(link_vec, link_vec)
    if link_len_sq < 1e-9:
        return p1
    t = np.dot(o - p1, link_vec) / link_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    return p1 + t_clamped * link_vec
def get_closest_points_between_segments(p1, p2, q1, q2):
    """
    Calculates the closest points between two line segments (p1, p2) and (q1, q2).
    This is a robust implementation that correctly handles all edge cases.
    """
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    a, b, c, d, e = np.dot(u,u), np.dot(u,v), np.dot(v,v), np.dot(u,w), np.dot(v,w)
    D = a*c - b*b
    s, t = 0.0, 0.0
    
    if D < 1e-7:
        s = np.clip(-d/a, 0, 1) if a > 1e-7 else 0
        t = (b*s + e)/c if c > 1e-7 else 0
    else:
        s = np.clip((b*e - c*d)/D, 0, 1)
        t = np.clip((a*e - b*d)/D, 0, 1)

        dist_s0 = np.sum(((q1 + t*v) - p1)**2)
        dist_s1 = np.sum(((q1 + t*v) - p2)**2)
        dist_t0 = np.sum(((p1 + s*u) - q1)**2)
        dist_t1 = np.sum(((p1 + s*u) - q2)**2)

        if s == 0 and dist_s0 > dist_t0 or s == 1 and dist_s1 > dist_t0:
             t = np.clip(-e/c, 0, 1) if c > 1e-7 else 0
        if t == 0 and dist_t0 > dist_s0 or t == 1 and dist_t1 > dist_s0:
             s = np.clip(-d/a, 0, 1) if a > 1e-7 else 0

    c1 = p1 + s * u
    c2 = q1 + t * v
    return c1, c2

def check_capsule_capsule_collision(p1_A, p2_A, r_A, p1_B, p2_B, r_B):
    """Checks if two capsules collide."""
    c_A, c_B = get_closest_points_between_segments(p1_A, p2_A, p1_B, p2_B)
    dist_sq = np.sum((c_A - c_B)**2)
    min_safe_dist_sq = (r_A + r_B)**2 + 0.02
    return dist_sq < min_safe_dist_sq

def get_link_endpoint_positions(q):
    """Calculates the world position of all necessary link frames for a given joint configuration."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    positions = {}
    for name, frame_id in frame_ids.items():
        positions[name] = data.oMf[frame_id].translation
    return positions

def main():
    package_root = os.path.dirname(os.path.abspath(__file__))
    full_output_dir = os.path.join(os.path.dirname(os.path.dirname(package_root)), "share", "cbf_safety_filter", OUTPUT_DIR)
    
    if os.path.exists(full_output_dir):
        shutil.rmtree(full_output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    print(f"Generating {NUM_SCENARIOS} scenarios in '{full_output_dir}'...")
    
    gamma_list = GAMMA_JS_RANGE
    beta_list = BETA_JS_RANGE
    if N_PARAMETERS_VARIATIONS == 1:
        gamma_list = [2.0]
        beta_list = [3.0]

        initial_endpoint_positions = get_link_endpoint_positions(Q_INITIAL)
    initial_ee_pos = initial_endpoint_positions[EE_FRAME_NAME]
    
    generated_count = 0
    while generated_count < NUM_SCENARIOS:
        # --- Generate Goal ---
        goal_pos = np.array([random.uniform(GOAL_X_RANGE[0], GOAL_X_RANGE[1]),
                             random.uniform(GOAL_Y_RANGE[0], GOAL_Y_RANGE[1]),
                             random.uniform(GOAL_Z_RANGE[0], GOAL_Z_RANGE[1])])
        
        # --- Generate "Relevant" Obstacles ---
        num_obstacles = random.randint(NUM_OBSTACLES_RANGE[0], NUM_OBSTACLES_RANGE[1])
        obstacles = []
        generation_successful = True
        
        for j in range(num_obstacles):
            is_relevant_and_valid = False
            for _ in range(MAX_TRIES_PER_OBSTACLE):
                # 1. Generate a candidate obstacle
                is_sphere = random.random() < 0.5
                p1 = np.array([random.uniform(X_RANGE[0], X_RANGE[1]), random.uniform(Y_RANGE[0], Y_RANGE[1]), random.uniform(Z_RANGE[0], Z_RANGE[1])])
                p2 = p1 if is_sphere else p1 + random.uniform(0.1, MAX_CAPSULE_LENGTH) * (lambda v: v/np.linalg.norm(v))(np.random.randn(3))
                p2 = np.clip(p2, [X_RANGE[0], Y_RANGE[0], Z_RANGE[0]], [X_RANGE[1], Y_RANGE[1], Z_RANGE[1]])
                radius = round(random.uniform(RADIUS_RANGE[0], RADIUS_RANGE[1]), 4)
                
                # 2. Check for immediate collision with initial robot pose
                in_initial_collision = False
                for link_def in LINKS_DEF:
                    p1_robot, p2_robot = initial_endpoint_positions[link_def['start_frame_name']], initial_endpoint_positions[link_def['end_frame_name']]
                    if check_capsule_capsule_collision(p1_robot, p2_robot, link_def['radius'], p1, p2, radius):
                        in_initial_collision = True
                        break
                # Check for initial distance between goal position and end effector
                if check_capsule_capsule_collision(initial_ee_pos, initial_ee_pos, 0.03, goal_pos, goal_pos, 0.1):
                    in_initial_collision = True
                if in_initial_collision: continue # Try again

                # 3. Check for relevance
                is_relevant = False
                # Check proximity to any robot link
                for link_def in LINKS_DEF:
                    p1_robot, p2_robot = initial_endpoint_positions[link_def['start_frame_name']], initial_endpoint_positions[link_def['end_frame_name']]
                    c_robot, c_obs = get_closest_points_between_segments(p1_robot, p2_robot, p1, p2)
                    if np.linalg.norm(c_robot - c_obs) < (radius + RELEVANCE_DISTANCE_THRESHOLD):
                        is_relevant = True; break
                if is_relevant:
                    is_relevant_and_valid = True; break

                # Check proximity to trajectory line (EE start to goal)
                c_traj, c_obs = get_closest_points_between_segments(initial_ee_pos, goal_pos, p1, p2)
                if np.linalg.norm(c_traj - c_obs) < (radius):
                    is_relevant_and_valid = True; break

            if is_relevant_and_valid:
                obstacles.append({
                    'name': f"obstacle_{j}",
                    'pose_start': {'position': [round(x, 4) for x in p1.tolist()]},
                    'pose_end': {'position': [round(x, 4) for x in p2.tolist()]},
                    'size': {'radius': radius}, 'velocity': {'linear': [0.0, 0.0, 0.0]}
                })
            else:
                generation_successful = False
                # print(f"Warning: Could not place a relevant and valid obstacle {j+1}/{num_obstacles}. Retrying scenario.")
                break # Failed to place this obstacle, so restart the whole scenario generation

        if not generation_successful: continue

        # --- Final Validation Checks (Goal related) ---
        is_valid = True
        for obs in obstacles:
            p1_obs, p2_obs = np.array(obs['pose_start']['position']), np.array(obs['pose_end']['position'])
            if check_capsule_capsule_collision(goal_pos, goal_pos, 0.02, p1_obs, p2_obs, obs['size']['radius']):
                is_valid = False; break
        if not is_valid: continue

        # --- Assemble and Write Scenario File ---
        if generated_count >= NUM_SCENARIOS: break
        
        for gamma in gamma_list:
            for beta in beta_list:
                if generated_count >= NUM_SCENARIOS:
                    break
                
                scenario_name = f"scenario_{generated_count:04d}"
                scenario_data = {
                    'hocbf_controller': {
                        'ros__parameters': {
                            'goal_ee_pos': [round(x, 4) for x in goal_pos.tolist()],
                            'gamma_js': float(gamma),
                            'beta_js': float(beta),
                            'd_margin': 0.0,
                            'output_data_basename': scenario_name,
                            'goal_maintolerance_m': 0.02,
                            'goal_settle_time_s': 1.5,
                            'max_sim_duration_s': 30.0
                        }
                    },
                    'obstacles': obstacles
                }

                file_path = os.path.join(full_output_dir, f"{scenario_name}.yaml")
                with open(file_path, 'w') as f:
                    yaml.dump(scenario_data, f, default_flow_style=False, sort_keys=False)

                generated_count += 1
            if generated_count >= NUM_SCENARIOS:
                break

    print(f"Successfully generated {generated_count} scenario files.")

if __name__ == '__main__':
    main()

