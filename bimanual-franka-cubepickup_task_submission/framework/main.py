import mujoco
import mujoco.viewer
import numpy as np
import time

# Integration timestep in seconds
integration_dt: float = 0.1

# Damping term for the pseudoinverse
damping: float = 1e-4

# Gains for the twist computation
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation
gravity_compensation: bool = True

# Simulation timestep in seconds
dt: float = 0.002

# Nullspace P gain
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s
max_angvel = 0.785

# Additional variables for gripper control
GRIPPER_CLOSED_POS = 0.0  # Fully closed
GRIPPER_OPEN_POS = 0.04   # Fully open

# Simulated input function - this will be replaced with your actual input
def get_simulated_input(t):
    """
    Generate simulated input values based on time
    Returns: Dict with left and right arm positions and grip values
    """
    # Simple oscillating pattern
    freq = 0.2  # oscillation frequency
    
    # Left arm moves in a small circle
    left_x = 0.6 + 0.1 * np.sin(freq * t)
    left_y = -0.2 + 0.1 * np.cos(freq * t)
    left_z = 1.0 + 0.05 * np.sin(freq * t * 2)
    
    # Right arm moves in a similar circle but out of phase
    right_x = 0.6 + 0.1 * np.sin(freq * t + np.pi)
    right_y = 0.2 + 0.1 * np.cos(freq * t + np.pi)
    right_z = 1.0 + 0.05 * np.sin(freq * t * 2 + np.pi)
    
    # Grippers open and close periodically
    left_grip = (np.sin(freq * t * 0.5) + 1) * 0.5 * GRIPPER_OPEN_POS
    right_grip = (np.sin(freq * t * 0.5 + np.pi) + 1) * 0.5 * GRIPPER_OPEN_POS
    
    return {
        'left': {'pos': [left_x, left_y, left_z], 'grip': left_grip},
        'right': {'pos': [right_x, right_y, right_z], 'grip': right_grip}
    }

def main() -> None:
    # Load the model and data
    model = mujoco.MjModel.from_xml_path(
        "/Users/jiafeid/Downloads/bimanual-franka-cubepickup_task_submission/resources/bi-franka/scene_with_line_and_2_cubes.xml"
    )
    
    data = mujoco.MjData(model)

    # Enable gravity compensation
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # Joint and actuator names for both arms
    left_joint_names = [f"fa-joint{i}" for i in range(1, 8)]
    right_joint_names = [f"fb-joint{i}" for i in range(1, 8)]

    left_actuator_names = [f"fa-actuator{i}" for i in range(1, 8)]
    right_actuator_names = [f"fb-actuator{i}" for i in range(1, 8)]

    # Get joint and actuator IDs correctly
    dof_ids_left = np.array([model.joint(name).qposadr[0] for name in left_joint_names])
    dof_ids_right = np.array([model.joint(name).qposadr[0] for name in right_joint_names])

    actuator_ids_left = np.array([model.actuator(name).id for name in left_actuator_names])
    actuator_ids_right = np.array([model.actuator(name).id for name in right_actuator_names])

    # Initial joint positions (home keyframe)
    key_id = model.key("home").id
    q0_left = model.key("home").qpos[dof_ids_left]
    q0_right = model.key("home").qpos[dof_ids_right]

    # Mocap targets for both arms
    mocap_id_left = model.body("target_left").mocapid[0]
    mocap_id_right = model.body("target_right").mocapid[0]

    # Pre-allocate numpy arrays
    jac_left, jac_right = np.zeros((6, model.nv)), np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    twist_left, twist_right = np.zeros(6), np.zeros(6)
    body_quat_left, body_quat_right = np.zeros(4), np.zeros(4)
    body_quat_conj_left = np.zeros(4)
    body_quat_conj_right = np.zeros(4)
    error_quat_left, error_quat_right = np.zeros(4), np.zeros(4)
    dq = np.zeros(model.nv)

    # Get gripper actuator IDs
    left_gripper_actuator = model.actuator("fa-actuator8").id
    right_gripper_actuator = model.actuator("fb-actuator8").id

    # Initial positions
    initial_left_pos = np.array([0.6, -0.2, 1.0])
    initial_right_pos = np.array([0.6, 0.2, 1.0])

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Reset simulation
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        
        # Set initial mocap positions
        data.mocap_pos[mocap_id_left] = initial_left_pos
        data.mocap_pos[mocap_id_right] = initial_right_pos
        
        mujoco.mj_forward(model, data)
        
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        
        body_id_left = model.body("fa-hand").id
        body_id_right = model.body("fb-hand").id
        
        print("Starting simulation with controlled movement...")
        
        # Start time for simulation
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # Get simulated input based on elapsed time
            elapsed_time = time.time() - start_time
            control_input = get_simulated_input(elapsed_time)
            
            # Set target positions and gripper states from input
            data.mocap_pos[mocap_id_left] = control_input['left']['pos']
            data.mocap_pos[mocap_id_right] = control_input['right']['pos']
            
            data.ctrl[left_gripper_actuator] = control_input['left']['grip']
            data.ctrl[right_gripper_actuator] = control_input['right']['grip']
            
            # --- Left Hand ---
            dx_left = data.mocap_pos[mocap_id_left] - data.body("fa-hand").xpos
            twist_left[:3] = Kpos * dx_left / integration_dt
            body_quat_left[:] = data.body("fa-hand").xquat
            mujoco.mju_negQuat(body_quat_conj_left, body_quat_left)
            mujoco.mju_mulQuat(error_quat_left, data.mocap_quat[mocap_id_left], body_quat_conj_left)
            mujoco.mju_quat2Vel(twist_left[3:], error_quat_left, 1.0)
            twist_left[3:] *= Kori / integration_dt

            mujoco.mj_jacBody(model, data, jac_left[:3], jac_left[3:], body_id_left)
            dq[dof_ids_left] = jac_left[:, dof_ids_left].T @ np.linalg.solve(
                jac_left[:, dof_ids_left] @ jac_left[:, dof_ids_left].T + diag, twist_left
            )

            dq[dof_ids_left] += (np.eye(7) - np.linalg.pinv(jac_left[:, dof_ids_left]) @ jac_left[:, dof_ids_left]) @ (
                Kn * (q0_left - data.qpos[dof_ids_left])
            )

            # --- Right Hand ---
            dx_right = data.mocap_pos[mocap_id_right] - data.body("fb-hand").xpos
            twist_right[:3] = Kpos * dx_right / integration_dt
            body_quat_right[:] = data.body("fb-hand").xquat
            mujoco.mju_negQuat(body_quat_conj_right, body_quat_right)
            mujoco.mju_mulQuat(error_quat_right, data.mocap_quat[mocap_id_right], body_quat_conj_right)
            mujoco.mju_quat2Vel(twist_right[3:], error_quat_right, 1.0)
            twist_right[3:] *= Kori / integration_dt

            mujoco.mj_jacBody(model, data, jac_right[:3], jac_right[3:], body_id_right)
            dq[dof_ids_right] = jac_right[:, dof_ids_right].T @ np.linalg.solve(
                jac_right[:, dof_ids_right] @ jac_right[:, dof_ids_right].T + diag, twist_right
            )

            dq[dof_ids_right] += (np.eye(7) - np.linalg.pinv(jac_right[:, dof_ids_right]) @ jac_right[:, dof_ids_right]) @ (
                Kn * (q0_right - data.qpos[dof_ids_right])
            )

            # Clamp max joint velocity
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            # Set control signal
            data.ctrl[actuator_ids_left] = q[dof_ids_left]
            data.ctrl[actuator_ids_right] = q[dof_ids_right]

            # Step the simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            # Print positions every second (for debugging)
            if int(elapsed_time) != int(elapsed_time - (time.time() - step_start)) and int(elapsed_time) % 2 == 0:
                print(f"Time: {int(elapsed_time)}s - Left pos: {data.mocap_pos[mocap_id_left]}, Right pos: {data.mocap_pos[mocap_id_right]}")

            # Control simulation speed
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()