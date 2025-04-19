import mujoco
import mujoco.viewer
import numpy as np
import time
import socket
import threading
import json

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785

def map_spacemouse_to_mujoco(delta_pos, rotation):
    """
    Maps SpaceMouse control inputs to MuJoCo coordinate system.
    
    Args:
        delta_pos: SpaceMouse position deltas [dx, dy, dz]
        rotation: SpaceMouse rotation [rx, ry, rz]
        
    Returns:
        mapped_delta: Mapped position deltas
        mapped_rotation: Mapped rotation
    """
    # Position mapping (adjust these values based on testing)
    # Configure for x-forward, y-right
    mapped_delta = [
        delta_pos[1],    # SpaceMouse y-axis -> MuJoCo x-axis (forward)
        -delta_pos[0],   # SpaceMouse x-axis -> MuJoCo y-axis (right)
        delta_pos[2]     # SpaceMouse z-axis -> MuJoCo z-axis (up)
    ]
    
    # Rotation mapping (adjust these values based on testing)
    mapped_rotation = [
        rotation[0],     # SpaceMouse pitch -> MuJoCo roll (around x)
        rotation[1],    # SpaceMouse roll -> MuJoCo pitch (around y)
        rotation[2]      # SpaceMouse yaw -> MuJoCo yaw (around z)
    ]
    
    return mapped_delta, mapped_rotation

def euler_to_quat(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion for MuJoCo.
    
    Args:
        roll (float): Rotation around x-axis in radians.
        pitch (float): Rotation around y-axis in radians.
        yaw (float): Rotation around z-axis in radians.
        
    Returns:
        list: Quaternion [w, x, y, z].
    """
    # Standard ZYX Euler angles to quaternion conversion
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return [w, x, y, z]


class SingleArmServer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.client_thread = None
        
        # Initial values for the arm
        self.latest_data = {
            'pos': [0.6, -0.2, 1.0],  # Current position
            'delta_pos': [0.0, 0.0, 0.0],  # Position change
            'rot': [0.0, 0.0, 0.0],  # Rotation as rx,ry,rz
            'grip': 0.0,  # Gripper state
            'quat': [1.0, 0.0, 0.0, 0.0]  # Quaternion (calculated from rot)
        }
        self.lock = threading.Lock()
        
    def start(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            
            self.running = True
            print(f"Server Started: {self.host}:{self.port}")
            
            self.accept_thread = threading.Thread(target=self._accept_connections)
            self.accept_thread.daemon = True
            self.accept_thread.start()
            
            return True
        except Exception as e:
            print(f"Server Start Error: {e}")
            return False
    
    def _accept_connections(self):
        while self.running:
            try:
                print("Waiting for controller connect...")
                client_socket, client_addr = self.server_socket.accept()
                print(f"Controller Connected: {client_addr}")
                
                if self.client_thread and self.client_thread.is_alive():
                    self.client_socket.close()
                
                self.client_socket = client_socket
                self.client_thread = threading.Thread(target=self._handle_client, args=(client_socket,))
                self.client_thread.daemon = True
                self.client_thread.start()
            except Exception as e:
                if self.running:
                    print(f"Receive Connect Error: {e}")
                    time.sleep(1)
    
    def _handle_client(self, client_socket):
        try:
            while self.running:
                data = client_socket.recv(1024)
                if not data:
                    print("Client Disconnected")
                    break
                
                try:
                    controller_data = json.loads(data.decode('utf-8'))
                    
                    with self.lock:
                        has_delta = 'delta_pos' in controller_data
                        has_rot = 'rot' in controller_data
                        
                        # Apply mappings if both position and rotation are present
                        if has_delta and has_rot:
                            mapped_delta, mapped_rot = map_spacemouse_to_mujoco(
                                controller_data['delta_pos'], 
                                controller_data['rot']
                            )
                            
                            self.latest_data['delta_pos'] = mapped_delta
                            self.latest_data['rot'] = mapped_rot
                            
                            # Update current position based on mapped delta
                            for i in range(3):
                                self.latest_data['pos'][i] += mapped_delta[i]
                            
                            # Update quaternion from mapped rotation
                            self.latest_data['quat'] = euler_to_quat(*mapped_rot)
                        else:
                            # Handle partial updates
                            if has_delta:
                                mapped_delta, _ = map_spacemouse_to_mujoco(
                                    controller_data['delta_pos'], 
                                    [0, 0, 0]
                                )
                                self.latest_data['delta_pos'] = mapped_delta
                                for i in range(3):
                                    self.latest_data['pos'][i] += mapped_delta[i]
                            
                            if has_rot:
                                _, mapped_rot = map_spacemouse_to_mujoco(
                                    [0, 0, 0],
                                    controller_data['rot']
                                )
                                self.latest_data['rot'] = mapped_rot
                                self.latest_data['quat'] = euler_to_quat(*mapped_rot)
                        
                        # Process grip
                        if 'grip' in controller_data:
                            self.latest_data['grip'] = controller_data['grip']
                                
                except json.JSONDecodeError:
                    print("Invalid JSON data")
                except Exception as e:
                    print(f"Data Process Error: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"Client Process Error: {e}")
        finally:
            client_socket.close()
    
    def get_arm_data(self):
        with self.lock:
            return self.latest_data.copy()
    
    def stop(self):
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        print("Server Stopped")

def main(host='localhost', port=12345, model_path=None):
    if model_path is None:
        model_path = "/Users/jiafeid/Downloads/bimanual-franka-cubepickup_task_submission/resources/bi-franka/scene_with_line_and_2_cubes.xml"
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"Model Load Error: {e}")
        print("Please check model path")
        return
        
    data = mujoco.MjData(model)

    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    server = SingleArmServer(host, port)
    if not server.start():
        print("Server Starts Error")
        return

    try:
        # Setup only for the left arm
        left_joint_names = [f"fa-joint{i}" for i in range(1, 8)]
        left_actuator_names = [f"fa-actuator{i}" for i in range(1, 8)]

        dof_ids_left = np.array([model.joint(name).qposadr[0] for name in left_joint_names])
        actuator_ids_left = np.array([model.actuator(name).id for name in left_actuator_names])

        key_id = model.key("home").id
        q0_left = model.key("home").qpos[dof_ids_left]

        mocap_id_left = model.body("target_left").mocapid[0]

        jac_left = np.zeros((6, model.nv))
        diag = damping * np.eye(6)
        twist_left = np.zeros(6)
        body_quat_left = np.zeros(4)
        body_quat_conj_left = np.zeros(4)
        error_quat_left = np.zeros(4)
        dq = np.zeros(model.nv)

        left_gripper_actuator = model.actuator("fa-actuator8").id

        initial_left_pos = np.array([0.6, -0.2, 1.0])
        initial_quat = np.array([1.0, 0.0, 0.0, 0.0])

        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            
            data.mocap_pos[mocap_id_left] = initial_left_pos
            data.mocap_quat[mocap_id_left] = initial_quat
            
            mujoco.mj_forward(model, data)
            
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            
            body_id_left = model.body("fa-hand").id
            
            print("Start to receive left arm input...")
  
            start_time = time.time()
            last_print_time = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                arm_data = server.get_arm_data()
                
                # Update target position and orientation for the left arm
                data.mocap_pos[mocap_id_left] = arm_data['pos']
                data.mocap_quat[mocap_id_left] = arm_data['quat']
                
                # Update gripper
                data.ctrl[left_gripper_actuator] = arm_data['grip']
                
                # --- Left Hand Control Logic ---
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

                # Limit joint velocities
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

                # Integrate to get new joint positions
                q = data.qpos.copy()
                mujoco.mj_integratePos(model, q, dq, integration_dt)

                # Apply joint positions to actuators
                data.ctrl[actuator_ids_left] = q[dof_ids_left]

                # Step simulation and update viewer
                mujoco.mj_step(model, data)
                viewer.sync()

                # Print status info periodically
                current_time = time.time()
                if current_time - last_print_time >= 2.0:
                    elapsed_time = current_time - start_time

                    print(f"Time: {elapsed_time:.1f}s")
                    print("Left Arm Status:")
                    print(f"  Original Position: {arm_data['pos']}")
                    print(f"  Delta Position: {arm_data['delta_pos']}")
                    print(f"  Rotation (rx,ry,rz): {arm_data['rot']}")
                    print(f"  Quaternion: {arm_data['quat']}")
                    print(f"  Gripper: {arm_data['grip']:.4f}")
                    print("-------------------------------------------------")

                    last_print_time = current_time

                # Maintain simulation timing
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except Exception as e:
        print(f"Simulator Running Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        server.stop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Franka single arm simulator server")
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--model_path', type=str, default='/Users/jiafeid/Downloads/bimanual-franka-cubepickup_task_submission/resources/bi-franka/scene_with_line_and_2_cubes.xml')
    
    args = parser.parse_args()
    
    main(args.host, args.port, args.model_path)