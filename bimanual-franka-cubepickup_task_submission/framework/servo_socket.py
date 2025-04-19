import mujoco
import mujoco.viewer
import numpy as np
import time
import socket
import threading
import json


integration_dt: float = 0.1
damping: float = 1e-4

Kpos: float = 0.95
Kori: float = 0.95

gravity_compensation: bool = True

dt: float = 0.002

Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

max_angvel = 0.785

GRIPPER_CLOSED_POS = 0.0
GRIPPER_OPEN_POS = 0.04

def map_grip_value(controller_value):
    controller_min = 4.377
    controller_max = -26.6
    
    gripper_min = 0.0
    gripper_max = 0.04
    
    if controller_max == controller_min:
        normalized = 0.5
    else:
        normalized = (controller_value - controller_min) / (controller_max - controller_min)
    
    mapped_value = gripper_min + (1 - normalized) * (gripper_max - gripper_min)
    mapped_value = max(gripper_min, min(gripper_max, mapped_value))
    
    return mapped_value

def map_position(controller_pos):

    x, y, z = controller_pos
    
    x_sim = map_axis_value(x, 0.0, 1.0, 0.3, 0.8)
    y_sim = map_axis_value(y, -0.5, 0.5, -0.3, 0.3)
    z_sim = map_axis_value(z, 0.2, 0.4, 0.98, 2)
    
    return [x_sim, y_sim, z_sim]

def map_axis_value(value, src_min, src_max, dst_min, dst_max):

    if src_max == src_min:
        normalized = 0.5
    else:
        normalized = (value - src_min) / (src_max - src_min)
    
    mapped_value = dst_min + normalized * (dst_max - dst_min)
    mapped_value = max(dst_min, min(dst_max, mapped_value))
    
    return mapped_value

class RightArmServer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.client_thread = None
        
        self.latest_data = {
            'right': {'pos': [0.6, 0.2, 1.0], 'grip': 0.02, 'raw_grip': 0.0, 'raw_pos': [0.5, 0.0, 0.2]}
        }
        self.lock = threading.Lock()
        
    def start(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            
            self.running = True
            print(f"服务器已启动: {self.host}:{self.port}")
            
            self.accept_thread = threading.Thread(target=self._accept_connections)
            self.accept_thread.daemon = True
            self.accept_thread.start()
            
            return True
        except Exception as e:
            print(f"服务器启动失败: {e}")
            return False
    
    def _accept_connections(self):
        while self.running:
            try:
                print("等待控制器连接...")
                client_socket, client_addr = self.server_socket.accept()
                print(f"控制器已连接: {client_addr}")
                
                if self.client_thread and self.client_thread.is_alive():
                    self.client_socket.close()
                
                self.client_socket = client_socket
                self.client_thread = threading.Thread(target=self._handle_client, args=(client_socket,))
                self.client_thread.daemon = True
                self.client_thread.start()
            except Exception as e:
                if self.running:
                    print(f"接受连接错误: {e}")
                    time.sleep(1)
    
    def _handle_client(self, client_socket):
        try:
            while self.running:
                data = client_socket.recv(1024)
                if not data:
                    print("客户端断开连接")
                    break
                
                try:
                    controller_data = json.loads(data.decode('utf-8'))
                    
                    with self.lock:
                        if 'right' in controller_data:
                            if 'pos' in controller_data['right']:
                                self.latest_data['right']['raw_pos'] = controller_data['right']['pos']
                                self.latest_data['right']['pos'] = map_position(controller_data['right']['pos'])
                            
                            if 'grip' in controller_data['right']:
                                self.latest_data['right']['raw_grip'] = controller_data['right']['grip']
                                self.latest_data['right']['grip'] = map_grip_value(controller_data['right']['grip'])
                except json.JSONDecodeError:
                    print("接收到无效的JSON数据")
                except Exception as e:
                    print(f"处理数据错误: {e}")
        except Exception as e:
            print(f"客户端处理错误: {e}")
        finally:
            client_socket.close()
    
    def get_right_arm_data(self):
        with self.lock:
            return self.latest_data['right'].copy()
    
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
        print("服务器已停止")

def main(host='localhost', port=12345):
    model_path = "/Users/jiafeid/Downloads/bimanual-franka-cubepickup_task_submission/resources/bi-franka/scene_with_line_and_2_cubes.xml"
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保模型路径正确，退出程序")
        return
        
    data = mujoco.MjData(model)

    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    server = RightArmServer(host, port)
    if not server.start():
        print("服务器启动失败，退出程序")
        return

    try:
        right_joint_names = [f"fb-joint{i}" for i in range(1, 8)]
        right_actuator_names = [f"fb-actuator{i}" for i in range(1, 8)]

        dof_ids_right = np.array([model.joint(name).qposadr[0] for name in right_joint_names])
        actuator_ids_right = np.array([model.actuator(name).id for name in right_actuator_names])

        key_id = model.key("home").id
        q0_right = model.key("home").qpos[dof_ids_right]

        mocap_id_right = model.body("target_right").mocapid[0]

        jac_right = np.zeros((6, model.nv))
        diag = damping * np.eye(6)
        twist_right = np.zeros(6)
        body_quat_right = np.zeros(4)
        body_quat_conj_right = np.zeros(4)
        error_quat_right = np.zeros(4)
        dq = np.zeros(model.nv)

        right_gripper_actuator = model.actuator("fb-actuator8").id

        initial_right_pos = np.array([0.6, 0.2, 1.0])

        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            
            data.mocap_pos[mocap_id_right] = initial_right_pos
            
            mujoco.mj_forward(model, data)
            
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            
            body_id_right = model.body("fb-hand").id
            
            print("开始接收右臂控制器输入的模拟...")
            print(f"夹爪映射: 控制器 [{4.377} 到 {-26.6}] -> 夹爪 [{GRIPPER_CLOSED_POS} 到 {GRIPPER_OPEN_POS}]")
            print(f"位置映射:")
            print(f"  X轴: 控制器 [0.0 到 1.0] -> 模拟器 [0.3 到 0.8]")
            print(f"  Y轴: 控制器 [-0.5 到 0.5] -> 模拟器 [-0.3 到 0.3]")
            print(f"  Z轴: 控制器 [0.0 到 0.4] -> 模拟器 [0.98 到 1.2]")
            
            start_time = time.time()
            last_print_time = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                right_arm_data = server.get_right_arm_data()
                
                data.mocap_pos[mocap_id_right] = right_arm_data['pos']
                data.ctrl[right_gripper_actuator] = right_arm_data['grip']
                
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

                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

                q = data.qpos.copy()
                mujoco.mj_integratePos(model, q, dq, integration_dt)

                data.ctrl[actuator_ids_right] = q[dof_ids_right]

                mujoco.mj_step(model, data)
                viewer.sync()

                current_time = time.time()
                if current_time - last_print_time >= 2.0:
                    elapsed_time = current_time - start_time
                    raw_pos = right_arm_data.get('raw_pos', ['N/A']*3)
                    mapped_pos = right_arm_data['pos']
                    raw_grip = right_arm_data.get('raw_grip', 'N/A')
                    mapped_grip = right_arm_data['grip']
                    print(f"时间: {elapsed_time:.1f}s")
                    print(f"位置: 原始值 {raw_pos} -> 映射值 {mapped_pos}")
                    print(f"夹爪: 原始值 {raw_grip} -> 映射值 {mapped_grip:.4f}")
                    last_print_time = current_time

                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except Exception as e:
        print(f"模拟运行错误: {e}")
    finally:
        server.stop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Franka右臂模拟器服务器")
    parser.add_argument('--host', type=str, default='localhost', help="服务器主机")
    parser.add_argument('--port', type=int, default=12345, help="服务器端口")
    
    args = parser.parse_args()
    
    main(args.host, args.port)