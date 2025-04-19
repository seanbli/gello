# ruff: noqa
import time
from typing import List, Optional
import socket
import json
import threading

import mujoco
import mujoco.viewer
import numpy as np

from STservo_sdk import *


DEFAULT_NORMALIZATION_LIMITS = [
    [481, 3696],
    [1151, 2706],
    [2330, 812],
    [767, 3295],
    [1533, 3623],
    [919, 3528],
    [2125, 1881],  # Gripper
]

DEFAULT_JOINT_LIMITS = [
    [-235, 35],
    [0, 135],
    [-135, 0],
    [-202.5, 22.5],
    [-90, 90],
    [-202.5, 22.5],
    [180, -180],
]

class GelloController:
    def __init__(
        self,
        device_name: str,
        arm_id: str = "right",
        baud_rate: int = 1000000,
    ):
        self.baud_rate = baud_rate
        self.device_name = device_name
        self.arm_id = arm_id
        self.portOpen = False
        self.num_motors = 7
        self.motor_pos = np.zeros(self.num_motors, dtype=np.float32)
        self.motor_speed = np.zeros(self.num_motors, dtype=np.float32)
        
        try:
            self.portHandler = PortHandler(self.device_name)
            self.packetHandler = sts(self.portHandler)

            if self.portHandler.openPort():
                print(f"Successfully opened {arm_id} arm port: {device_name}")
                self.portOpen = True
            else:
                print(f"Failed to open {arm_id} arm port, using simulated data")
                return

            if self.portHandler.setBaudRate(self.baud_rate):
                print(f"{arm_id} arm baud rate set successfully")
            else:
                print(f"{arm_id} arm baud rate setting failed, using simulated data")
                self.portHandler.closePort()
                self.portOpen = False
                return

            self.groupSyncRead = GroupSyncRead(self.packetHandler, STS_PRESENT_POSITION_L, 4)
        except Exception as e:
            print(f"Failed to initialize {arm_id} arm controller: {e}")
            print("Will use simulated data")

    def read_joints(self):
        if not self.portOpen:
            t = time.time()
            for i in range(self.num_motors):
                self.motor_pos[i] = 2000 + 500 * np.sin(t * 0.5 + i * 0.5)
            return self.motor_pos, self.motor_speed
        
        try:
            for sts_id in range(1, self.num_motors + 1):
                sts_addparam_result = self.groupSyncRead.addParam(sts_id)
                if not sts_addparam_result:
                    print(f"[{self.arm_id}][ID:{sts_id:03d}] groupSyncRead addparam Failed")
                    return None, None

            sts_comm_result = self.groupSyncRead.txRxPacket()
            if sts_comm_result != COMM_SUCCESS:
                print(f"[{self.arm_id}]{self.packetHandler.getTxRxResult(sts_comm_result)}")

            for sts_id in range(1, 8):
                sts_data_result, sts_error = self.groupSyncRead.isAvailable(sts_id, STS_PRESENT_POSITION_L, 4)
                if sts_data_result:
                    sts_present_position = self.groupSyncRead.getData(sts_id, STS_PRESENT_POSITION_L, 2)
                    sts_present_speed = self.groupSyncRead.getData(sts_id, STS_PRESENT_SPEED_L, 2)
                    self.motor_pos[sts_id - 1] = sts_present_position
                    self.motor_speed[sts_id - 1] = self.packetHandler.sts_tohost(sts_present_speed, 15)
                else:
                    print(f"[{self.arm_id}][ID:{sts_id:03d}] groupSyncRead getdata Failed")
                    return None, None
                if sts_error != 0:
                    print(f"[{self.arm_id}]{self.packetHandler.getRxPacketError(sts_error)}")
            self.groupSyncRead.clearParam()
        except Exception as e:
            print(f"[{self.arm_id}]Read Joint Data Error: {e}")
            t = time.time()
            for i in range(self.num_motors):
                self.motor_pos[i] = 2000 + 500 * np.sin(t * 0.5 + i * 0.5)

        return self.motor_pos, self.motor_speed

    def close(self):
        if self.portOpen:
            print(f"Close{self.arm_id}Arm Port")
            self.portHandler.closePort()
            self.portOpen = False

    def __del__(self):
        self.close()


class GelloControllerWrapper:
    def __init__(
        self,
        controller,
        motor_limits=None,
        joint_limits=None,
        use_mujoco=True,
        xml_path="gamify/controllers/gello_arm.xml",
    ):
        super().__init__()

        self.controller = controller
        self.motor_limits = motor_limits
        self.joint_limits = joint_limits
        if self.motor_limits is None:
            self.motor_limits = DEFAULT_NORMALIZATION_LIMITS
        if self.joint_limits is None:
            self.joint_limits = DEFAULT_JOINT_LIMITS

        self.motor_limits = np.array(self.motor_limits)
        self.joint_limits = np.radians(self.joint_limits)

        if use_mujoco:
            try:
                self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
                self.mujoco_data = mujoco.MjData(self.mujoco_model)
            except Exception as e:
                print(f"加载MuJoCo模型失败: {e}")
                print("使用简单默认模型")
                default_xml = """
                <mujoco>
                  <worldbody>
                    <body name="link6">
                      <geom type="sphere" size="0.05"/>
                    </body>
                  </worldbody>
                </mujoco>
                """
                self.mujoco_model = mujoco.MjModel.from_xml_string(default_xml)
                self.mujoco_data = mujoco.MjData(self.mujoco_model)

    def get_joint_pos(self):
        pos, _ = self.controller.read_joints()
        if pos is None:
            return np.zeros(7)
        pos = (pos - self.motor_limits[:, 1]) / (self.motor_limits[:, 0] - self.motor_limits[:, 1])
        pos = self.joint_limits[:, 0] + pos * (self.joint_limits[:, 1] - self.joint_limits[:, 0])
        return pos

    def close(self):
        self.controller.close()

    def get_ee_pos_rmat_gripper(self):
        qpos = self.get_joint_pos()
        try:
            self.mujoco_data.qpos[:] = qpos[:-1]
            mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
            link6_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "link6")
            pos = np.array(self.mujoco_data.xpos[link6_id])
            rmat = np.array(self.mujoco_data.xmat[link6_id]).reshape(3, 3)

            pos = np.array([pos[1], pos[0], pos[2]])
            rmat = np.array([rmat[1], -rmat[0], rmat[2]])
            gripper = qpos[-1]
        except Exception as e:
            print(f"[{self.controller.arm_id}]正向运动学计算错误: {e}")
            if self.controller.arm_id == "left":
                pos = np.array([0.6, -0.2, 1.0])
            else:
                pos = np.array([0.6, 0.2, 1.0])
            rmat = np.eye(3)
            gripper = 0.02

        return pos, rmat, gripper


class BimanualArmClient:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"已连接到模拟器: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connect Failed: {e}")
            return False
            
    def send_data(self, left_pos=None, left_grip=None, right_pos=None, right_grip=None):
        if not self.connected:
            if not self.connect():
                return False
                
        try:
            data = {}
            
            if left_pos is not None and left_grip is not None:
                data['left'] = {
                    'pos': left_pos.tolist() if isinstance(left_pos, np.ndarray) else left_pos,
                    'grip': float(left_grip)
                }
            
            if right_pos is not None and right_grip is not None:
                data['right'] = {
                    'pos': right_pos.tolist() if isinstance(right_pos, np.ndarray) else right_pos,
                    'grip': float(right_grip)
                }
            
            if not data:
                return False
                
            json_data = json.dumps(data)
            self.socket.sendall(json_data.encode('utf-8'))
            return True
        except Exception as e:
            print(f"发送数据失败: {e}")
            self.connected = False
            return False
            
    def close(self):
        if self.connected and self.socket:
            try:
                self.socket.close()
                print("客户端连接已关闭")
            except Exception as e:
                print(f"关闭连接错误: {e}")
            finally:
                self.connected = False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual Arm Gello Controller Client")
    parser.add_argument('--left_port', type=str, default="/dev/cu.usbmodem58FA0957621")
    parser.add_argument('--right_port', type=str, default="/dev/cu.usbmodem58FA0820271")
    parser.add_argument('--host', type=str, default="localhost", help="Simulator Host")
    parser.add_argument('--socket_port', type=int, default=12345, help="Simulator")
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--left_only', action='store_true')
    parser.add_argument('--right_only', action='store_true')
    parser.add_argument('--view', action='store_true')
    
    args = parser.parse_args()
    
    use_left = not args.right_only
    use_right = not args.left_only
    
    if args.left_only and args.right_only:
        print("Error: Cannot set both --left_only and --right_only")
        exit(1)
    
    left_controller = None
    right_controller = None
    left_wrapper = None
    right_wrapper = None
    
    try:
        if use_left:
            if args.sim:
                left_controller = GelloController(device_name="sim", arm_id="left", baud_rate=1000000)
            else:
                left_controller = GelloController(device_name=args.left_port, arm_id="left", baud_rate=1000000)
            
            left_wrapper = GelloControllerWrapper(controller=left_controller)
            print("左臂控制器初始化完成")
        
        if use_right:
            if args.sim:
                right_controller = GelloController(device_name="sim", arm_id="right", baud_rate=1000000)
            else:
                right_controller = GelloController(device_name=args.right_port, arm_id="right", baud_rate=1000000)
            
            right_wrapper = GelloControllerWrapper(controller=right_controller)
            print("右臂控制器初始化完成")
            
    except Exception as e:
        print(f"控制器创建失败: {e}")
        if use_left and left_controller is None:
            print("使用左臂模拟控制器")
            left_controller = GelloController(device_name="sim", arm_id="left", baud_rate=1000000)
            left_wrapper = GelloControllerWrapper(controller=left_controller)
            
        if use_right and right_controller is None:
            print("使用右臂模拟控制器")
            right_controller = GelloController(device_name="sim", arm_id="right", baud_rate=1000000)
            right_wrapper = GelloControllerWrapper(controller=right_controller)
    
    client = BimanualArmClient(host=args.host, port=args.socket_port)
    
    try:
        if args.view:
            view_model = None
            view_data = None
            if right_wrapper is not None:
                view_model = right_wrapper.mujoco_model
                view_data = right_wrapper.mujoco_data
            elif left_wrapper is not None:
                view_model = left_wrapper.mujoco_model
                view_data = left_wrapper.mujoco_data
            
            if view_model is not None:
                with mujoco.viewer.launch_passive(view_model, view_data) as viewer:
                    print("开始发送控制器数据到模拟器...")
                    
                    while True:
                        left_pos = None
                        left_gripper = None
                        if left_wrapper is not None:
                            left_pos, left_rmat, left_gripper = left_wrapper.get_ee_pos_rmat_gripper()
                        
                        right_pos = None
                        right_gripper = None
                        if right_wrapper is not None:
                            right_pos, right_rmat, right_gripper = right_wrapper.get_ee_pos_rmat_gripper()
                        
                        client.send_data(left_pos, left_gripper, right_pos, right_gripper)
                        
                        if left_pos is not None:
                            print(f"Left Arm: Pose {left_pos}, Grisper {left_gripper:.4f}")
                        if right_pos is not None:
                            print(f"Right Arm: Pose {right_pos}, Grisper {right_gripper:.4f}")
                        
                        time.sleep(0.02)
                        viewer.sync()
            else:
                print("无法启动查看器，没有可用的模型")
        else:
            print("开始发送控制器数据到模拟器...")
            
            last_print_time = 0
            
            while True:
                left_pos = None
                left_gripper = None
                if left_wrapper is not None:
                    left_pos, left_rmat, left_gripper = left_wrapper.get_ee_pos_rmat_gripper()
                
                right_pos = None
                right_gripper = None
                if right_wrapper is not None:
                    right_pos, right_rmat, right_gripper = right_wrapper.get_ee_pos_rmat_gripper()
                
                client.send_data(left_pos, left_gripper, right_pos, right_gripper)
                
                current_time = time.time()
                if current_time - last_print_time >= 2.0:
                    print("-" * 50)
                    if left_pos is not None:
                        print(f"Left Arm: Pose {left_pos}, Grisper {left_gripper:.4f}")
                    if right_pos is not None:
                        print(f"Right Arm: Pose {right_pos}, Grisper {right_gripper:.4f}")
                    last_print_time = current_time
                
                time.sleep(0.02)
                
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        client.close()
        if left_wrapper is not None:
            left_wrapper.close()
        if right_wrapper is not None:
            right_wrapper.close()