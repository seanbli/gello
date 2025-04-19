# ruff: noqa
import time
from typing import List, Optional
import socket
import json
import threading

import mujoco
import mujoco.viewer
import numpy as np

try:
    from STservo_sdk import *
except ImportError:
    print("警告: STservo_sdk 未找到，将使用模拟数据")

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
        baud_rate: int = 1000000,
    ):
        self.baud_rate = baud_rate
        self.device_name = device_name
        self.portOpen = False
        self.num_motors = 7
        self.motor_pos = np.zeros(self.num_motors, dtype=np.float32)
        self.motor_speed = np.zeros(self.num_motors, dtype=np.float32)
        
        try:
            self.portHandler = PortHandler(self.device_name)
            self.packetHandler = sts(self.portHandler)

            if self.portHandler.openPort():
                print("成功打开端口")
                self.portOpen = True
            else:
                print("打开端口失败，使用模拟数据")
                return

            if self.portHandler.setBaudRate(self.baud_rate):
                print("成功设置波特率")
            else:
                print("设置波特率失败，使用模拟数据")
                self.portHandler.closePort()
                self.portOpen = False
                return

            self.groupSyncRead = GroupSyncRead(self.packetHandler, STS_PRESENT_POSITION_L, 4)
        except Exception as e:
            print(f"初始化控制器失败: {e}")
            print("将使用模拟数据")

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
                    print(f"[ID:{sts_id:03d}] groupSyncRead addparam失败")
                    return None, None

            sts_comm_result = self.groupSyncRead.txRxPacket()
            if sts_comm_result != COMM_SUCCESS:
                print(f"{self.packetHandler.getTxRxResult(sts_comm_result)}")

            for sts_id in range(1, 8):
                sts_data_result, sts_error = self.groupSyncRead.isAvailable(sts_id, STS_PRESENT_POSITION_L, 4)
                if sts_data_result:
                    sts_present_position = self.groupSyncRead.getData(sts_id, STS_PRESENT_POSITION_L, 2)
                    sts_present_speed = self.groupSyncRead.getData(sts_id, STS_PRESENT_SPEED_L, 2)
                    self.motor_pos[sts_id - 1] = sts_present_position
                    self.motor_speed[sts_id - 1] = self.packetHandler.sts_tohost(sts_present_speed, 15)
                else:
                    print(f"[ID:{sts_id:03d}] groupSyncRead getdata失败")
                    return None, None
                if sts_error != 0:
                    print(f"{self.packetHandler.getRxPacketError(sts_error)}")
            self.groupSyncRead.clearParam()
        except Exception as e:
            print(f"读取关节数据失败: {e}")
            t = time.time()
            for i in range(self.num_motors):
                self.motor_pos[i] = 2000 + 500 * np.sin(t * 0.5 + i * 0.5)

        return self.motor_pos, self.motor_speed

    def close(self):
        if self.portOpen:
            print("关闭端口")
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
                # 创建简单默认模型
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
            print(f"正向运动学计算错误: {e}")
            pos = np.array([0.6, 0.2, 1.0])
            rmat = np.eye(3)
            gripper = 0.02

        return pos, rmat, gripper


class RightArmClient:
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
            print(f"连接失败: {e}")
            return False
            
    def send_data(self, right_pos, right_grip):
        if not self.connected:
            if not self.connect():
                return False
                
        try:
            data = {
                'right': {
                    'pos': right_pos.tolist() if isinstance(right_pos, np.ndarray) else right_pos,
                    'grip': float(right_grip)
                }
            }
            
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
    
    parser = argparse.ArgumentParser(description="Gello右臂控制器客户端")
    parser.add_argument('--port', type=str, default="/dev/cu.usbmodem58A60701011", help="控制器端口")
    parser.add_argument('--host', type=str, default="localhost", help="模拟器主机")
    parser.add_argument('--socket_port', type=int, default=12345, help="模拟器端口")
    parser.add_argument('--sim', action='store_true', help="使用模拟控制器数据")
    parser.add_argument('--view', action='store_true', help="显示MuJoCo视图")
    
    args = parser.parse_args()
    
    try:
        if args.sim:
            controller = GelloController(device_name="sim", baud_rate=1000000)
        else:
            controller = GelloController(device_name=args.port, baud_rate=1000000)
    except Exception as e:
        print(f"控制器创建失败: {e}")
        print("使用模拟控制器")
        controller = GelloController(device_name="sim", baud_rate=1000000)
    
    controller_wrapper = GelloControllerWrapper(
        controller=controller,
    )
    
    client = RightArmClient(host=args.host, port=args.socket_port)
    
    try:
        if args.view:
            with mujoco.viewer.launch_passive(controller_wrapper.mujoco_model, controller_wrapper.mujoco_data) as viewer:
                print("开始发送右臂控制器数据到模拟器...")
                
                while True:
                    pos, rmat, gripper = controller_wrapper.get_ee_pos_rmat_gripper()
                    
                    client.send_data(pos, gripper)
                    
                    print(f"右臂位置: {pos}, 抓取器: {gripper}")
                    time.sleep(0.02)
                    viewer.sync()
        else:
            print("开始发送右臂控制器数据到模拟器...")
            
            while True:
                pos, rmat, gripper = controller_wrapper.get_ee_pos_rmat_gripper()
                
                client.send_data(pos, gripper)
                
                if int(time.time()) % 5 == 0:
                    print(f"右臂位置: {pos}, 抓取器: {gripper}")
                
                time.sleep(0.02)
                
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        client.close()
        controller_wrapper.close()