# ruff: noqa
import time
from typing import List, Optional

import mujoco
import mujoco.viewer
import numpy as np

import time
import signal
import sys
import rtc_sdk
import argparse
import json

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
    """
    Class for reading the GELLO controller using STservo_sdk
    """

    def __init__(
        self,
        device_name: str,
        baud_rate: int = 1000000,
    ):
        self.baud_rate = baud_rate
        self.device_name = device_name
        self.portHandler = PortHandler(self.device_name)
        self.packetHandler = sts(self.portHandler)
        self.portOpen = False

        if self.portHandler.openPort():
            print("Succeeded to open the port")
            self.portOpen = True
        else:
            print("Failed to open the port; quitting")
            quit()

        if self.portHandler.setBaudRate(self.baud_rate):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate; quitting")
            quit()

        self.groupSyncRead = GroupSyncRead(self.packetHandler, STS_PRESENT_POSITION_L, 4)

        self.num_motors = 7
        self.motor_pos = np.zeros(self.num_motors, dtype=np.float32)
        self.motor_speed = np.zeros(self.num_motors, dtype=np.float32)

    def read_joints(self):
        for sts_id in range(1, self.num_motors + 1):
            # Add parameter storage for STServo#1~8 present position value
            sts_addparam_result = self.groupSyncRead.addParam(sts_id)
            if not sts_addparam_result:
                print("[ID:%03d] groupSyncRead addparam failed" % sts_id)
                return None, None

        sts_comm_result = self.groupSyncRead.txRxPacket()
        if sts_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(sts_comm_result))

        for sts_id in range(1, 8):
            # Check if groupsyncread data of STServo#1~8 is available
            sts_data_result, sts_error = self.groupSyncRead.isAvailable(scs_id, STS_PRESENT_POSITION_L, 4)
            if sts_data_result:
                # Get STServo#scs_id present position value
                sts_present_position = self.groupSyncRead.getData(sts_id, STS_PRESENT_POSITION_L, 2)
                sts_present_speed = self.groupSyncRead.getData(sts_id, STS_PRESENT_SPEED_L, 2)
                self.motor_pos[sts_id - 1] = sts_present_position
                self.motor_speed[sts_id - 1] = self.packetHandler.sts_tohost(sts_present_speed, 15)
            else:
                print("[ID:%03d] groupSyncRead getdata failed" % sts_id)
                return None, None
            if sts_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(sts_error))
        self.groupSyncRead.clearParam()

        return self.motor_pos, self.motor_speed

    def close(self):
        if self.portOpen:
            print("Closing port")
            self.portHandler.closePort()
            self.portOpen = False

    def __del__(self):
        self.close()


class GelloControllerRemote:
    RTC_CONNECTION_STATE = [
        "NONE",
        "DISCONNECTED",
        "CONNECTING",
        "CONNECTED",
        "RECONNECTING",
        "FAILED",
    ]

    def __init__(self, secret_id, secret_key, room_id):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.room_id = room_id
        self.connected = False
        self._last_msg = {}
        self._last_motor_pos = None
        self._last_motor_speed = None

        rtc_sdk.register_event_callback(self.on_message_received, self.on_connection_state_changed)
        rtc_sdk.connect(secret_id, secret_key, room_id)
        self.connected = True
        time.sleep(5)

    def close(self):
        if self.connected:
            print("Disconnecting")
            rtc_sdk.disconnect()
            self.connected = False

    def __del__(self):
        self.close()

    def on_message_received(self, msg):
        self._last_msg = json.loads(msg.decode("utf-8"))

    def on_connection_state_changed(self, state):
        print("Connection state changed:", self.RTC_CONNECTION_STATE[state])

    def read_joints(self):
        self._last_motor_pos = np.array(self._last_msg["angles"])
        # Not setting motor speed for now
        return self._last_motor_pos, self._last_motor_speed


class GelloControllerWrapper:
    """
    Wrapper class for GelloController for:
    1) Normalizing motor positions
    2) Computing forward kinematics
    """

    def __init__(
        self,
        controller: GelloController,
        motor_limits: Optional[List[float]] = None,
        joint_limits: Optional[List[float]] = None,
        use_mujoco: bool = True,
        xml_path: str = "gamify/controllers/gello_arm.xml",
    ):
        super().__init__()

        self.controller = controller
        # If motor/joint limits not provided, use default values
        self.motor_limits = motor_limits
        self.joint_limits = joint_limits
        if self.motor_limits is None:
            self.motor_limits = DEFAULT_NORMALIZATION_LIMITS
        if self.joint_limits is None:
            self.joint_limits = DEFAULT_JOINT_LIMITS

        self.motor_limits = np.array(self.motor_limits)
        self.joint_limits = np.radians(self.joint_limits)

        if use_mujoco:
            self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
            self.mujoco_data = mujoco.MjData(self.mujoco_model)

    def get_joint_pos(self):
        pos, _ = self.controller.read_joints()
        if pos is None:
            return None, None
        pos = (pos - self.motor_limits[:, 1]) / (self.motor_limits[:, 0] - self.motor_limits[:, 1])
        pos = self.joint_limits[:, 0] + pos * (self.joint_limits[:, 1] - self.joint_limits[:, 0])
        return pos

    def close(self):
        self.controller.close()

    def get_ee_pos_rmat_gripper(self):
        qpos = self.get_joint_pos()
        self.mujoco_data.qpos[:] = qpos[:-1]

        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
        link6_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        pos = np.array(self.mujoco_data.xpos[link6_id])
        rmat = np.array(self.mujoco_data.xmat[link6_id]).reshape(3, 3)

        pos = np.array([pos[1], pos[0], pos[2]])
        rmat = np.array([rmat[1], -rmat[0], rmat[2]])
        gripper = qpos[-1]

        return pos, rmat, gripper


if __name__ == "__main__":
    # Check which port is being used on your controller
    # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
    # controller = GelloController(device_name="/dev/ttyACM0", baud_rate=1000000)
    # controller = GelloController(device_name="/dev/cu.usbmodem58A60701011", baud_rate=1000000)
    controller = GelloController(device_name="/dev/cu.usbmodem58FA0957621", baud_rate=1000000)


    controller = GelloControllerWrapper(
        controller=controller,
    )

    try:
        with mujoco.viewer.launch_passive(controller.mujoco_model, controller.mujoco_data) as viewer:
            while True:
                pos, rmat, gripper = controller.get_ee_pos_rmat_gripper()
                print(pos, gripper)
                time.sleep(0.02)
                viewer.sync()
    except KeyboardInterrupt:
        controller.close()
