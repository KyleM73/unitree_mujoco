import time
import sys
import os
from threading import Lock, Event
from scipy.interpolate import make_interp_spline, PPoly
from itertools import accumulate
import argparse

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from typing import Tuple, List, Optional
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)

import numpy as np

parser = argparse.ArgumentParser(prog="g1_dancing")
parser.add_argument("dance_move", choices=["disco", "sprinkler", "cabbage_patch"])
parser.add_argument("-r", "--robot")
parser.add_argument("-sg", "--skip_graph", action="store_true")

RIGHT_ARM_BROKEN = True

SPRINKLER_BPM = 200
DISCO_BPM = 50
CABBAGE_PATCH_BPM = 90

DISCO_UP = [
    -3 * np.pi / 4,
    2 * np.pi / 4,
    -np.pi / 8,
    2 * np.pi / 5,
    0.0,
    0.0,
    -3 * np.pi / 8,
    np.pi / 2,
    -np.pi / 8,
    0.0,
]

DISCO_DOWN = [
    -np.pi / 2,
    0.0,  # -np.pi / 2,
    -5 * np.pi / 8,
    np.pi / 8,
    0.0,
    0.0,
    -3 * np.pi / 8,
    np.pi / 2,
    -np.pi / 8,
    0.0,
]

SPRINKLER_OPEN = [
    -np.pi / 2,
    1 * np.pi / 4,
    0.0,
    np.pi / 2,
    0.0,  # -np.pi / 2,
    3 * -np.pi / 4,
    -np.pi / 8,
    0.0,
    -np.pi / 3,
    0.0,
]

SPRINKLER_CLOSED = [
    2 * -np.pi / 3,
    0.0,
    0.0,
    np.pi / 2,
    0.0,
    -np.pi / 2,
    np.pi / 16,
    0.0,
    -np.pi / 3,
    0.0,
]

# CABBAGE_PATCH_OUT = [
#     -np.pi / 3,
#     0.0,
#     0.0,
#     np.pi / 3,
#     0.0,
#     -np.pi / 3,
#     0.0,
#     0.0,
#     np.pi / 3,
#     0.0,
# ]

CABBAGE_PATCH_IN = [
    np.pi / 4,
    np.pi / 8,
    -np.pi / 8,
    -np.pi / 3,
    0.0,
    np.pi / 4,
    -np.pi / 8,
    np.pi / 8,
    -np.pi / 3,
    0.0,
]

CABBAGE_PATCH_LEFT = [
    -np.pi / 4,
    np.pi / 8,
    0.0,
    np.pi / 4,
    0.0,
    -np.pi / 4,
    np.pi / 24,
    np.pi / 8,
    np.pi / 4,
    0.0,
]

CABBAGE_PATCH_RIGHT = [
    -np.pi / 4,
    -np.pi / 24,
    -np.pi / 8,
    np.pi / 4,
    0.0,
    -np.pi / 4,
    -np.pi / 8,
    0.0,
    np.pi / 4,
    0.0,
]

DISCO_CMD = ([60.0 / DISCO_BPM, 60.0 / DISCO_BPM], [DISCO_DOWN, DISCO_UP])
SPRINKLER_CMD = (
    [60.0 / SPRINKLER_BPM, 60.0 / SPRINKLER_BPM],
    [SPRINKLER_OPEN, SPRINKLER_CLOSED],
)
CABBAGE_PATCH_CMD = (
    [
        60.0 / CABBAGE_PATCH_BPM,
        60.0 / CABBAGE_PATCH_BPM,
        60.0 / CABBAGE_PATCH_BPM,
    ],
    [CABBAGE_PATCH_IN, CABBAGE_PATCH_RIGHT, CABBAGE_PATCH_LEFT],
)

DANCES = {
    "disco": DISCO_CMD,
    "sprinkler": SPRINKLER_CMD,
    "cabbage_patch": CABBAGE_PATCH_CMD,
}


class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20  # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21  # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28  # NOTE: INVALID for g1 23dof

    kNotUsedJoint = 29  # NOTE: Weight


class JointAnglesController:
    def __init__(self, using_robot: bool):
        self.kp: float = 20.0
        self.kd: float = 1.0
        self.hip_kp: float = 40.0
        self.hip_kd: float = 1.0
        self._n_iters: int = 8
        self._using_robot: bool = using_robot

        self.mode_machine_ = 0

        self._time: float = 0.0
        self._control_dt: float = 0.02
        self._low_cmd: LowCmd_ = unitree_hg_msg_dds__LowCmd_()
        self._low_state: Optional[LowState_] = None
        self._crc = CRC()

        self._mutex: Lock = Lock()
        self._done_first_update: Event = Event()
        self._low_cmd_write_thread_ptr: Optional[RecurrentThread] = None

        self._cmd_queue: Optional[Tuple[List[float], List[List[float]]]] = None
        self._init_cmd_queue: Optional[Tuple[List[float], List[List[float]]]] = None
        self._finish_cmd_queue: Optional[Tuple[List[float], List[List[float]]]] = None

        self._interp_init: Optional[PPoly] = None
        self._interp: Optional[PPoly] = None
        self._interp_finish: Optional[PPoly] = None
        self._start_time: float = 1.0
        self._max_time: float = 0.0
        self._loop: bool = False

        self._joint_data: List[Tuple[float, List[float], List[float]]] = []

        self.arm_joints: List[int] = [
            G1JointIndex.LeftShoulderPitch,
            G1JointIndex.LeftShoulderRoll,
            G1JointIndex.LeftShoulderYaw,
            G1JointIndex.LeftElbow,
            G1JointIndex.LeftWristRoll,
            G1JointIndex.RightShoulderPitch,
            G1JointIndex.RightShoulderRoll,
            G1JointIndex.RightShoulderYaw,
            G1JointIndex.RightElbow,
            G1JointIndex.RightWristRoll,
            # G1JointIndex.WaistYaw,
            # G1JointIndex.WaistRoll,
            # G1JointIndex.WaistPitch,
        ]
        self._right_arm_joints: List[int] = [
            G1JointIndex.RightShoulderPitch,
            G1JointIndex.RightShoulderRoll,
            G1JointIndex.RightShoulderYaw,
            G1JointIndex.RightElbow,
            G1JointIndex.RightWristRoll,
        ]
        self._waist_yaw: List[int] = [
            G1JointIndex.WaistYaw,
        ]
        self._leg_joints: List[int] = [
            G1JointIndex.LeftHipPitch,
            G1JointIndex.LeftHipRoll,
            G1JointIndex.LeftHipYaw,
            G1JointIndex.LeftKnee,
            G1JointIndex.LeftAnklePitch,
            G1JointIndex.LeftAnkleRoll,
            G1JointIndex.RightHipPitch,
            G1JointIndex.RightHipRoll,
            G1JointIndex.RightHipYaw,
            G1JointIndex.RightKnee,
            G1JointIndex.RightAnklePitch,
            G1JointIndex.RightAnkleRoll,
        ]
        self._hip_joints: List[int] = [
            G1JointIndex.WaistRoll,
            G1JointIndex.WaistPitch,
        ]

    @property
    def _done_time(self) -> float:
        return self._max_time * self._n_iters + self._start_time

    @property
    def done(self) -> bool:
        return self._time >= self._done_time + self._start_time

    def init(self, cmd_queue: Tuple[List[float], List[List[float]]], loop=True) -> None:
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while self._using_robot and result["name"]:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # create publisher
        self._arm_sdk_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._arm_sdk_publisher.Init()

        # create subscriber
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        self._cmd_queue = cmd_queue
        self._loop = loop

    def start(self):
        self._low_cmd_write_thread_ptr = RecurrentThread(
            interval=self._control_dt, target=self._low_cmd_write, name="control"
        )
        self._done_first_update.wait()
        self._low_cmd_write_thread_ptr.Start()

    def _low_state_handler(self, msg: LowState_) -> None:
        self._low_state = msg
        self._joint_data.append(
            (
                self._time,
                self._arm_joint_pos_from_msg(msg),
                self._arm_joint_vel_from_msg(msg),
            )
        )
        if not self._done_first_update.is_set():
            self._compute_interpolation(msg)
            self.mode_machine_ = self._low_state.mode_machine
            self._done_first_update.set()

    def _low_cmd_write(self) -> None:
        assert self._interp is not None, (
            "Invalid initialization -- self._interp is None!"
        )
        self._time += self._control_dt

        for i, joint in enumerate(self.arm_joints):
            if RIGHT_ARM_BROKEN and joint in self._right_arm_joints:
                self._update_low_cmd(joint, 0.0, 0.0, 0.0, 0.0, False)
            else:
                self._update_low_cmd(
                    joint,
                    self.interp(self._time)[i],
                    self.interp(self._time, 1)[i],
                    self.kp,
                    self.kd,
                    True,
                )
        for joint in self._waist_yaw:
            self._update_low_cmd(joint, 0.0, 0.0, self.hip_kp, self.hip_kd, True)
        for joint in self._leg_joints:
            self._update_low_cmd(joint, 0.0, 0.0, self.kp, self.kd)
        for joint in self._hip_joints:
            self._update_low_cmd(joint, 0.0, 0.0, 0.0, 0.0)

        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._arm_sdk_publisher.Write(self._low_cmd)

    def _update_low_cmd(
        self,
        joint: int,
        q_des: float,
        dq_des,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
        enable: bool = False,
    ) -> None:
        if enable:
            self._low_cmd.motor_cmd[joint].mode = 1
            self._low_cmd.mode_machine = self.mode_machine_

        if kp is None:
            kp = self.kp
        if kd is None:
            kd = self.kd
        # q = self._low_state.motor_state[joint].q
        # dq = self._low_state.motor_state[joint].dq
        # self._low_cmd.motor_cmd[joint].tau = kp * (q_des - q) + kd * (dq_des - dq)
        self._low_cmd.motor_cmd[joint].q = q_des
        self._low_cmd.motor_cmd[joint].dq = dq_des
        self._low_cmd.motor_cmd[joint].kp = kp
        self._low_cmd.motor_cmd[joint].kd = kd

    def _arm_joint_pos_from_msg(self, msg: LowState_) -> List[float]:
        return np.array(list(map(lambda state: state.q, msg.motor_state)))[
            self.arm_joints
        ].tolist()

    def _arm_joint_vel_from_msg(self, msg: LowState_) -> List[float]:
        return np.array(list(map(lambda state: state.dq, msg.motor_state)))[
            self.arm_joints
        ].tolist()

    def _compute_interpolation(self, init_msg: LowState_) -> None:
        assert self._cmd_queue is not None
        timesteps, joint_cmds = self._cmd_queue
        assert len(timesteps) == len(joint_cmds)

        init_pos = self._arm_joint_pos_from_msg(init_msg)
        abs_timesteps = list(accumulate(timesteps))

        self._max_time = max(abs_timesteps)
        self._init_cmd_queue = ([0.0, self._start_time], [init_pos, joint_cmds[0]])
        self._finish_cmd_queue = (
            [0.0, self._start_time],
            [joint_cmds[0], np.zeros(len(self.arm_joints)).tolist()],
        )
        self._interp = make_interp_spline(
            [0.0, *abs_timesteps],
            [*joint_cmds, joint_cmds[0]],
            bc_type="periodic",
            k=5,
        )
        self._interp_init = make_interp_spline(
            self._init_cmd_queue[0],
            self._init_cmd_queue[1],
            k=5,
            bc_type=(
                [(1, np.zeros(10)), (2, np.zeros(10))],
                [
                    (1, self._interp(0.0, 1)),
                    (2, self._interp(0.0, 2)),
                ],
            ),
        )
        self._interp_finish = make_interp_spline(
            self._finish_cmd_queue[0],
            self._finish_cmd_queue[1],
            k=5,
            bc_type=(
                [
                    (1, self._interp(0.0, 1)),
                    (2, self._interp(0.0, 2)),
                ],
                [(1, np.zeros(10)), (2, np.zeros(10))],
            ),
        )

    def interp(self, time: float, order: int = 0) -> np.ndarray:
        assert (
            self._interp is not None
            and self._interp_init is not None
            and self._interp_finish is not None
        )
        if time < self._start_time:
            return self._interp_init(time, order)
        elif time < self._done_time:
            return self._interp((time - self._start_time) % self._max_time, order)
        elif time < self._done_time + self._start_time:
            return self._interp_finish(time - self._done_time, order)
        else:
            return self._interp_finish(self._start_time, order)

    def vectorized_interp(self, time: np.ndarray, order: int = 0) -> np.ndarray:
        assert self._interp is not None and self._interp_init is not None
        time_start_ar = np.repeat(
            (time > self._start_time)[:, np.newaxis], len(self.arm_joints), -1
        )
        time_end_ar = np.repeat(
            (time <= self._done_time)[:, np.newaxis], len(self.arm_joints), -1
        )
        time_terminate_ar = np.repeat(
            (time < self._done_time + self._start_time)[:, np.newaxis],
            len(self.arm_joints),
            -1,
        )
        return np.where(
            time_terminate_ar,
            np.where(
                time_end_ar,
                np.where(
                    time_start_ar,
                    self._interp((time - self._start_time) % self._max_time, order),
                    self._interp_init(time, order),
                ),
                self._interp_finish(time - self._done_time, order),
            ),
            self._interp_finish(self._start_time, order),
        )

    def _graph_all_interp(
        self,
        interp: PPoly,
        *,
        xmin: float,
        xmax: float,
        x_pts: List[float],
        y_pts: List[List[float]],
        save: bool = False,
        include_actual: bool = False,
        plot_prefix: str = "",
        vlines: List[float] = [],
    ) -> None:
        xs = np.linspace(xmin, xmax, 100)
        pos_fig, pos_ax = plt.subplots(figsize=(6.5, 4))
        vel_fig, vel_ax = plt.subplots(figsize=(6.5, 4))
        acc_fig, acc_ax = plt.subplots(figsize=(6.5, 4))

        labels = np.arange(len(self.arm_joints))
        pos_ax.plot(x_pts, y_pts, marker="o", linestyle="none")
        if include_actual:
            joint_time, joint_pos, joint_vel = zip(*self._joint_data.copy())
            pos_ax.plot(joint_time, joint_pos)
            vel_ax.plot(joint_time, joint_vel)
            pos_ax.plot(xs, interp(xs), label=labels, linestyle="--")
            vel_ax.plot(xs, interp(xs, 1), label=labels, linestyle="--")
        else:
            pos_ax.plot(xs, interp(xs), label=labels)
            vel_ax.plot(xs, interp(xs, 1), label=labels)
        acc_ax.plot(xs, interp(xs, 2), label=labels)

        for ax in [pos_ax, vel_ax, acc_ax]:
            ax.set_xlim(xmin, xmax)
            ax.legend(loc="lower left", ncol=2)
            for vline in vlines:
                ax.axvline(vline, ls=":", c=(1, 0, 0))

        pos_ax.set_title("Joint Position Interpolation")
        vel_ax.set_title("Joint Velocity Interpolation")
        acc_ax.set_title("Joint Acceleration Interpolation")
        if save:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            pos_fig.savefig(f"{dir_path}/graphs/{plot_prefix}joint_pos.png")
            vel_fig.savefig(f"{dir_path}/graphs/{plot_prefix}joint_vel.png")
            acc_fig.savefig(f"{dir_path}/graphs/{plot_prefix}joint_acc.png")
        else:
            plt.show()

    def graph_full_interp(
        self, *, save: bool = False, prefix: str = "", include_actual: bool = False
    ) -> None:
        x_pts = [0.0, self._start_time]
        y_pts = [*self._init_cmd_queue[1]]
        for i in range(self._n_iters):
            x_pts.extend(
                list(
                    accumulate(
                        self._cmd_queue[0],
                        initial=self._start_time + i * self._max_time,
                    )
                )[1:]
            )
            y_pts.extend([*self._cmd_queue[1][1:], self._cmd_queue[1][0]])
        x_pts.extend([self._done_time, self._done_time + self._start_time])
        y_pts.extend(self._finish_cmd_queue[1])
        self._graph_all_interp(
            self.vectorized_interp,
            xmin=0.0,
            xmax=self._done_time + self._start_time + 1,
            x_pts=x_pts,
            y_pts=y_pts,
            save=save,
            plot_prefix=prefix,
            vlines=[self._start_time, self._done_time],
            include_actual=include_actual,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    if args.robot:
        ChannelFactoryInitialize(0, args.robot)
    else:
        ChannelFactoryInitialize(1, "lo")

    controller = JointAnglesController(args.robot is not None)
    controller.init(DANCES[args.dance_move])
    controller.start()

    while not controller.done:
        time.sleep(1)
    print("Done!")
    if not args.skip_graph:
        import matplotlib.pyplot as plt

        controller.graph_full_interp(
            save=True, prefix=f"{args.dance_move}_", include_actual=True
        )
        print("Created graphs")
    sys.exit(-1)
