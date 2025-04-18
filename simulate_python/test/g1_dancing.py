import time
import sys
from threading import Lock, Event
from scipy.interpolate import CubicSpline, PchipInterpolator, make_interp_spline
from itertools import accumulate
import matplotlib.pyplot as plt

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

import numpy as np

SPRINKLER_BPM = 220
DISCO_BPM = 110

DISCO_UP = [
    -np.pi / 2,
    2 * np.pi / 4,
    0.0,
    np.pi / 2,
    0.0,  # -np.pi / 2,
    0.0,
    -3 * np.pi / 8,
    np.pi / 2,
    -np.pi / 8,
    0.0,
    # 0.0,
    # 0.0,
    # 0.0,
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
    # 0.0,
    # 0.0,
    # 0.0,
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
    # 0.0,
    # 0.0,
    # 0.0,
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
    # 0.0,
    # 0.0,
    # 0.0,
]

DISCO_CMD = ([1.1 * 60.0 / DISCO_BPM, 0.9 * 60.0 / DISCO_BPM], [DISCO_DOWN, DISCO_UP])
SPRINKLER_CMD = (
    [60.0 / SPRINKLER_BPM, 60.0 / SPRINKLER_BPM],
    [SPRINKLER_OPEN, SPRINKLER_CLOSED],
)


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
    def __init__(self):
        self.kp: float = 20.0
        self.kd: float = 1.5

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
        self._interp_init: Optional[CubicSpline] = None
        self._interp: Optional[CubicSpline] = None
        self._start_time: float = 0.0
        self._max_time: float = 0.0
        self._loop: bool = False

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
            G1JointIndex.WaistYaw,
            G1JointIndex.WaistRoll,
            G1JointIndex.WaistPitch,
        ]

    @property
    def done(self) -> bool:
        return not self._loop and self._time > self._max_time

    def init(self, cmd_queue: Tuple[List[float], List[List[float]]], loop=True) -> None:
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
        if not self._done_first_update.is_set():
            self._compute_interpolation(msg)
            self._done_first_update.set()

    def _low_cmd_write(self) -> None:
        assert self._interp is not None, (
            "Invalid initialization -- self._interp is None!"
        )
        self._time += self._control_dt

        for i, joint in enumerate(self.arm_joints):
            self._update_low_cmd(
                joint,
                self.interp(self._time)[i],
                self.interp(self._time, 1)[i],
                self.kp,
                self.kd,
            )
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
    ) -> None:
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

    def _compute_interpolation(self, init_msg: LowState_) -> None:
        assert self._cmd_queue is not None
        timesteps, joint_cmds = self._cmd_queue
        assert len(timesteps) == len(joint_cmds)

        init_pos = np.array(list(map(lambda state: state.q, init_msg.motor_state)))[
            self.arm_joints
        ].tolist()
        abs_timesteps = list(accumulate(timesteps))

        self._start_time = timesteps[0]
        self._max_time = max(abs_timesteps)
        self._init_cmd_queue = ([0.0, self._start_time], [init_pos, joint_cmds[0]])
        self._interp = make_interp_spline(
            [*abs_timesteps, self._start_time + self._max_time],
            [*joint_cmds, joint_cmds[0]],
            bc_type="periodic",
        )
        self._interp_init = CubicSpline(
            self._init_cmd_queue[0],
            self._init_cmd_queue[1],
            bc_type=((1, np.zeros(10)), (1, self._interp(self._start_time, 1))),
        )

    def interp(self, time: float, order: int = 0) -> np.ndarray:
        assert self._interp is not None and self._interp_init is not None
        interp_time = (time - self._start_time) % self._max_time + self._start_time
        return (
            self._interp(interp_time, order)
            if time >= self._start_time
            else self._interp_init(time, order)
        )

    def vectorized_interp(self, time: np.ndarray, order: int = 0) -> np.ndarray:
        assert self._interp is not None and self._interp_init is not None
        time_ar = np.repeat(
            (time > self._start_time)[:, np.newaxis], len(self.arm_joints), -1
        )
        return np.where(
            time_ar,
            self._interp(
                (time - self._start_time) % self._max_time + self._start_time, order
            ),
            self._interp_init(time, order),
        )

    def graph_interp(self, joint_idx: int, save: bool = False):
        assert self._interp is not None and joint_idx < len(self.arm_joints)
        xs = np.arange(
            self._start_time - 0.5, self._max_time + self._start_time + 0.5, 0.1
        )
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(xs, self._interp(xs)[:, joint_idx], label="S")
        ax.plot(xs, self._interp(xs, 1)[:, joint_idx], label="S'")
        ax.plot(xs, self._interp(xs, 2)[:, joint_idx], label="S''")
        ax.plot(xs, self._interp(xs, 3)[:, joint_idx], label="S'''")
        ax.set_xlim(self._start_time - 0.5, self._max_time + self._start_time + 0.5)

        ax.legend(loc="lower left", ncol=2)

        if save:
            plt.savefig("joints.png")
        else:
            plt.show()

    def _graph_all_interp(
        self,
        interp: CubicSpline,
        *,
        xmin: float,
        xmax: float,
        x_pts: List[float],
        y_pts: List[List[float]],
        save: bool = False,
        plot_prefix: str = "",
        vlines: List[float] = [],
    ) -> None:
        xs = np.linspace(xmin, xmax, 100)
        pos_fig, pos_ax = plt.subplots(figsize=(6.5, 4))
        vel_fig, vel_ax = plt.subplots(figsize=(6.5, 4))
        acc_fig, acc_ax = plt.subplots(figsize=(6.5, 4))

        labels = np.arange(len(self.arm_joints))
        pos_ax.plot(x_pts, y_pts, marker="o", linestyle="none")
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
            pos_fig.savefig(f"{plot_prefix}joint_pos.png")
            vel_fig.savefig(f"{plot_prefix}joint_vel.png")
            acc_fig.savefig(f"{plot_prefix}joint_acc.png")
        else:
            plt.show()

    def graph_main_interp(
        self,
        *,
        save: bool = False,
    ) -> None:
        self._graph_all_interp(
            self._interp,
            xmin=self._start_time,
            xmax=self._max_time + self._start_time,
            x_pts=[
                *list(accumulate(self._cmd_queue[0])),
                self._max_time + self._start_time,
            ],
            y_pts=[*self._cmd_queue[1], self._cmd_queue[1][0]],
            save=save,
        )

    def graph_init_interp(
        self,
        *,
        save: bool = False,
    ) -> None:
        self._graph_all_interp(
            self._interp_init,
            xmin=0.0,
            xmax=self._start_time,
            x_pts=self._init_cmd_queue[0],
            y_pts=self._init_cmd_queue[1],
            save=save,
            plot_prefix="init_",
        )

    def graph_full_interp(self, *, save: bool = False) -> None:
        x_pts = [
            *self._init_cmd_queue[0],
            *list(accumulate(self._cmd_queue[0])),
            self._max_time + self._start_time,
        ]
        y_pts = [*self._init_cmd_queue[1], *self._cmd_queue[1], self._cmd_queue[1][0]]
        self._graph_all_interp(
            self.vectorized_interp,
            xmin=0.0,
            xmax=self._max_time + self._start_time,
            x_pts=x_pts,
            y_pts=y_pts,
            save=save,
            plot_prefix="full_",
            vlines=[self._start_time],
        )


if __name__ == "__main__":
    ChannelFactoryInitialize(1, "lo")

    controller = JointAnglesController()
    controller.init(DISCO_CMD)
    controller.start()

    time.sleep(1)
    # controller.graph_main_interp(save=True)
    # controller.graph_init_interp(save=True)
    controller.graph_full_interp(save=True)
    print("Created graphs")

    while True:
        time.sleep(1)
        if controller.done:
            print("Done!")
            sys.exit(-1)
