from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

from robot_gait_vis.robot import Robot


class Simulate():
    """Class to simulate the movement of a Robot."""

    def __init__(self,
                 robot: Robot,
                 dt: float,
                 init_thetas: Dict[str, List[float]]) -> None:
        """
        Args:
            robot (Robot): Robot to simulate.
            dt (float): Time between each robot command in seconds.
            init_thetas (Dict[str, List[float]]): Dictionary of initial joint 
            angles for each leg on the robot.

        Raises:
            KeyError: Raises error if not all legs of the robot are given 
            initial joint angles.
        """
        self.robot = robot
        self.dt = dt

        for leg_name in self.robot.leg_names:
            if init_thetas.get(leg_name):
                self._update_leg_pos(leg_name, init_thetas[leg_name])
            else:
                raise KeyError(
                    f'Leg {leg_name} was not given initial joint angles')

        # Ground depth is at lowest leg in Z direction
        self.ground = np.min([self.robot.legs_pos_global[leg_name][0][-1][2]
                              for leg_name in self.robot.legs_pos_global])
        for leg_name in self.robot.leg_names:
            self._update_leg_down(leg_name)
        # Increase number of frames
        self.robot.frames += 1

    def move_robot(self, thetas: Dict[str, List[float]]) -> None:
        """Updates the position of the robot given joint angles for the legs.

        Args:
            thetas (Dict[str, List[float]]): Dictionary of joint angles for 
            each leg to move to in radians.
        """
        num_legs_stance = 0
        for leg_name in self.robot.leg_names:
            if thetas.get(leg_name):
                self._update_leg_pos(leg_name, thetas[leg_name])
                self._update_leg_down(leg_name)

            # If no joint angles for leg are provided, leave leg at last position
            else:
                self.robot.joint_angles[leg_name].append(
                    self.robot.joint_angles[leg_name][-1])
                self.robot.legs_pos_local[leg_name].append(
                    self.robot.legs_pos_local[leg_name][-1])
                self.robot.legs_pos_global[leg_name].append(
                    self.robot.legs_pos_global[leg_name][-1])
                self.robot.leg_down[leg_name].append(
                    self.robot.leg_down[leg_name][-1])

            # Stance occurs when leg was on ground and is still on ground
            if self.robot.leg_down[leg_name][-1] \
                    and self.robot.leg_down[leg_name][-2]:
                num_legs_stance += 1

        translate = 0
        for leg_name in self.robot.leg_names:
            # If stance occurred
            if self.robot.leg_down[leg_name][-1] \
                    and self.robot.leg_down[leg_name][-2]:
                stance_dist = np.subtract(
                    self.robot.legs_pos_local[leg_name][-1][-1],
                    self.robot.legs_pos_local[leg_name][-2][-1])
                # Stance in opposite direction of motion, total body
                #   translation is average of stance distance by each leg
                #   that is in stance
                translate += -1 * stance_dist / num_legs_stance

        # Update the leg attachment and leg points
        #   to move with the body
        self._translate_robot(translate)
        # Increase number of frames
        self.robot.frames += 1

    def _update_leg_pos(self, leg_name: str, joint_angles: List[float]):
        """Updates the history of a leg with the given joint angles.

        Args:
            leg_name (str): Key for the leg name in the robot.
            joint_angles (List[float]): Joint angles to move the leg to.
        """
        self.robot.joint_angles[leg_name].append(joint_angles)
        # Local frame is found with forward kinematics
        local_pos = self.robot.leg_type.forward_kinematics(joint_angles)
        self.robot.legs_pos_local[leg_name].append(local_pos)

        # Global position is found by moving from
        #   local frame base to leg attachment point
        global_pos = [np.add(local,
                             self.robot.leg_attach[leg_name][-1]).tolist()
                      for local in local_pos]
        self.robot.legs_pos_global[leg_name].append(global_pos)

    def _update_leg_down(self, leg_name: str):
        """Updates the leg_down property.

        Args:
            leg_name (str): Key for the leg name in the robot.
        """
        # Give a buffer to the ground
        global_pos = self.robot.legs_pos_global[leg_name][-1]
        down = global_pos[-1][2] - self.ground < 0.001
        self.robot.leg_down[leg_name].append(down)

    def _translate_robot(self, distance: List[float]):
        """Translates the robot by the given distance.

        Args:
            distance (List[float]): Vector to translate the robot as 
            (X, Y, Z) in meters.
        """
        new_pos = np.add(distance, self.robot.body_pos[-1])
        self.robot.body_pos.append(new_pos.tolist())
        for leg_name in self.robot.leg_names:
            self.robot.leg_attach[leg_name].append(
                np.add(self.robot.leg_attach[leg_name][-1],
                       distance).tolist())
            for i in range(self.robot.leg_type.num_joints):
                global_pos = self.robot.legs_pos_global[leg_name][-1][i]
                self.robot.legs_pos_global[leg_name][-1][i] = \
                    np.add(distance, global_pos).tolist()

    def animate(self, save_file: str):
        """Animates the history of the robot.

        Args:
            save_file (str): A .gif or a .mp4 file to save animation to.
        """
        return

    def plot_xyz_history(self) -> None:
        """Plots the position of the robot center of 
            body and the end-effector of each leg over time.
        """
        # Left to right is X, Y, Z vs time
        _, ax = plt.subplots(ncols=3, nrows=1)
        plt.subplots_adjust(left=0.05,
                            bottom=0.05,
                            right=0.95,
                            top=0.95,
                            wspace=0.3)

        time = np.linspace(0,
                           (self.robot.frames-1)*self.dt,
                           self.robot.frames)
        # For x, y, z
        for i in range(3):
            # Plot body pos
            ax[i].plot(time,
                       [self.robot.body_pos[j][i]
                        for j in range(self.robot.frames)],
                       label='Center of Body',
                       linestyle='--',
                       marker='o')
            # Plot end-effector pos for each leg
            for leg_name in self.robot.leg_names:
                ax[i].plot(time,
                           [self.robot.legs_pos_global[leg_name][j][-1][i]
                            for j in range(self.robot.frames)],
                           label=leg_name,
                           linestyle='--',
                           marker='o')

            ax[i].set(xlabel='Time (s)',
                      ylabel=f'{chr(ord("X")+i)} position (m)')
        # Plot legend on X plot
        ax[0].legend(loc="upper left")
        plt.show()

    def plot_joint_history(self) -> None:
        """Plots the joint angles of each leg over time."""
        # Left to right joint1, joint2, ... , jointN
        num_joints = self.robot.leg_type.num_joints
        _, ax = plt.subplots(ncols=num_joints, nrows=1)
        plt.subplots_adjust(left=0.05,
                            bottom=0.05,
                            right=0.95,
                            top=0.95)

        time = np.linspace(0,
                           (self.robot.frames-1)*self.dt,
                           self.robot.frames)
        # For each joint
        for i in range(num_joints):
            for leg_name in self.robot.leg_names:
                p, = ax[i].plot(time,
                                [self.robot.joint_angles[leg_name][j][i]
                                 for j in range(self.robot.frames)],
                                label=leg_name,
                                linestyle='--',
                                marker='o')

            ax[i].set(xlabel='Time (s)',
                      ylabel=f'Joint {i+1} (radians)')
        # Plot legend on X plot
        ax[0].legend(loc="upper left")
        plt.show()
