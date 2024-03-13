from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

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

            # If no joint angles for leg are provided,
            #   leave leg at last position
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

    def _update_leg_pos(self,
                        leg_name: str,
                        joint_angles: List[float]) -> None:
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

    def _update_leg_down(self, leg_name: str) -> None:
        """Updates the leg_down property.

        Args:
            leg_name (str): Key for the leg name in the robot.
        """
        # Give a buffer to the ground
        global_pos = self.robot.legs_pos_global[leg_name][-1]
        down = global_pos[-1][2] - self.ground < 0.001
        self.robot.leg_down[leg_name].append(down)

    def _translate_robot(self, distance: List[float]) -> None:
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

    def animate(self, save_file: str) -> None:
        """Animates the history of the robot.

        Args:
            save_file (str): A .gif or a .mp4 file to save animation to.

        Raises:
            ValueError: Raises error if file type is not .gif or .mp4.
        """
        if save_file[-3:] == 'mp4':
            writer = animation.FFMpegWriter(fps=1/self.dt)
        elif save_file[-3:] == 'gif':
            writer = animation.PillowWriter(fps=1/self.dt)
        else:
            raise ValueError('File must be either .gif or .mp4')

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d", computed_zorder=False)

        # Find plot limits
        lims = []
        for i in range(3):
            joints = [self.robot.legs_pos_global[leg_name][frame][joint][i]
                      for joint in range(self.robot.leg_type.num_joints)
                      for frame in range(self.robot.frames)
                      for leg_name in self.robot.leg_names]
            leg_attach = [self.robot.leg_attach[leg_name][frame][i]
                          for frame in range(self.robot.frames)
                          for leg_name in self.robot.leg_names]
            low = np.min(joints+leg_attach)
            # Don't lower ground
            if i < 2:
                low -= self.robot.body_dims[i]
            high = np.max(joints+leg_attach) + self.robot.body_dims[i]
            lims.append([low, high])
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.zaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.set(xlim=lims[0],
               ylim=lims[1],
               zlim=lims[2],

               xlabel='X [m]',
               ylabel='Y [m]',
               zlabel='Z [m]',
               title=f'Time = {0}s',
               aspect='equal')

        # Plot ground
        x_grid_ground, y_grid_ground = np.meshgrid(lims[0], lims[1])
        z_grid_ground = np.array([self.ground]*4).reshape(2, 2)
        ax.plot_surface(x_grid_ground,
                        y_grid_ground,
                        z_grid_ground,
                        alpha=1,
                        color='C0')

        # Fill in leg coordinates and plot
        legs = []
        for leg_name in self.robot.leg_names:
            leg_pos = self._get_leg_positions(leg_name, 0)
            # Plot right legs further out, hide left legs behind body
            if leg_name[0] == 'R':
                legs.append(ax.plot(leg_pos[0],
                                    leg_pos[1],
                                    leg_pos[2],
                                    color='C3',
                                    zorder=3,
                                    linewidth=2)[0])
            else:
                legs.append(ax.plot(leg_pos[0],
                                    leg_pos[1],
                                    leg_pos[2],
                                    color='C3',
                                    zorder=1,
                                    linewidth=2)[0])
        # Plot body
        box_surfaces = []
        surfs = self._get_box_surfaces(0)
        for surf in surfs:
            x_grid, y_grid, z_grid = surf
            box_surfaces.append(ax.plot_surface(x_grid,
                                                y_grid,
                                                z_grid,
                                                color='k',
                                                alpha=0.75,
                                                zorder=2))

        def update(frame):
            # Update title
            ax.set(title=f'Time = {self.dt*frame: .3f}s')
            # Update legs
            for i, leg_name in enumerate(self.robot.leg_names):
                leg_pos = self._get_leg_positions(leg_name, frame)
                legs[i].set_xdata(leg_pos[0])
                legs[i].set_ydata(leg_pos[1])
                legs[i].set_3d_properties(leg_pos[2])
            # Update body
            surfs = self._get_box_surfaces(frame)
            for i, surf in enumerate(surfs):
                box_surfaces[i].remove()
                x_grid, y_grid, z_grid = surf
                box_surfaces[i] = ax.plot_surface(x_grid,
                                                  y_grid,
                                                  z_grid,
                                                  color='k',
                                                  alpha=0.75,
                                                  zorder=2)

            return (legs, box_surfaces)

        # Animate and save
        ani = animation.FuncAnimation(fig=fig,
                                      func=update,
                                      frames=self.robot.frames,
                                      interval=self.dt*1000)
        ani.save(save_file, writer=writer)
        plt.close()

    def _get_leg_positions(self,
                           leg_name: str,
                           frame: int) -> List[List[float]]:
        """Transforms positions of the joints of a leg for plotting.

        Args:
            leg_name (str): Name of leg to compute leg positions for.
            frame (int): Frame to compute leg positions for.

        Returns:
            List[List[float]]: List of length 3 specifying the x, y, z 
            positions of the leg attachment point and each joint on leg. 
        """
        leg_pos = []
        # For x, y, z
        for i in range(3):
            legs = [self.robot.leg_attach[leg_name][frame][i]] \
                + [self.robot.legs_pos_global[leg_name][frame][j][i]
                   for j in range(self.robot.leg_type.num_joints)]
            leg_pos.append(legs)
        return leg_pos

    def _get_box_surfaces(self, frame: int) -> List[List[List[float]]]:
        """Generates six surfaces to be plotted that represent the body of 
        the robot.

        Args:
            frame (int): Frame to plot the body.

        Returns:
            List[List[List[float]]]: List of surfaces to be plotted.
        """
        body_pos = self.robot.body_pos[frame]
        body_dims = self.robot.body_dims
        # Determine the max and min of the body in x, y, z
        ranges = []
        for i in range(3):
            range_i = [body_pos[i]-body_dims[i]/2, body_pos[i]+body_dims[i]/2]
            ranges.append(range_i)

        surfaces = []
        x_order = [0, 0, 1]
        y_order = [1, 2, 2]
        z_order = [2, 1, 0]

        for x, y, z in zip(x_order, y_order, z_order):
            a_grid, b_grid = np.meshgrid(ranges[x], ranges[y])
            # Get the two parallel surfaces (top and bottom / left and right)
            for i in range(2):
                c_grid = np.array([ranges[z][i]]*4).reshape(2, 2)
                temp = [0, 0, 0]
                temp[x] = a_grid
                temp[y] = b_grid
                temp[z] = c_grid
                surfaces.append(temp)

        return surfaces

    def plot_xyz_history(self) -> None:
        """Plots the position of the robot center of 
            body and the end-effector of each leg over time.
        """
        # Left to right is X, Y, Z vs time
        fig, ax = plt.subplots(ncols=3, nrows=1)
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
        fig.suptitle('Position vs Time')
        plt.show()

    def plot_joint_history(self) -> None:
        """Plots the joint angles of each leg over time."""
        # Left to right joint1, joint2, ... , jointN
        num_joints = self.robot.leg_type.num_joints
        fig, ax = plt.subplots(ncols=num_joints, nrows=1)
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
                ax[i].plot(time,
                           [self.robot.joint_angles[leg_name][j][i]
                            for j in range(self.robot.frames)],
                           label=leg_name,
                           linestyle='--',
                           marker='o')

            ax[i].set(xlabel='Time (s)',
                      ylabel=f'Joint {i+1} (radians)')
        # Plot legend on X plot
        ax[0].legend(loc="upper left")
        fig.suptitle('Joint Angles vs Time')
        plt.show()
