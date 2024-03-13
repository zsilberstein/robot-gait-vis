"""Example module that animates the walking motion of a biped robot."""
import numpy as np
from robot_gait_vis.planar_leg import PlanarLeg
from robot_gait_vis.robot import Robot
from robot_gait_vis.simulate import Simulate
from robot_gait_vis import gait_gen


def biped_example() -> None:
    # Create a PlanarLeg and a biped Robot
    l1 = 0.5
    l2 = l1
    planar_leg = PlanarLeg((l1, l2))
    biped = Robot((0.25, 0.25, 0.5), planar_leg, 2)

    # Set gait parameters (m)
    stance_dist = 1/3
    raise_height = 1/12

    # Alternating gait
    duty_factor = 1/2
    stance_start = {'R0': 0,
                    'L0': 1/2}

    # Number of points to compute
    points = 30

    # Number of steps to take
    n_steps = 3

    # Time for one gait cycle (s)
    cycle_time = 2

    # Find initial joint angles for each leg
    start_thetas = {}
    gait_height = biped.leg_type.forward_kinematics(
        np.deg2rad([45, 45]))[-1]
    for leg_name in biped.leg_names:
        start_thetas[leg_name] = biped.leg_type.inverse_kinematics(
            [stance_dist, 0, gait_height[2]])[0]
    stance_vec = (-stance_dist, 0, 0)
    raise_vec = (stance_dist/2, 0, raise_height)

    # Create trajectory for each leg
    trajectory = gait_gen.create_ellipse_trajectory(
        biped,
        start_thetas,
        stance_vec,
        raise_vec,
        stance_vec,
        raise_vec,
        points,
        duty_factor)

    # Create gait
    gait = gait_gen.get_gait(trajectory, stance_start)

    # Create simulation
    dt = cycle_time / points
    biped_sim = Simulate(biped, dt, gait[-1])
    
    # Move robot
    for _ in range(n_steps):
        for thetas in gait:
            biped_sim.move_robot(thetas)

    # Save simulation
    biped_sim.animate('biped_ex.gif')
    # Plot history
    biped_sim.plot_joint_history()
    biped_sim.plot_xyz_history()


def main() -> None:
    biped_example()


if __name__ == '__main__':
    main()
