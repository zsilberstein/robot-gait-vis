"""Example module that animates the walking motion of a 
biped and hexapod robot."""
import numpy as np
import robot_gait_vis as rgv


def biped_example() -> None:
    # Create a PlanarLeg and a biped Robot
    l1 = 0.5
    l2 = l1
    planar_leg = rgv.PlanarLeg((l1, l2))
    biped = rgv.Robot((0.25, 0.25, 0.5), planar_leg, 2)

    # Set gait parameters (m)
    stance_dist = 1/3
    raise_height = 1/12

    # Alternating gait
    gait_type = rgv.AlternatingGait(biped, 30)

    # Number of steps to take
    n_steps = 3

    # Time for one gait cycle (s)
    cycle_time = 2

    # Find initial joint angles for each leg
    start_thetas = {}
    gait_height = planar_leg.forward_kinematics((np.pi/4, np.pi/4))[-1]
    for leg_name in biped.leg_names:
        start_thetas[leg_name] = planar_leg.inverse_kinematics(
            (stance_dist, 0, gait_height[2]))[0]
    stance_vec = (-stance_dist, 0, 0)
    raise_vec = (stance_dist/2, 0, raise_height)

    # Create gait for each leg
    gait = rgv.gait_gen.create_gait(biped,
                                    gait_type,
                                    start_thetas,
                                    stance_vec,
                                    raise_vec,
                                    stance_vec,
                                    raise_vec)

    # Create simulation
    dt = cycle_time / gait_type.points
    biped_sim = rgv.Simulate(biped, dt, gait[-1])

    # Move robot
    for _ in range(n_steps):
        for thetas in gait:
            biped_sim.move_robot(thetas)

    # Save simulation
    biped_sim.animate('biped_ex.gif')


def hexapod_example() -> None:
    # Create a ThreeDOFLeg and a hexapod Robot
    l1 = 0.5
    l2 = 1.5
    l3 = l2 * 0.63 / 0.37
    three_dof_leg = rgv.ThreeDOFLeg((l1, l2, l3))
    hex_robot = rgv.Robot((4, 1, 1), three_dof_leg, 6)

    # Set gait parameters (m)
    stance_dist = 0.5
    raise_height = 0.5

    # Alternating gait
    gait_type = rgv.RippleGait(hex_robot, 30)

    # Number of steps to take
    n_steps = 3

    # Time for one gait cycle (s)
    cycle_time = 2

    start_thetas = {}
    gait_height = three_dof_leg.forward_kinematics(
        (0, np.pi/4, np.pi/4))[-1]
    for leg_name in hex_robot.leg_names:
        # Flip y coord of right leg so that it maps in the world frame
        if leg_name[0] == 'R':
            y = -gait_height[1]
        else:
            y = gait_height[1]
        start_thetas[leg_name] = three_dof_leg.inverse_kinematics(
            (stance_dist/2, y, gait_height[2]))[0]

    stance_vec = (-stance_dist, 0, 0)
    raise_vec_right = (stance_dist/2, -raise_height/2, raise_height)
    raise_vec_left = (stance_dist/2, raise_height/2, raise_height)

    # Create gait
    gait = rgv.gait_gen.create_gait(hex_robot,
                                    gait_type,
                                    start_thetas,
                                    stance_vec,
                                    raise_vec_right,
                                    stance_vec,
                                    raise_vec_left)

    # Create simulation
    dt = cycle_time / gait_type.points
    hex_sim = rgv.Simulate(hex_robot, dt, gait[-1])

    # Move robot
    for _ in range(n_steps):
        for thetas in gait:
            hex_sim.move_robot(thetas)

    # Save simulation
    hex_sim.animate('hexapod_ex.gif')


def main() -> None:
    biped_example()
    hexapod_example()


if __name__ == '__main__':
    main()
