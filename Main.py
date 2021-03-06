import matplotlib.pyplot as plt
import numpy as np

import PathGeneration
import TrajectoryGeneration
from CombinedDriveController import CombinedDriveController
from PurePursuitController import PurePursuit
from Util import *
import random

class DriveBase(object):
    pose = Pose(0, 0, 0)
    drive_base = 0.0
    max_v = 0.0

    def __init__(self, start_pose: Pose, drive_base, max_v):
        self.pose = start_pose
        self.drive_base = drive_base
        self.max_v = max_v

    def simulate_step(self, vel_left, vel_right, dt):
        eam = (vel_left + vel_right) / 2
        ead = vel_left - vel_right

        if vel_right - vel_left == 0 or vel_right + vel_left == 0:
            if random.random() > .5:
                vel_right += .01
            else:
                vel_left += .01

            r = self.drive_base / 2 * (vel_right + vel_left) / (vel_right - vel_left)
            omega = (vel_right - vel_left) / self.drive_base

            x_icc = self.pose.x - r * np.sin(self.pose.theta)
            y_icc = self.pose.y + r * np.cos(self.pose.theta)
            # theta = theta + d_theta
            theta = self.pose.theta + omega * dt
            x = np.cos(omega * dt) * (self.pose.x - x_icc) - np.sin(omega * dt) * (self.pose.y - y_icc) + x_icc
            y = np.sin(omega * dt) * (self.pose.x - x_icc) + np.cos(omega * dt) * (self.pose.y - y_icc) + y_icc

            new_pose = Pose(x, y, theta)
            self.pose = new_pose
        else:
            r = self.drive_base / 2 * (vel_right + vel_left) / (vel_right - vel_left)
            omega = (vel_right - vel_left) / self.drive_base

            x_icc = self.pose.x - r * np.sin(self.pose.theta)
            y_icc = self.pose.y + r * np.cos(self.pose.theta)
            # theta = theta + d_theta
            theta = self.pose.theta + omega * dt
            x = np.cos(omega * dt) * (self.pose.x - x_icc) - np.sin(omega * dt) * (self.pose.y - y_icc) + x_icc
            y = np.sin(omega * dt) * (self.pose.x - x_icc) + np.cos(omega * dt) * (self.pose.y - y_icc) + y_icc

            new_pose = Pose(x, y, theta)
            self.pose = new_pose


if __name__ == "__main__":

    max_v = 6
    max_a = 10
    dt = 1 / 60.0
    drive_base_width = .4
    lookahead_at_max_vel = .3
    minimum_lookahead = .05

    ws = WaypointSequence()
    ws.add_waypoint(Pose(0, 0, math.pi / 2))
    # ws.add_waypoint(Pose(2, 2, 3 * math.pi/2))
    ws.add_waypoint(Pose(-1, 1,  math.pi))

    config = TrajectoryConfig(max_v, max_a, dt)

    trajectory = PathGeneration.generate_from_waypoints(ws, config, 0, 0)

    left_right_traj = TrajectoryGeneration.make_left_right_trajectories(trajectory, drive_base_width)
    csv = CsvWriter("C:/Users/Stanl/PycharmProjects/PurePursuit/traj.csv")
    time_list = []
    left_list = []
    right_list = []
    for i in range(len(left_right_traj[0])):
        left_list.append(left_right_traj[0][i].vel)
        right_list.append(left_right_traj[1][i].vel)
        time_list.append(left_right_traj[0][1].dt * i)

    csv.write_to_csv(time_list, left_list, right_list)

    robot = DriveBase(Pose(0, 0, 0), drive_base_width, max_v)
    pp = PurePursuit(minimum_lookahead, lookahead_at_max_vel, left_right_traj[2], drive_base_width, max_v)
    controller = CombinedDriveController(pp, left_right_traj, max_v)
    cx = []
    cy = []

    for seg in left_right_traj[2]:
        coords = [seg.x, seg.y]
        print(coords)
        cx.append(coords[0])
        cy.append(coords[1])

    time = 0
    robot_x_positions = []
    robot_y_positions = []
    for i in range(int(len(left_right_traj[2]))):
        left_right = controller.calc_combined_output(robot_pose=robot.pose)
        if controller.is_finished(3):
            break
        print(left_right)
        robot.simulate_step(left_right[0], left_right[1], dt)
        robot_x_positions.append(robot.pose.x)
        robot_y_positions.append(robot.pose.y)

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(cx, cy, "-r", label="course")
        plt.plot(robot_x_positions, robot_y_positions, "-b", label="trajectory")
        # print(controller.lookahead_pose)
        # print(str(controller.closest_seg.x) + " " +  str(controller.closest_seg.y))
        plt.plot(controller.lookahead_pose.x, controller.lookahead_pose.y, "b", label="lookahead")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(.005)
    plt.pause(5)
