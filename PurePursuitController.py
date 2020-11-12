from typing import List

import numpy as np

from TrajectoryGeneration import Trajectory
from Util import *

class PurePursuit(object):
    robot_pose = Pose(0, 0, 0)
    lookahead_distance = 0.0
    track_width = 0.0
    traj: Trajectory = None
    pose_list: List[Pose] = []
    last_closest_point_idx = 1
    last_lookahead_frac_inx = 0
    max_v = 0

    def __init__(self, lookahead_distance: float, traj: Trajectory, track_width, max_vel: float):

        self.lookahead_distance = lookahead_distance
        self.track_width = track_width
        self.seg_list = traj.segments
        self.traj = traj
        self.max_v = max_vel
        for seg in traj.segments:
            self.pose_list.append(Pose(seg.x, seg.y, seg.heading))

    #call once per update cycle
    def update_pose(self, robot_pose: Pose):
        self.robot_pose = robot_pose
        print(self.calc_closest_point_idx(robot_pose))

    def closest_point_idx(self):
        return min(self.last_closest_point_idx, len(self.seg_list))

    def calc_closest_point_idx(self, pose: Pose) -> int:
        min_dist = float("inf")
        min_idx = self.last_closest_point_idx

        for i in range(min_idx, len(self.seg_list) - 1):
            path_pose = self.pose_list[i]
            if pose.distance(path_pose) < min_dist:
                min_dist = pose.distance(path_pose)
                min_idx = i

        self.last_closest_point_idx = min_idx
        return min_idx

    def calc_lookahead_point(self) -> Pose:
        robot_pose = self.robot_pose
        closest_idx = self.closest_point_idx()
        path: List[Pose] = self.pose_list
        intersect_t = 0
        fractional_idx = 0
        point = Pose(0, 0, 0)
        for i in range(math.floor(self.last_lookahead_frac_inx), len(path) - 1):
            seg_start = [path[i].x, path[i].y]
            seg_end = [path[i + 1].x, path[i + 1].y]
            c = [robot_pose.x, robot_pose.y]
            r = self.lookahead_distance

            d = [seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]]
            f = [seg_start[0] - c[0], seg_start[1] - c[1]]

            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - pow(r, 2)

            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                # no intersection
                pass
            else:
                discriminant = math.sqrt(discriminant)
                t1 = (-b - discriminant) / (2 * a)
                t2 = (-b + discriminant) / (2 * a)
                if 0 <= t1 <= 1:
                    fractional_idx = i + t1
                    point = Pose(seg_start[0] + t1 * d[0], seg_start[1] + t1 * d[1],
                                 path[closest_idx].theta)
                    break
                if 0 <= t2 <= 1:
                    fractional_idx = i + t2
                    point = Pose(seg_start[0] + t2 * d[0], seg_start[1] + t2 * d[1],
                                 path[closest_idx].theta)
                    break

        self.last_lookahead_frac_inx = fractional_idx
        return point

    def left_right_speeds(self, target_vel, curvature) -> []:
        left = target_vel * (2.0 + curvature * self.track_width) / 2.0
        right = target_vel * (2.0 - curvature * self.track_width) / 2.0
        if left > self.max_v or right > self.max_v:
            if left > right:
                right = (right / left) * self.max_v
                left = self.max_v
            elif right > left:
                left = (left / right) * self.max_v
                right = self.max_v

        return [left, right]

    def curvature_to_lookahead(self, lookahead_point: Pose):
        # positive curvature is a right turn for these calcs
        robot_pose = self.robot_pose

        a = -math.tan(robot_pose.theta)
        b = 1
        c = math.tan(robot_pose.theta) * robot_pose.x - robot_pose.y
        # point line distance formula, x = |ax + by +c| / sqrt(a^2 + b^2), where x is an unsigned distance
        x = abs(a * lookahead_point.x + b * lookahead_point.y + c) / math.sqrt(pow(a, 2) + pow(b, 2))
        side = np.sign(math.sin(robot_pose.theta) * (lookahead_point.x - robot_pose.x)
                       - math.cos(robot_pose.theta) * (lookahead_point.y - robot_pose.y))

        curvature = 2.0 * x / (math.pow(self.lookahead_distance, 2))
        return side * curvature
