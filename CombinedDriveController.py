from PurePursuitController import PurePursuit
from typing import List

from TrajectoryGeneration import Segment
from Util import Pose
import numpy as np


class CombinedDriveController(object):
    pure_pursuit: PurePursuit = None
    left_right_traj: List[List[Segment]] = None
    max_vel = 0
    lookahead_pose = []
    closest_seg = []

    def __init__(self, pure_pursuit_controller: PurePursuit, left_right_traj: List[List[Segment]], max_vel):
        self.pure_pursuit = pure_pursuit_controller
        self.left_right_traj = left_right_traj
        self.max_vel = max_vel

    def calc_combined_output(self, robot_pose: Pose):
        pp = self.pure_pursuit
        pp.update_pose(robot_pose)
        closest_point = pp.closest_point_idx()
        self.closest_seg = pp.seg_list[closest_point]
        vel = self.left_right_traj[2][closest_point].vel
        self.lookahead_pose = pp.calc_lookahead_point(vel)
        left_right_vel = pp.left_right_speeds(vel, pp.curvature_to_lookahead(self.lookahead_pose, vel))
        left = (left_right_vel[0] + self.left_right_traj[0][closest_point].vel)/2
        right = (left_right_vel[1] + self.left_right_traj[1][closest_point].vel)/2
        left = np.sign(left) * min(abs(left), self.max_vel)
        right = np.sign(right) * min(abs(right), self.max_vel)
        return [left, right]

    def is_finished(self, points_till_end) -> bool:
        return self.pure_pursuit.closest_point_idx() > len(self.pure_pursuit.seg_list) - points_till_end - 1

