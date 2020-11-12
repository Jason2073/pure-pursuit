import math
from typing import List

from Util import TrajectoryConfig


class Segment(object):
    pos = 0.0
    vel = 0.0
    acc = 0.0
    heading = 0.0
    dt = 0.0
    x = 0.0
    y = 0.0

    def __init__(self, pos=0.0, vel=0.0, acc=0.0, heading=0.0, dt=0.0, x=0.0, y=0.0):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.heading = heading
        self.dt = dt
        self.x = x
        self.y = y

    def __str__(self):
        return str(self.pos) + "," + str(self.vel) + "," + str(
            self.acc) + "," + str(self.x) + "," + str(self.y)

    def human_readable_str(self):
        return "pos, vel, acc, (x, y): " + str(self.pos) + ", " + str(self.vel) + ", " + str(
            self.acc) + ", " + " (" + str(self.x) + ", " + str(self.y) + ")"


class Trajectory(object):
    segments: List[Segment] = []

    def __init__(self, length: int):
        for i in range(length):
            self.segments.append(Segment())

    def scale(self, factor: float):
        for i in range(len(self.segments)):
            self.segments[i].pos *= factor
            self.segments[i].vel *= factor
            self.segments[i].acc *= factor

    def relative_pos(self, start: float):
        for i in range(len(self.segments)):
            self.segments[i].pos += start

    def get_index(self, idx: int) -> Segment:
        return self.segments[idx]


class MotionProfile(object):
    total_time = 0
    config: TrajectoryConfig = None
    distance = 0.0
    start_vel = 0.0
    end_vel = 0.0
    t1 = 0
    t2 = 0
    p1 = 0
    p2 = 0

    def __init__(self, config: TrajectoryConfig, distance: float, start_vel: float, end_vel: float):
        self.config = config
        self.distance = distance
        self.start_vel = start_vel
        self.end_vel = end_vel
        self.t1 = config.max_v / config.max_a
        self.p1 = .5 * self.t1 * config.max_v
        self.p2 = distance - self.p1
        self.t2 = self.t1 + (self.p2 - self.p1) / config.max_v
        self.total_time = self.t2 + self.t1

    def at_time(self, t):
        if 0 <= t < self.t1:
            acc = self.config.max_a
            vel = acc * t
            pos = .5 * acc * pow(t, 2)
        elif self.t1 <= t < self.t2:
            acc = 0
            vel = self.config.max_v
            pos = self.p1 + vel * (t - self.t1)
        elif self.t2 <= t < self.total_time:
            acc = -self.config.max_a
            vel = self.config.max_v + acc * (t - self.t2)
            pos = self.p2 + vel * (t - self.t2) - .5 * acc * pow(t - self.t2, 2)
        else:
            acc = 0
            vel = 0
            pos = self.distance

        return [pos, vel, acc]


def make_left_right_trajectories(traj: Trajectory, wheelbase_width):
    output = [traj, traj, traj]
    left = output[0]
    right = output[1]

    for i in range(len(traj.segments)):
        current = traj.segments[i]
        cos_angle = math.cos(current.heading)
        sin_angle = math.sin(current.heading)

        left_seg = left.segments[i]
        left_seg.x = current.x - wheelbase_width / 2.0 * sin_angle
        left_seg.y = current.y + wheelbase_width / 2.0 * cos_angle

        if i > 0:
            dist = math.sqrt(pow(left_seg.x - left.segments[i - 1].x, 2)
                             + pow((left_seg.y - left.segments[i - 1].y), 2))
            left_seg.pos = left.segments[i - 1].pos + dist
            left_seg.vel = dist / left_seg.dt
            left_seg.acc = (left_seg.vel - left.segments[i - 1].vel) / left_seg.dt

        right_seg = right.segments[i]
        right_seg.x = current.x + wheelbase_width / 2.0 * sin_angle
        right_seg.y = current.y - wheelbase_width / 2.0 * cos_angle

        if i > 0:
            dist = math.sqrt(pow(right_seg.x - right.segments[i - 1].x, 2)
                             + pow((right_seg.y - right.segments[i - 1].y), 2))
            right_seg.pos = right.segments[i - 1].pos + dist
            right_seg.vel = dist / right_seg.dt
            right_seg.acc = (right_seg.vel - right.segments[i - 1].vel) / right_seg.dt

    return output


def choose_strategy(start_vel, goal_vel, max_vel):
    strat = "trapezoid"
    if start_vel == goal_vel and start_vel == max_vel:
        strat = "step"
    elif start_vel == goal_vel and start_vel == 0:
        strat = "scurve"

    return strat
