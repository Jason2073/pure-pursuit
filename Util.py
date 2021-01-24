from __future__ import annotations

import math


class Pose(object):
    x = 0.0
    y = 0.0
    theta = 0.0

    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta

    def __str__(self):
        return "(x, y, theta): (" + str(self.x) + ", " + str(self.y) + ", " + str(self.theta) + ")"

    @classmethod
    def copy_of(cls, waypoint: Pose):
        return cls(waypoint.x, waypoint.y, waypoint.theta)

    def distance(self, other: Pose):
        return math.hypot(self.x - other.x, self.y - other.y)


class SplineSegment(object):
    curvature = 0
    pose = Pose(0, 0, 0)
    vel = 0
    vel_lim = 0
    time = 0

    def __init__(self, pose: Pose, curvature: float, vel: float, vel_lim: float, time: float):
        self.curvature = curvature
        self.pose = pose
        self.vel = vel
        self.vel_lim = vel_lim
        self.time = time

    def __str__(self):
        return "Pose, vel, time " + ", " + str(self.vel) + ", " + str(self.time)

    def copy(self):
        return SplineSegment(self.pose, self.curvature, self.vel, self.vel_lim, self.time)


class WaypointSequence(object):
    waypoints = []
    num_waypoints = 0

    def add_waypoint(self, waypoint: Pose) -> WaypointSequence:
        self.waypoints.append(waypoint)
        self.num_waypoints = len(self.waypoints)
        return self


class TrajectoryConfig(object):
    max_v = 0.0
    max_a = 0.0
    interval = 0.0

    def __init__(self, max_v: float, max_a: float, interval: float):
        self.max_v = max_v
        self.max_a = max_a
        self.interval = interval


class CsvWriter(object):
    file_path = ""

    def __init__(self, file_path):
        self.file_path = file_path

    def write_to_csv(self, *lists):
        f = open(self.file_path, 'w+')
        for i in range(len(lists[0])):
            line = ""
            for data in lists:
                line += str(data[i]) + ','
            line += "\n"
            f.write(line)
        f.close()

    def write_lines(self, lines):
        f = open(self.file_path, 'w+')
        for line in lines:
            f.write(line)
        f.close()
