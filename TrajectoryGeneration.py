from __future__ import annotations

import math
from typing import List

from Util import SplineSegment


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

    @staticmethod
    def from_spline_segment(ss: SplineSegment) -> Segment:
        return Segment(0, ss.vel, 0, ss.pose.theta, 0, ss.pose.x, ss.pose.y)

    def __str__(self):
        return str(self.pos) + "," + str(self.vel) + "," + str(
            self.acc) + "," + str(self.x) + "," + str(self.y)

    def human_readable_str(self):
        return "pos, vel, acc, (x, y): " + str(self.pos) + ", " + str(self.vel) + ", " + str(
            self.acc) + ", " + " (" + str(self.x) + ", " + str(self.y) + ")"


def make_left_right_trajectories(traj: List[SplineSegment], wheelbase_width):
    output = [[], [], []]
    for seg in traj:
        output[0].append(Segment.from_spline_segment(seg))
        output[1].append(Segment.from_spline_segment(seg))
        output[2].append(Segment.from_spline_segment(seg))

    left = output[0]
    right = output[1]

    for i in range(len(traj)):
        current = Segment.from_spline_segment(traj[i])
        cos_angle = math.cos(current.heading)
        sin_angle = math.sin(current.heading)

        left_seg = left[i]
        if i != 0:
            left_seg.dt = traj[i].time - traj[i - 1].time
        else:
            left_seg.dt = traj[i + 1].time - traj[i].time
        left_seg.x = current.x - wheelbase_width / 2.0 * sin_angle
        left_seg.y = current.y + wheelbase_width / 2.0 * cos_angle

        if i > 0:
            dist = math.sqrt(pow(left_seg.x - left[i - 1].x, 2)
                             + pow((left_seg.y - left[i - 1].y), 2))

            left_seg.pos = left[i - 1].pos + dist
            left_seg.vel = dist / left_seg.dt
            left_seg.acc = (left_seg.vel - left[i - 1].vel) / left_seg.dt

        right_seg = right[i]

        if i != 0:
            right_seg.dt = traj[i].time - traj[i - 1].time
        else:
            right_seg.dt = traj[i + 1].time - traj[i].time
        right_seg.x = current.x + wheelbase_width / 2.0 * sin_angle
        right_seg.y = current.y - wheelbase_width / 2.0 * cos_angle

        if i > 0:
            dist = math.sqrt(pow(right_seg.x - right[i - 1].x, 2)
                             + pow((right_seg.y - right[i - 1].y), 2))

            right_seg.pos = right[i - 1].pos + dist
            right_seg.vel = dist / right_seg.dt
            right_seg.acc = (right_seg.vel - right[i - 1].vel) / right_seg.dt

    for i in range(len(output[0])):
        output[2][i].pos = (output[0][i].pos + output[1][i].pos) / 2
        output[2][i].vel = (output[0][i].vel + output[1][i].vel) / 2
        output[2][i].acc = (output[0][i].acc + output[1][i].acc) / 2
        output[2][i].heading = (output[0][i].heading + output[1][i].heading) / 2
        output[2][i].dt = (output[0][i].dt + output[1][i].dt) / 2
        output[2][i].x = (output[0][i].x + output[1][i].x) / 2
        output[2][i].y = (output[0][i].y + output[1][i].y) / 2

    return output
