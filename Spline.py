from __future__ import annotations

import numpy as np

from Util import *

SEGMENT_RESOLUTION_FACTOR = 3


class HermiteSpline(object):

    # based on https://ieeexplore.ieee.org/document/5641305
    SMOOTHNESS_FACTOR = 1.5
    coeffs = np.zeros([4, 2])
    xu: np.polynomial.Polynomial = None
    yu: np.polynomial.Polynomial = None
    M = np.array([
        [2, -2, 1, 1],
        [-3, 3, -2, -1],
        [0, 0, 1, 0],
        [1, 0, 0, 0]])

    @staticmethod
    def interpolate(p0: Pose, p1: Pose):
        # TODO: does the scale on cx0... matter? or is it ONLY orientation, paper says orientation, but im not sure.
        # TODO: HOW DOES THIS WORK? what does the two represent?
        cx0, cy0 = [HermiteSpline.SMOOTHNESS_FACTOR * math.sin(p0.theta),
                    HermiteSpline.SMOOTHNESS_FACTOR * math.cos(p0.theta)]
        cx1, cy1 = [HermiteSpline.SMOOTHNESS_FACTOR * math.sin(p1.theta),
                    HermiteSpline.SMOOTHNESS_FACTOR * math.cos(p1.theta)]

        G = np.array([
            [p0.x, p0.y],
            [p1.x, p1.y],
            [cx0, cy0],
            [cx1, cy1]])

        coeffs = np.matmul(HermiteSpline.M, G)
        result = HermiteSpline()
        result.coeffs = coeffs
        result.xu = np.polynomial.Polynomial([coeffs[3, 0], coeffs[2, 0], coeffs[1, 0], coeffs[0, 0]])
        result.yu = np.polynomial.Polynomial([coeffs[3, 1], coeffs[2, 1], coeffs[1, 1], coeffs[0, 1]])
        return result

    def get_xy(self, u):
        return [self.xu(u), self.yu(u)]

    def calc_curvature(self, u: float):
        num = self.xu.deriv(1)(u) * self.yu.deriv(2)(u) - self.yu.deriv(1)(u) * self.xu.deriv(2)(u)
        denom = (self.xu.deriv(1)(u) ** 2 + self.yu.deriv(1)(u) ** 2) ** (3 / 2)
        return num / denom

    def velocity_profile(self, start_vel: float, end_vel: float, dt: float, max_v: float, max_a: float):
        us = np.linspace(0, 1, int(3 / dt))
        list = []

        for u in us:
            curvature = self.calc_curvature(u)
            list.append(
                SplineSegment(Pose(self.xu(u), self.yu(u), math.atan2(self.yu.deriv(1)(u), self.xu.deriv(1)(u))),
                              curvature, max_v, max_v, 0))

        # forward pass

        start_pose = Pose(self.xu(0), self.yu(0), math.atan2(self.yu.deriv(1)(0), self.xu.deriv(1)(0)))
        list[0] = SplineSegment(start_pose,
                                self.calc_curvature(0),
                                start_vel,
                                start_vel,
                                0)
        first_at = math.sqrt(max_a ** 2 - start_vel ** 2 * self.calc_curvature(us[1]) ** 2)
        first_ds = Pose(self.xu(us[1]), self.yu(us[1]), 0.0).distance(start_pose)
        first_dt = ((-start_vel + math.sqrt(start_vel ** 2 + 2 * first_at * first_ds)) / first_at)
        if dt < 0:
            first_dt = ((-start_vel - math.sqrt(start_vel ** 2 + 2 * first_at * first_ds)) / first_at)

        list[1] = SplineSegment(
            Pose(self.xu(us[1]), self.yu(us[1]), math.atan2(self.yu.deriv(1)(us[1]), self.xu.deriv(1)(us[1]))),
            self.calc_curvature(us[1]),
            start_vel + first_at * first_dt,
            max_v,
            first_dt)
        sc = list[0].copy()
        sn = list[1].copy()
        at_list = []
        at_list.append(first_at)
        for i in range(1, len(us) - 1):
            ds = list[i + 1].pose.distance(list[i].pose)
            ar = sc.vel ** 2 * sc.curvature
            if abs(ar) > max_a:
                ar = np.sign(ar) * max_a
            at_lim = math.sqrt(max_a ** 2 - ar ** 2)
            v_lim = math.sqrt(sc.vel ** 2 + 2 * at_lim * ds)
            sn.vel = min(sn.vel, v_lim, sn.vel_lim, math.sqrt(max_a) / abs(sc.curvature))
            at_list.append(at_lim)
            list[i] = sn.copy()
            sc = sn.copy()
            sn = list[i + 1].copy()

        end_pose = Pose(self.xu(1), self.yu(1), math.atan2(self.yu.deriv(1)(1), self.xu.deriv(1)(1)))
        list[len(list) - 1] = SplineSegment(end_pose,
                                            self.calc_curvature(1),
                                            end_vel,
                                            end_vel,
                                            list[len(list) - 2].time
                                            + end_pose.distance(list[len(list) - 2].pose))
        # backward pass
        sc = list[len(list) - 1].copy()
        sn = list[len(list) - 2].copy()
        for i in range(len(us) - 1, 1, -1):
            ds = list[i - 1].pose.distance(list[i].pose)
            ar = sc.vel ** 2 * sc.curvature
            if abs(ar) > max_a:
                ar = np.sign(ar) * max_a
            at_lim = math.sqrt(max_a ** 2 - ar ** 2)
            v_lim = math.sqrt(sc.vel ** 2 + 2 * at_lim * ds)
            sn.vel = min(sn.vel, v_lim, sn.vel_lim, math.sqrt(max_a) / abs(sc.curvature))
            list[i] = sn.copy()
            sc = sn.copy()
            sn = list[i - 1].copy()

        list[0].time = 0
        for i in range(1, len(list)):
            ds = list[i - 1].pose.distance(list[i].pose)
            if at_list[i - 1] != 0:
                dt = ((-list[i - 1].vel + math.sqrt(list[i - 1].vel ** 2 + 2 * at_list[i - 1] * ds)) / at_list[
                    i - 1])
                if dt < 0:
                    dt = ((-list[i - 1].vel - math.sqrt(list[i - 1].vel ** 2 + 2 * at_list[i - 1] * ds)) /
                          at_list[i - 1])
                list[i].time = list[i - 1].time + dt
            else:
                list[i].time = list[i - 1].time + (ds / list[i - 1].vel)

        return list[:-1]

    def velocity_generation(self, start_vel: float, end_vel: float, dt: float, max_v: float, max_a: float):
        us = np.linspace(0, 1, int(SEGMENT_RESOLUTION_FACTOR / dt))
        seg_list = []

        for u in us:
            curvature = self.calc_curvature(u)
            seg_list.append(
                SplineSegment(Pose(self.xu(u), self.yu(u), math.atan2(self.yu.deriv(1)(u), self.xu.deriv(1)(u))),
                              curvature, start_vel, min(max_v, math.sqrt(max_a) / abs(curvature)), 0))

        # forward pass
        sc = seg_list[0]
        sn = seg_list[1]
        start_pose = Pose(self.xu(0), self.yu(0), math.atan2(self.yu.deriv(1)(0), self.xu.deriv(1)(0)))
        seg_list[0] = SplineSegment(start_pose,
                                    self.calc_curvature(0),
                                    start_vel,
                                    start_vel,
                                    0)
