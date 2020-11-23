from __future__ import annotations

import numpy as np

import Math
from TrajectoryGeneration import *
from PathGeneration import *
from Util import *


class Spline:
    # the number of samples for integration
    num_samples = 1000
    # ax^5 + bx^4 +cx^3 + dx^2 + ex
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    e = 0.0

    y_offset = 0.0

    x_offset = 0.0

    knot_distance = 0.0
    theta_offset = 0.0
    arc_length = 0.0

    def __init__(self):
        self.arc_length = -1

    @staticmethod
    def almost_equal(x, y):
        return abs(x - y) < 1e-6

    @staticmethod
    def reticulate_spline(x0: float, y0: float, theta0: float,
                          x1: float, y1: float, theta1: float):
        print("reticulating splines...")
        result = Spline()
        result.x_offset = x0
        result.y_offset = y0

        x1_hat = math.sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2))

        if x1_hat == 0:
            return None

        result.knot_distance = x1_hat
        result.theta_offset = math.atan2(y1 - y0, x1 - x0)

        theta0_hat = Math.get_difference_in_angle_radians(result.theta_offset, theta0)
        theta1_hat = Math.get_difference_in_angle_radians(result.theta_offset, theta1)

        # Currently doesnt support vertical slope, (meaning a 90 degrees off the straight line between p0 and p1
        if Spline.almost_equal(abs(theta0_hat), math.pi / 2) or Spline.almost_equal(abs(theta1_hat), math.pi / 2):
            return None

        # Currently doesnt support turning more than 90 degrees in a single path
        if abs(Math.get_difference_in_angle_radians(theta0_hat, theta1_hat)) >= math.pi / 2:
            return None

        yp0_hat = math.tan(theta0_hat)
        yp1_hat = math.tan(theta1_hat)

        result.a = -(3 * (yp0_hat + yp1_hat)) / (x1_hat ** 4)
        result.b = (8 * yp0_hat + 7 * yp1_hat) / (x1_hat ** 3)
        result.c = -(6 * yp0_hat + 4 * yp1_hat) / (x1_hat ** 2)
        result.d = 0
        result.e = yp0_hat

        return result

    def derivative_at(self, percentage: float):
        percentage = max(min(percentage, 1), 0)

        x_hat = percentage * self.knot_distance
        yp_hat = (5 * self.a * x_hat + 4 * self.b) * x_hat * x_hat * x_hat \
                 + 3 * self.c * x_hat * x_hat + 2 * self.d * x_hat + self.e

        return yp_hat

    def second_derivative_at(self, percentage: float):
        percentage = max(min(percentage, 1), 0)

        x_hat = percentage * self.knot_distance
        ypp_hat = (20 * self.a * x_hat + 12 * self.b) * x_hat * x_hat + 6 * self.c * x_hat + 2 * self.d
        return ypp_hat

    def angle_at(self, percentage: float):
        percentage = max(min(percentage, 1), 0)
        angle = Math.bound_angle_0_to_2pi_radians(math.atan(self.derivative_at(percentage)) + self.theta_offset)
        return angle

    def calculate_length(self):
        if self.arc_length >= 0:
            return self.arc_length

        num_samples = self.num_samples
        arc_length = 0.0

        integrand = math.sqrt(1 + self.derivative_at(0) * self.derivative_at(0)) / num_samples
        last_integrand = integrand

        for i in range(1, num_samples + 1):
            t = float(i) / num_samples
            dydt = self.derivative_at(t)
            integrand = math.sqrt(1 + dydt * dydt) / num_samples
            arc_length += (integrand + last_integrand) / 2.0
            last_integrand = integrand
        self.arc_length = self.knot_distance * arc_length

        return self.arc_length

    def get_percentage_for_distance(self, distance: float):
        num_samples = self.num_samples
        arc_length = 0.0
        t = 0.0
        last_arc_length = 0.0
        dydt = 0.0

        integrand = math.sqrt(1 + self.derivative_at(0) * self.derivative_at(0)) / num_samples
        last_integrand = integrand
        distance /= self.knot_distance
        for i in range(1, num_samples + 1):
            t = float(i) / num_samples
            dydt = self.derivative_at(t)
            integrand = math.sqrt(1 + dydt * dydt) / num_samples
            arc_length += (integrand + last_integrand) / 2.0
            if arc_length > distance:
                break

            last_integrand = integrand
            last_arc_length = arc_length

        interpolated = t
        if arc_length != last_arc_length:
            interpolated += ((distance - last_arc_length) / (arc_length - last_arc_length) - 1) / float(num_samples)
        return interpolated

    def get_xy(self, percentage: float) -> []:
        result = [0.0] * 2
        percentage = max(min(percentage, 1), 0)

        x_hat = percentage * self.knot_distance
        y_hat = (self.a * x_hat + self.b) * x_hat * x_hat * x_hat * x_hat \
                + self.c * x_hat * x_hat * x_hat + self.d * x_hat * x_hat + self.e * x_hat

        cos_theta = math.cos(self.theta_offset)
        sin_theta = math.sin(self.theta_offset)

        result[0] = x_hat * cos_theta - y_hat * sin_theta + self.x_offset
        result[1] = x_hat * sin_theta + y_hat * cos_theta + self.y_offset

        return result


class AlexSpline(object):
    num_samples = 1000
    knot_distance = 0.0
    theta_offset = 0.0
    arc_length = 0.0
    polynomial_x: np.polynomial.Polynomial = None
    polynomial_y: np.polynomial.Polynomial = None
    trajectory: Trajectory = None

    A = np.linalg.inv(np.array([
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5],
        [0, 0, 2, 0, 0, 0],
        [0, 0, 2, 6, 12, 20]]))

    @staticmethod
    def interpolate(state, desired_state, start_speed, end_speed):
        # t is between 0 and 1
        x_constraint = np.array([
            [state[0, 0]],
            [desired_state[0, 0]],
            [start_speed * math.cos(state[2])],
            [end_speed * math.cos(desired_state[2])],
            [0],
            [0]])
        y_constraint = np.array([
            [state[1, 0]],
            [desired_state[1, 0]],
            [start_speed * math.sin(state[2])],
            [end_speed * math.sin(desired_state[2])],
            [0],
            [0]])
        coeffs_x = HermiteSpline.A @ x_constraint
        coeffs_y = HermiteSpline.A @ y_constraint
        result = AlexSpline()
        result.knot_distance = math.sqrt(
            (desired_state[0, 0] - state[0, 0]) ** 2 + (desired_state[1, 0] - state[1, 0]) ** 2)
        result.polynomial_x, result.polynomial_y = np.polynomial.Polynomial(
            coeffs_x.squeeze()), np.polynomial.Polynomial(coeffs_y.squeeze())
        return result

    def populate_trajectory(self, dt, wheelbase, wheel_radius):
        fx = self.polynomial_x
        fy = self.polynomial_y
        time_per_waypoint = 4
        ts = np.linspace(0, 1, int(time_per_waypoint / dt))
        xs = fx(ts)
        ys = fy(ts)

        ax, ay, au, av = [], [], [], []
        pos_list = []
        wheel_vel_list = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            rt = i * dt / time_per_waypoint
            v = np.array([fx.deriv(1)(rt), fy.deriv(1)(rt)])
            dx2 = fx.deriv(2)(rt)
            dy2 = fy.deriv(2)(rt)
            speed2 = (v[0] ** 2 + v[1] ** 2)
            wr = (v[0] * dy2 - v[1] * dx2) / speed2 / time_per_waypoint
            thetar = math.atan2(v[1], v[0])
            desired_pos = np.array([[x, y, thetar]]).T
            speedr = np.sqrt(speed2) / time_per_waypoint
            pos_list.append(desired_pos)
            v_l = ((2 * speedr) - (wr * wheelbase)) / (2 * wheel_radius)
            v_r = ((2 * speedr) + (wr * wheelbase)) / (2 * wheel_radius)
            wheel_vel_list.append([v_l, v_r])
        return pos_list, wheel_vel_list, ts


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


        list[1] = SplineSegment(Pose(self.xu(us[1]), self.yu(us[1]), math.atan2(self.yu.deriv(1)(us[1]), self.xu.deriv(1)(us[1]))),
                                self.calc_curvature(us[1]),
                                start_vel + first_at*first_dt,
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
            # print(list[i].time)
        return list

    def velocity_generation(self, start_vel: float, end_vel: float, dt: float, max_v: float, max_a: float):
        us = np.linspace(0, 1, int(3 / dt))
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


if __name__ == "__main__":
    graph = CsvWriter("C:/Users/stanl/PycharmProjects/PurePursuit/paperspline.csv")
    lines = []

    MAX_VEL = 6
    MAX_ACC = 30

    # spline = HermiteSpline.interpolate(Pose(0, 0, math.pi / 2), Pose(1.5, 1, 3 * math.pi / 4))
    # segment_list = spline.velocity_profile(0, 0, 1 / 60, 5., 8)
    ws = WaypointSequence()
    ws.add_waypoint(Pose(0, 0, math.pi / 2))
    ws.add_waypoint(Pose(1.5/2, 1/2, 3 * math.pi / 4))
    ws.add_waypoint(Pose(3/2, 1.5/2,  math.pi/2))
    config = TrajectoryConfig(MAX_VEL, MAX_ACC, 1/60)
    segs = generate_from_waypoints(ws, config, 0, 0)

    for seg in segs:
        line = str(segs.index(seg)) + "," + str(seg.vel) + "," + str(seg.time) + "," + str(
            seg.pose.x) + "," + str(seg.pose.y) + ","
        print(line)
        lines.append(line + "\n")

        graph.write_lines(lines)
    # for seg in segment_list:
    #     line = str(segment_list.index(seg)) + "," + str(seg.vel) + "," + str(seg.time) + "," + str(
    #         seg.pose.x) + "," + str(seg.pose.y) + ","
    #     print(line)
    #     lines.append(line + "\n")
    #
    #     graph.write_lines(lines)
