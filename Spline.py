from __future__ import annotations

import numpy as np

import Math
from TrajectoryGeneration import *
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

    def calc_traj(self, start_vel: float, end_vel: float, max_v: float, a_rad_max: float, a_tan_max: float, dt: float):
        ts = np.linspace(0, 1, int(10 / dt))
        xs = self.xu(ts)
        ys = self.yu(ts)
        curv_list = []
        max_vel_profile = []
        delta_ts = []
        v_kappa_max_1 = []
        v_kappa_max_2 = []
        v_max = []
        v_start = []
        v_end = []

        for t in ts:
            curv = self.calc_curvature(t)
            curv_list.append(curv)
            max_vel_profile.append(math.sqrt(a_rad_max / abs(curv)))

        min_idx = max_vel_profile.index(min(max_vel_profile))
        v_start.append(start_vel)
        v_kappa_max_1.append(max_vel_profile[0])
        v_kappa_max_2.append(max_vel_profile[min_idx])
        v_max = [max_v] * len(ts)
        v_end.append( end_vel)

        A = a_tan_max ** 2
        B = a_rad_max ** 2
        a_tan = [0]
        a_rad = [(start_vel ** 2) * abs(curv_list[0])]
        delta_ts.append(0)
        self.calc_tangential_vel(v_kappa_max_1, ts, curv_list, a_tan_max, a_rad_max)
        self.calc_tangential_vel(v_kappa_max_2, ts, curv_list, a_tan_max, a_rad_max)
        self.calc_tangential_vel(v_start, ts, curv_list, a_tan_max, a_rad_max)

        return curv_list, max_vel_profile, ts, v_kappa_max_1, v_kappa_max_2

    def calc_tangential_vel(self, vel_list: [], ts, curv_list, a_tan_max, a_rad_max):
        a_tan = [0]
        a_rad = [(vel_list[0] ** 2) * abs(curv_list[0])]

        A = a_tan_max ** 2
        B = a_rad_max ** 2
        t = 0
        for i in range(1, len(ts)):
            delta_s = math.sqrt((self.xu(ts[i + 1]) - self.xu(ts[i])) ** 2 +
                                (self.yu(ts[i + 1]) - self.yu(ts[i])) ** 2)
            if abs(a_tan[i - 1]) < .0001:
                delta_ts = delta_s / vel_list[i - 1]
            else:
                delta_ts = (-vel_list[i - 1] + math.sqrt(vel_list[i - 1] ** 2 + 2 * a_tan[i - 1] * delta_s)) / a_tan[
                    i - 1]
                if delta_ts <= 0:
                    delta_ts = (-vel_list[i - 1] - math.sqrt(vel_list[i - 1] ** 2 + 2 * a_tan[i - 1] * delta_s)) / \
                               a_tan[i - 1]

            t += delta_ts
            a_rad.append(vel_list[i - 1] ** 2 * curv_list[i - 1])
            det = ((A * (B ** 2) * (t ** 2)) - (A * B * (vel_list[i - 1] ** 2 * curv_list[i - 1]) ** 2))
            vel_list.append((B * vel_list[0] + math.sqrt(det)) / B)
            a_tan.append((vel_list[i] - vel_list[i - 1]) / delta_ts)


if __name__ == "__main__":
    graph = CsvWriter("C:/Users/stanl/PycharmProjects/PurePursuit/paperspline.csv")
    lines = []
    spline = HermiteSpline.interpolate(Pose(0, 0, math.pi / 2), Pose(1.5, 1, 3 * math.pi / 4))
    curv, max_vel_rad, ts, vk1, vk2 = spline.calc_traj(0.5, 0.75, 1.5, 2, 1.5, 1 / 60)

    for i in range(len(ts)):
        x, y = spline.get_xy(ts[i])
        lines.append(
            str(x) + "," + str(y) + "," + str(ts[i]) + "," + str(curv[i]) + "," + str(max_vel_rad[i]) + "," + str(
                vk1[i]) + "," + str(vk2[i]) + ",\n")
    graph.write_lines(lines)

    # spline = AlexSpline.interpolate(np.array([[0], [0], [0]]), np.array([[1], [2], [0]]), 0, 0)
    # graph = CsvWriter("C:/Users/stanl/PycharmProjects/PurePursuit/spline.csv")
    # lines = []
    # pos, vel, t = spline.populate_trajectory(.01, .3, .05)
    # for i in range(len(pos)):
    #     lines.append(str(pos[i][0][0]) + ',' + str(pos[i][1][0]) + ',' + str(t[i]) + ',' + str(vel[i][0]) + ','
    #                  + str(vel[i][1]) + ',\n')
    #
    # graph.write_lines(lines)

    # for i in range(1000):
    #     xy = spline.get_xy(i/1000)[0]
    #     spline_x.append(xy[0])
    #     spline_y.append(xy[1])
    #
    #     plt.cla()
    #     # for stopping simulation with the esc key.
    #     plt.gcf().canvas.mpl_connect(
    #         'key_release_event',
    #         lambda event: [exit(0) if event.key == 'escape' else None])
    #     # plt.plot(cx, cy, "-r", label="course")
    #     plt.plot(spline_x, spline_y, "-b", label="trajectory")
    #     plt.axis("equal")
    #     plt.grid(True)
    #     plt.pause(.01)
