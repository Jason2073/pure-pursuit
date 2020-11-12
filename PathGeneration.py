from Spline import *
from TrajectoryGeneration import *
from Util import *


def generate_from_path(path: WaypointSequence, config: TrajectoryConfig, start_vel, end_vel):
    if path.num_waypoints < 2:
        return None

    splines: List[Spline] = []
    for i in range(path.num_waypoints - 1):
        splines.append(Spline())
    spline_lengths = []
    for i in range(len(splines)):
        spline_lengths.append(0.0)
    total_distance = 0.0

    for i in range(path.num_waypoints - 1):
        splines[i] = Spline()
        start = path.waypoints[i]
        end = path.waypoints[i + 1]
        print(splines[i].arc_length)
        splines[i] = Spline.reticulate_spline(start.x, start.y, start.theta, end.x, end.y, end.theta)
        if splines[i] is None:
            return None

        spline_lengths[i] = splines[i].calculate_length()
        total_distance += spline_lengths[i]

    print("GENERATING TRAJECTORY...")
    print(str(total_distance) + "vel" + str(start_vel) + "end_vel" + str(end_vel))
    profile = MotionProfile(config, total_distance, start_vel, end_vel)
    trajectory = Trajectory(math.floor(profile.total_time / config.interval))
    cur_spline = 0
    cur_spline_start_pos = 0
    length_of_splines_finished = 0
    for i in range(len(trajectory.segments)):
        dt = config.interval
        trajectory.segments[i].pos = profile.at_time(dt * i)[0]
        trajectory.segments[i].vel = profile.at_time(dt * i)[1]
        trajectory.segments[i].acc = profile.at_time(dt * i)[2]
        trajectory.segments[i].dt = dt

        cur_pos = trajectory.segments[i].pos
        found_spline = False
        while not found_spline:
            cur_pos_relative = cur_pos - cur_spline_start_pos
            if cur_pos_relative <= spline_lengths[cur_spline]:
                percentage = splines[cur_spline].get_percentage_for_distance(cur_pos_relative)

                trajectory.segments[i].heading = splines[cur_spline].angle_at(percentage)
                coords = splines[cur_spline].get_xy(percentage)

                trajectory.segments[i].x = coords[0]
                trajectory.segments[i].y = coords[1]
                found_spline = True
            elif cur_spline < len(splines) - 1:
                length_of_splines_finished += spline_lengths[cur_spline]
                cur_spline_start_pos = length_of_splines_finished
                cur_spline += 1
            else:
                trajectory.segments[i].heading = splines[len(splines) - 1].angle_at(1.0)
                coords = splines[len(splines) - 1].get_xy(1.0)
                trajectory.segments[i].x = coords[0]
                trajectory.segments[i].y = coords[1]
                found_spline = True

    return trajectory
