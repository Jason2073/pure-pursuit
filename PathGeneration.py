from Spline import *
from TrajectoryGeneration import *
from Util import *


def generate_from_waypoints(path: WaypointSequence, config: TrajectoryConfig, start_vel: float, end_vel: float) \
        -> List[SplineSegment]:
    if path.num_waypoints < 2:
        return None

    splines = []
    total_time = 0.0
    segments: List[SplineSegment] = []
    last_velocity = start_vel
    for i in range(path.num_waypoints - 1):
        splines.append(HermiteSpline())
        start = path.waypoints[i]
        end = path.waypoints[i + 1]
        splines[i] = splines[i].interpolate(Pose(start.x, start.y, start.theta), Pose(end.x, end.y, end.theta))
        if splines[i] is None:
            return None
        seg_list = splines[i].velocity_profile(last_velocity, config.max_v if i < path.num_waypoints - 2 else end_vel,
                                               config.interval, config.max_v, config.max_a)
        last_velocity = seg_list[len(seg_list) - 1].vel
        for seg in seg_list:
            seg.time += total_time
            segments.append(seg)

        total_time += seg_list[len(seg_list) - 1].time

    return segments
