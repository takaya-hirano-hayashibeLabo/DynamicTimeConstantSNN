import numpy as np

def generate_trajectory(config:dict):
    trajectory_type = config["type"]
    origin = config.get("origin", (0, 0))
    num_loops = config.get("num_loops", 1)
    loop_duration = config.get("loop_duration", 10.0)
    delta_time = config.get("delta_time", 0.1)
    angle = config.get("angle", 0.0)
    noise_std = config.get("noise_std", 0.01)

    if "8" in trajectory_type.casefold() or "eight" in trajectory_type.casefold():
        max_distance = config.get("max_distance", (1, 1))
        fx,fy=config.get("fx",2),config.get("fy",4)
        trajectory = generate_eight_trajectory(
            origin, max_distance, num_loops, loop_duration, delta_time, angle, noise_std,fx,fy
        )
    elif "ellipse" in trajectory_type.casefold():
        semi_axes = config.get("semi_axes", (1, 0.5))
        trajectory = generate_ellipse_trajectory(
            origin, semi_axes, num_loops, loop_duration, delta_time, angle, noise_std
        )
    else:
        raise ValueError(f"Unsupported trajectory type: {trajectory_type}")

    return trajectory


def generate_ellipse_trajectory(origin, semi_axes, num_loops, loop_duration, delta_time, angle, noise_std=0.01):
    """
    generate ellipse trajectory
    :param origin: trajectory origin [x,y]
    :param semi_axes: ellipse semi-axes [a,b]
    :param num_loops: number of loops
    :param loop_duration: time to draw one trajectory [s]
    :param delta_time: time step of trajectory [s]
    :param angle: rotation angle around z-axis [rad]
    :param noise_std: standard deviation of white noise
    :return: trajectory list [(time, x, y)]
    """
    x_origin, y_origin = origin
    a, b = semi_axes
    trajectory = []
    total_time = 0.0

    # calculate rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    t = 0.0
    while t < loop_duration*num_loops:
        # generate ellipse trajectory
        x = a * np.cos(2 * np.pi * t / loop_duration,dtype=np.double)
        y = b * np.sin(2 * np.pi * t / loop_duration,dtype=np.double)

        # apply rotation (origin is center)
        x_rotated = cos_angle * x - sin_angle * y + x_origin
        y_rotated = sin_angle * x + cos_angle * y + y_origin

        # add white noise
        if noise_std > 0:
            x_rotated = x_rotated + np.random.normal(0, noise_std)
            y_rotated = y_rotated + np.random.normal(0, noise_std)

        trajectory.append((total_time, x_rotated, y_rotated))
        t += delta_time
        total_time += delta_time

    return trajectory


def generate_eight_trajectory(origin, max_distance, num_loops, loop_duration, delta_time, angle, noise_std=0.01,fx=2,fy=4):
    """
    generate 8 trajectory
    :param origin: trajectory origin [x,y]
    :param max_distance: maximum distance from origin [x,y]
    :param num_loops: number of loops
    :param loop_duration: time to draw one trajectory [s]
    :param delta_time: time step of trajectory [s]
    :param angle: rotation angle around z-axis [rad]
    :param noise_std: standard deviation of white noise
    :return: trajectory list [(time, x, y)]
    """
    x_origin, y_origin = origin
    max_x, max_y = max_distance
    trajectory = []
    total_time = 0.0

    # calculate rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    t = 0.0
    while t < loop_duration*num_loops:
        # generate 8 trajectory
        x = max_x * np.sin(fx * np.pi * t / loop_duration)
        y = max_y * np.sin(fy * np.pi * t / loop_duration)

        # apply rotation (origin为中心)
        x_rotated = cos_angle * x - sin_angle * y + x_origin
        y_rotated = sin_angle * x + cos_angle * y + y_origin

        # add white noise
        if noise_std > 0:
            x_rotated = x_rotated + np.random.normal(0, noise_std)
            y_rotated = y_rotated + np.random.normal(0, noise_std)

        trajectory.append((total_time, x_rotated, y_rotated))
        t += delta_time
        total_time += delta_time

    return trajectory


def animate_trajectory(trajectory):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.collections import LineCollection
    import numpy as np

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    max_val=np.max(np.array(trajectory)[:,1:])
    min_val=np.min(np.array(trajectory)[:,1:])
    ax.set_xlim([1.2*min_val,1.2*max_val])
    ax.set_ylim([1.2*min_val,1.2*max_val])

    # list to store trajectory
    x_data, y_data = [], []

    # initialize
    line_segments = LineCollection([], linewidths=2)
    ax.add_collection(line_segments)

    def init():
        line_segments.set_segments([])
        return line_segments,

    def update(frame):
        x, y = frame[1], frame[2]
        x_data.append(x)
        y_data.append(y)

        # create trajectory segments
        points = np.array([x_data, y_data]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # change color according to time
        norm = plt.Normalize(0, len(x_data))
        colors = plt.cm.viridis(norm(range(len(x_data))))

        line_segments.set_segments(segments)
        line_segments.set_color(colors)

        return line_segments,

    ani = animation.FuncAnimation(fig, update, frames=trajectory, init_func=init, blit=True, interval=50)
    plt.show()


if __name__=="__main__":
    # # example
    origin = (0.5, 0)
    max_distance = (0.3, 0.075)
    # max_distance = (0.3, 0.2)
    angle=np.pi/2
    num_loops = 50
    loop_duration = 5.0  # time to draw one trajectory [s]
    delta_time = 0.07
    noise_std=-1
    fx=1
    fy=2    

    trajectory = generate_eight_trajectory(
        origin, max_distance, num_loops, loop_duration, 
        delta_time,angle=angle,noise_std=noise_std,fx=fx,fy=fy
    )
    animate_trajectory(trajectory)

    # # example
    # origin = (0.5, 0)
    # semi_axes = (0.15, 0.15)
    # angle=np.pi/2
    # num_loops = 50
    # loop_duration = 5.0  # 1つの軌道を描く時間 [s]
    # delta_time = 0.03
    # noise_std=-1

    # trajectory = generate_ellipse_trajectory(
    #     origin, semi_axes, num_loops, loop_duration, 
    #     delta_time,angle=angle,noise_std=noise_std
    # )

    # # trajectory_db=pd.DataFrame(
    # #     np.concatenate([np.array(trajectory)[:-1],np.array(trajectory)[1:,1:]],axis=1),
    # #     columns=["time","endpos_x","endpos_y","target_x","target_y"]
    # # )
    # # trajectory_db.to_csv(
    # #     "C:/Users/3meko/Dev/HayashibeLab/workspace/ur5e_ik_control/main/collect_dataset/20241017/eight_figure_ideal/output/datasets.csv",
    # #     index=False,
    # # )

    # animate_trajectory(trajectory)