"""
Script to collect pairs of robot joint angles and target trajectories as a dataset.
The rest is just supervised learning with the collected data.
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import yaml
import argparse
import os
import pandas as pd

import mink

EXP3_ROOT=Path(__file__).parent.parent
_XML = EXP3_ROOT / "env" / "scene.xml"

ROOT=EXP3_ROOT.parent.parent.parent
DATAPATH=ROOT/"data/original-data/Trajectory"

import sys
sys.path.append(str(EXP3_ROOT))
from exp3_utils import np2SE3_target
from trajectory_generator import generate_trajectory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--configpath",required=True)
    args=parser.parse_args()

    # Read the YAML file for trajectory settings
    config_path=args.configpath
    with open(config_path, 'r',encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # >> Trajectory generation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    trajectory_config=config["trajectory"]
    delta_time = trajectory_config['delta_time']
    trajectory = generate_trajectory(trajectory_config)

    _,target_x,target_y=trajectory.pop(0)
    T_wt=np2SE3_target( # Initial coordinates
        quaternion=np.array([1,0,0,0]),
        position=np.array([target_x,target_y,0.3])
    )
    # << Trajectory generation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    result_path=DATAPATH
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model = mujoco.MjModel.from_xml_path(_XML.as_posix()) # xml path can contain ONLY ENGLISH characters.

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # Enable collision avoidance between the following geoms:
    collision_pairs = [
        (["wrist_3_link"], ["floor", "wall"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    mid = model.body("target").mocapid[0]
    model = configuration.model
    data = configuration.data
    solver = "quadprog"


    datasets=[]
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=500.0, warn=False)
        elapsed_t_from_set_target=0 # Elapsed time since specifying the target
        while viewer.is_running():


            if len(trajectory)<1: break # End if there are no more trajectories


            # >> Specify target position and orientation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            elapsed_t_from_set_target+=rate.dt
            if elapsed_t_from_set_target>delta_time: # Control period and target trajectory period are different
                target_t,target_x,target_y=trajectory.pop(0)
                target_z=0.3 # z coordinate is fixed
                T_wt=np2SE3_target(
                    quaternion=np.array([1,0,0,0]), # Orientation is fixed
                    position=np.array([target_x,target_y,target_z])
                )
                elapsed_t_from_set_target=0.0 # Reset elapsed time
                # print(T_wt)

                datasets.append(( # Record robot state and target trajectory
                    target_t,
                    *data.qpos, # Angle of each joint
                    *data.site("attachment_site").xpos, # Position of end-effector
                    target_x,target_y,target_z, # Target end-effector position
                ))
                print(f"endpos:{data.site('attachment_site').xpos[:-1]}, target:{target_x,target_y}")
            # << Specify target position and orientation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


            end_effector_task.set_target(T_wt) # Just give the SE3-type quaternion and xyz coordinates here

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-3, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            # rate.sleep() # Synchronize real time and simulator time    


    datasets_db=pd.DataFrame(
        datasets,
        columns=[
            "time",
            *[f"joint{i}" for i in range(6)],
            *[f"endpos_{label}" for label in ["x","y","z"]],
            "target_x","target_y","target_z"
        ]
    )
    datasets_db.to_csv(result_path/"datasets.csv",index=False)


    #>> Save config >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    with open(config_path, 'r',encoding="utf-8") as file:
        config_lines = file.readlines()
    with open(result_path/"conf.yml", 'w',encoding="utf-8") as file:
        file.writelines(config_lines)
    # << Save config <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if __name__ == "__main__":
    main()