"""
Script to move the collected trajectory
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from tqdm import tqdm
import mink

EXP3_ROOT=Path(__file__).parent.parent
_XML = EXP3_ROOT / "env" / "scene.xml"

ROOT=EXP3_ROOT.parent.parent.parent

import sys
sys.path.append(str(EXP3_ROOT))
from exp3_utils import np2SE3_target,save_video

import argparse
import pandas as pd
sys.path.append(str(ROOT/"codes")) #put model here
from utils import load_json2dict


def add_visual_line(scene, point1, point2, width, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        type=mujoco.mjtGeom.mjGEOM_LINE,
                        size=[1, 1, 1],
                        pos=np.array([0.5,0,0.2]),
                        mat=np.eye(3).flatten(),
                        rgba=rgba,
                        )
    # print(scene.geoms[scene.ngeom-1])
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                           mujoco.mjtGeom.mjGEOM_LINE, width,
                           point1[0], point1[1], point1[2], point2[0], point2[1], point2[2])



def modify_viewer_scene(scn, point1,point2,rgba=[0,0,1,1],width=4):
    """
    :param point1: [x,y,z]
    :param point2: [x,y,z]
    """
    
    # 線を描画
    add_visual_line(scn, point1, point2, width, np.array(rgba))



def modify_renderer_scene(scn, pos_trajectory,rgba_trj,width=4):
    """
    :param trajectory: [[x,y,z],...]
    :param rgba_trj: [[r,g,b,a],...]
    """
    for i in range(len(pos_trajectory)-1):
        modify_viewer_scene(scn, pos_trajectory[i], pos_trajectory[i+1],rgba_trj[i],width)


if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--trjpath",required=True, help="absolute path")
    args=parser.parse_args()

    trj_datapath=Path(args.trjpath)



    #>> data preparation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    
    datasets=pd.read_csv(trj_datapath/"trajectory.csv")
    target_datas=datasets[["target_x","target_y"]].values

    config=load_json2dict(trj_datapath/"args.json")
    delta_time=config["args"]["deltatime"]
    sequence=config["nn_conf"]["train"]["sequence"] if config["args"]["nhead"] is None else config["args"]["nhead"]
    #>> data preparation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    model = mujoco.MjModel.from_xml_path(_XML.as_posix())


    #>> capture >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    width,height=240,240
    renderer=mujoco.Renderer(model,width,height)
    frames=[]
    #>> capture >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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


    print("trj_datapath:\n",trj_datapath)
    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Initialize to the home keyframe.
            configuration.update_from_keyframe("home")

            # Initialize the mocap target at the end-effector site.
            mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

            rate = RateLimiter(frequency=500.0, warn=False)
            run_count=0
            elapsed_time=0
            total_time=0

            target_trajectory=[
                [*target_datas[0],0.3]
            ]
            target_x,target_y=target_datas[0]
            run_count+=1


            is_line=True
            ideal_color=[1,0,0,1]
            actual_color=[0,0,1,1]
            # actual_color=[1,0,0,1]
            rgba_trajectory=[]

            with tqdm(total=len(target_datas)) as pbar:
                while viewer.is_running():

                    if run_count>=len(target_datas): break

                    total_time+=rate.dt
                    elapsed_time+=rate.dt
                    if elapsed_time>delta_time:
                        target_x,target_y=target_datas[run_count]
                        target_trajectory.append(np.array([target_x,target_y,0.3]))
                        run_count+=1
                        elapsed_time=0.0

                        is_line=True
                        pbar.update(1)

                    # Update task target.
                    T_wt = np2SE3_target(
                        quaternion=[1,0,0,0],
                        position=[target_x,target_y,0.3]
                    )
                    end_effector_task.set_target(T_wt) #just give quaternion and xyz coordinates here

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


                    if len(target_trajectory)>1 and is_line:
                        is_line=False

                        line_color=ideal_color if run_count<sequence else actual_color
                        rgba_trajectory.append(line_color)

                        modify_viewer_scene(viewer.user_scn, target_trajectory[-1], target_trajectory[-2],rgba=line_color)

                        renderer.update_scene(data,camera="video_cam")
                        modify_renderer_scene(renderer.scene, target_trajectory,rgba_trajectory)
                        # print("renderer.scene.ngeom",renderer.scene.ngeom)
                        frame=renderer.render()
                        frames.append(frame)

                    # Visualize at fixed FPS.
                    viewer.sync() #if this is added, the arm motion is visualized
                    # rate.sleep()

            
    finally:
        print("frames",np.array(frames).shape)
        result_path=Path(__file__).parent / config["args"]["saveto"]
        filename=f"trjview_timescale{config['args']['end_timescale']}.mp4"
        save_video(frames,result_path,filename,scale=3,fps=24)