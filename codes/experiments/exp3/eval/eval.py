"""
change trajectory speed with dynamic forward
use joint angles as input
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

EXP3_ROOT=Path(__file__).parent.parent
_XML = EXP3_ROOT / "env" / "scene.xml"

ROOT=EXP3_ROOT.parent.parent.parent

import sys
sys.path.append(str(EXP3_ROOT))
from exp3_utils import np2SE3_target
from trajectory_generator import generate_trajectory

import argparse
import os
import torch    
import pandas as pd
import json
from copy import deepcopy
sys.path.append(str(ROOT/"codes")) #put model here
from models import DynamicSNN, ContinuousSNN, ThresholdEncoder, SNN
from utils import load_yaml,load_json2dict


def error_feedback(target_position, ref_trajectory):
    """
    calculate error from target position and reference trajectory and feedback
    :param target_position: target position [x,y]
    :param ref_trajectory: reference trajectory [x,y] array
    """

    if not isinstance(ref_trajectory,np.ndarray):
        ref_trajectory=np.array(ref_trajectory)
    if not isinstance(target_position,np.ndarray):
        target_position=np.array(target_position)

    error_arr=ref_trajectory-target_position.reshape(1,-1)
    min_error_idx=np.argmin(np.sum(error_arr**2,axis=1))
    min_error=error_arr[min_error_idx]

    # print(f"ref position: {ref_trajectory[min_error_idx]}")
    # print(f"min_error: {min_error}")

    return min_error


def calculate_timescale(current_timescale,delta_timescale,end_timescale):
    new_timescale=current_timescale+delta_timescale
    if (new_timescale-end_timescale) * (current_timescale-end_timescale)<=0:
        new_timescale=end_timescale
    return new_timescale


def add_volt2trajectory(volt:np.ndarray,volt_trajectory:list):
    """
    :param volt: membrane potential [T x hidden_size]
    """

    if None in volt_trajectory[-1]: #if None is in the latest volt_trajectory
        # print(volt_trajectory)
        # print(f"volt shape: {volt.shape}")
        volt_trj_np=np.array(volt_trajectory)
        volt_trj_np[-len(volt):]=volt.copy()
        volt_trajectory=deepcopy(volt_trj_np.tolist())
    else:
        volt_trajectory.append(list(volt[-1]))
    return volt_trajectory



if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--modelpath",required=True, help="absolute path")
    parser.add_argument("--trjpath",required=True, help="absolute path")
    parser.add_argument("--kfb",default=0.05,type=float,help="error feedback coefficient")
    parser.add_argument("--timescale_duration",default=1.0,type=float, help="timescale change time [s]")
    parser.add_argument("--end_timescale",default=1.0,type=float)
    parser.add_argument("--deltatime",default=0.07, type=float)
    parser.add_argument("--nloop",default=1,type=int,help="number of loops. not including the initial run")
    parser.add_argument("--saveto",required=True)
    parser.add_argument("--nhead",default=None,type=int,help="number of initial run data")
    args=parser.parse_args()

    nn_modelpath=Path(args.modelpath)
    trj_datapath=Path(args.trjpath)


    nn_conf=load_yaml(nn_modelpath/"conf.yml")

    if nn_conf["model"]["type"].casefold()=="snn":
        time_enc=SNN(conf=nn_conf["model"])
    else:
        time_enc=DynamicSNN(conf=nn_conf["model"])
    time_enc.eval()
    snn_hidden_size=nn_conf["model"]["out-size"]

    # # get the weights of the final layer of time_enc and display
    # time_enc_weights = list(time_enc.parameters())[-1]
    # print("Time Encoder Final Layer Weights:")
    # print(time_enc_weights)

    nn_model=ContinuousSNN(
        nn_conf["output-model"],time_encoder=time_enc
    )
    print(nn_model)

    weights=torch.load(
        nn_modelpath/"result/models/model_best.pth",
        map_location="cpu",
    )
    nn_model.load_state_dict(weights)
    nn_model.eval()

    sequence=nn_conf["train"]["sequence"]

    # # get the weights of the final layer of nn_model and display
    # nn_model_weights = list(nn_model.time_encoder.parameters())[-1]
    # print("SNN Model Final Layer Weights:")
    # print(nn_model_weights)


    #>> encoder preparation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    encoder_conf=nn_conf["encoder"]
    if encoder_conf["type"].casefold()=="thr":
        encoder=ThresholdEncoder(
            thr_max=encoder_conf["thr-max"],thr_min=encoder_conf["thr-min"],
            resolution=encoder_conf["resolution"]
        )
    else:
        raise ValueError(f"encoder type {encoder_conf['type']} is not supportated...")
    #<< encoder preparation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> data preparation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    trj_conf=load_yaml(trj_datapath.parent/"conf.yml")
    loop_duration=trj_conf["trajectory"]["loop_duration"] #time for one loop
    
    datasets=pd.read_csv(trj_datapath)
    
    input_labels=[f"joint{i}" for i in range(6)]
    input_datas=datasets[input_labels]

    input_nrm_js=load_json2dict(nn_modelpath/"result/input_nrm_params.json")
    input_max=pd.Series(input_nrm_js["max"],name="max")
    input_min=pd.Series(input_nrm_js["min"],name="min")

    input_datas=input_datas.values

    target_datas=datasets[["target_x","target_y"]]
    if nn_conf["output-model"]["out-type"].casefold()=="velocity":
        target_datas=target_datas.diff().iloc[1:] #first row is NaN, so exclude it
    elif nn_conf["output-model"]["out-type"].casefold()=="position":
        target_datas=target_datas.iloc[1:]
    target_max=target_datas.max()
    target_max.name="max"
    target_min=target_datas.min()
    target_min.name="min"

    target_positions=datasets[["target_x","target_y"]].values[1:] #target position coordinates

    n_head=sequence if args.nhead is None else int(args.nhead)
    in_trajectory=[]
    time_scales=[]
    endeffector_target_trajectory=[]
    endeffector_deltapos=[]
    time_trajectory=[]
    volt_trajectory=[]
    #>> data preparation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    #>> reference trajectory generation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ref_conf=deepcopy(trj_conf["trajectory"])
    ref_conf["delta_time"]=args.deltatime/10 #increase the time resolution of the reference trajectory
    ref_conf["num_loops"]=2
    ref_trajectory=generate_trajectory(
        ref_conf,
    )
    ref_trajectory=np.array(ref_trajectory)[:,1:] #extract x,y only
    k_fb=args.kfb #error feedback coefficient
    #<< reference trajectory generation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

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
            delta_time=args.deltatime
            elapsed_time=delta_time
            total_time=delta_time
            current_timescale=1.0 #initial timescale is 1
            end_timescale=args.end_timescale
            delta_timescale=(end_timescale-1.0)/(args.timescale_duration/delta_time) #change timescale every delta_time

            max_exptime=loop_duration*args.nloop*end_timescale #maximum time of experiment (not including the initial run)
            exptime=0.0 #elapsed time since the initial run
            while viewer.is_running():

                if exptime>max_exptime: break #if the experiment time exceeds the maximum time, exit

                #>> target trajectory estimation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                elapsed_time+=rate.dt
                total_time+=rate.dt
                if elapsed_time>delta_time:

                    time_trajectory.append(total_time)

                    elapsed_time=0.0 #reset elapsed time

                    if run_count<n_head: #first is initial run

                        if len(in_trajectory)>sequence:
                            in_x=np.array(in_trajectory)[-sequence:] if len(in_trajectory)>sequence else np.array(in_trajectory)
                            in_x=2*(in_x-input_min.values)/(input_max.values-input_min.values)-1
                            in_spike=encoder(torch.Tensor(in_x).unsqueeze(0))

                            in_scales=np.array(time_scales)[-sequence:] if len(time_scales)>sequence else np.array(time_scales)

                            with torch.no_grad():
                                if nn_conf["model"]["type"].casefold()=="snn":
                                    out_nrm, out_v=nn_model.forward(in_spike.flatten(start_dim=2),return_v=True)
                                    out_nrm=out_nrm[0,-1].to("cpu").detach().numpy()
                                    out_v=out_v[0].to("cpu").detach().numpy() #[T x hidden_size]
                                elif "dyna".casefold() in nn_conf["model"]["type"].casefold():
                                    out_nrm, out_v=nn_model.dynamic_forward_given_scale(
                                        in_spike.flatten(start_dim=2), torch.Tensor(in_scales),return_v=True
                                    )
                                    out_nrm=out_nrm[0,-1].to("cpu").detach().numpy()
                                    out_v=out_v[0].to("cpu").detach().numpy() #[T x hidden_size]
                            out=0.5*(out_nrm+1)*(target_max.values-target_min.values)+target_min.values
                            print(f"runcount: {run_count}, out nrm: {out_nrm}, out: {out}, in_spike count: {in_spike[0][-1].sum()}")

                            volt_trajectory=add_volt2trajectory(out_v,volt_trajectory)
                            endeffector_deltapos.append(list(out))
                        else:
                            volt_trajectory.append([None for _ in range(snn_hidden_size)])
                            endeffector_deltapos.append([None for _ in range(2)])

                        in_trajectory.append(list(data.qpos))
                        time_scales.append(current_timescale) 
                        endeffector_target_trajectory.append(target_positions[run_count])

                    else:        
                        exptime+=delta_time #update elapsed time since the initial run

                        sequence_t=sequence#int(sequence*current_timescale) #change sequence length according to timescale
                        in_x=np.array(in_trajectory)[-sequence_t:] if len(in_trajectory)>sequence_t else np.array(in_trajectory)
                        in_x=2*(in_x-input_min.values)/(input_max.values-input_min.values)-1
                        in_spike=encoder(torch.Tensor(in_x).unsqueeze(0))

                        in_scales=np.array(time_scales)[-sequence_t:] if len(time_scales)>sequence_t else np.array(time_scales)

                        with torch.no_grad():
                            if nn_conf["model"]["type"].casefold()=="snn":
                                out_nrm, out_v=nn_model.forward(in_spike.flatten(start_dim=2),return_v=True)
                                out_nrm=out_nrm[0,-1].to("cpu").detach().numpy()
                                out_v=out_v[0].to("cpu").detach().numpy() #[T x hidden_size]
                            elif "dyna".casefold() in nn_conf["model"]["type"].casefold():
                                out_nrm, out_v=nn_model.dynamic_forward_given_scale(
                                    in_spike.flatten(start_dim=2), torch.Tensor(in_scales),return_v=True
                                )
                                out_nrm=out_nrm[0,-1].to("cpu").detach().numpy()
                                out_v=out_v[0].to("cpu").detach().numpy() #[T x hidden_size]
                        out=0.5*(out_nrm+1)*(target_max.values-target_min.values)+target_min.values
                        print(f"total time: {total_time:.2f}, runcount: {run_count}, out nrm: {out_nrm}, out: {out}, in_spike count: {in_spike[0][-1].sum()}")

                        current_position=np.array(data.site("attachment_site").xpos)[:-1]
                        dposition= out + k_fb*error_feedback(current_position, ref_trajectory) #NN prediction + error feedback
                        next_target=current_position+dposition/current_timescale #add difference to current position. divide by current_timescale for mathematical correctness
                        
                        in_trajectory.append(list(data.qpos))
                        current_timescale=calculate_timescale(current_timescale,delta_timescale,end_timescale)
                        time_scales.append(current_timescale)
                        endeffector_target_trajectory.append(next_target)
                        volt_trajectory=deepcopy(add_volt2trajectory(out_v,volt_trajectory))
                        endeffector_deltapos.append(list(out))

                    target_x,target_y=endeffector_target_trajectory[-1]
                    run_count+=1
                #<< target trajectory estimation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


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

                # Visualize at fixed FPS.
                viewer.sync() #if this is added, the arm motion is visualized (the internal physical state is updated regardless of the viewer)
                # rate.sleep()

            
    finally:

        result_dir=Path(__file__).parent/Path(args.saveto)/f"trajectory_timescale{current_timescale:.2f}_duration{args.timescale_duration:.2f}_kfb{args.kfb:.5f}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        print(f"volt shape: {np.array(volt_trajectory).shape}")
        print(f"time_trajectory shape: {np.array(time_trajectory).reshape(-1,1).shape}")

        length_diff=np.array(time_trajectory).reshape(-1,1).shape[0]-np.array(volt_trajectory).shape[0]
        if length_diff>0:
            volt_trajectory=[[None for _ in range(snn_hidden_size)] for _ in range(length_diff)]+volt_trajectory

        trajectory=pd.DataFrame(
            np.concatenate([
                np.array(time_trajectory).reshape(-1,1),np.array(time_scales).reshape(-1,1),
                np.array(in_trajectory),np.array(endeffector_target_trajectory),
                np.array(volt_trajectory),
                np.array(endeffector_deltapos)
                ],axis=1),
            columns=["time","timescale",*input_labels,"target_x","target_y"] + [f"volt{i}" for i in range(snn_hidden_size)] + ["delta_x","delta_y"]
        )
        trajectory.to_csv(result_dir/"trajectory.csv",index=False)

        args_dict=vars(args)
        out_dict={
            "args":args_dict,
            "nn_conf":nn_conf,
            "trj_conf":trj_conf,
        }
        with open(result_dir/"args.json","w") as f:
            json.dump(out_dict,f,indent=4)
