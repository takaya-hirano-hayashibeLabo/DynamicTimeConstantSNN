import h5py
from multiprocessing import Pool
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
import numpy as np
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def print_terminal(contents="",pre="\033[96m",suffi="\033[0m"):
    """
    :param pre: prefix
    :param contentns: contents
    :param suffi: suffix
    """

    import shutil
    termianl_width=shutil.get_terminal_size().columns
    s= contents[:termianl_width] if len(contents)>termianl_width else contents

    print(pre+s+suffi)
    

def load_yaml(file_path):
    """
    Load a YAML file and return its contents as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML file contents.
    """
    import yaml
    with open(file_path, 'r',encoding="utf-8") as file:
        return yaml.safe_load(file)



def load_single_hdf5(path):
    with h5py.File(path, 'r') as f:
        data = f['events'][:]
        target = f['target'][()]
    return data, target

def load_hdf5(file_path_list: list, num_workers: int = 64):
    """
    read hdf5 files from a list of paths, return data
    :return datas: [batch x time_sequence x ...]
    :return targets: [batch]
    """
    with Pool(num_workers) as pool:
        results = pool.map(load_single_hdf5, file_path_list)

    datas, targets = zip(*results)
    return list(datas), list(targets)


def save_dict2json(data, saveto):
    """
    Save a dictionary to a specified path in JSON format.

    Parameters:
    data (dict): The dictionary to save.
    saveto (str or Path): The path where the JSON file will be saved.
    """
    import json
    with open(saveto, 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def load_json2dict(file_path):
    """function to load a JSON file as a dict"""
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def resample_scale(a: list, target_length: int) -> list:
    """
    function to resample a 1D list to a specified length
    use linear interpolation
    :param data: 1D list to resample
    :param target_length: length after resampling
    :return: resampled list
    """
    original_length = len(a)
    if original_length == target_length:
        return a

    original_indices = np.linspace(0, original_length - 1, num=original_length)
    target_indices = np.linspace(0, original_length - 1, num=target_length)
    
    resampled_data = np.interp(target_indices, original_indices, a)
    
    return resampled_data.tolist()

def spike2timestamp(spike:np.ndarray,dt:float):
    """
    function to convert a spike column of 0 or 1 to a spike timestamp
    :param spike: [time-sequence x ...]
    :return timestamp: timestamp of the spike
    :return idx_x: spatial position of the spike
    """
    idx_sp=np.argwhere(spike==1)
    idx_t=idx_sp[:,0]
    idx_x=idx_sp[:,1:]
    timestamp=dt*idx_t
    return timestamp.reshape(-1,1),idx_x


def timestamp2spike(timestamp:np.ndarray,idx_x,dt,spike_shape:tuple):
    """
    :param spike_shape: size of dimensions other than time
    """
    from math import ceil

    T=ceil(np.max(timestamp)/dt)+1
    idx_time=np.array((timestamp/dt).round(),dtype=np.int64)

    spike=np.zeros(shape=(T,*spike_shape))
    spike_idx=np.concatenate([idx_time,idx_x],axis=1)
    spike[tuple(spike_idx.T)]=1

    return spike


def scale_sequence(data:np.ndarray,a:list,dt:float):
    """
    :param data: [batch x time-sequence x ...]
    :param a: list of scaling. 1 or more [time-sequence]
    :param dt: delta t of data
    """
    from math import ceil

    elapsed = np.cumsum(np.concatenate(([0], a[:-1]))) * dt
    T_max=ceil(elapsed[-1]/dt)
    scaled_data=[]
    for data_i in tqdm(data): #process one batch at a time
        timestamp, idx_x=spike2timestamp(data_i,dt)

        scaled_timestamp = np.zeros_like(timestamp)

        for t in range(data_i.shape[0]):
            mask = (timestamp >= t * dt) & (timestamp < (t + 1) * dt)
            scaled_timestamp[mask] = elapsed[t]

        scaled_spike=timestamp2spike(
            scaled_timestamp,idx_x,
            dt,data_i.shape[1:]
        )

        if scaled_spike.shape[0]<T_max:
            scaled_spike=np.concatenate([
                scaled_spike, np.zeros(shape=(T_max-scaled_spike.shape[0], *data_i.shape[1:]))
                ],axis=0)

        scaled_data.append(scaled_spike)

    return np.array(scaled_data)


def calculate_accuracy(output, target):
    """
    function to check the accuracy of LSTM and others
    """
    import torch
    predicted:torch.Tensor = torch.argmax(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy




from math import ceil
class Event2Frame():
    def __init__(self, sensor_size,time_window):
        """
        :param sensor_size: (channel x h x w) ※give in this order
        """
        self.sensor_size=sensor_size
        self.time_window=time_window

    def __call__(self, events:np.ndarray):
        """
        :param events: [event_num x (x,y,p,t)]
        """

        t_start=events[0]["t"]
        t_end=events[-1]["t"]
        time_length=ceil((t_end-t_start)/self.time_window)

        frame=np.zeros(shape=(time_length,)+self.sensor_size,dtype=np.int16)
        current_time_window=t_start+self.time_window
        t=0
        for e in events:
            if e["t"]>current_time_window:
                current_time_window+=self.time_window
                t+=1
            frame[t,int(e["p"]),e["y"],e["x"]]=1

        return frame
    

class Pool2DTransform(nn.Module):
    def __init__(self, pool_size,pool_type="max"):
        super(Pool2DTransform,self).__init__()
        
        if pool_type=="max".casefold():
            self.pool_layer=nn.MaxPool2d(pool_size)
        if pool_type=="avg".casefold():
            self.pool_layer=nn.AvgPool2d(pool_size)

    def __call__(self, events):
        # tensor should be of shape (T, C, H, W)
        with torch.no_grad():
            events = self.pool_layer(events.to(torch.float))
        return events  # Remove batch dimension


def resize_heatmap(frame:np.ndarray, scale:int=5):
    """
    :param frame: [h x w], -1<=frame<=1
    """
    import cv2
    h,w=frame.shape
    frame=((frame+1)/2*255).astype(np.uint8) #convert [-1,1] to [0,255]
    resized_heatmap=cv2.resize(frame,(w*scale,h*scale),interpolation=cv2.INTER_NEAREST)
    return resized_heatmap

def apply_cmap2heatmap(frame:np.ndarray, cmap:str="viridis"):
    import matplotlib.pyplot as plt
    import numpy as np
    viridis_colormap = plt.get_cmap(cmap)
    colored_heatmap = viridis_colormap(frame / 255.0)  # Normalize to [0, 1] for colormap
    colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB
    return colored_heatmap


def save_heatmap(frame, output_path, file_name, scale=5, border_size=10):
    """
    :param frame: [h x w]
    :param border_size: Size of the white border to add around the heatmap.
    """
    import cv2
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    h, w = frame.shape
    normalized_frame = ((frame + 1) / 2 * 255).astype(np.uint8)
    resized_heatmap = cv2.resize(normalized_frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Apply the viridis colormap
    viridis_colormap = plt.get_cmap('viridis')
    colored_heatmap = viridis_colormap(resized_heatmap / 255.0)  # Normalize to [0, 1] for colormap
    colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB

    # Add a white border around the colored heatmap
    bordered_heatmap = cv2.copyMakeBorder(
        colored_heatmap, 
        border_size, border_size, border_size, border_size, 
        cv2.BORDER_CONSTANT, 
        value=[255, 255, 255]  # White color
    )

    # Save the final image
    plt.imsave(str(output_path / file_name), bordered_heatmap)

    
def save_heatmap_video(frames, output_path, file_name, fps=30, scale=5, frame_label_view=True):
    import cv2
    import subprocess
    import os

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    height, width = frames[0].shape
    new_height, new_width = int(height * scale), int(width * scale)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    tmpout = str(output_path / "tmp.avi")
    video = cv2.VideoWriter(tmpout, fourcc, fps, (new_width, new_height), isColor=True)

    for i, frame in enumerate(frames):
        # Normalize frame to range [0, 255] with original range [-1, 1]
        normalized_frame = ((frame + 1) / 2 * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_JET)
        resized_heatmap = cv2.resize(heatmap, (new_width, new_height))

        # Add frame number text
        if frame_label_view:
            cv2.putText(resized_heatmap, f"Frame: {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        video.write(resized_heatmap)

    video.release()

    # Re-encode the video using ffmpeg
    file_name = file_name + ".mp4" if not ".mp4" in file_name else file_name
    ffmpeg_command = [
        'ffmpeg', '-y', '-i', tmpout,
        '-pix_fmt', 'yuv420p', '-vcodec', 'libx264',
        '-crf', '23', '-preset', 'medium', str(output_path / file_name)
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Remove the temporary file
    os.remove(tmpout)


def create_windows(data:torch.Tensor, window, overlap=0.5):
    """
    function to split time series data into windows
    :param data: time series data (torch tensor) [T x m]
    :param window: window size
    :param overlap: overlap rate
    :return: data split into windows (torch tensor) [N x window x m]
    """
    T, m = data.shape
    step = window - int(window*overlap)
    num_windows = int((T - overlap) // step)
    
    windows = []
    for i in range(num_windows):
        start = i * step
        end = start + window
        window_data=data[start:end, :]
        if window_data.shape[0]<window: #if the window size is smaller than the window size, end
            break
        windows.append(window_data)
    
    return torch.stack(windows)