import numpy as np
from scipy.spatial.transform import Rotation as R

import mink

def transform_matrix(quaternion, position):
    """
    calculate transformation matrix from quaternion and position
    :param quaternion: [x,y,z,w]
    :param position: [x,y,z]
    :return: 4x4 transformation matrix
    """
    # Create a rotation object from the quaternion
    rotation = R.from_quat(quaternion)

    # Convert the rotation object to a 3x3 rotation matrix
    rotation_matrix = rotation.as_matrix()

    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    return transformation_matrix


def np2SE3_target(quaternion:np.ndarray,position:np.ndarray):
    """
    convert quaternion and position to mink target
    :param quaternion: [x,y,z,w]
    :param position: [x,y,z]
    :return: 4x4 transformation matrix
    """
    t_matrix = transform_matrix(np.array(quaternion), np.array(position))
    target=mink.SE3.from_matrix(t_matrix)
    
    return target


def save_video(frames, output_path, file_name, fps=30, scale=5, frame_label_view=True):
    """
    :param frames: list of frames [t x h x w x c]
    :param output_path: output path
    :param file_name: file name
    :param fps: frame rate
    :param scale: scale
    :param frame_label_view: frame label view
    """
    import cv2
    import subprocess
    import os

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    frames=np.array(frames)
    height, width,channel = frames[0].shape
    new_height, new_width = int(height * scale), int(width * scale)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    tmpout = str(output_path / "tmp.avi")
    video = cv2.VideoWriter(tmpout, fourcc, fps, (new_width, new_height), isColor=True)

    for i, frame in enumerate(frames):
        # Normalize frame to range [0, 255] with original range [-1, 1]
        bgr_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        resized_frame = cv2.resize(bgr_frame, (new_width, new_height))

        # Add frame number text
        if frame_label_view:
            cv2.putText(resized_frame, f"Frame: {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        video.write(resized_frame)

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