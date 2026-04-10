import os 
from glob import glob
from decord import VideoReader, cpu
import torch


def filter_paths(data_paths, target_view_id, min_par_id):
    path_lists = []
    for path in data_paths:
        # if view point is not egocentric, view id is specified in the end of video name.
        # video name: {participant id}_{protocol number}_{trial number}_{view id}
        video_name = os.path.splitext(os.path.basename(path))[0]
        
        # Filter view id
        view_id = "T0"
        if len(video_name.split('_')) == 4: # get view id if tpv
            view_id = video_name.split('_')[3] 
        if target_view_id != view_id: # skip unwanted view id
            continue
        
        # Filter participant id
        par_id = int(video_name.split('_')[0][1:])
        if par_id < min_par_id: # skip participant id < min_par_id
            continue
        
        # Filter duplicated path
        if path in set(path_lists): # skip duplicate paths
            continue
        
        # Collect valid path
        path_lists.append(path)

    assert len(path_lists) > 0, f"Either view id or participant id meet the requirements."
    return sorted(path_lists)

def get_video_paths(data_dir, target_view_id:str="T0", min_par_id:int=0):
    data_paths = glob(os.path.join(data_dir, "*.mp4"))
    assert len(data_paths) > 0, f"No mp4 files found in {data_dir}!"
    
    path_lists = filter_paths(data_paths, target_view_id, min_par_id)
    return path_lists

def decode_video_to_clips(video_path,
                        frames_per_clip:int,
                        clip_stride:int,
                        sampling_rate:int=1):
    """Decode and split video to multiple clips.
    
    - Number of clips depends on clip_stride.
    - If clip length is shorter than frames_per_clip, padding that clip with black frames to get full clip.
    - Instead of return the whole buffer which consume a lot of memory. Yield the result.
    - Number of clips is calculated by this equation:
    ```python
    _num_clips = (total_frames + clip_stride -1) // clip_stride
    ```
    Args:
        video_path (str): path to video
        frames_per_clip (int): number of frames per output clip. **IMPORTANT**: It is not number of frames that clip spanning in original video fps.
        sampling_rate (int, default=1): if sampling_rate=1, sampling fps=video fps. Meaning no frames are skipped.
        clip_stride (int): number of skipped frames to get the next clip in original video fps.
    
    Returns:
        dict(buffer=buffer, num_clips=_num_clips)
        
        - buffer shape: T,H,W,3
    """
    assert os.path.exists(video_path), "Unvalid video path!"
    assert frames_per_clip > 0, "frames_per_clip must be greater than 0"
    assert clip_stride > 0, "clip stride must be greater than 0."
    assert sampling_rate > 0, "sampling rate must be greater than 0."
    
    # === decode video ===
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    _num_clips = (total_frames + clip_stride -1) // clip_stride
    
    # calculate clip spanning in original video fps
    clip_duration = int(frames_per_clip * sampling_rate)

    # grab start indices 
    start_indices = range(0, total_frames, clip_stride)    
    for start_frame in start_indices:
        end_frame = start_frame + clip_duration
        frame_indices = list(range(start_frame, min(end_frame, total_frames), sampling_rate))
        
        buffer = vr.get_batch(frame_indices).asnumpy() # T,H,W,C
        buffer = torch.from_numpy(buffer)
        t,h,w,c = buffer.shape
        # padding
        if end_frame > total_frames:
            dummy = torch.zeros((end_frame - total_frames, h, w, c))
            buffer = torch.cat((buffer, dummy), dim=0) # add dummy on time dim

        yield dict(buffer=buffer, num_clips=_num_clips)