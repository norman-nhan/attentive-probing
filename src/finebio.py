import json
import os
import random
from decord import VideoReader, cpu
import random

class FineBioDataset():
    def __init__(self,  
        json_file: str,
        label2id_dir: str,
        video_dir: tuple,
        split: tuple,
        sampling_rate: int = 1,
        frames_per_clip: int = 16,
        allow_clip_overlap: bool = True,
        random_view: bool = False, # sample different views if possible
        view: str = "T0" # T0 is fpv, T1-4 are tpv
    ):
        # assign dataset attributes
        self.json_file = json_file
        self.label2id_dir = label2id_dir
        self.video_dir = video_dir
        self.split = split
        self.sampling_rate = sampling_rate
        self.fpc = frames_per_clip
        # self.allow_clip_overlap = allow_clip_overlap
        self.random_view = random_view
        self.view = view if random_view == False else "use_random_view"
        
        # main
        self.data = []
        # 1. load json file
        with open(self.json_file, 'r') as f:
            anno = json.load(f)
        
        self.type_info = anno['type_info']
        
        # 2. filter videos by split
        videos = {video_id: info for video_id, info in anno["database"].items() if info["subset"] in self.split}
                
        # 3. assign video path to each video
        for video_name in videos:
            if random_view:
                view = f"T{random.choice(range(0, 5))}"
                # search for video path matching view
                paths = []
                count = 0
                while count < 3 and len(paths) == 0:
                    for vid_dir in self.video_dir:
                        if view == "T0":
                            video_path = os.path.join(vid_dir, f"{video_name}.mp4")
                        else:
                            video_path = os.path.join(vid_dir, f"{video_name}_{view}.mp4")
                        if os.path.isfile(video_path):
                            paths.append(video_path)
                    count += 1
                    view = f"T{random.choice(range(0, 5))}"
                
                assert len(paths) > 0, f'No video found for {video_name} with view {view} in {self.video_dir}.'
                videos[video_name]["path"] = paths[0]
            else:
                if self.view == "T0":
                    video_path = os.path.join(self.video_dir[0], f"{video_name}.mp4")
                else:
                    video_path = os.path.join(self.video_dir[0], f"{video_name}_{self.view}.mp4")
                assert os.path.isfile(video_path), f'No video found for {video_name} with view {self.view} in {self.video_dir[0]}.'
                videos[video_name]["path"] = video_path

        # 4. flatten video segments into one sequence. Meaning total data length is a sum of the number of segments of all videos.
        for item in videos.values():
            vr = VideoReader(item["path"], ctx=cpu(0), num_threads=8)
            segments = item["annotations"]
            for seg in segments:
                ts, te = seg["segment"] # sec
                sf = max(0, int(ts * item["fps"])) # frame index
                ef = min(len(vr), int(te * item["fps"])) # frame index
                indices = list(range(sf, ef, self.sampling_rate))
                
                if len(indices) == 0:
                    continue
                
                clips = self._split_into_clips(indices)

                for clip in clips:
                    self._register_data(item=item, seg=seg, indices=clip)
    
    def _split_into_clips(self, indices):
        
        # if segment length < frames per clip, pad then return it
        if len(indices) <= self.fpc:
            indices = self._padding_clip(indices)
            return [indices]
        
        # if segment length > frames per clip, split into multiple clips, pad if necessary and return a list of clips
        if len(indices) > self.fpc:
            clips = []
            for i in range(0, len(indices), self.fpc):
                end_idx = i + self.fpc
                # check out of bound
                if end_idx > len(indices):
                    end_idx = len(indices)
                
                sub_indices = indices[i:end_idx]
                clips.append(self._padding_clip(sub_indices))
            return clips
    
    def _padding_clip(self, indices):
        assert len(indices) > 0
        if len(indices) < self.fpc:
            last_frame_idx = indices[-1]
            indices += [last_frame_idx] * (self.fpc - len(indices))
        return indices
    
    def _register_data(self, item, seg, indices):
        self.data.append({
            # 'start': sf,
            # 'end': ef,
            # 'duration': ef-sf,
            'indices': indices,
            'path': item['path'],
            'verb_id': seg['verb_label_id'],
            'manipulated_id': seg['manipulated_label_id'],
            'affected_id': seg['affected_label_id'],
            'atomic_operation_id': seg['atomic_operation_label_id'],
            'hand_id': seg['hand_label_id']
        })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]