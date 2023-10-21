import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
from decord import VideoReader
import cv2

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from animatediff.utils.util import zero_rank_print
from torchvision.io import read_image



class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        random.shuffle(self.dataset)    
        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        print("length",len(self.dataset))
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    


    def get_batch_big(self, idx):
        while True:
            video_dict = self.dataset[idx]
            videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        
            video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
            # If the video doesn't exist, return pixel values filled with zeros
            if not os.path.exists(video_dir):
                
                idx = idx+ 1
                continue
               
                
            video_reader = VideoReader(video_dir)
            video_length = len(video_reader)
        
            if not self.is_image:
                clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
                start_idx = random.randint(0, video_length - clip_length)
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
            else:
                batch_index = [random.randint(0, video_length - 1)]
        
            pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
            pixel_values = self.center_crop(pixel_values)
            pixel_values = pixel_values / 255.
            del video_reader
        
            if self.is_image:
                pixel_values = pixel_values[0]
        
            return pixel_values, name

    def center_crop(self,img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top:top+min_dim, left:left+min_dim]

    def get_batch_webvid_video(self, idx):
        while True:
            video_dict = self.dataset[idx]
            videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
            
            video_dir = os.path.join(self.video_folder, page_dir, f"{videoid}.mp4")
            
            # Check if the file exists
            if not os.path.exists(video_dir):
                #print(f"File not found: {video_dir}")
                idx = random.randint(0, len(self.dataset) - 1)
                continue  # try the next index
            
            video_reader = VideoReader(video_dir)
            video_length = len(video_reader)
            
            if not self.is_image:
                clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
                start_idx = random.randint(0, video_length - clip_length)
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
            else:
                batch_index = [random.randint(0, video_length - 1)]
            
            pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.
            del video_reader
            
            if self.is_image:
                pixel_values = pixel_values[0]
            
            return pixel_values, name
            
    def get_batch(self, idx):
        def sort_frames(frame_name):
            # Extract the frame number and convert it to an integer
            return int(frame_name.split('_')[1].split('.')[0])
    
        while True:
            video_dict = self.dataset[idx]
            videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
    
            # Directory where preprocessed images for the video are stored
            preprocessed_dir = os.path.join(self.video_folder, videoid)
    
            # Check if the directory exists
            if not os.path.exists(preprocessed_dir):
                idx = random.randint(0, len(self.dataset) - 1)
                continue  # try the next index
    
            if not self.is_image:
                image_files = sorted(os.listdir(preprocessed_dir), key=sort_frames)
                total_frames = len(image_files)
                clip_length = min(total_frames, (self.sample_n_frames - 1) * self.sample_stride + 1)
                start_idx = random.randint(0, total_frames - clip_length)
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
            else:
                image_files = random.choice(os.listdir(preprocessed_dir))
                batch_index = [image_files]
    
            # Read images from the directory
            pixel_values = torch.stack([read_image(os.path.join(preprocessed_dir, image_files[int(i)])) for i in batch_index])
            pixel_values = pixel_values.float() / 255.  # Convert from uint8 to float and normalize
    
            if self.is_image:
                pixel_values = pixel_values[0]
    
            return pixel_values, name

    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                print (e)
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample



if __name__ == "__main__":
    from animatediff.utils.util import save_videos_grid

    dataset = WebVid10M(
        csv_path="/data/webvid/results_2M_train.csv",
        video_folder="/data/webvid/data/videos",
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=True,
    )
    import pdb
    pdb.set_trace()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)
