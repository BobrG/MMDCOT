import os
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from training.point_tokenizer.point_m2ae import Point_M2AE
from PIL import Image


class Sk3DDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        self.point_encoder = Point_M2AE(group_sizes=[16, 8, 8], num_groups=[512, 256, 64], encoder_depths=[5, 5, 5], encoder_dims=[96, 192, 384], local_radius=[0.32, 0.64, 1.28]).cuda()
        state_dict = torch.load('../ckpts/', map_location=map_location)
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items() if 'encoder' in k}
        self.point_encoder.load_state_dict(base_ckpt, strict = True)
        
        self.data = []
        
        self.clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
        self.clip.load_model()
        
        objects = os.listdir(os.path.join(root_dir, 'dataset'))
        for object_name in objects:
            rgb_path = os.path.join(root_dir, 'dataset', object_name, 'tis_right', 'rgb', 'undistorted', 'ambient@best')
            points_path = os.path.join(root_dir, 'point_clouds', object_name)
            texts_path = os.path.join(root_dir, 'texts', object_name)
            
            if not (os.path.isdir(rgb_path) and os.path.isdir(points_path) and os.path.isdir(texts_path)):
                continue

            rgb_files = os.listdir(rgb_path)
            points_files = os.listdir(points_path)

            with open(os.path.join(texts_path, f'{object_name}_conversations.json'), 'r') as f:
                conversations = json.load(f)
            with open(os.path.join(texts_path, f'{object_name}_cots.json'), 'r') as f:
                cots = json.load(f)

            for idx, (img_file, point_file) in enumerate(zip(rgb_files, points_files)):
                self.data.append({
                    'image_path': os.path.join(rgb_path, img_file),
                    'point_path': os.path.join(points_path, point_file),
                    'conversations': conversations[idx],
                    'cots': cots[idx]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(len(self.data))
        idx = random.randint(0, len(self.data)-1)
        item = self.data[idx]
        image = Image.open(item['image_path'])
        image_features = self.clip.image_processor(np.array(image), return_tensors='pt')
        vision_embs = self.clip(image_features['pixel_values'])
        
        point_cloud = torch.FloatTensor(np.load(item['point_path'])[None, ...]).cuda()
        point_embs = self.point_encoder(point_cloud)

        conversation = item['conversations']
                
        query = conversation.split("**User:**")[1].split("\n\n**Assistant:**")[0].replace('\n', ' ')
        
        query += ' Chain-Of-Thought: ' + item['cots'] # add chain of thought
        
        answer = conversation.split("**User:**")[1].split("\n\n**Assistant:**")[1].replace('\n', ' ') + '</s>'

        return query, vision_embs, point_embs, answer

