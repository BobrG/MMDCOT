import os
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from ..point_tokenizer.point_m2ae import Point_M2AE
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root_dir, point_encoder):
        self.root_dir = root_dir
        self.point_encoder = point_encoder
        self.data = []
        
        objects = os.listdir(os.path.join(root_dir, 'dataset'))
        for object_name in objects:
            rgb_path = os.path.join(root_dir, 'dataset', object_name, 'tis_right', 'rgb')
            points_path = os.path.join(root_dir, 'points', object_name)
            texts_path = os.path.join(root_dir, 'texts', object_name)

            if not (os.path.isdir(rgb_path) and os.path.isdir(points_path) and os.path.isdir(texts_path)):
                continue

            rgb_files = os.listdir(rgb_path)
            points_files = os.listdir(points_path)

            with open(os.path.join(texts_path, f'{object_name}_conversations.json'), 'r') as f:
                conversations = json.load(f)
            with open(os.path.join(texts_path, f'{object_name}_cots.json'), 'r') as f:
                cots = json.load(f)

            for img_file, point_file in zip(rgb_files, points_files):
                self.data.append({
                    'image_path': os.path.join(rgb_path, img_file),
                    'point_path': os.path.join(points_path, point_file),
                    'conversations': conversations[idx],
                    'cots': cots[idx]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.datasets)-1)
        item = self.data[idx]
        image = Image.open(item['image_path'])
        point_cloud = np.load(item['point_path'])
        point_embs = self.point_encoder(point_cloud)

        conversation = item['conversations']
        query = conversation.split("**User:**")[1].split("\n\n**Assistant:**")[0].replace('\n', ' ')
        query += ' Chain-Of-Thought: ' + item['cots'][idx] # add chain of thought

        answer = conversation.split("**User:**")[1].split("\n\n**Assistant:**")[1].replace('\n', ' ') + '</s>'

        return query, point_embs, answer
