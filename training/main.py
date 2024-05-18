import os
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, early_stopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningDataModule
from test.MMDCOT.training.pl_model import VQ_VAE
from models import CLIPVisionTower
from omegaconf import OmegaConf

# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.datasets = dataset
        self.clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
        self.clip.load_model()

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, i):
        i = random.randint(0, len(self.datasets)-1)
        image = None
        while not image:
            i = random.randint(0, len(self.datasets)-1)
            item = self.datasets[i]
            image = item['image']
        conversation = item['conversations']
        if type(conversation) == str:
            conversation = ast.literal_eval(conversation)
        query = conversation[0]['value'].replace('\n<image>', '').replace('\n', ' ')
        answer = item['solution'].replace('\n', ' ').strip() + '</s>'
        image_features = self.clip.image_processor(np.array(image), return_tensors='pt')
        vision_embs = self.clip(image_features['pixel_values'])
        return query, vision_embs, answer

class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ds = {'train': None, 'test': None, 'val': None}
        self.img_folder = 'images'

    def setup(self, stage: str):
        sub_folder_dict = {'fit': 'train', 'test': 'test', 'validate': 'validation'}
        dataset_science = load_dataset("cnut1648/ScienceQA-LLAVA")
        for stage, mode in sub_folder_dict.items():
            
            self.ds[mode] = MyDataset(dataset_science[mode])
    def train_dataloader(self):
        # noinspection PyTypeChecker
        return DataLoader(self.ds['train'], batch_size=self.args.batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available(), )

    def val_dataloader(self):
        # noinspection PyTypeChecker
        return DataLoader(self.ds['val'], batch_size=self.args.batch_size_val, num_workers=8)

        
if __name__ == "__main__":
    args = OmegaConf.load("configs/config_vq_vae_train.yaml")
    # set_cuda_devices(args)
    seed_everything(args.seed)

    data_module = DataModule(args)
    ckpt_path = args.ckpt_path

    callbacks = None
    model_ckpt = None
    logger = WandbLogger(name=args.exp_name, version=args.exp_name, save_dir=args.output_dir,
                               project='vq_vae_training')
    model = None
    if ckpt_path is not None:
        print('check_eval_inf_load')
        model = VQ_VAE.load_from_checkpoint(ckpt_path)
    else:
        model = VQ_VAE(args)

    callbacks = []
    if args.exec_mode == "train":
        early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, verbose=True, mode="min")
        callbacks.append(early_stopping)
        model_ckpt = ModelCheckpoint(filename="best_{epoch}_{val_loss:.2f}", monitor="val_dice",
                                         mode="min", save_last=True, every_n_epochs=10, save_top_k=1)
        callbacks.append(model_ckpt)


    # main class that contain all training logic and processes
    trainer = Trainer(
        logger=logger,
        precision=16,
        benchmark=True,
        deterministic=False,
        min_epochs=args.n_epochs,
        max_epochs=args.n_epochs,
        sync_batchnorm=False,
        gradient_clip_val=0,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        accelerator="gpu",
        devices=args.devices,
    )

    if args.exec_mode == "train":
        trainer.fit(model, data_module)
        # trainer.test(model, data_module)
        print('train is FINISHED')
