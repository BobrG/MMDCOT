import os
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import Adafactor, AdafactorSchedule
from huggingface_hub import hf_hub_download
import diffusers 
from IPython.display import clear_output
import matplotlib.pyplot as plt
import nltk
import ast
from models import CLIPVisionTower
import wandb
nltk.download('punkt')

hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/projection.pt", local_dir='./')
hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="models.py", local_dir='./')
        

class VQ_VAE(pl.LightningModule):
    """
    Module organizes PyTorch code includding train/ validation/test loop. This class also contain model, loss and optimizer initialization
    
    """
    def __init__(self, cfg, triton=False, data_dir=None):
        super(VQ_VAE, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        if data_dir is not None:
            self.args.data = data_dir
        
        self.learning_rate = cfg.learning_rate
        self.model = AutoModelForCausalLM.from_pretrained("AIRI-Institute/OmniFusion", subfolder="OmniMistral-v1_1/tuned-model", torch_dtype=torch.bfloat16)
        self.projection = torch.load("OmniMistral-v1_1/projection.pt")
        self.tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/OmniFusion", subfolder="OmniMistral-v1_1/tokenizer", use_fast=False)

        self.clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
        self.clip.load_model()

        self.model_ae = diffusers.VQModel(1, 1,vq_embed_dim= 1, scaling_factor=cfg.scaling_factor)
        self.reconstruction_loss = nn.CrossEntropyLoss()


    def prepare_embedings(self, batch, ):
        query, vision_embs, answer = batch
        caption_ids = list(self.tokenizer.encode(answer, add_special_tokens=False))
        caption_ids = torch.tensor(caption_ids, dtype=torch.long, device=self.device)
        user_query_ids = self.tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(self.device)
        projected_vision_embs = self.projection(vision_embs[0])
        text_embeddings = self.model.model.embed_tokens(caption_ids)
        prompt_embeddings = self.model.model.embed_tokens(user_query_ids)
        bs = projected_vision_embs.shape[0]
         
        embeddings1 = torch.cat([
                        projected_vision_embs,
                        prompt_embeddings
                        ],
                        dim=1)
            
        embeddings2 = text_embeddings.repeat(bs,1,1)
        embeddings = torch.cat([embeddings1, embeddings2], dim=1)
        return embeddings
    
    def prepare_embedings_3d(self, batch, ):
        query, vision_embs, point_embs, answer = batch
        caption_ids = list(self.tokenizer.encode(answer, add_special_tokens=False))
        caption_ids = torch.tensor(caption_ids, dtype=torch.long, device=self.device)
        user_query_ids = self.tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(self.device)
        projected_vision_embs = self.projection(vision_embs[0])
        text_embeddings = self.model.model.embed_tokens(caption_ids)
        prompt_embeddings = self.model.model.embed_tokens(user_query_ids)
        bs = projected_vision_embs.shape[0]
         
        embeddings1 = torch.cat([
                        point_embs,
                        projected_vision_embs,
                        prompt_embeddings
                        ],
                        dim=1)
            
        embeddings2 = text_embeddings.repeat(bs,1,1)
        embeddings = torch.cat([embeddings1, embeddings2], dim=1)
        return embeddings


                          
    def training_step(self, batch, batch_idx):
        if self.cfg.data_3d:
            embeddings = self.prepare_embedings_3d(batch)
        else:
            embeddings = self.prepare_embedings(batch)
        
        h = self.model_ae.encode(embeddings[None, ...].float()).latents
        _, vq_loss, _ = self.model_ae.quantize(h)
        out = self.model_ae.decode(h).sample
        reconst_loss = self.reconstruction_loss(out, embeddings[None, ...].float())
        loss = reconst_loss + vq_loss
        
        self.log("train_loss", loss)
        self.log("train_reconst_loss", reconst_loss)
        self.log("train_vq_loss", vq_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings = self.prepare_embedings(batch)
        
        h = self.model_ae.encode(embeddings[None, ...].float()).latents
        _, vq_loss, _ = model_ae.quantize(h)
        out = self.model_ae.decode(h).sample
        reconst_loss = self.reconstruction_loss(out, embeddings[None, ...].float())
        loss = reconst_loss + vq_loss
        
        # (2) computing DICE (averaging over the batch):
        dlist = []
        for i in range(len(imgs)):
            dice, _, _ = self._compute_dice_iou(i, imgs, masks, preds, trans)
            dlist.append(dice)

        mdice = np.mean(dlist)

        # (3) logging metric:
        self.log("val_dice", mdice, batch_size=self.args.val_batch_size, on_epoch=True)
        self.log("val_loss", loss)
        self.log("val_reconst_loss", reconst_loss)
        self.log("val_vq_loss", vq_loss)
   

    def configure_optimizers(self):
        optimizer =  optim.AdamW(self.model_ae.parameters(), lr=self.learning_rate, weight_decay=self.cfg.weight_decay)
        
        scheduler = {
                "scheduler": WarmupCosineSchedule(
                    optimizer=optimizer,
                    warmup_steps=250,
                    t_total=self.cfg.n_epochs * len(self.trainer.datamodule.train_dataloader()),
                ),
                "interval": "step",
                "frequency": 1,
            }
        return {"optimizer": optimizer, "monitor": "val_loss", "lr_scheduler": scheduler}
