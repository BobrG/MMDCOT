import os
import numpy as np
import random
import PIL
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from IPython.display import clear_output
%matplotlib inline
import open_clip
import nltk
nltk.download('punkt')

PIL.Image.MAX_IMAGE_PIXELS = 889730000

DEVICE = torch.device("cuda:0")

hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/projection.pt", local_dir='./')
hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="models.py", local_dir='./')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autotokenizer', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    
    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet','vit'], help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-E', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args

def preprocess_image(image_path, preprocess_function=preprocess_train):
    image = Image.open(image_path).convert("RGB")
    return preprocess_function(image)

def encode_image(path=None, image=None, preprocess_function=preprocess_train):
    if path:
        image_tensor = preprocess_image(path, preprocess_function).to(DEVICE_NEW)
        with torch.no_grad():
            hidden_states = clip.visual(image_tensor[None])[1].to(DEVICE).squeeze(0)
        return hidden_states
    
    if image:
        image_tensor = preprocess_function(image).to(DEVICE_NEW)
        with torch.no_grad():
            hidden_states = clip.visual(image_tensor[None])[1].to(DEVICE).squeeze(0)
        return hidden_states


class MyDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.datasets = dataset

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, i):
        i  = random.randint(0, len(self.datasets)-1)
        image = None
        while not image:
                i  = random.randint(0, len(self.datasets)-1)
                item = self.datasets[i]
                image = item['image']
        conversation = item['conversations']
        if type(conversation) == str:
                conversation = ast.literal_eval(conversation)
        query = conversation[0]['value'].replace('\n<image>', '').replace('\n', ' ')
        answer = item['solution'].replace('\n', ' ').strip() + '</s>'
        return query, image, answer

         
def train_loop():
    N_EPOCHS = 10
    N_ITERS = 1000000
    EXP_NAME = 
    skip = 128
    
    model = AutoModelForCausalLM.from_pretrained("AIRI-Institute/OmniFusion", subfolder="OmniMistral-v1_1/tuned-model", torch_dtype=torch.bfloat16, device_map=DEVICE)

    projection = torch.load("OmniMistral-v1_1/projection.pt", map_location=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/OmniFusion", subfolder="OmniMistral-v1_1/tokenizer", use_fast=False)

    clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
    clip.load_model()
    clip = clip.to(device=DEVICE, dtype=torch.bfloat16)

    model_ae = diffusers.VQModel(1, 1).to(DEVICE)
    for epoch in range(N_EPOCHS):
        for query, image, answer in dataloader:               
            with torch.no_grad():
                image_features = clip.image_processor(image, return_tensors='pt')
                image_embedding = clip(image_features['pixel_values']).to(device=DEVICE, dtype=torch.bfloat16)
            
                caption_ids = list(tokenizer.encode(answer, add_special_tokens=False))
                caption_ids = torch.LongTensor(caption_ids).to(DEVICE)
                user_query_ids = tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(DEVICE)

            vision_embs = image_embedding
            projected_vision_embs = projection(vision_embs)
            text_embeddings = model.model.embed_tokens(caption_ids)
            prompt_embeddings = model.model.embed_tokens(user_query_ids)
            bs = projected_vision_embs.shape[0]
         
            embeddings1 = torch.cat([
                        projected_vision_embs,
                        prompt_embeddings
                        ],
                        dim=1)
            
            embeddings2 = text_embeddings.repeat(bs,1,1)
            embeddings = torch.cat([embeddings1, embeddings2], dim=1)

            
            outputs = model(inputs_embeds = embeddings.half(), output_hidden_states=True)
            logits = outputs.get("logits")
            logits = logits[..., embeddings1.shape[1] - 1 :-1, :].contiguous()
            labels = caption_ids.contiguous()
            loss = loss_ae(logits.view(-1, N_EMBEDDINGS), labels.view(-1)).mean()
            
            loss.backward(retain_graph=True)
            losses_batch.append(loss.item())
    
            if iters % skip == 0:
                clear_output(True)
                opt.step()
                opt.zero_grad()
                losses.append(np.mean(losses_batch))
                losses_batch = []
                print(f"EPOCH: {epoch}")
                plt.title("train loss")
                plt.semilogy(losses)
                plt.grid()
                plt.savefig(f'{EXP_NAME}_loss')
                plt.show()
                if iters % (skip * 10) == 0:
                        test()
                        torch.save(projection, f'ckpts/{EXP_NAME}_projection')
                        model.save_pretrained(f'ckpts/{EXP_NAME}-AE')
                        tokenizer.save_pretrained(f'ckpts/{EXP_NAME}-AE')
                        print("SAVED!")
            iters += 1


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
    
    dataset_science = load_dataset("cnut1648/ScienceQA-LLAVA")
    dataset = MyDataset(dataset_science['train'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



    
    