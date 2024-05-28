# MMDCOT

For all experiment please clone this repo: ```git clone https://github.com/BobrG/MMDCOT```

### Early experiments

* 3D tokenizer:
  Before applying tokenizer to 3D point cloud data, it was debugged following guidelines from the official repo of the picked model - [Point Masked AutoEncoder](https://github.com/ZrrSkywalker/Point-M2AE/tree/main). The quality of tokenizer was checked by visualization of produced reconstructions on ShapeNet dataset. To reproduce this experiment please follow these steps:
  1. prepare Point_M2AE code and download ShapeNet data as described in the original repo
  
  2. replace /path/to/Point-M2AE/models/Point_M2AE.py with /path/to/MMDCOT/Point_M2AE.py and create directory /path/to/Point-M2AE/pointmae_reconstructions
 
  3. launch pre-training on ShapeNet with the script:
      ```
      CUDA_VISIBLE_DEVICES=0 python main.py --config /path/to/MMDCOT/point-m2ae.yaml --exp_name pre-train --start_ckpts ckpts/pre-train.pth --ckpt ckpts/pre-train.pth
      ```
  4. Now you are able to visualize reconstructions and original data, stored in /path/to/Point-M2AE/pointmae_reconstructions as plys via i.e meshlab.
  

### 3D Dataset experiments

* Dataset preparation:
  Please find explanations on how to prepare dataset [here](https://github.com/BobrG/MMDCOT/tree/main/ae_training/dataset#readme).
* 
### Install

To run training process you need to go throught the installation process from [LLAVA](https://github.com/PolinaDruzhinina/LLaVA?tab=readme-ov-file#install)

### Inference

1. You need to prepare dataset, for example ScienceQA and save test part as in Llava pipeline under ```.LLaVA/playground/data/eval/scienceqa```, download ```images```, ```pid_splits.json```, ```problems.json``` from the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Go to the LLaVA directory to run the following commands

3. Convert it into LLaVA conversation-style format.

```
python scripts/convert_sqa_to_llava.py convert_to_llava --base-dir playground/data/eval/scienceqa --prompt-format "CQM-A" --split test
```
4. Run inference and evaluation.

```
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa.sh
```

### Trainig and Fine-Tuning
####  VQ-VAE training
Before running experiments with VQ-VAE model you need to specify config (/home/polina/test/MMDCOT/ae_training/configs/config_vq_vae_train.yaml), set your results folder and so on, then run ``` cd ae_training & python main.py ```


####  Fine-Tuning
The training consists of two stages: (1) feature alignment stage: connecting a frozen pretrained quntized encoder with CoT to a frozen LLM; (2) tuning stage: fine-tuning LLM, teaching the model to follow multimodal instructions based on compressed CoT sequentes.

1. For adapter training like in LLaVA pipeline we used the same datasets, so you need prepare them: download the 558K subset of the LAION-CC-SBU dataset with BLIP captions (you can find it on [huggingface](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain))

2. On feature alignment step you need to train projector, for that you need to run this command with primary argument ```--tune_mm_mlp_adapter True``` 
```
 deepspeed llava/train/train_mem.py --tune_mm_mlp_adapter True

 ```
 The example of this command you can find in the [script](https://github.com/PolinaDruzhinina/LLaVA/blob/main/scripts/pretrain.sh) 

 2. To finetune LLM you need to prepare another set of data, so please download the annotation of the data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from dataset COCO, GQA, OCR-VQA, TextVQA, VisualGenome under the ``` ./playground/data``` folder.

 3. Then run the training on the largest number of gpu's available to you. 

 ```
deepspeed llava/train/train_mem.py --lora_enable True 
 ```

 The example of this command you can find in the [script](https://github.com/PolinaDruzhinina/LLaVA/blob/main/scripts/finetune_lora.sh) 


