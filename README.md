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
  Please find explanations on how to prepare dataset [here](https://github.com/BobrG/MMDCOT/tree/main/dataset#readme).
* 

### Trainig and Fine-Tuning
First of all you need to go throught the installation process from [LLAVA](https://github.com/haotian-liu/LLaVA/tree/main)


####  VQ-VAE training
Before running experiments with VQ-VAE model you need to specify config (/home/polina/test/MMDCOT/ae_training/configs/config_vq_vae_train.yaml) and set your results folder, then run ``` cd ae_training & python main.py ```


####  Fine-Tuning
The training consists of two stages: (1) feature alignment stage: connecting a frozen pretrained quntized encoder with CoT to a frozen LLM; (2) tuning stage: fine-tuning LLM, teaching the model to follow multimodal instructions based on compressed CoT sequentes.

1. On feature alignment step you need to train projector, for that you need to run this command with primary argument ```--tune_mm_mlp_adapter True``` 
```
 deepspeed llava/train/train_mem.py --deepspeed ./scripts/zero2.json --model_name_or_path vq_vae_vicuna-13b-v1.5  --version plain   --data_path ./path_to_data_json/blip_laion_cc_sbu_558k.json --image_folder ./path_to_images/images --vision_tower openai/clip-vit-large-patch14-336 --mm_projector_type mlp2x_gelu --tune_mm_mlp_adapter True --mm_vision_select_layer -2 --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir ./results/llava-v1.5-13b-pretrain --num_train_epochs 1 --per_device_train_batch_size 32 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy "no"   --save_strategy "steps" --save_steps 24000 --save_total_limit 1 --learning_rate 1e-3 --weight_decay 0. --warmup_ratio 0.03   --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True    --dataloader_num_workers 4 --lazy_preprocess True --report_to wandb

 ```
 The example of this command you can find in the [script]() 

 2. 