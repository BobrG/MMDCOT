import os
import sys
import wandb
import time
import torch
import pytorch_lightning as pl
from utils.visualisation import get_pseudo_normals as get_normal_map, preprocess_image

class LogPredictionsCallback(pl.Callback):
    def __init__(
        self,
        log_image_frequency
    ):
        """
        Args:
            log_image_frequency: how often to save images
        """
        self.log_image_frequency = log_image_frequency
    
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the train batch ends."""
        
        stage = "train" if trainer.datamodule.stage == 'fit' else trainer.datamodule.stage

        keep_metric_maps = False
        if trainer.global_step % trainer.log_every_n_steps == 0 and trainer.global_step % self.log_image_frequency == 0:
            keep_metric_maps = True
        if trainer.global_step % trainer.log_every_n_steps == 0:
            for metric_f in pl_module.metrics:
                # log per pixel metric
                batch_metric_map = {f'stage_{i}': metric_f(outputs[f'stage_{i}']['depth'], batch['depths'][f'stage_{i}'],
                                                           mask=batch['masks'][f'stage_{i}'], interval=batch['initial_interval'][0]) 
                                    for i in range(pl_module.stages_n)}
                
                if metric_f.get_name() != 'Signed_Deviation':
                    # log scalar metric
                    metric = {f'stage_{i}': metric_f.get_scalar(batch_metric_map[f'stage_{i}'], mask=batch['masks'][f'stage_{i}']) for i in range(pl_module.stages_n)}
                    if 'val' not in trainer.datamodule.stage: # TODO: move to some function to avoid repetitions...?
                        [self.log(f'{stage}_{metric_f.get_name()}_stage_{i}', metric[f'stage_{i}'], on_step=True, on_epoch=False) for i in range(pl_module.stages_n-1)]
                        self.log(f'{stage}_{metric_f.get_name()}', metric[f'stage_{pl_module.stages_n-1}'], on_step=True, on_epoch=False)
                        
                    [self.log(f'{stage}_avg_{metric_f.get_name()}_stage_{i}', metric[f'stage_{i}'], on_step=False, on_epoch=True) for i in range(pl_module.stages_n-1)]
                    self.log(f'{stage}_avg_{metric_f.get_name()}', metric[f'stage_{pl_module.stages_n-1}'], on_step=False, on_epoch=True)
                    
                if keep_metric_maps:
                    if metric_f.get_name().lower() == 'signed_deviation':
                        signed_errormap = batch_metric_map.copy()
                    elif metric_f.get_name().lower() == 'l1_error':
                        abs_errormap = batch_metric_map.copy()
                    
        if trainer.global_step % self.log_image_frequency == 0:
            # draw only i-th batch; TODO: take elements from batch at one place before the visualisation
            sample_i = 0
            # normalization function for depths
            norm_depth = lambda d: (d - batch['dmin'][sample_i]) / (batch['dmax'][sample_i] - batch['dmin'][sample_i])
            # get mask & normal maps
            mask = batch['masks']['stage_2'][sample_i]

            _, h, w = outputs['depth'].shape
            normalmap_gt = get_normal_map(batch['depths']['stage_2'][sample_i].view(1, 1, h, w) / batch['initial_interval'][0], scale=10)
            normalmap_est = get_normal_map(outputs['depth'][sample_i].view(1, 1, h, w) / batch['initial_interval'][0], scale=10)
            
            # if we haven't logged metrics on this step, then compute them to draw error maps 
            if not keep_metric_maps:
                for metric_f in pl_module.metrics:
                    # TODO: reduce repeated code
                    if metric_f.get_name().lower() == 'signed_deviation':
                        signed_errormap = {f'stage_{i}': metric_f(outputs[f'stage_{i}']['depth'], batch['depths'][f'stage_{i}'],
                                                                  mask=batch['masks'][f'stage_{i}'][sample_i], interval=batch['initial_interval'][0]) for i in range(pl_module.stages_n)}
                    elif metric_f.get_name().lower() == 'l1_error':
                        abs_errormap = {f'stage_{i}': metric_f(outputs[f'stage_{i}']['depth'], batch['depths'][f'stage_{i}'],
                                                mask=batch['masks'][f'stage_{i}'][sample_i], interval=batch['initial_interval'][0]) for i in range(pl_module.stages_n)}
            
            captions = ['Rerence Image', # fix typo later
                        'Source Images',
                        'Ground Truth',
                        'Estimation',
                        'Estimation (No Mask)',
                        'Signed Error Map',
                        'Abs Error Map',
                        'Normal Map GT',
                        'Normal Map Est',
                        'Confidence Est', 
                        'TSDFs']

            # TODO: log depth & metrics & error maps for all steps
            for i in range(pl_module.stages_n-1):
                for c in ['Estimation', 'Estimation (No Mask)', 'Signed Error Map', 'Abs Error Map']:
                    captions.append(c + f' Stage_{i}')
                
            # preprocessing images so that they have appropriate colormap
            preprocessed_images = map(lambda a: preprocess_image(*a),
                                      [('img', batch['images'][sample_i][0]),
                                      ('img', batch['images'][sample_i][1:]),
                                       ('depth', norm_depth(batch['depths']['stage_2'][sample_i])),
                                       ('depth', norm_depth(outputs['depth'][sample_i])*mask),
                                       ('depth', norm_depth(outputs['depth'][sample_i])),
                                       ('signed_errormap', signed_errormap['stage_2'][sample_i]),
                                       ('errormap', abs_errormap['stage_2'][sample_i]),
                                       ('normalmap', normalmap_gt[sample_i]),
                                       ('normalmap', normalmap_est[sample_i]),
                                       ('conf', outputs['photometric_confidence'][sample_i]),
                                       ('tsdf', outputs['stage_0']['tsdfs'][sample_i])]
                                      )
            preprocessed_images = list(preprocessed_images)
            
            for i in range(pl_module.stages_n - 1):
                preprocessed_images.append(preprocess_image('depth', norm_depth(outputs[f'stage_{i}']['depth'][sample_i])))
                preprocessed_images.append(preprocess_image('depth', norm_depth(outputs[f'stage_{i}']['depth'][sample_i]*batch['masks'][f'stage_{i}'][sample_i])))
                preprocessed_images.append(preprocess_image('signed_errormap', signed_errormap[f'stage_{i}'][sample_i]))
                preprocessed_images.append(preprocess_image('errormap', abs_errormap[f'stage_{i}'][sample_i]))
             
            trainer.logger.log_image(key=f'{trainer.datamodule.stage}_images',
                                          images=preprocessed_images,
                                          caption=captions)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        return self.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        return self.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
   
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        trainer.logger.log({'loaded_on_step', float(checkpoint['global_step'])})
   

# TODO: fix lock file approach
# class FileLockCallback(pl.Callback):
#     def __init__(self, lock_file_path):
#         super().__init__()
#         self.lock_file_path = lock_file_path

#     def on_save_checkpoint(self, trainer, pl_module, checkpoint):
#         # Create a lock file before saving the checkpoint
#         with open(self.lock_file_path, "w") as lock_file:
#             lock_file.write("writing")

#     def on_fit_end(self, trainer, pl_module):
#         # Remove the lock file after the checkpoint is saved
#         print('file lock delete')
#         if os.path.exists(self.lock_file_path):
#             print('exists')
#             print(self.lock_file_path)
#             os.remove(self.lock_file_path)
        
        
