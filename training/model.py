import pytorch_lightning as pl
from torch import nn
import torch
from image_plotting_callback import ImageSampler
from argparse import ArgumentParser


class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):
        super().__init__()

        self.save_hyperparameters()
        self.model= diffusers.VQModel(1, 1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        h = model_ae.encode(embeddings[None, ...].float()).latents
        _, vq_loss, _ = model_ae.quantize(h)
        out = model_ae.decode(h).sample
        loss = loss_vae(out, embeddings[None, ...].float(), vq_loss)
        

        self.log_dict(loss)

        return elbo


def train():
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10')
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        dataset = CIFAR10DataModule('.')
    if args.dataset == 'imagenet':
        dataset = ImagenetDataModule('.')

    sampler = ImageSampler()

    vae = VAE()
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=20, callbacks=[sampler])
    trainer.fit(vae, dataset)


if __name__ == '__main__':
    train()