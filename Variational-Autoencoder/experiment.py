import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

from torchvision.transforms import ToPILImage

from inception_score import inception_score as IS
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        
        self.IS_handler = InceptionScore()
        self.FID_handler = FrechetInceptionDistance(feature=64)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
    
        # compute the FID score
        # generate two slightly overlapping image intensity distributions
#         imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
#         imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
        
#         imgs_dist1 = imgs_dist1.to(self.curr_device)
#         imgs_dist2 = imgs_dist2.to(self.curr_device)
        
#         self.FID_handler.update(imgs_dist1, real=True)
#         self.FID_handler.update(imgs_dist2, real=False)
#         print("FID Score:", self.FID_handler.compute())
        
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            
            sample_imgs = samples.cpu().data
            
            max_val = torch.max(sample_imgs)
            min_val = torch.min(sample_imgs)
            sample_imgs_uint8 = (sample_imgs - min_val) / (max_val - min_val)
            sample_imgs_uint8 = torch.tensor(sample_imgs_uint8*255, dtype=torch.uint8)
            
            max_val = torch.max(test_input)
            min_val = torch.min(test_input)
            test_input_uint8 = (test_input - min_val) / (max_val - min_val)
            test_input_uint8 = torch.tensor(test_input_uint8*255, dtype=torch.uint8)
            
            imgs_dist1 = test_input_uint8.to(self.curr_device)
            imgs_dist2 = sample_imgs_uint8.to(self.curr_device)
            
            self.FID_handler.update(imgs_dist1, real=True)
            self.FID_handler.update(imgs_dist2, real=False)
            print("FID Score from Torchmetrics:", self.FID_handler.compute())

            self.IS_handler.update(imgs_dist2)
            print("Inception Score from TorchMetrics:", self.IS_handler.compute())
            print("Inception Scores from Github:", IS(sample_imgs, resize=True))
            
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
