import torch
from PIL import Image
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import wandb
import os
from typing import List
from pytorch_lightning.callbacks import ModelCheckpoint
import kornia
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from random import randint
import argparse
from util import accuracy

class GetBinaryEdges(object):
    def __call__(self, pic):
        id = randint(0,1000)
        filtered = kornia.sobel(pic.unsqueeze(0)).squeeze()
        filtered = filtered > 0.25
        filtered = filtered.float()
        return filtered
    def __repr__(self):
        return self.__class__.__name__+'()'

class GetMaskedEdges(object):
    def __call__(self, pic):
        edges = kornia.sobel(pic.unsqueeze(0)).squeeze()
        mask = edges > 0.25
        filtered = edges * mask.int().float()
        return filtered
    def __repr__(self):
        return self.__class__.__name__+'()'

class GetEdgesWithColor(object):
    def __call__(self, pic):
        edges = kornia.sobel(pic.unsqueeze(0)).squeeze()
        mask = edges > 0.25
        filtered = pic * mask.int().float()
        return filtered
    def __repr__(self):
        return self.__class__.__name__+'()'



class TinyimagenetModule(pl.LightningModule):
    def __init__(self, model_arch, debug=False):
        super().__init__()
        self.debug = debug
        if model_arch == 'mobilenet':
            self.model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=False)
            self.model.classifier[1] = nn.Linear(1280, 200)
        elif model_arch == 'resnet':
            self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
            self.model.fc = nn.Linear(512, 200)
        else:
            print(f"Unrecognized model architecture {model_arch}")
            exit(1)

        self.criterion = nn.NLLLoss()
    def forward(self, x):
        predictions = self.model(x)
        return predictions
    def training_step(self, batch, batch_idx):
        inputs, labels = batch  
        if self.debug:
            save_image(inputs[0], '/tmp/sample_filt.png')
            #exit(0)
        output = nn.functional.log_softmax(self.model.forward(inputs))
        loss = self.criterion(output, labels)
        probs = torch.exp(output)
        return {'loss': loss, 'probs': probs, 'labels': labels}

    def training_epoch_end(self, outputs):
        total_examples = 0
        total_correct = 0
        for x in outputs:
            probs = x['probs']
            labels = x['labels']
            top_p, top_class = probs.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            total_examples += equals.shape[0]
            total_correct += torch.sum(equals.type(torch.FloatTensor)).item()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss, prog_bar=True)
        self.log('avg_train_acc', total_correct / total_examples)


    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        output = nn.functional.log_softmax(self.model.forward(inputs))
        loss = self.criterion(output, labels)

        probs = torch.exp(output)
        # top_p, top_class = probs.topk(1, dim=1)
        # equals = top_class == labels.view(*top_class.shape)
        # accuracy = torch.mean(equals.type(torch.FloatTensor)).item()

        #self.log('val_loss', loss)
        #self.log('val_acc', accuracy)
        return {'loss': loss, 'probs': probs, 'labels': labels}

    def validation_epoch_end(self, outputs):
        total_examples = 0
        total_correct = 0
        top_1_accs = 0
        top_5_accs = 0

        for x in outputs:
            top_1_5_acc = accuracy(x['probs'], x['labels'])
            top_1_accs += (top_1_5_acc[0].item())
            top_5_accs += (top_1_5_acc[1].item())
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        self.log('avg_val_loss', avg_loss, prog_bar=True)
        self.log('avg_val_acc_1', top_1_accs / len(outputs))
        self.log('avg_val_acc_5', top_5_accs / len(outputs))

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        output = nn.functional.log_softmax(self.model.forward(inputs))
        loss = self.criterion(output, labels)

        top_p, top_class = torch.exp(output).topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()

        self.log('test_loss', loss)
        self.log('test_acc', accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer




arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--modelarch', action='store', type=str, required=True)
arg_parser.add_argument('--runid', action='store', type=str, required=True)
arg_parser.add_argument('--datadir', action='store', type=str, required=True)
arg_parser.add_argument('--transform', action='store', type=str, required=True)
arg_parser.add_argument('--resume', type=bool, action='store', default=False)
arg_parser.add_argument('--debug', type=bool, action='store', default=False)
arg_parser.add_argument('--numworkers', type=int, action='store', required=True)


args = arg_parser.parse_args()

if args.transform == 'binedges':
    transformations = transforms.Compose([
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        GetBinaryEdges(),
    ])
elif args.transform == 'maskededges':
    transformations = transforms.Compose([
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        GetMaskedEdges(),
    ])
elif args.transform == 'edgeswithcolor':
    transformations = transforms.Compose([
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        GetEdgesWithColor(),
    ])
else:
    print("Using null transformation")
    transformations = transforms.Compose([
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

data_dir = args.datadir
train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform = transformations)
val_set = datasets.ImageFolder(os.path.join(data_dir, 'val_organized'), transform = transformations)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=args.numworkers)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =32, shuffle=False, num_workers=args.numworkers)

module = TinyimagenetModule(model_arch=args.modelarch, debug=args.debug)

run_id = args.runid
should_resume = args.resume
model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=5, monitor="avg_val_acc_1")

if should_resume:
    os.environ["WANDB_RESUME"] = "must"
    os.environ["WANDB_RUN_ID"] = run_id
    if not args.debug:
        wandb.init(project="cs230")
        wandb_logger = WandbLogger(project='cs230', log_model=True)
    ckpt_path = '/home/anthony/Documents/cs230/wandb/run-20210502_005627-anthonyrun0/files/cs230/anthonyrun0/checkpoints/epoch=2-step=7489.ckpt'
    if not args.debug:
        trainer = pl.Trainer(gpus=1, max_epochs=200, resume_from_checkpoint=ckpt_path, logger=wandb_logger, callbacks=[model_checkpoint])
    else:
        trainer = pl.Trainer(gpus=1, max_epochs=200, resume_from_checkpoint=ckpt_path, callbacks=[model_checkpoint])
    trainer.fit(module, train_loader, val_loader)
else:
    if not args.debug:
        wandb.init(project="cs230", id=run_id, resume="allow")
        wandb_logger = WandbLogger(project='cs230', log_model=True)
        trainer = pl.Trainer(gpus=1, max_epochs=200, progress_bar_refresh_rate=20, logger=wandb_logger, callbacks=[model_checkpoint])
    else:
        trainer = pl.Trainer(gpus=1, max_epochs=200, progress_bar_refresh_rate=20, callbacks=[model_checkpoint])
    trainer.fit(module, train_loader, val_loader)