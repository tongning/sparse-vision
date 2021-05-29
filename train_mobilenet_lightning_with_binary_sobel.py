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
class GetEdges(object):
    def __call__(self, pic):
        id = randint(0,1000)
        filtered = kornia.sobel(pic.unsqueeze(0)).squeeze()
        filtered = filtered > 0.25
        filtered = filtered.float()
        return filtered
    def __repr__(self):
        return self.__class__.__name__+'()'


class MobilenetModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=False)
        self.model.classifier[1] = nn.Linear(1280, 200)
        self.criterion = nn.NLLLoss()
    def forward(self, x):
        predictions = self.model(x)
        return predictions
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        #save_image(inputs[0], '/tmp/sample.png')
        inputs = kornia.sobel(inputs)        
        #save_image(inputs[0], '/tmp/sample_filt.png')
        exit(0)
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
        inputs = kornia.sobel(inputs)
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
            top_1_5_acc = self.accuracy(x['probs'], x['labels'])
            top_1_accs += (top_1_5_acc[0].item())
            top_5_accs += (top_1_5_acc[1].item())
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        self.log('avg_val_loss', avg_loss, prog_bar=True)
        self.log('avg_val_acc_1', top_1_accs / len(outputs))
        self.log('avg_val_acc_5', top_5_accs / len(outputs))

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = kornia.sobel(inputs)
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

    def accuracy(self, output: torch.Tensor, target: torch.Tensor, topk=(1,5)) -> List[torch.FloatTensor]:
        """
        Computes the accuracy over the k top predictions for the specified values of k
        In top-5 accuracy you give yourself credit for having the right answer
        if the right answer appears in your top five guesses.

        ref:
        - https://pytorch.org/docs/stable/generated/torch.topk.html
        - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
        - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
        - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
        - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

        :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
        :param target: target is the truth
        :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
        e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
        So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
        but if it were either cat or dog you'd accumulate +1 for that example.
        :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
        """
        with torch.no_grad():
            # ---- get the topk most likely labels according to your model
            # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
            maxk = max(topk)  # max number labels we will consider in the right choices for out model
            batch_size = target.size(0)

            # get top maxk indicies that correspond to the most likely probability scores
            # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
            _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
            y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

            # - get the credit for each example if the models predictions is in maxk values (main crux of code)
            # for any example, the model will get credit if it's prediction matches the ground truth
            # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
            # if the k'th top answer of the model matches the truth we get 1.
            # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
            target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
            # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
            correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
            # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

            # -- get topk accuracy
            list_topk_accs = []  # idx is topk1, topk2, ... etc
            for k in topk:
                # get tensor of which topk answer was right
                ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
                # flatten it to help compute if we got it correct for each example in batch
                flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
                # get if we got it right for any of our top k prediction for each example in batch
                tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
                # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
                topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
                list_topk_accs.append(topk_acc)
            return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

transformations = transforms.Compose([
    transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    GetEdges(),
])
data_dir = '../tiny-imagenet-200'
train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform = transformations)
val_set = datasets.ImageFolder(os.path.join(data_dir, 'val_organized'), transform = transformations)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =32, shuffle=False, num_workers=8)

module = MobilenetModule()

run_id = "mobilenetbinarysobel1"
should_resume = False
model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=5, monitor="avg_val_acc_1")

if should_resume:
    os.environ["WANDB_RESUME"] = "must"
    os.environ["WANDB_RUN_ID"] = run_id
    wandb.init(project="cs230")
    wandb_logger = WandbLogger(project='cs230', log_model=True)
    ckpt_path = '/home/anthony/Documents/cs230/wandb/run-20210502_005627-anthonyrun0/files/cs230/anthonyrun0/checkpoints/epoch=2-step=7489.ckpt'
    trainer = pl.Trainer(gpus=1, max_epochs=200, resume_from_checkpoint=ckpt_path, logger=wandb_logger, callbacks=[model_checkpoint])
    trainer.fit(module, train_loader, val_loader)
else:
    wandb.init(project="cs230", id=run_id, resume="allow")
    wandb_logger = WandbLogger(project='cs230', log_model=True)
    trainer = pl.Trainer(gpus=1, max_epochs=200, progress_bar_refresh_rate=20, logger=wandb_logger, callbacks=[model_checkpoint])
    trainer.fit(module, train_loader, val_loader)