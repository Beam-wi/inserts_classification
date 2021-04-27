import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchmetrics
import pytorch_lightning as pl
import yaml
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
import numpy as np

from dataset.dataset import Dataset


class LitClassifier(pl.LightningModule):

    def __init__(self, num_classes, lr, lr_milestones):

        super().__init__()
        self.save_hyperparameters()

        self.model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        self.train_acc = torchmetrics.Accuracy()        
        self.val_acc = torchmetrics.Accuracy()        


    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, targets)

        preds = torch.argmax(logits, dim=1)
        accuracy = self.train_acc(preds, targets)
        self.log('train_acc', accuracy, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self(imgs)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, targets)
        return preds.cpu().numpy(), targets.cpu().numpy()


    def validation_epoch_end(self, outputs):
        if len(outputs) == 0:
            print('no outputs at validation_epoch_end')
            return
        accuracy = self.val_acc.compute()
        self.log('val_acc', accuracy, prog_bar=True)
        self.val_acc.reset()

        preds = []
        targets = []
        for out in outputs:
        # do something with these
            preds.append(out[0])
            targets.append(out[1])
        
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        print(classification_report(targets, preds))

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD([{'params': self.model.parameters()}], momentum=0.9, lr=self.hparams['lr'], weight_decay=1e-4)
        # optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams['lr_milestones'], gamma=0.1)
        return [optimizer], [scheduler]
    



def main():

    with open('cfg.yaml', 'r') as fd:
        cfg = yaml.load(fd, Loader=yaml.FullLoader)
    print(cfg)

    train_dataset = Dataset(cfg['train_data_dir'], cfg['image_size'], cfg['class_names'], phase='train')
    print('images nums:', train_dataset.__len__())
    num_classes = len(train_dataset.cls_names)
    sample_weight = train_dataset.sample_weight
    sample_weight = torch.DoubleTensor(sample_weight)                                                                                          
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))                     
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg['batch_size'],
        # shuffle=True,
        num_workers=4,
        sampler=sampler
    )

    val_dataset = Dataset(cfg['val_data_dir'], cfg['image_size'], cfg['class_names'], phase='val')
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=4,
    )

    test_dataset = Dataset(cfg['test_data_dir'], cfg['image_size'], cfg['class_names'], phase='test')
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=4,
    )

    litmodel = LitClassifier(num_classes, lr=cfg['lr'], lr_milestones=cfg['lr_milestones'])
    pretrained_dict = torch.load('resnet18-5c106cde.pth')
    pretrained_dict.pop('fc.weight')
    pretrained_dict.pop('fc.bias')
    status = litmodel.model.load_state_dict(pretrained_dict, strict=False)
    print(status)

    trainer = pl.Trainer(gpus=1,
                         benchmark=True,
                         max_epochs=cfg['total_epochs'],
                         check_val_every_n_epoch=1
                         )

    trainer.fit(litmodel, train_loader, val_loader)                 

if __name__ == '__main__':
    main()