import torch
from torch.nn import functional as F
import torchvision
import pytorch_lightning as pl

import yaml


class LitClassifier(pl.LightningModule):

    def __init__(self, cfg_file):

        super().__init__()
        self.save_hyperparameters()

        with open(cfg_file, 'r') as fd:
            cfg = yaml.load(fd, Loader=yaml.FullLoader)
        print(cfg)

        torchvision.datasets.ImageFolder(cfg['train_data_dir'])

        self.model = torchvision.models.resnet18(pretrained=True, num_classes=)



    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def cli_main():
    cli = LightningCLI(LitClassifier, MNISTDataModule, seed_everything_default=1234)
    result = cli.trainer.test(cli.model, datamodule=cli.datamodule)
    print(result)


if __name__ == '__main__':
    cli_lightning_logo()
    cli_main()