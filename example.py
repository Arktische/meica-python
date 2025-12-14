import torch.nn.functional as F
from meica import Trainer
import argparse
import torch


class MyDataset(torch.utils.data.TensorDataset):
    def __init__(self, num):
        super().__init__(torch.randn(num, 2), torch.zeros(num, 1))


class MyTrainer(Trainer):
    def train_step(self, batch, step):
        x, y = batch
        out = self.module(x)
        loss = F.mse_loss(out, y)
        return loss
    def val_step(self, batch, step):
        x, y = batch
        out = self.module(x)
        loss = F.mse_loss(out, y)
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", nargs="+", type=str, required=True, help="Path to YAML config"
    )
    args = parser.parse_args()
    trainer = MyTrainer()
    trainer.configure(*args.config)
    trainer.fit()
