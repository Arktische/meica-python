from accelerate.utils import ProjectConfiguration
import torch.nn.functional as F
from meica import Trainer
import argparse
import torch


class MyDataset(torch.utils.data.TensorDataset):
    def __init__(self, num):
        super().__init__(torch.randn(num, 2), torch.zeros(num, 1))


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    parser.add_argument(
        "--project_dir",
        "-p",
        type=str,
        default=None,
        help="Project directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--resume_dir",
        "-r",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    args = parser.parse_args()
    trainer = MyTrainer(
        project_dir=args.project_dir,
        project_config=ProjectConfiguration(
            logging_dir="logs", total_limit=5
        ),
    )
    trainer.configure(*args.config)
    trainer.fit(resume_dir=args.resume_dir)