from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from adamp import AdamP
from pytorch_lightning import LightningModule
from torchmetrics import F1, Accuracy, Recall, Specificity

from src.models.generator import Generator


class GeneratorModule(LightningModule):
    def __init__(
        self,
        lambda_cd=1,
        dim_neck=16,
        dim_emb=256,
        dim_pre=512,
        freq=16,
        lr=0.0001,
    ) -> None:
        super().__init__()
        self.lambda_cd = lambda_cd
        self.G = Generator(dim_neck, dim_emb, dim_pre, freq)
        self.lr = lr
        
    def forward(self, spmel, sp_emb) -> Any:
        x_identic, x_identic_psnt, code_real = self.G(spmel, sp_emb, sp_emb)
        code_reconst = self.G(x_identic_psnt, sp_emb, None)
        return x_identic, x_identic_psnt, code_real, code_reconst
    
    def training_step(self, batch, batch_idx):
        spmel, sp_emb, label = batch
        x_identic, x_identic_psnt, code_real, code_reconst = self(spmel, sp_emb)
        g_loss_id = F.mse_loss(spmel, x_identic)
        g_loss_id_psnt = F.mse_loss(spmel, x_identic_psnt)
        g_loss_cd = F.l1_loss(code_real, code_reconst)
        g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
        self.log_dict(
            dictionary={
                "train/g_loss": g_loss,
                "train/g_loss_id": g_loss_id,
                "train/g_loss_id_psnt": g_loss_id_psnt,
                "train/g_loss_cd": g_loss_cd,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return g_loss

    def validation_step(self, batch, batch_idx):
        spmel, sp_emb, label = batch
        x_identic, x_identic_psnt, code_real, code_reconst = self(spmel, sp_emb)
        g_loss_id = F.mse_loss(spmel, x_identic)
        g_loss_id_psnt = F.mse_loss(spmel, x_identic_psnt)
        g_loss_cd = F.l1_loss(code_real, code_reconst)
        g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
        self.log_dict(
            dictionary={
                "val/g_loss": g_loss,
                "val/g_loss_id": g_loss_id,
                "val/g_loss_id_psnt": g_loss_id_psnt,
                "val/g_loss_cd": g_loss_cd,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return g_loss

    def test_step(self, batch, batch_idx):
        spmel, sp_emb, label = batch
        x_identic, x_identic_psnt, code_real, code_reconst = self(spmel, sp_emb)
        g_loss_id = F.mse_loss(spmel, x_identic)
        g_loss_id_psnt = F.mse_loss(spmel, x_identic_psnt)
        g_loss_cd = F.l1_loss(code_real, code_reconst)
        g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
        self.log_dict(
            dictionary={
                "test/g_loss": g_loss,
                "test/g_loss_id": g_loss_id,
                "test/g_loss_id_psnt": g_loss_id_psnt,
                "test/g_loss_cd": g_loss_cd,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return g_loss

    def configure_optimizers(self):
        return AdamP(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    def get_preds_and_labels(self, dataloader):
        pass
        preds, labels = [], []
        self.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                logit, _, _ = self(batch)
            true_labels = batch[1].to(dtype=torch.long)
            preds.append(torch.sigmoid(logit).cpu().detach().numpy())
            labels.append(true_labels.cpu().detach().numpy())
        return (preds, labels)
