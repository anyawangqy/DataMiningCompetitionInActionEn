import torch
import torch.nn as nn

from mmaction.models.builder import LOSSES


@LOSSES.register_module()
class VideoTextContrastLoss(nn.Module):
    def __init__(self, temperature=2, **kwargs):
        super().__init__(**kwargs)
        self.T = temperature
    def forward(self, sim_matrix):
        exp_sim = torch.exp(sim_matrix/self.T)
        row_sum = torch.sum(exp_sim, dim=0)
        col_sum = torch.sum(exp_sim, dim=1)
        diag = torch.diag(exp_sim)
        loss = (-torch.log(diag/row_sum) - torch.log(diag/col_sum)) / 2
        return loss.mean()