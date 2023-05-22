import gc

import torch.nn as nn
import torch


class GradExplainer(nn.Module):
    def __init__(
            self,
            model
    ):
        super(GradExplainer, self).__init__()
        self.model = model

    def forward(self, feat, adj):
        self.model.zero_grad()
        self.adj.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
        ypred = self.model(feat, adj)
        loss = ypred.sum()
        loss.backward()
        edge_weight_matrix = adj.grad
        return edge_weight_matrix


def grad_explainer_explain(model, feat, adj):
    explainer = GradExplainer(
        model=model
    )

    masked_adj = explainer()
    del explainer
    gc.collect()
    a = torch.sum(masked_adj, 2)
    return torch.sum(masked_adj, 2) / masked_adj.shape[2]









