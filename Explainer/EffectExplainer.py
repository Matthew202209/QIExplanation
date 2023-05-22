import gc

import numpy as np
import torch.nn as nn
import torch


class EffectExplainer(nn.Module):
    def __init__(
            self,
            model
    ):
        super(EffectExplainer, self).__init__()
        self.model = model

    def forward(self, feat, adj):
        with torch.no_grad():
            self.model.using_attention_explanation()
            _ = self.model(feat, adj)
        edge_weight_matrix = self.model.attention_weight
        return edge_weight_matrix


def attention_explainer_explain(model, feat, adj):
    explainer = EffectExplainer(
        model=model
    )

    masked_adj = explainer()
    del explainer
    gc.collect()
    return masked_adj



