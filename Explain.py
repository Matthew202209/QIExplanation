import pickle

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from Config import parse_args, config
from Explainer.EffectExplainer import EffectExplainer
from Explainer.GNNExplainer import GNNExplainer
from Explainer.GradExplainer import GradExplainer
from Explainer.InputGradientExplainer import InputGradientExplainer
from PredModel.rsr_model import NRSR
from dataloader import DataLoader


class Explanation:
    def __init__(self, args, explainer_name='InputGradientExplainer'):
        self.args = args
        # device
        self.device = args.device
        #data
        self.data = None
        self.graph_data = None
        self.feature_data_path = args.feature_data_path
        self.year = args.year
        self.graph_data_path = args.graph_data_path
        # model
        self.d_feat = args.d_feat
        self.num_layers = args.num_layers
        self.model_dir = args.model_dir
        self.num_relation = None
        self.pred_model = None

        # explainer
        self.explainer_name = explainer_name
        self.explainer = None

        #
        self.explained_graph_dict = {explainer_name:[]}
        # Explanation preparation
        self.load_data()
        self.get_pred_model()
        self.get_explainer()

    def get_pred_model(self):
        with torch.no_grad():
            model = NRSR(num_relation=self.num_relation,
                         d_feat=self.d_feat,
                         num_layers=self.num_layers)

            model.to(self.device)
            model.load_state_dict(torch.load(self.model_dir + '/model.bin', map_location=self.device))

    def get_explainer(self):
        if self.explainer_name == 'GnnExplainer':
            self.explainer = GNNExplainer(self.pred_model, self.args)

        elif self.explainer_name == 'InputGradientExplainer':
            self.explainer = InputGradientExplainer(self.pred_model)

        elif self.explainer_name == 'GradExplainer':
            self.explainer = GradExplainer(self.pred_model)

        elif self.explainer_name == 'EffectExplainer':
            self.explainer = EffectExplainer(self.pred_model)

    def load_data(self):
        data_path = r"{}/{}.pkl".format(self.feature_data_path, self.year)
        f = open(data_path, 'rb')
        self.data = pickle.load(f)
        f.close()
        self.graph_data = torch.Tensor(np.load(self.graph_data_path)).to(self.device)
        self.num_relation = self.graph_data.shape[2]

    def explain(self):
        start_index = len(self.data.groupby(level=0).size())
        data_loader = DataLoader(self.data["feature"], self.data["label"],
                                 self.data['market_value'], self.data['stock_index'],
                                 pin_memory=True, start_index=start_index, device=self.device)
        for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
            feature, label, market_value, stock_index, index = data_loader.get(slc)
            expl_graph = self.explainer(self.pred_model, feature,
                                        self.graph_data[stock_index][:, stock_index], args)
            EG = nx.from_numpy_array(expl_graph.detach().numpy())
            self.explained_graph_dict[self.explainer_name] += [EG]

    def save_explanation(self):
        pass

    def evaluate(self):
        pass


if __name__ == '__main__':
    args = parse_args(config.NRSR_dict)
    ex = Explanation(args)