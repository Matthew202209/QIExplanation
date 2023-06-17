import math
import pickle

import pandas as pd
import scipy.stats as stats
import networkx as nx
import numpy as np
import torch
import random
from tqdm import tqdm
from Config import parse_args, config
from Explainer.EffectExplainer import EffectExplainer
from Explainer.GNNExplainer import GNNExplainer
from Explainer.GradExplainer import GradExplainer
from Explainer.InputGradientExplainer import InputGradientExplainer
from PredModel.rsr_model import NRSR
from dataloader import DataLoader


class Explanation:
    def __init__(self, args, year, explainer_name='InputGradientExplainer'):
        self.args = args
        # device
        self.device = args.device
        # data
        self.data = None
        self.graph_data = None
        self.data_loader = None
        self.feature_data_path = args.feature_data_path
        self.year = year
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
        self.EG = None

        #
        self.explained_graph_list = []
        self.explained_graph_dict = {explainer_name: []}
        self.evaluation_results = []
        self.eval_results_df = None
        # Explanation preparation
        self.load_data()
        self.get_pred_model()
        self.get_explainer()
        self.get_data_loader()

    def get_pred_model(self):
        with torch.no_grad():
            self.pred_model = NRSR(num_relation=self.num_relation,
                                   d_feat=self.d_feat,
                                   num_layers=self.num_layers)

            self.pred_model.to(self.device)
            self.pred_model.load_state_dict(torch.load(self.model_dir + '/model.bin', map_location=self.device))

    def get_explainer(self):
        if self.explainer_name == 'GnnExplainer':
            self.explainer = GNNExplainer(self.pred_model, self.args)

        elif self.explainer_name == 'InputGradientExplainer':
            self.explainer = InputGradientExplainer(self.pred_model)

        elif self.explainer_name == 'GradExplainer':
            self.explainer = GradExplainer(self.pred_model)

        elif self.explainer_name == 'EffectExplainer':
            self.explainer = EffectExplainer(self.pred_model)

        elif self.explainer_name == 'random':
            pass

    def load_data(self):
        data_path = r"{}/{}.pkl".format(self.feature_data_path, self.year)
        f = open(data_path, 'rb')
        self.data = pickle.load(f)
        f.close()
        self.graph_data = torch.Tensor(np.load(self.graph_data_path)).to(self.device)
        self.num_relation = self.graph_data.shape[2]

    def get_data_loader(self):
        start_index = len(self.data.groupby(level=0).size())
        self.data_loader = DataLoader(self.data["feature"], self.data["label"],
                                      self.data['market_value'], self.data['stock_index'],
                                      pin_memory=True, start_index=start_index, device=self.device)

    def explain(self):
        start_index = len(self.data.groupby(level=0).size())
        data_loader = DataLoader(self.data["feature"], self.data["label"],
                                 self.data['market_value'], self.data['stock_index'],
                                 pin_memory=True, start_index=start_index, device=self.device)
        for i, slc in tqdm(self.data_loader.iter_daily(), total=self.data_loader.daily_length):
            feature, label, market_value, stock_index, index = data_loader.get(slc)
            graph = self.graph_data[stock_index][:, stock_index]
            original_pred = self.pred_model(feature, graph)
            if self.explainer_name == 'random':
                ran_k_graph, ran_k_comp_graph = self.random_edges_extraction(graph)
                fidelity_L1, causality_L1 = self.cal_random_selection_metrics(original_pred, label,
                                                                              feature, ran_k_graph,
                                                                              ran_k_comp_graph)
                self.evaluation_results += [[fidelity_L1, causality_L1]]

            else:
                expl_graph = self.explainer.run_explain(feature, graph)
                self.EG = nx.from_numpy_array(expl_graph.detach().numpy())
                self.explained_graph_dict[self.explainer_name] += [self.EG]
                fidelity_L1, causality_L1 = self.evaluate(original_pred, graph, feature, label)
                self.evaluation_results += [[fidelity_L1, causality_L1]]

    def save_explanation(self):
        file = r'{}/{}-{}.pkl'.format(self.args.expl_results_dir, self.explainer_name, self.year)
        f = open(file, 'wb')
        pickle.dump(self.explained_graph_dict[self.explainer_name], f)
        f.close()
        print('Save Explanation {}'.format(file))

    def save_evaluation(self):
        self.eval_results_df = pd.DataFrame(self.evaluation_results, columns=['fidelity_L1', 'causality_L1'])
        self.eval_results_df.to_csv(r'{}/{}_{}.csv'.format(r'./result',
                                                           self.explainer_name, self.year), index=False)

    def get_eval_results_df(self):
        self.eval_results_df = pd.DataFrame(self.evaluation_results, columns=['fidelity_L1', 'causality_L1'])


    def evaluate(self, original_pred, graph, feature, label):
        top_k_graph, top_k_comp_graph = self.edges_extraction(graph)
        pred_top_k = self.pred_model(feature, top_k_graph)
        pred_top_k_comp = self.pred_model(feature, top_k_comp_graph)

        fidelity_L1 = Explanation.cal_fidelity(pred_top_k, pred_top_k_comp, original_pred)
        causality_L1 = Explanation.cal_causality(pred_top_k_comp, original_pred, label)
        return fidelity_L1, causality_L1

    def edges_extraction(self, origin_graph):
        sorted_edges = sorted(self.EG.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        top = range(0, math.ceil(int(len(sorted_edges)) * self.args.top_k))
        top_k_edges = [sorted_edges[x] for x in top]
        top_k_comp_edges = [sorted_edges[x] for x in range(0, len(sorted_edges)) if x not in top]

        top_k_graph = Explanation.edges_2_graph(top_k_comp_edges, origin_graph.clone().detach())
        top_k_comp_graph = Explanation.edges_2_graph(top_k_edges, origin_graph.clone().detach())
        return top_k_graph, top_k_comp_graph

    def cal_random_selection_metrics(self, original_pred, label, feature, random_k_complement, random_k_graph):
        pred_top_k_comp = self.pred_model(feature, random_k_complement)
        pred_top_k = self.pred_model(feature, random_k_graph)
        fidelity_l1 = Explanation.cal_fidelity(pred_top_k_comp, pred_top_k, original_pred)
        causality_L1 = Explanation.cal_causality(pred_top_k_comp, original_pred, label)

        return float(fidelity_l1), float(causality_L1)

    def random_edges_extraction(self, graph):
        new_graph = torch.zeros([graph.shape[0], graph.shape[1]], dtype=torch.int)
        mask = torch.sum(graph, 2)  # mask that could have relation value
        index = torch.t((mask == 1).nonzero())
        new_graph[index[0], index[1]] = 1
        G_edge = nx.from_numpy_array(new_graph.detach().numpy())
        G_edge = list(G_edge.edges)
        edge_num = len(G_edge)
        ran = random.sample(range(0, edge_num), math.ceil(edge_num * self.args.top_k))
        ran_k_edges = [G_edge[x] for x in ran]
        ran_k_comp_edges = [G_edge[x] for x in range(0, edge_num) if x not in ran]

        ran_k_graph = Explanation.edges_2_graph(ran_k_comp_edges, graph.clone().detach())
        ran_k_comp_graph = Explanation.edges_2_graph(ran_k_edges, graph.clone().detach())
        return ran_k_graph, ran_k_comp_graph

    def cal_mean_evaluation(self):
        mean_fidelity_L1 = self.eval_results_df['fidelity_L1'].mean()
        mean_causality_L1 = self.eval_results_df['causality_L1'].mean()
        return mean_fidelity_L1, mean_causality_L1

    @staticmethod
    def cal_fidelity(top_k_comp_pred, top_k_pred, original_pred):
        # L1
        loss_func = torch.nn.L1Loss(reduction='mean')
        top_k_l1 = loss_func(top_k_pred, original_pred)
        top_k_comp_l1 = loss_func(top_k_comp_pred, original_pred)
        fidelity_l1 = 1 - top_k_l1 / top_k_comp_l1
        return float(fidelity_l1)

    @staticmethod
    def cal_causality(top_k_comp_pred, original_pred, label):
        # L1
        loss_func = torch.nn.L1Loss(reduction='mean')
        tok_k_comp_l1 = loss_func(top_k_comp_pred, label)
        original_l1 = loss_func(original_pred, label)
        causality_l1 = 1 - original_l1 / tok_k_comp_l1

        return float(causality_l1)

    @staticmethod
    def edges_2_graph(comp_edges, origin_graph):
        for edge in comp_edges:
            origin_graph[edge[0], edge[1]] = 0
        return origin_graph

def run_explain():
    args = parse_args(config.NRSR_dict)
    for year in ['2020', '2021', '2022']:
        args.year = year
        df_mean_expl_result_dict = []
        for explainer_name in ['GnnExplainer', 'InputGradientExplainer(Our)', 'GradExplainer', 'EffectExplainer']:
            Explainer = Explanation(args, year, explainer_name=explainer_name)
            Explainer.explain()
            Explainer.get_eval_results_df()
            mean_fidelity_L1, mean_causality_L1 = Explainer.cal_mean_evaluation()
            df_mean_expl_result_dict.append([mean_fidelity_L1, mean_causality_L1])

        df_mean_expl_result_dict = pd.DataFrame(df_mean_expl_result_dict, columns=['fidelity_L1',
                                                                                   'causality_L1'])
        df_mean_expl_result_dict.index = ['GnnExplainer', 'InputGradientExplainer(Our)',
                                          'GradExplainer', 'EffectExplainer']

        df_mean_expl_result_dict.to_csv(r'{}/{}_{}.csv'.format('./result',
                                                               'mean_evaluation',
                                                               year), index=True)


