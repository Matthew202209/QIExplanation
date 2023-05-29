import argparse


class config:
    NRSR_dict = {
        'model_name': 'NRSR',
        'model_dir': './CheckPoints/csi300_NRSR_3',
        'feature_data_path': './Data',
        'year': '2020',
        'graph_data_path': './data/csi300_multi_stock2stock_all.npy',
        'd_feat': 6,
        'num_layers': 2,
        'device': 'cpu'
    }


def parse_args(param_dict):
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_name', default=param_dict['model_name'])
    parser.add_argument('--model_dir', default=param_dict['model_dir'])
    # feature
    parser.add_argument('--feature_data_path', default=param_dict['feature_data_path'])
    parser.add_argument('--year', default='2020')
    parser.add_argument('--graph_data_path', default=param_dict['graph_data_path'])
    parser.add_argument('--d_feat', type=int, default=param_dict['d_feat'])
    parser.add_argument('--num_layers', type=int, default=param_dict['num_layers'])
    # explanation

    parser.add_argument('--expl_results_dir', type=str, default='./ExplanationResults')
    parser.add_argument('--save_name', type=str, default='pred_loss2021')
    parser.add_argument('--init_strategy', type=str, default='normal')
    parser.add_argument('--mask_act', type=str, default='sigmoid')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--size_lamda', type=float, default=0.000001)
    parser.add_argument('--density_lamda', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=0.2)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    return args