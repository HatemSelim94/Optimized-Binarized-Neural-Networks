import argparse
import os
import json
parser = argparse.ArgumentParser('Plots')
parser.add_argument('--exp_folder', type=str, default='eval/darts')
parser.add_argument('--exp_name', type=str)
parser.add_argument('--exp_num', type=int, default=3)
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()
def main():
    data_file = '_data.json'
    model_info_file = 'model_info.json'
    exp_path = os.path.join(args.exp_folder,args.exp_name)
    experiments = []
    for i in range(args.exp_num):
        experiments.append(exp_path+'_i')
    parameter_name = [f'lr: {i}' for i in [0.6, 0.7, 0.8]]
    data_key = 'Validatoin Mean IoU'
    with open(os.path.join(experiments[i], data_file)) as f:
        data = json.load(f)
        v_miou = data[data_key][:51]

    



if __name__ == '__main__':
    main()