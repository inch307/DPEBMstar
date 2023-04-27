import argparse
import pandas as pd
import csv
import numpy as np
import ebm as ebm
import os
import sys
import pickle
from utils import *

from dputils import DPUtils
import time

parser = argparse.ArgumentParser(description='DP_EBM')
## 
parser.add_argument('-p', '--data_path', default='data/adult.csv', help='dataset to use: adult, ...')
parser.add_argument('-l', '--label', help='the name of label column')
parser.add_argument('--regression', default=False, action='store_true', help='default is binary classification')
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--train_test_split', type=int, default=0.8)

## privacy parameters
parser.add_argument('--privacy', default=False, action='store_true')
parser.add_argument('-r', '--range_label', type=float, help='the range of label for sensitivity')
parser.add_argument('-e', '--eps', type=float, default=0.1, help='epsilon privacy budget of building EBM (0 or negative for non-privacy)')
parser.add_argument('-d', '--delta', type=float, default=1e-6, help='delta for privacy')
parser.add_argument('--hist_ebm_ratio', type=float, default=0.9, help='ebm_eps = eps * hist_ebm_ratio, hist_eps = eps - ebm_eps')
parser.add_argument('--k', type=float, default=1, help='calc when hist_ebm_ratio=0')
parser.add_argument('--a', type=float, default=0.3, help='sensitivity calc when hist_ebm_ratio=0')
# parser.add_argument('--residual_eps_ratio', type=float, default=0.5, help='residual_eps = ebm_eps * ratio, hessian_eps = ebm_eps - residual_eps ;; only for privacy and classification_hessian')
parser.add_argument('--adaptive_feature', default=False, action='store_true')
parser.add_argument('--af_prob', default=0, type=float, help='prune probabilty bound')
parser.add_argument('--max_bins', default=32, type=int)
parser.add_argument('--min_cf', type=float, default=0)
parser.add_argument('--explain', default=False, action='store_true')
parser.add_argument('--bias', default=False, action='store_true')
parser.add_argument('--num_sm', type=int, default=0, help='num of sample mean')

## tree building
parser.add_argument('--lr', type=float, default=0.01, help='learning rate of EBM')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--max_leaves', type=int, default=3, help='the number of max leaves (max_split +1)')
parser.add_argument('--regularization_score', type=float, default=0, help='regularization term for similarity score')
parser.add_argument('--gamma', type=float, default=0, help='parameter for pruning (gain-gamma)')
# parser.add_argument('outer_bags', type=int, default=0, help='outer bagging')


def main():
    args = parser.parse_args()
    print(args)
    
    df, data_name = get_dataset(args)
    
    # print(df)
    file_name = sys.argv[0].split('.')[0]
    csv_name = 'experiment_'+ data_name + '_' + file_name +'.csv'
    pkl_name = data_name + '_' + file_name + '.pkl'
    if os.path.exists(csv_name):
        csv_f = open(csv_name, 'a', newline='')
        wr = csv.writer(csv_f)
    else:
        csv_f = open(csv_name, 'w', newline='')
        wr = csv.writer(csv_f)
        write_columns(wr)

    write_lst = make_write_lst(args)

    rmse_lst = []
    acc_lst = []
    auroc_lst = []
    f1_lst = []
    num_trees_lst = []
    time_lst = []
    # eps_lst = []

    if args.seed is not None:
        random.seed(args.seed) # random seed for experiment
    train_idx, test_idx = train_test_idx(df, args.train_test_split)
    
    if args.cv != 0:
        cv = CrossValidation(df, train_idx, args.cv)
    
    if args.cv == 0:
        for i in range(args.n_runs):
            df_train, df_test = get_train_test_df(df, train_idx, test_idx)
            
            if args.seed is not None:
                random.seed(args.seed + i)
                np.random.seed(args.seed + i)
            
            model = ebm.EBM(df_train, args)
            start_time = time.time()
            model.fit()
            end_time = time.time()
            time_lst.append(end_time - start_time)

            test_X = df_test.drop(columns=[args.label], axis=1)
            test_y = df_test[args.label]

            # predict
            if args.regression:
                rmse = model.predict(test_X, test_y)
                rmse_lst.append(rmse)
                
            else:
                accuracy, auroc, f1 = model.predict(test_X, test_y)
                acc_lst.append(accuracy)
                auroc_lst.append(auroc)
                f1_lst.append(f1)
            num_trees_lst.append(model.num_trees)

    else:
        for c in range(args.cv):
            for i in range(args.n_runs):
                df_train, df_test = cv.get_train_test(c)
                
                if args.seed is not None:
                    random.seed(args.seed + i)
                    np.random.seed(args.seed + i)
                model = ebm.EBM(df_train, args)
                model.fit()

                test_X = df_test.drop(columns=[args.label], axis=1)
                test_y = df_test[args.label]

                # predict
                if args.regression:
                    rmse = model.predict(test_X, test_y)
                    rmse_lst.append(rmse)
                    
                else:
                    accuracy, auroc, f1 = model.predict(test_X, test_y)
                    acc_lst.append(accuracy)
                    auroc_lst.append(auroc)
                    f1_lst.append(f1)
                num_trees_lst.append(model.num_trees)
    
    if args.regression:
        rmse = np.array(rmse_lst)
        write_lst.append(np.mean(rmse))
        write_lst.append(np.std(rmse))
        write_lst.append('')
        write_lst.append('')
        write_lst.append('')
        write_lst.append('')
        write_lst.append('')
        write_lst.append('')

    else:
        acc = np.array(acc_lst)
        auroc = np.array(auroc_lst)
        f1 = np.array(f1_lst)
        write_lst.append('')
        write_lst.append('')
        write_lst.append(np.mean(acc))
        write_lst.append(np.std(acc))
        write_lst.append(np.mean(auroc))
        write_lst.append(np.std(auroc))
        write_lst.append(np.mean(f1))
        write_lst.append(np.std(f1))
        # print(f'Test accuracy is: {acc*100} %')
    # epsnp = np.array(eps_lst)
    if args.cv == 0:
        num_trees = np.array(num_trees_lst)
        write_lst.append(np.mean(num_trees))
        avg_time = np.array(time_lst)
        write_lst.append(np.mean(avg_time))
    # write_lst.append(np.mean(epsnp))
        
    wr.writerow(write_lst)
    csv_f.close()


    if args.explain:
        model.explain()

    return

if __name__=='__main__':
    main()