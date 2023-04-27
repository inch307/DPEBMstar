import pandas as pd
import random
import math
import cubic_solver
from sklearn import datasets

def category_bin_splits(s):
    x = len(s)
    lst = []
    for i in range(1 << x):
        left = []
        right = []
        for j in range(x):
            if (i & (1 << j)):
                left.append(s[j])
            else:
                right.append(s[j])
        # lst.append([s[j] for j in range(x) if (i & (1 << j))])
        lst.append([left, right])
    return lst[:-1]

def train_test_idx(df, train_ratio):
    # return train, test idx list (not df)
    num_data = len(df)
    slicer = int(num_data * train_ratio)
    shuffled = random.sample(range(num_data), num_data)
    # train = df.loc[shuffled[:slicer]]
    # test = df.loc[shuffled[slicer:]]

    return shuffled[:slicer], shuffled[slicer:]

def get_train_test_df(df, train_idx, test_idx):
    train = df.loc[train_idx]
    test = df.loc[test_idx]

    return train, test

def get_dataset(args):
    if args.data_path == 'syn_cls':
        data_name = 'syn_cls'
        # n_features = 40
        n_features = 20
        X, y = datasets.make_classification(n_samples=10000, n_features=n_features, n_informative=10, n_redundant=5, n_clusters_per_class=2, random_state=args.seed)
        column_name = []
        for i in range(n_features):
            column_name.append(str(i))
        X_df = pd.DataFrame(X, index=None, columns=column_name)
        y_df = pd.DataFrame(y, index=None, columns=[args.label])
        df =pd.concat([X_df, y_df], axis=1)

    elif args.data_path == 'syn_reg':
        data_name= 'syn_reg'
        # n_features = 40
        n_features = 20
        X, y = datasets.make_regression(n_samples=10000, n_features=n_features, n_informative=10, random_state=args.seed)
        column_name = []
        for i in range(n_features):
            column_name.append(str(i))
        X_df = pd.DataFrame(X, index=None, columns=column_name)
        y_df = pd.DataFrame(y, index=None, columns=[args.label])
        df =pd.concat([X_df, y_df], axis=1)
    
    else:
        data_name = args.data_path[5:].split('.')[0]
        df = pd.read_csv(args.data_path)

    return df, data_name

class CrossValidation:
    def __init__(self, df, train_idx, k_fold):
        self.df = df
        self.train_idx = train_idx
        self.k_fold = k_fold
        self.slicer = int(len(self.train_idx) / self.k_fold)
    
    def get_train_test(self, i):
        train_left = self.df.loc[self.train_idx[:self.slicer * i]]
        train_right = self.df.loc[self.train_idx[min(self.slicer * (i+1), len(self.df)):]]
        train = pd.concat([train_left, train_right])
        test = self.df.loc[self.train_idx[self.slicer * i: min(self.slicer * (i+1), len(self.df))]]

        return train, test

def write_columns(wr):
    # wr.writerow(['data', 'n_runs', 'privacy', 'eps', 'delta', 'af', 'af_prob', 'lr', 'epo', 're_train', 'rmse', 'rmse_std', 'acc', 'acc_std', 'auroc', 'auroc_std', 'remain_eps', 're_rmse', 'std', 're_acc', 'std', 're_roc', 'std'])
    wr.writerow(['data', 'n_runs', 'cv', 'seed', 'max_leaves', 'hist_ebm_ratio', 'k', 'a', 'privacy', 'eps', 'delta', 'af', 'af_prob', 'min_cf', 'lr', 'epo', 'num_sm', 'bias_mod', 'rmse', 'rmse_std', 'acc', 'acc_std', 'auroc', 'auroc_std', 'f1', 'f1_std', 'avg_trees', 'avg_time'])

def make_write_lst(args):
    lst = [args.data_path, args.n_runs, args.cv, args.seed, args.max_leaves, args.hist_ebm_ratio, args.k, args.a, args.privacy, args.eps, args.delta, args.adaptive_feature, args.af_prob, args.min_cf, args.lr, args.epochs, args.num_sm, args.bias]

    return lst

def get_hist_ebm_ratio_eps(h, a, epoch):
    # a = epoch*feature**3*h*A**2 + epoch**3*feature**3
    # b = -3*eps*epoch**2*feature**2
    # c = 3*eps**2*epoch*feature
    # d = -eps**3

    a1=epoch**2 + h*a**2
    a2=-3*epoch**2
    a3=3*epoch**2
    a4=-epoch**2

    del0 = a2**2 - 3*a1*a3
    del1 = 2*a2**3 - 9*a1*a2*a3 + 27*a1**2*a4
    C = ((del1 + math.sqrt(del1**2 - (4*del0**3))) / 2) **(1./3)
    sol = -(a2+C+del0/C) / (3*a1)

    # return epoch * feature * sol / eps
    return sol

def get_hist_ebm_ratio_mu(h, a, epoch):

    # mu_2 = mu**2
    # sol1 = (-mu_2 * epoch + a*mu_2*math.sqrt(k*epoch)) / (a**2*k*epoch-epoch**2)
    # # sol2 = (-mu_f_2 * epoch - a*mu_f_2*math.sqrt(k*epoch)) / (a**2*k*epoch-epoch**2)
    # R = sol1 * epoch
    
    # variance
    # mu_2_per_tree = (mu_2 * i) / (epoch*feature)
    # mu_2_per_hist = (mu_2 * (1-i)) / feature
    # print(f'ratio: {i} and k is {k} the variance is {1/(mu_2_per_tree) + (k*a**2)/mu_2_per_hist}')

    return ( (epoch - a*math.sqrt(epoch*h)) / (epoch - h*a**2) )
        

if __name__ == '__main__':
    print(powerset([2,3,4]))
