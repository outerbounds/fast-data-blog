import lightgbm as lgb
from dataframe_utils import GroupTimeSeriesSplit, StratifiedGroupKFold, GroupTimeSeriesSplit
from collections import defaultdict
from sklearn.model_selection import GroupKFold
import joblib
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import gc

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# TODO: replace with feval_pearsonr
def feval_rmse(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'rmse', rmse(y_true, y_pred), False

# https://www.kaggle.com/c/ubiquant-market-prediction/discussion/302480
def feval_pearsonr(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'pearsonr', pearsonr(y_true, y_pred)[0], True

# https://www.kaggle.com/gogo827jz/jane-street-supervised-autoencoder-mlp?scriptVersionId=73762661
# weighted average as per Donate et al.'s formula
# https://doi.org/10.1016/j.neucom.2012.02.053
# [0.0625, 0.0625, 0.125, 0.25, 0.5] for 5 fold
def weighted_average(a):
    w = []
    n = len(a)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        w.append(1 / (2**(n + 1 - j)))
    return np.average(a, weights = w)

def run(info, args, train, features, cat_features):    

    # hyperparams from: https://www.kaggle.com/valleyzw/ubiquant-lgbm-optimization
    params = {
        'learning_rate':0.05,
        "objective": "regression",
        "metric": "rmse",
        'boosting_type': "gbdt",
        'verbosity': -1,
        'n_jobs': -1, 
        'seed': args.seed,
        'lambda_l1': 0.03627602394442367, 
        'lambda_l2': 0.43523855951142926, 
        'num_leaves': 114, 
        'feature_fraction': 0.9505625064462319, 
        'bagging_fraction': 0.9785558707339647, 
        'bagging_freq': 7, 
        'max_depth': -1, 
        'max_bin': 501, 
        'min_data_in_leaf': 374,
        'n_estimators': 1000, 
    }
    
    y = train['target']
    train['preds'] = -1000
    scores = defaultdict(list)
    features_importance= []
    
    def run_single_fold(fold, trn_ind, val_ind):
        train_dataset = lgb.Dataset(train.loc[trn_ind, features], y.loc[trn_ind], categorical_feature=cat_features)
        valid_dataset = lgb.Dataset(train.loc[val_ind, features], y.loc[val_ind], categorical_feature=cat_features)
        model = lgb.train(
            params,
            train_set = train_dataset, 
            valid_sets = [train_dataset, valid_dataset], 
            verbose_eval=100,
            early_stopping_rounds=50,
            feval = feval_pearsonr
        )
        joblib.dump(model, f'lgbm_seed{args.seed}_{fold}_{info}.pkl')
        preds = model.predict(train.loc[val_ind, features])
        train.loc[val_ind, "preds"] = preds
        scores["rmse"].append(rmse(y.loc[val_ind], preds))
        scores["pearsonr"].append(pearsonr(y.loc[val_ind], preds)[0])
        fold_importance_df= pd.DataFrame({'feature': features, 'importance': model.feature_importance(), 'fold': fold})
        features_importance.append(fold_importance_df)
        del train_dataset, valid_dataset, model
        gc.collect()

    if args.cv_method=="stratified":
        for fold in range(args.folds):
            print(f"=====================fold: {fold}=====================")
            trn_ind, val_ind = train.fold!=fold, train.fold==fold
            print(f"train length: {trn_ind.sum()}, valid length: {val_ind.sum()}")
            run_single_fold(fold, trn_ind, val_ind)
    elif args.cv_method=="time":
        tscv = TimeSeriesSplit(args.folds)
        for fold, (trn_ind, val_ind) in enumerate(tscv.split(train[features])):
            print(f"=====================fold: {fold}=====================")
            print(f"train length: {len(trn_ind)}, valid length: {len(val_ind)}")
            run_single_fold(fold, trn_ind, val_ind)
    elif args.cv_method=="group":
        # https://www.kaggle.com/lucamassaron/eda-target-analysis/notebook
        kfold = GroupKFold(args.folds)
        for fold, (trn_ind, val_ind) in enumerate(kfold.split(train[features], y, train.time_id)):
            print(f"=====================fold: {fold}=====================")
            print(f"train length: {len(trn_ind)}, valid length: {len(val_ind)}")
            run_single_fold(fold, trn_ind, val_ind)
    elif args.cv_method=="group_time":
        # https://www.kaggle.com/joelqv/grouptimeseriescv-catboost-gpu
        # kfold = GroupTimeSeriesSplit(n_splits=args.folds)
        # for fold, (trn_ind, val_ind) in enumerate(kfold.split(train[features], y, train.time_id)):
        # https://www.kaggle.com/c/ubiquant-market-prediction/discussion/304036
        kfold = GroupTimeSeriesSplit(n_folds=args.folds, holdout_size=args.holdout_size, groups=train.time_id)
        for fold, (trn_ind, val_ind) in enumerate(kfold.split(train)):
            print(f"=====================fold: {fold}=====================")
            print(f"train length: {len(trn_ind)}, valid length: {len(val_ind)}")
            run_single_fold(fold, trn_ind, val_ind)
    elif args.cv_method=="kfold":
        kfold = KFold(args.folds)
        for fold, (trn_ind, val_ind) in enumerate(kfold.split(train[features], train.investment_id)):
            print(f"=====================fold: {fold}=====================")
            print(f"train length: {len(trn_ind)}, valid length: {len(val_ind)}")
            run_single_fold(fold, trn_ind, val_ind)
    # TODO: add another fold to train with the whole training dataset with a tiny learning rate
    elif args.cv_method=="time_range":
        ranges = train.time_range.unique()
        kfold = TimeSeriesSplit(n_splits=args.folds-1)
        for fold, (trn_ind, val_ind) in enumerate(kfold.split(ranges)):
            trn_ind, val_ind = train.time_range.isin(ranges[trn_ind]), train.time_range.isin(ranges[val_ind])
            assert trn_ind.idxmin()-1 < val_ind.idxmax()
            print(f"train length: {trn_ind.sum()}, valid length: {val_ind.sum()}")
            run_single_fold(fold, trn_ind, val_ind)
        
    print(f"lgbm {info} {args.folds} folds mean rmse: {np.mean(scores['rmse'])}, mean pearsonr: {np.mean(scores['pearsonr'])}")
    if "time" in args.cv_method:
        print(f"lgbm {info} {args.folds} folds weighted mean rmse: {weighted_average(scores['rmse'])}, weighted mean pearsonr: {weighted_average(scores['pearsonr'])}")
    train.filter(regex=r"^(?!f_).*").to_csv(f"preds_{info}.csv", index=False)
    return pd.concat(features_importance, axis=0)