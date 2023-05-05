# See the original source of modeling and visualization code here - https://www.kaggle.com/code/valleyzw/ubiquant-lgbm-baseline
# The Ubiquant Kaggle competition produced a lot of good resources:
    # https://www.kaggle.com/datasets/robikscube/ubiquant-parquet/code?datasetId=1875082&sortBy=voteCount

from metaflow import FlowSpec, step, batch, current, card, IncludeFile, Parameter
from metaflow.cards import Image, Markdown, Table
from custom_decorators import pip
from table_loader import load_table

# environment
PANDAS_VERSION = '2.0.1'
# POLARS_VERSION = '0.17.11'
PYTHON_VERSION = '3.10.10'
PYARROW_VERSION = '12.0.0'
LIGHTGBM_VERSION = '3.3.5'
MATPLOTLIB_VERSION = '3.7.1'
SEABORN_VERSION = '0.12.2'
SCIPY_VERSION = '1.10.1'

# data and compute 
DEFAULT_URL = "s3://outerbounds-datasets/ubiquant/investment_ids"
RESOURCES = dict(memory=48000, cpu=16, use_tmpfs=True, tmpfs_size=24000)

def _set_plot_config():

    import seaborn as sns

    # data viz config
    YELLOW = '#FFBC00'
    GREEN = '#37795D'
    PURPLE = '#5460C0'
    BACKGROUND = '#F4EBE6'
    colors = [GREEN, PURPLE, YELLOW]
    custom_params = {
        'axes.spines.right': False, 'axes.spines.top': False,
        'axes.facecolor':BACKGROUND, 'figure.facecolor': BACKGROUND, 
        'figure.figsize':(8, 8)
    }
    sns_palette = sns.color_palette(colors, len(colors))
    sns.set_theme(style='ticks', rc=custom_params)

    return tuple(colors)

def target_prediction_viz(train, calendar_df):

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    from scipy.signal import find_peaks

    GREEN, PURPLE, YELLOW = _set_plot_config()

    # leave only national holidays
    calendar_df = calendar_df.loc[(calendar_df.type.isin(["National holiday", "Common local holiday"]))]

    # fill with everyday from 2014 to 2022
    calendar_df = (
        pd.DataFrame({"date": pd.date_range(start="2014-01-01", end="2022-01-01")}).merge(calendar_df, on="date", how="left")
        .assign(weekday=lambda x: x.date.dt.day_name(), year=lambda x: x.date.dt.year)
    )

    # remove weekends and national holidays and align with time_id
    calendar_df = (
        calendar_df.loc[(~calendar_df.weekday.isin(["Sunday", "Saturday"]))&(calendar_df.name.isna())]
        .reset_index(drop=True)
        .head(train.time_id.max()+1)
        .dropna(axis=1)
    )

    full_time_id_range = range(train.time_id.min(), train.time_id.max()+1)
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1,1,6]}, figsize=(10,10), sharex=True, dpi=128)
    _df = (
        train[['time_id', 'investment_id']]
        .groupby("time_id")
        .count()
        .reindex(full_time_id_range)
        .set_index(calendar_df.date)
    )
    peeks, _ = find_peaks(-_df.values.squeeze(), threshold=200)
    _df.plot(ax=ax0, color=PURPLE)
    ax0.set_xticks(ticks=_df.iloc[peeks].index.values, minor=True)
    ax0.set_ylabel("count")
    ax0.legend(loc='upper left')

    (
        train[['time_id', 'target']]
        .groupby("time_id")
        .mean()
        .reindex(full_time_id_range)
        .set_index(calendar_df.date)
        .plot(ax=ax1, color=PURPLE)
    )
    ax1.axvspan(*mdates.date2num(_df.loc[(_df.index>"2015-06")&(_df.index<"2016-03")].index[[0,-1]]), fill=True, alpha=0.9, color=YELLOW)
    ax1.set_ylabel("mean")
    ax1.legend(loc='upper left')

    _df = (
        train[['investment_id', 'time_id', "target"]]
        .pivot_table(index="time_id", columns="investment_id", values="target", aggfunc="count")
        .reindex(full_time_id_range)
        .set_index(calendar_df.date)
    )
    ax2.imshow(_df.T, cmap='winter', interpolation='nearest', aspect="auto", origin="lower", alpha=0.6, extent=[*mdates.date2num([calendar_df.date.min(), calendar_df.date.max()]), train.investment_id.min(), train.investment_id.max()])
    ax2.set_xlabel("date")
    ax2.set_ylabel("investment_id")
    ax2.xaxis_date()
    plt.tight_layout()

    return fig

def outlier_viz(train, args):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    GREEN, PURPLE, YELLOW = _set_plot_config()

    fig1, ax1 = plt.subplots(1,1, figsize=(8, 8))
    fig2, ax2 = plt.subplots(1,1, figsize=(8, 8))

    df = train[["investment_id", "target"]].groupby("investment_id").target.mean()
    upper_bound, lower_bound = df.quantile([1-args.outlier_threshold, args.outlier_threshold])
    df.plot(figsize=(16, 8), color=PURPLE, ax=ax1)
    ax1.axhspan(lower_bound, upper_bound, fill=False, linestyle="--", color="k")
    plt.show()

    outlier_investments = df.loc[(df>upper_bound)|(df<lower_bound)|(df==0)].index
    _ = pd.pivot(
        train.loc[train.investment_id.isin(outlier_investments), ["investment_id", "time_id", "target"]],
        index='time_id', columns='investment_id', values='target'
    ).plot(figsize=(16,12), subplots=True, sharex=True, ax=ax2)

    return fig1, fig2, outlier_investments

def remove_outliers(train, outlier_investments, args):

    import gc
    import numpy as np

    outlier_list = []
    outlier_col = []

    for col in (f"f_{i}" for i in range(300)):
        _mean, _std = train[col].mean(), train[col].std()
        
        temp_df = train.loc[(train[col] > _mean + _std * 70) | (train[col] < _mean - _std * 70)]
        temp2_df = train.loc[(train[col] > _mean + _std * 35) | (train[col] < _mean - _std * 35)]
        if len(temp_df) >0 : 
            outliers = temp_df.index.to_list()
            outlier_list.extend(outliers)
            outlier_col.append(col)
            print(col, len(temp_df))
        elif len(temp2_df)>0 and len(temp2_df) <6 :
            outliers = temp2_df.index.to_list()
            outlier_list.extend(outliers)
            outlier_col.append(col)
            print(col, len(temp2_df))

    outlier_list = list(set(outlier_list))
    train.drop(train.index[outlier_list], inplace = True)
    print(len(outlier_col), len(outlier_list), train.shape)

    if args.min_time_id is not None:
        train = train.query("time_id>=@args.min_time_id").reset_index(drop=True)
        gc.collect()
        
    train = train.loc[~train.investment_id.isin(outlier_investments)].reset_index(drop=True)

    time_id_df = (
        train[["investment_id", "time_id"]]
        .groupby("investment_id")
        .agg(["min", "max", "count", np.ptp])
        .assign(
            time_span=lambda x: x.time_id.ptp,
            time_count=lambda x: x.time_id["count"]
        )
        .drop(columns="ptp", level=1)
        .reset_index()
    )
    train = train.merge(time_id_df.drop(columns="time_id", level=0).droplevel(level=1, axis=1), on="investment_id", how='left')

    max_time_span=time_id_df.time_id["max"].max()
    outlier_investments = time_id_df.loc[time_id_df.time_id["count"]<32, "investment_id"].to_list()
    del time_id_df
    gc.collect()

    return train

def filter_features(train):

    import gc
    import numpy as np
    from dataframe_utils import reduce_mem_usage
    
    cat_features = []
    num_features = list(train.filter(like="f_").columns)
    features = num_features + cat_features

    combination_features = ["f_231-f_250", "f_118-f_280", "f_155-f_297", "f_25-f_237", "f_179-f_265", "f_119-f_270", "f_71-f_197", "f_21-f_65"]
    for f in combination_features:
        f1, f2 = f.split("-")
        train[f] = train[f1] + train[f2]
    features += combination_features

    to_drop = ["f_148", "f_72", "f_49", "f_205", "f_228", "f_97", "f_262", "f_258"]
    features = list(sorted(set(features).difference(set(to_drop))))

    # train = reduce_mem_usage(train.drop(columns="time_span"))
    # train[["investment_id", "time_id"]] = train[["investment_id", "time_id"]].astype(np.int16)

    train = train.drop(columns=["row_id"]+to_drop)
    gc.collect()

    return features, cat_features, train


def missing_times_viz(train):

    import pandas as pd
    import matplotlib.pyplot as plt
    GREEN, _, _ = _set_plot_config()

    fig, ax = plt.subplots(1,1, figsize=(8, 8))

    _=pd.pivot(
        train.loc[train.investment_id.isin([1,17,19,3011,3151]), ["investment_id", "time_id", "target"]],
        index='time_id', columns='investment_id', values='target'
    ).plot(figsize=(16,12), subplots=True, sharex=True, color=GREEN, ax=ax)

    return fig


def interesting_times_viz(train):

    import matplotlib.pyplot as plt
    GREEN, PURPLE, YELLOW = _set_plot_config()

    fig, ax = plt.subplots(1,1, figsize=(16, 8))
    _features = ["f_74", "f_153", "f_183", "f_145"]
    df=train[["time_id", "target"]+_features].groupby("time_id").mean()
    time_to_cheer_up, time_to_calm_down = df.target.idxmax(), df.target.idxmin()
    _, *_ = df.plot(figsize=(16,12), subplots=True, sharex=True, color=[PURPLE] + [YELLOW] * (df.shape[1] - 1) , alpha=.5, ax=ax)
    ax.axhline(0, linestyle="--", color="k", linewidth=1)
    ax.scatter(time_to_cheer_up, df.loc[time_to_cheer_up, "target"], marker="^", color=GREEN)
    ax.scatter(time_to_calm_down, df.loc[time_to_calm_down, "target"], marker="v", color='red')

    return fig


def train_model(args, train, features, cat_features, save_feature_importance=False):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy.stats import pearsonr

    # custom modeling stuff
    from lgbm_ops import run, rmse

    GREEN, PURPLE, YELLOW = _set_plot_config()

    investment_ids = train.investment_id.unique()
    info = "without_investment_id"

    features_importance = run(info=info, args=args, train=train, features=features, cat_features=cat_features)

    df = train[["target", "preds", "time_id"]].query("preds!=-1000")
    
    score = df.groupby("time_id").apply(lambda x: pearsonr(x.target, x.preds)[0]).mean()
    print(f"lgbm {info} {args.cv_method} {args.folds} folds mean rmse: {rmse(df.target, df.preds):.4f}, mean pearsonr: {pearsonr(df.target, df.preds)[0]:.4f}, mean pearsonr by time_id: {score:.4f}")

    folds_mean_importance = (
        features_importance.groupby("feature", as_index=False)
        .importance.mean()
        .sort_values(by="importance", ascending=False)
    )

    if save_feature_importance:
        features_importance.to_csv(f"features_importance_{info}.csv", index=False)
        folds_mean_importance.to_csv(f"folds_mean_feature_importance_{info}.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.barplot(x="importance", y="feature", data=folds_mean_importance.head(50), ax=ax[0])
    ax[0].set_title(f'Head LightGBM Features {info} (avg over {args.folds} folds)')

    sns.barplot(x="importance", y="feature", data=folds_mean_importance.tail(50), ax=ax[1])
    ax[1].set_title(f'Tail LightGBM Features {info} (avg over {args.folds} folds)')
    plt.tight_layout()
    
    return fig


class FastDataModeling(FlowSpec):

    num_files = Parameter(
        "num-files", default=1000000, help="Maximum number of files to download"
    )
    url = Parameter("s3src", default=DEFAULT_URL, help="S3 prefix to Parquet files")
    only_download = Parameter("only-download", default=False, is_flag=True)
    chinese_calendar_data = IncludeFile("cal", default='holidays_of_china_from_2014_to_2030.csv')
    
    @step
    def start(self):
        self.next(self.model)

    @batch(**RESOURCES)
    @pip(
        libraries={
            'pandas': PANDAS_VERSION, 
            'pyarrow': PYARROW_VERSION, 
            'scipy': SCIPY_VERSION,
            'lightgbm': LIGHTGBM_VERSION, 
            'matplotlib': MATPLOTLIB_VERSION, 
            'seaborn': SEABORN_VERSION
        }
    )
    @card
    @step
    def model(self):

        # I/O & data dependencies
        from metaflow import S3
        from io import StringIO
        import pandas as pd
        from datetime import datetime

        # Modeling dependencies
        import lightgbm as lgb

        from pathlib import Path
        from argparse import Namespace
        self.args = Namespace(
            debug=False,
            seed=21,
            folds=2,
            workers=RESOURCES['cpu'],
            min_time_id=None,
            cv_method="group",
            num_bins=16,
            holdout_size=100,
            outlier_threshold=0.001,
            trading_days_per_year=250,
            data_path=Path("."),
        )

        # Load data
        print('Loading data...', end=' ')
        table = load_table(
            self.url, self.num_files, num_threads=RESOURCES['cpu'], only_download=self.only_download
        )
        train_df = table.to_pandas()
        print(f'Loaded {train_df.shape[0]} rows of data.')

        # Visualize the target and columns
        self.calendar_df = pd.read_csv(StringIO(self.chinese_calendar_data), parse_dates=["date"], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'))
        fig = target_prediction_viz(train_df, self.calendar_df)
        current.card.append(Markdown("# Target prediction visualization"))
        current.card.append(Markdown("See the original analysis [here](https://www.kaggle.com/code/valleyzw/ubiquant-lgbm-baseline)"))
        current.card.append(Image.from_matplotlib(fig))

        # Visualize outliers
        fig1, fig2, self.outlier_investments = outlier_viz(train_df, self.args)
        current.card.append(Markdown("# Outlier visualization"))
        current.card.append(Image.from_matplotlib(fig1))
        current.card.append(Image.from_matplotlib(fig2))

        # Remove outliers and filter
        train_df = remove_outliers(train_df, self.outlier_investments, self.args)
        features, cat_features, train_df = filter_features(train_df)

        # Visualize missing and interesting times
        fig1 = missing_times_viz(train_df)
        fig2 = interesting_times_viz(train_df)
        current.card.append(Markdown("# Missing and interesting times visualization"))
        current.card.append(Table([
            [Image.from_matplotlib(fig1), Image.from_matplotlib(fig2)]
        ]))

        # Train LGBM model
        fig = train_model(self.args, train_df, features, cat_features)        
        current.card.append(Markdown("# Feature importance visualization"))
        current.card.append(Image.from_matplotlib(fig))

        self.next(self.end)

    @step
    def end(self):
        print("Done")

if __name__ == '__main__':
    FastDataModeling()