#%%
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import smogn
import os

import time


#%%
'''
function
    lgbm: Perform LightGBM.
    bootstrap_lgbm: Repeat LightGBM on a randomly divided dataset.
'''

def bayes_objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 2, 10),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 3, 15),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 15),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 1),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 1),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 0.2)
    }

    if sm:
        xy = pd.concat([y,x], axis=1)
        xy.reset_index(inplace=True, drop=True)
        # smogn
        rg_mtrx = [
            [0.5,  1, 0],  ## over-sample ("minority")
            [1, 0, 0],  ## under-sample ("majority")
            [0, 0, 0],  ## under-sample ("majority")
        ]
        for trial in range(0, 10):
            try:
                xy_smogn = smogn.smoter(
                    data=xy,
                    y=y.name,
                    samp_method="extreme",
                    rel_thres=0.3,
                    rel_method="manual",
                    rel_ctrl_pts_rg=rg_mtrx,
                    replace=False
                    )
                break
            except ValueError as e:
                print("error:{e} retry:{t}/10".format(e=e, t=trial))
                continue
        x_resmp = xy_smogn.loc[:, cols]
        y_resmp = xy_smogn["gymno_freq"]
        x_train, x_test, y_train, y_test = train_test_split(x_resmp, y_resmp, test_size=0.3, shuffle=True)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

    # validate model
    fit_params = {
        'callbacks': [lgb.early_stopping(stopping_rounds=10, verbose=0)],
        'eval_metric': 'rmse',
        'eval_set': [(x_test, y_test)]
        }
    model.set_params(**params)
    scores = cross_val_score(model, x_train, y_train, cv=cv, scoring=scoring, fit_params=fit_params, n_jobs=-1)
    val = scores.mean()

    return val


def lgbm(x_train, x_test, y_train, y_test, params):
    # data
    train_data = lgb.Dataset(x_train, y_train)
    valid_set = lgb.Dataset(x_test, y_test)

    # LightGBM
    params.update(
        objective="regression",
        metric="rmse",
        random_state=777,
        learning_rate=0.01
    )
    process = {}
    bst = lgb.train(
        params=params,
        train_set = train_data,
        valid_sets = [train_data, valid_set],
        num_boost_round = 10000,
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=0)],
        evals_result = process
    )

    # feature importance
    raw_imp = bst.feature_importance(importance_type = "gain")
    max_imp = max(raw_imp)
    norm = lambda z: z / max_imp
    imp = norm(raw_imp)
    # prediction
    pred = pd.DataFrame(y_test)
    y_pred = bst.predict(x_test, num_iteration=bst.best_iteration)
    pred["Prediction"] = y_pred

    return(imp, pred, process)


def bootstrap_lgbm(x, y, params, seed, n_trials=100):
    # dataframes to store result
    imp_idx = x.columns.to_list()
    imp_trials = pd.DataFrame(index=imp_idx)
    met_trials = pd.DataFrame(index=["R2", "MAE", "MSE", "RMSE"])
    pred_trials = pd.DataFrame(index=y.index)

    for i in range(n_trials):
        # split data for training and test.
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=seed+i, shuffle=True)
        # perform LightGBM
        imp, pred, _ = lgbm(x_train, x_test, y_train, y_test, params)

        # feature importance
        imp_trials["trial_{}".format(i+1)] = imp
        # metrics
        r2 = r2_score(pred["gymno_freq"], pred["Prediction"])
        mae = mean_absolute_error(pred["gymno_freq"], pred["Prediction"])
        mse = mean_squared_error(pred["gymno_freq"], pred["Prediction"], squared=True)
        rmse = mean_squared_error(pred["gymno_freq"], pred["Prediction"], squared=False)
        met_trials["trial_{}".format(i+1)] = [r2, mae, mse, rmse]
        # prediction
        pred.drop(["gymno_freq"], axis=1, inplace=True)
        pred.rename(columns={"Prediction": "trial_{}".format(i+1)}, inplace=True)
        pred_trials = pd.merge(pred_trials, pred, how="left", left_index=True, right_index=True)

    res = [imp_trials, met_trials, pred_trials]

    for r in res:
        mean = r.mean(axis=1)
        var = r.var(axis=1)
        r.insert(loc=0, column="mean", value=mean)
        r.insert(loc=1, column="var", value=var)

    res[2].insert(loc=0, column="gymno_freq", value=y)

    return(res)


def smogn_lgbm(x, y, data, params, smogn_trials=100, lgbm_trials=10):
    # dataframes to store result
    imp_idx = x.columns.to_list()
    imp_df = pd.DataFrame(index=imp_idx)
    met_df = pd.DataFrame(index=["R2", "MAE", "MSE", "RMSE"])
    pred_df = pd.DataFrame(index=y.index)
    dfs = [imp_df, met_df, pred_df]

    xy = pd.concat([y,x], axis=1)
    xy.reset_index(inplace=True, drop=True)

    for l in range(smogn_trials):
        # smogn
        rg_mtrx = [
            [0.5,  1, 0],  ## over-sample ("minority")
            [1, 0, 0],  ## under-sample ("majority")
            [0, 0, 0],  ## under-sample ("majority")
        ]
        for trial in range(0, 10):
            try:
                xy_smogn = smogn.smoter(
                    data=xy,
                    y=y.name,
                    samp_method="extreme",
                    rel_thres=0.3,
                    rel_method="manual",
                    rel_ctrl_pts_rg=rg_mtrx,
                    replace=False
                    )
                break
            except ValueError as e:
                print("error:{e} retry:{t}/10".format(e=e, t=trial))
                continue

        xy_resmp = pd.merge(
            data, xy_smogn,
            how="right",
            left_on=xy.columns.to_list(),
            right_on=xy.columns.to_list()
            )
        # reassign index
        for s in range(xy_resmp["Project_code"].isnull().sum()):
            xy_resmp.loc[s, "Project_code"] = f"synthetic_sample_{s+1}"
        xy_resmp.set_index("Project_code", inplace=True)
        sp_idx = y.index.tolist()
        x_resmp = xy_resmp.loc[:, x.columns].astype("int")
        y_resmp = xy_resmp["gymno_freq"]

        # bootstrap cycle for a over-sampled dataset
        bs_cycle = bootstrap_lgbm(x_resmp, y_resmp, params, l, n_trials=lgbm_trials)

        # store result
        for m, (df, res) in enumerate(zip(dfs, bs_cycle)):
            if m == 2:
                res.drop(["mean", "var", "gymno_freq"], axis=1, inplace=True)
            else:
                res.drop(["mean", "var"], axis=1, inplace=True)
            cols = []
            for n in range(len(res.columns)):
                col = "trial_{}".format(l * 10 + n + 1)
                cols.append(col)
            res.columns = cols
            dfs[m] = pd.merge(df, res, how="left", left_index=True, right_index=True)

    for m, df in enumerate(dfs):
        mean = df.mean(axis=1)
        var = df.var(axis=1)
        df.insert(loc=0, column="mean", value=mean)
        df.insert(loc=1, column="var", value=var)

    dfs[2].insert(loc=0, column="gymno_freq", value=y)

    return(dfs)

##########################################################
#%%
'''
Preparation for analysis.
ASS: Dataset of CAZymes gene counts for Gymnosperm specialist / Angiosperm specialist / Generalist
["ass90_10"] = Angiosperm specialist: 0, Gymnosperm specialist: 1, Generalist: 2
'''
DF = pd.read_csv("../data/host_association.csv", header=0)
DF.set_index("Project_code", inplace=True)
ASS = DF.dropna(subset=["gymno_freq"])

# Column "ass90_10" is replaced from str to int.
host_map = {"ass90_10": {"A": 0, "G": 1, "Gen":2}}
ASS.replace(host_map, inplace=True)

# Remove class categories from explanatory variables.
cols = ASS.loc[:, "CAZymes":].columns.tolist()
classes = ["CAZymes", "AA", "CBM", "CE", "GH", "GT", "PL"]
for c in classes:
    cols.remove(c)


##########################################################
#%%
'''
Ex.1
LightGBM on original dataset.
'''
start = time.time()
#data
x = ASS.loc[:, cols].astype("int")
y = ASS["gymno_freq"]

# model
model = lgb.LGBMRegressor(
    boosting_type='gbdt',
    objective='regression',
    random_state=777,
    bagging_freq=1,
    max_depth=5,
    n_estimators=10000,
    learning_rate=0.01
    )

cv = KFold(n_splits=3, shuffle=True, random_state=777)
scoring = "neg_root_mean_squared_error"
sm = False

# parameter tuning.
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=777))
study.optimize(bayes_objective, n_trials=400, show_progress_bar=True)

best_params = study.best_trial.params
best_score = study.best_trial.value

os.makedirs("imbalanced_LGBM", exist_ok=True)
file = "imbalanced_LGBM/parameters.txt"
with open(file, "w") as f:
    f.write("Parameters:\n")
    for key, value in best_params.items():
        f.write(f"\t{key}: {value}\n")
    f.write(f"Best Score: {best_score}")

# perform LightGBM
result_1 = bootstrap_lgbm(x, y, best_params, 0, n_trials=100)
result_1[0].to_csv("imbalanced_LGBM/imb_LGBM_importance.csv")
result_1[1].to_csv("imbalanced_LGBM/imb_LGBM_metrics.csv")
result_1[2].to_csv("imbalanced_LGBM/imb_LGBM_prediction.csv")


##########################################################
#%%
'''
Ex.2
LightGBM on over-sampled dataset.
'''
start = time.time()
#data
ass_ex2 = ASS.reset_index()
x = ASS.loc[:, cols].astype("int")
y = ASS["gymno_freq"]

# model
model = lgb.LGBMRegressor(
    boosting_type='gbdt',
    objective='regression',
    random_state=777,
    bagging_freq=1,
    max_depth=5,
    n_estimators=10000,
    learning_rate=0.01
    )

cv = KFold(n_splits=3, shuffle=True, random_state=777)
scoring = "neg_root_mean_squared_error"
sm = True

# parameter tuning.
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=777))
study.optimize(bayes_objective, n_trials=400)

best_params = study.best_trial.params
best_score = study.best_trial.value

os.makedirs("smogn_LGBM", exist_ok=True)
file = "smogn_LGBM/parameters.txt"
with open(file, "w") as f:
    f.write("Parameters:\n")
    for key, value in best_params.items():
        f.write(f"\t{key}: {value}\n")
    f.write(f"Best Score: {best_score}")

# perform LightGBM
result_2 = smogn_lgbm(x, y, ass_ex2, best_params, smogn_trials=100, lgbm_trials=10)
result_2[0].to_csv("smogn_LGBM/smogn_LGBM_importance.csv")
result_2[1].to_csv("smogn_LGBM/smogn_LGBM_metrics.csv")
result_2[2].to_csv("smogn_LGBM/smogn_LGBM_prediction.csv")

# %%
