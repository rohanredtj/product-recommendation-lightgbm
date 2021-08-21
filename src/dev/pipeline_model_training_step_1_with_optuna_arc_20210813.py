import pandas as pd
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import pickle
import optuna
from sklearn.metrics import roc_auc_score
import json

train_split_format_path = '../data/processed/model_data/train_split_format.csv'
train_heldout_format_path = '../data/processed/model_data/train_heldout_format.csv'
test_split_format_path = '../data/processed/model_data/test_split_format.csv'
test_heldout_format_path = '../data/processed/model_data/test_heldout_format.csv'
model_save_prepath = '../models/'
features_lable_path = '../data/feature_labels.json'

with open(features_lable_path, "rb") as fh:
    features = json.load(fh)
features = features['features_current']

seed = 7
n_trials = 15


def callback_xgboost_split_format(study, trial):
    if study.best_trial == trial:
        model.save_model(model_save_prepath+"xgboost_with_optuna_split_format.json")
        
def callback_balanced_rf_split_format(study, trial):
    if study.best_trial == trial:
        with open(model_save_prepath+"balanced_rf_with_optuna_split_format.pickle", 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def callback_xgboost_heldout_format(study, trial):
    if study.best_trial == trial:
        model.save_model(model_save_prepath+"xgboost_with_optuna_heldout_format.json")
        
def callback_balanced_rf_heldout_format(study, trial):
    if study.best_trial == trial:
        with open(model_save_prepath+"balanced_rf_with_optuna_heldout_format.pickle", 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('\n### split train/test format (XGBoost)')


train = pd.read_csv(train_split_format_path)
y_train = train["bought_in_the_visit"]
train = train[features]

test = pd.read_csv(test_split_format_path)
y_test = test["bought_in_the_visit"]
test = test[features]

def objective(trial):
    
    learning_rate = trial.suggest_loguniform("lr", 1e-3, 1)
    n_estimators = trial.suggest_int("n_est", 1, 750)
    max_depth = trial.suggest_int("m_depth", 1, 20)
    reg_alpha = trial.suggest_loguniform("alpha", 1e-10, 1)
    reg_lamb = trial.suggest_loguniform("reg_lamb", 1e-10, 1)
    gamma = trial.suggest_float("gamma", 0.1, 1)
    col_tree = trial.suggest_float("colt_tree", 0.5, 1)
    min_child = trial.suggest_loguniform("min_child", 0.1, 20)
    subsamp = trial.suggest_float("subsamp", 0.01, 1)

    global model

    model = XGBClassifier(
        use_label_encoder=False,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=min_child,
        gamma=gamma,
        subsample=subsamp,
        colsample_bytree=col_tree,
        objective="binary:logistic",
        eval_metric='logloss',
        seed=seed,
        reg_alpha = reg_alpha,
        max_depth = max_depth,
        reg_lambda = reg_lamb
    )
    
    model.fit(train, y_train)
    y_pred_proba = model.predict_proba(test)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials, callbacks=[callback_xgboost_split_format])




print('\n### split train/test format (Balanced Random Forest)')

def objective(trial):
    
    n_estimators = trial.suggest_int("n_est", 1, 750)
    
    global model
    
    model = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1
        
    )

    model.fit(train, y_train)
    y_pred_proba = model.predict_proba(test)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials, callbacks=[callback_balanced_rf_split_format])

    
    
print('\n### heldout format (XGBoost)')

train = pd.read_csv(train_heldout_format_path)
y_train = train["bought_in_the_visit"]
train = train[features]

test = pd.read_csv(test_heldout_format_path)
y_test = test["bought_in_the_visit"]
test = test[features]

def objective(trial):
    
    learning_rate = trial.suggest_loguniform("lr", 1e-3, 1)
    n_estimators = trial.suggest_int("n_est", 1, 750)
    max_depth = trial.suggest_int("m_depth", 1, 20)
    reg_alpha = trial.suggest_loguniform("alpha", 1e-10, 1)
    reg_lamb = trial.suggest_loguniform("reg_lamb", 1e-10, 1)
    gamma = trial.suggest_float("gamma", 0.1, 1)
    col_tree = trial.suggest_float("colt_tree", 0.5, 1)
    min_child = trial.suggest_loguniform("min_child", 0.1, 20)
    subsamp = trial.suggest_float("subsamp", 0.01, 1)

    global model

    model = XGBClassifier(
        use_label_encoder=False,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=min_child,
        gamma=gamma,
        subsample=subsamp,
        colsample_bytree=col_tree,
        objective="binary:logistic",
        eval_metric='logloss',
        seed=seed,
        reg_alpha = reg_alpha,
        max_depth = max_depth,
        reg_lambda = reg_lamb
    )
    
    model.fit(train, y_train)
    y_pred_proba = model.predict_proba(test)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials, callbacks=[callback_xgboost_heldout_format])


print('\n### heldout format (Balanced Random Forest)')

def objective(trial):
    
    n_estimators = trial.suggest_int("n_est", 1, 750)

    global model

    model = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1
    )
    
    model.fit(train, y_train)
    y_pred_proba = model.predict_proba(test)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials, callbacks=[callback_balanced_rf_heldout_format])

