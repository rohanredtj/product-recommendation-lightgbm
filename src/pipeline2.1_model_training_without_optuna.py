# script to train the models and save the trained models

import json
import re
import timeit

import lightgbm as lgb
import pandas as pd

if __name__ == "__main__":

    starttime = timeit.default_timer()
    print("The start time is :", starttime)

    train_path = "../data/processed/train_heldout_format.csv"
    model_save_prepath = "../models/"
    features_lable_path = "../data/feature_labels.json"
    exp_id = "20210822v1_lgb_without_optuna_normalized_weeks_without_tsfresh_features_and_categorical_original_saved_model_on_heldout_train_format"

    with open(features_lable_path, "rb") as fh:
        features = json.load(fh)
    features = features["features_current"]

    seed = 7

    train = pd.read_csv(train_path)
    print("Training Data Loaded")
    y_train = train["bought_in_the_visit"]
    train = train[features]

    train.columns = list(range(0, len(list(train.columns))))
    train = lgb.Dataset(train, y_train)
    print("Model Train Start")
    model = lgb.train(params={"learning_rate": 0.05}, train_set=train)
    model.save_model(
        model_save_prepath + exp_id + ".txt", num_iteration=model.best_iteration
    )

    print("Total time is :", timeit.default_timer() - starttime)
