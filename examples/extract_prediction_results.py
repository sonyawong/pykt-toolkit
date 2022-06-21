#!/usr/bin/env python
# coding=utf-8

from logging.config import dictConfig
from pyexpat import model
import wandb
import pandas as pd
import numpy as np
# from tqdm import tqdm_notebook

import wandb 
import os, sys
import json
import socket
import yaml

with open("../configs/wandb.json") as fin:
    wandb_config = json.load(fin)
uid = wandb_config["uid"]
os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
pred_dir = "pred_wandbs"

def get_runs_result(runs):
    result_list = []
    for run in runs:#tqdm_notebook(runs):
        result = {}
        result.update(run.summary._json_dict)
        model_config = {k: v for k, v in run.config.items()
                    if not k.startswith('_') and type(v) not in [list,dict]}
        result.update(model_config)
        result['Name'] = run.name
        result_list.append(result)
    runs_df = pd.DataFrame(result_list)#.drop_duplicates(list(model_config.keys()))
    return runs_df


def get_df(sweep_name,sweep_dict):
    df_list = []
    for sweep_id in sweep_dict[sweep_name]:
        sweep = api.sweep("{}/{}/{}".format(uid, project_name,sweep_id))
        df = get_runs_result(sweep.runs)
        df_list.append(df)
    df = pd.concat(df_list)
#     print(df.shape)
    return df

def get_sweep_dict(project):
    '''获取sweep的id'''
    sweep_dict = {}
    for sweep in project.sweeps():
        if sweep.name not in sweep_dict:
            sweep_dict[sweep.name] = []
        sweep_dict[sweep.name].append(sweep.id)
    return sweep_dict

def downloads(sweep_dict, dataset_name="", model_name=""):
    all_res = dict()
    for key in sweep_dict:
        emb_type = key.split("_")[-2]
        if model_name !="dkt_forget":
            if key.find("pred_wandbs") == -1 or (dataset_name != "" and model_name != "" and (key.find(dataset_name+"_") == -1 or key.find("_"+model_name+"_fold") == -1)) :
                continue
        else:
            if key.find("pred_wandbs") == -1 or ("pred_wandbs" in key and dataset_name != "" and model_name != "" and model_name not in key) :
                continue            
        try:
            df = get_df(key,sweep_dict)
            df = df.dropna(subset=["testauc"])
            print(f"key: {key}, df dropna: {df.shape}")
            tmps = key.split("_")
            key = "_".join(tmps[0:-1])
            all_res.setdefault(key, dict())
            fold = tmps[-1]
            print(f"key: {key}, fold: {fold}, df: {df.shape}")
            all_res[key][fold] = df
        except:
            print(f"error: {key}")
            continue
    return all_res

def cal_results(dataset_name, model_name, all_res):
    all_res = all_res.drop_duplicates(["save_dir"])
    repeated_aucs = np.unique(all_res["testauc"].values)
    repeated_accs = np.unique(all_res["testacc"].values)
    repeated_window_aucs = np.unique(all_res["window_testauc"].values)
    repeated_window_accs = np.unique(all_res["window_testacc"].values)
    repeated_auc_mean, repeated_auc_std = round(np.mean(repeated_aucs), 4), round(np.std(repeated_aucs, ddof=0), 4)
    repeated_acc_mean, repeated_acc_std = round(np.mean(repeated_accs), 4), round(np.std(repeated_accs, ddof=0), 4)
    repeated_winauc_mean, repeated_winauc_std = round(np.mean(repeated_window_aucs), 4), round(np.std(repeated_window_aucs, ddof=0), 4)
    repeated_winacc_mean, repeated_winacc_std = round(np.mean(repeated_window_accs), 4), round(np.std(repeated_window_accs, ddof=0), 4)
    key = dataset_name + "_" + model_name
    print(key + "_repeated:", str(repeated_auc_mean) + "±" + str(repeated_auc_std) + "," + str(repeated_acc_mean) + "±" + str(repeated_acc_std) + "," + str(repeated_winauc_mean) + "±" + str(repeated_winauc_std) + "," + str(repeated_winacc_mean) + "±" + str(repeated_winacc_std)) 
    
    question_aucs = np.unique(all_res["oriaucconcepts"].values)
    question_accs = np.unique(all_res["oriaccconcepts"].values)
    question_window_aucs = np.unique(all_res["windowaucconcepts"].values)
    question_window_accs = np.unique(all_res["windowaccconcepts"].values)
    question_auc_mean, question_auc_std = round(np.mean(question_aucs), 4), round(np.std(question_aucs, ddof=0), 4)
    question_acc_mean, question_acc_std = round(np.mean(question_accs), 4), round(np.std(question_accs, ddof=0), 4)
    question_winauc_mean, question_winauc_std = round(np.mean(question_window_aucs), 4), round(np.std(question_window_aucs, ddof=0), 4)
    question_winacc_mean, question_winacc_std = round(np.mean(question_window_accs), 4), round(np.std(question_window_accs, ddof=0), 4)
    key = dataset_name + "_" + model_name
    print(key + "_concepts:", str(question_auc_mean) + "±" + str(question_auc_std) + "," + str(question_acc_mean) + "±" + str(question_acc_std) + "," + str(question_winauc_mean) + "±" + str(question_winauc_std) + "," + str(question_winacc_mean) + "±" + str(question_winacc_std)) 

    try:
        early_aucs = np.unique(all_res["oriaucearly_preds"].values)
        early_accs = np.unique(all_res["oriaccearly_preds"].values)
        early_window_aucs = np.unique(all_res["windowaucearly_preds"].values)
        early_window_accs = np.unique(all_res["windowaccearly_preds"].values)
        early_auc_mean, early_auc_std = round(np.mean(early_aucs), 4), round(np.std(early_aucs, ddof=0), 4)
        early_acc_mean, early_acc_std = round(np.mean(early_accs), 4), round(np.std(early_accs, ddof=0), 4)
        early_winauc_mean, early_winauc_std = round(np.mean(early_window_aucs), 4), round(np.std(early_window_aucs, ddof=0), 4)
        early_winacc_mean, early_winacc_std = round(np.mean(early_window_accs), 4), round(np.std(early_window_accs, ddof=0), 4)
        key = dataset_name + "_" + model_name
        print(key + "_early:", str(early_auc_mean) + "±" + str(early_auc_std) + "," + str(early_acc_mean) + "±" + str(early_acc_std) + "," + str(early_winauc_mean) + "±" + str(early_winauc_std) + "," + str(early_winacc_mean) + "±" + str(early_winacc_std))
    except:
        print(f"{model_name} don't have early fusion!!!")

    late_all_aucs = np.unique(all_res["oriauclate_all"].values)
    late_all_accs = np.unique(all_res["oriacclate_all"].values)
    late_all_window_aucs = np.unique(all_res["windowauclate_all"].values)
    late_all_window_accs = np.unique(all_res["windowacclate_all"].values)
    lateall_auc_mean, lateall_auc_std = round(np.mean(late_all_aucs), 4), round(np.std(late_all_aucs, ddof=0), 4)
    lateall_acc_mean, lateall_acc_std = round(np.mean(late_all_accs), 4), round(np.std(late_all_accs, ddof=0), 4)
    lateall_winauc_mean, lateall_winauc_std = round(np.mean(late_all_window_aucs), 4), round(np.std(late_all_window_aucs, ddof=0), 4)
    lateall_winacc_mean, lateall_winacc_std = round(np.mean(late_all_window_accs), 4), round(np.std(late_all_window_accs, ddof=0), 4)
    key = dataset_name + "_" + model_name
    print(key + "_lateall:", str(lateall_auc_mean) + "±" + str(lateall_auc_std) + "," + str(lateall_acc_mean) + "±" + str(lateall_acc_std) + "," + str(lateall_winauc_mean) + "±" + str(lateall_winauc_std) + "," + str(lateall_winacc_mean) + "±" + str(lateall_winacc_std))
    

    late_mean_aucs = np.unique(all_res["oriauclate_mean"].values)
    late_mean_accs = np.unique(all_res["oriacclate_mean"].values)
    late_mean_window_aucs = np.unique(all_res["windowauclate_mean"].values)
    late_mean_window_accs = np.unique(all_res["windowacclate_mean"].values)
    latemean_auc_mean, latemean_auc_std = round(np.mean(late_mean_aucs), 4), round(np.std(late_mean_aucs, ddof=0), 4)
    latemean_acc_mean, latemean_acc_std = round(np.mean(late_mean_accs), 4), round(np.std(late_mean_accs, ddof=0), 4)
    latemean_winauc_mean, latemean_winauc_std = round(np.mean(late_mean_window_aucs), 4), round(np.std(late_mean_window_aucs, ddof=0), 4)
    latemean_winacc_mean, latemean_winacc_std = round(np.mean(late_mean_window_accs), 4), round(np.std(late_mean_window_accs, ddof=0), 4)
    key = dataset_name + "_" + model_name
    print(key + "_latemean:", str(latemean_auc_mean) + "±" + str(latemean_auc_std) + "," + str(latemean_acc_mean) + "±" + str(latemean_acc_std) + "," + str(latemean_winauc_mean) + "±" + str(latemean_winauc_std) + "," + str(latemean_winacc_mean) + "±" + str(latemean_winacc_std))
    
    late_vote_aucs = np.unique(all_res["oriauclate_vote"].values)
    late_vote_accs = np.unique(all_res["oriacclate_vote"].values)
    late_vote_window_aucs = np.unique(all_res["windowauclate_vote"].values)
    late_vote_window_accs = np.unique(all_res["windowacclate_vote"].values)
    latevote_auc_mean, latevote_auc_std = round(np.mean(late_vote_aucs), 4), round(np.std(late_vote_aucs, ddof=0), 4)
    latevote_acc_mean, latevote_acc_std = round(np.mean(late_vote_accs), 4), round(np.std(late_vote_accs, ddof=0), 4)
    latevote_winauc_mean, latevote_winauc_std = round(np.mean(late_vote_window_aucs), 4), round(np.std(late_vote_window_aucs, ddof=0), 4)
    latevote_winacc_mean, latevote_winacc_std = round(np.mean(late_vote_window_accs), 4), round(np.std(late_vote_window_accs, ddof=0), 4)
    key = dataset_name + "_" + model_name
    print(key + "_latevote:", str(latevote_auc_mean) + "±" + str(latevote_auc_std) + "," + str(latevote_acc_mean) + "±" + str(latevote_acc_std) + "," + str(latevote_winauc_mean) + "±" + str(latevote_winauc_std) + "," + str(latevote_winacc_mean) + "±" + str(latevote_winacc_std))
    

def print_model_res(dataset_name, model_names, update):
    for model_name in model_names:
        fname = dataset_name + "_" + model_name + "_prediction.pkl"
        if update == "1" or not os.path.exists(fname):
            all_res = downloads(sweep_dict, dataset_name, model_name)
            pd.to_pickle(all_res, fname)
        else:
            all_res = pd.read_pickle(fname)
        # print("all_res", all_res)
        for key in all_res:
            # print(f"key{key}")
            if model_name == "dkt_forget":
                cal_results(dataset_name, model_name, all_res[key]["forget"])
            else:
                print(f"key:{key}")
                cal_results(dataset_name, model_name, all_res[key]["fold"])
            # print("="*20)

if __name__ == "__main__":
    api = wandb.Api()
    dataset_name = sys.argv[1]
    model_names = sys.argv[2]
    model_names = model_names.split(",")
    update = sys.argv[3]
    try:
        project_name = sys.argv[4]
    except:
        project_name = "kt_toolkits"
    project = api.project(name=project_name)
    sweep_dict = get_sweep_dict(project)
    # print(f"sweep_dict: {sweep_dict}")
    print_model_res(dataset_name, model_names, update)