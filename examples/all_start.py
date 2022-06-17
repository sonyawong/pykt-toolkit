#!/usr/bin/env python
# coding=utf-8

import os, sys

# 启动agent
import json
with open("../configs/wandb.json") as fin:
    wandb_config = json.load(fin)

logf = sys.argv[1]
outf = open(sys.argv[2], "w")
start = int(sys.argv[3])
end = int(sys.argv[4])
# endf = open(sys.argv[5], "w")
dataset_name = sys.argv[5]
model_name = sys.argv[6]
nums = sys.argv[7].split(",")
# emb_type = sys.argv[8]
print(len(sys.argv))
if len(sys.argv) == 8:
    project_name = "kt_toolkits"
else:
    project_name = sys.argv[8]

cmdpre = "WANDB_API_KEY=" + wandb_config["api_key"] + " nohup wandb agent " + wandb_config["uid"] + "/{}/".format(project_name)
endcmdpre = "WANDB_API_KEY=" + wandb_config["api_key"] + " wandb sweep " + wandb_config["uid"] + "/{}/".format(project_name)

idx = 0
with open(logf, "r") as fin:
    i = 0
    lines = fin.readlines()
    l = []
    num = 0
    while i < len(lines):
        if lines[i].strip().startswith("wandb: Creating sweep from: "):
            fname = lines[i].strip().split(": ")[-1].split("/")[-1]
        else:
            print("error!")
        if lines[i+1].strip().startswith("wandb: Created sweep with ID: "):
            sweepid = lines[i+1].strip().split(": ")[-1]
        else:
            print("error!")
        fname = fname.split(".")[0]
        # print(fname)
        if not fname.startswith(dataset_name) or fname.find("_" + model_name + "_") == -1:
            i += 4
            continue
        # if num == totalnum + 1:
        #     num = 0
        # l.append([fname, sweepid, cmd])
        print(f"dataset_name: {dataset_name}, model_name: {model_name}, fname: {fname}")
        if idx >= start and idx < end:
            cmd = "CUDA_VISIBLE_DEVICES=" + str(nums[num]) +" " + cmdpre + sweepid + " &"
            # end_cmd = endcmdpre + sweepid + " --stop"
            outf.write(cmd + "\n")
            # endf.write(end_cmd + "\n")
            num += 1
        idx += 1
        i += 4
l
