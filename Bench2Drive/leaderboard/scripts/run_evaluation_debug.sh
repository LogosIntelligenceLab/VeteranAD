#!/bin/bash
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/vis #bench2drive220
TEAM_AGENT=leaderboard/pad_team_code/pad_b2d_agent_visualize.py
#TEAM_CONFIG=your_team_agent_ckpt.pth   # for TCP and ADMLP
TEAM_CONFIG=leaderboard/pad_team_code/pad_config.py+/home/ke/PAD/exp/B2d_onlyP_map88_learnpos_scorekeyval2_closemul_epoch=13-step=10752.ckpt
BASE_CHECKPOINT_ENDPOINT=eval
SAVE_PATH=./eval_v1/
PLANNER_TYPE=only_traj

GPU_RANK=0
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="./eval_v1/${BASE_CHECKPOINT_ENDPOINT}.json"
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK
