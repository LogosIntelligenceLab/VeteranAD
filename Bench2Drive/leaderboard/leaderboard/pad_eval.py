#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
import sys
import os



CARLA_ROOT=os.environ.get("CARLA_ROOT")+"/"
Bench2Drive_ROOT=os.environ.get("Bench2Drive_ROOT")+"/"

sys.path.append(Bench2Drive_ROOT)

sys.path.append(CARLA_ROOT + "PythonAPI")
sys.path.append(CARLA_ROOT + "PythonAPI/carla")
sys.path.append(CARLA_ROOT + "PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")

sys.path.append(Bench2Drive_ROOT + 'leaderboard')
sys.path.append(Bench2Drive_ROOT + 'leaderboard/pad_team_code')
sys.path.append(Bench2Drive_ROOT + 'scenario_runner')


ROUTES=Bench2Drive_ROOT +"leaderboard/data/shuffle.xml"

os.environ["SAVE_PATH"] = Bench2Drive_ROOT+"/eval_pad/"

if not os.path.exists(os.environ.get("SAVE_PATH")):
    os.mkdir(os.environ.get("SAVE_PATH"))


visualize=False

if visualize:
    os.environ["TEAM_AGENT"]  = Bench2Drive_ROOT + "leaderboard/pad_team_code/pad_b2d_agent_vis.py"
else:
    os.environ["TEAM_AGENT"]= Bench2Drive_ROOT + "leaderboard/pad_team_code/pad_b2d_agent.py"

os.environ["IS_BENCH2DRIVE"] = "True"
os.environ["ROUTES"] = ROUTES
os.environ["CHECKPOINT_ENDPOINT"]=os.environ["SAVE_PATH"]+"eval.json"

os.environ["SCENARIO_RUNNER_ROOT"] = "scenario_runner"
os.environ["LEADERBOARD_ROOT"] = "leaderboard"

checkpoint_path=None

os.environ["TEAM_CONFIG"]=Bench2Drive_ROOT +"leaderboard/pad_team_code/pad_config.py+"+checkpoint_path

from leaderboard_evaluator import main

main()
