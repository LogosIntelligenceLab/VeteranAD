# Perception in Plan: Coupled Perception and Planning for End-to-End Autonomous Driving
### [[Paper]](https://arxiv.org/abs/2508.11488)

> [**Perception in Plan: Coupled Perception and Planning for End-to-End Autonomous Driving**](https://arxiv.org/abs/2508.11488)            
> [Bozhou Zhang](https://zbozhou.github.io/), [Jingyu Li](https://sii-whaleice.github.io/), [Nan Song](https://scholar.google.com/citations?hl=zh-CN&user=wLZVtjEAAAAJ), [Li Zhang](https://lzrobots.github.io)<br>
> **AAAI 2026**

## Abstract
End-to-end autonomous driving has achieved remarkable advancements in recent years. Existing methods primarily follow a perception–planning paradigm, where perception and planning are executed sequentially within a fully differentiable framework for planning-oriented optimization. We further advance this paradigm through a "perception-in-plan" framework design, which integrates perception into the planning process. This design facilitates targeted perception guided by evolving planning objectives over time, ultimately enhancing planning performance. Building on this insight, we introduce **VeteranAD**, a coupled perception and planning framework for end-to-end autonomous driving. By incorporating multi-mode anchored trajectories as planning priors, the perception module is specifically designed to gather traffic elements along these trajectories, enabling comprehensive and targeted perception. Planning trajectories are then generated based on both the perception results and the planning priors. To make perception fully serve planning, we adopt an autoregressive strategy that progressively predicts future trajectories while focusing on relevant regions for targeted perception at each step. With this simple yet effective design, VeteranAD fully unleashes the potential of planning-oriented end-to-end methods, leading to more accurate and reliable driving behavior. Extensive experiments on the NAVSIM and Bench2Drive datasets demonstrate that our VeteranAD achieves state-of-the-art performance.

## Pipeline
<div align="center">
  <img src="assets/main.png"/>
</div><br/>

## News
- 2025.12, the code for VeteranAD+ has been released. VeteranAD+ achieves a PDMS of 91.1, outperforming VeteranAD (90.2), thanks to the excellent codebase provided by [iPad](https://github.com/Kguo-cs/iPad).
- 2025.08, the paper is released on arXiv, and the code will be made publicly available in December 2025.

## Install
```
conda create -n VeteranAD python=3.8
conda activate VeteranAD

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install -e ./nuplan-devkit
pip install -e .
```

## Download
```
bash download/download_maps.sh
bash download/download_navtrain.sh
bash download/download_navtest.sh
```

## Data process
- Remember to set `cache_data=False` in `navsim/agents/pad/pad_agent.py` during training and evaluation, and set `cache_data=True` when caching data and computing metrics.
```
bash scripts_myself/cache_train_data.sh
bash scripts_myself/cache_train_metric.sh
bash scripts_myself/cache_eval_metric.sh
```

## Train and eval
```
bash scripts_myself/train.sh
bash scripts_myself/eval.sh
```

## Checkpoints
[ResNet-34](https://drive.google.com/file/d/1MD3Eb-ogO8mhPzg2i_h-X3sLFRQQijJ7/view?usp=sharing)

|Model   | NC   | DAC  | EP   | TTC  | Comfort | PDMS |
|:----:|:----:|:----:|:----:|:----:|:-------:|:----:|
|[VeteranAD+](https://drive.google.com/file/d/1yCTnnkZeF1oCxNSL8fjV2zcDMCYflLEA/view?usp=drive_link)| **98.6** | **97.8** | **87.8** | **94.7** | **99.2**  | **91.1** |

## BibTeX
```bibtex
@inproceedings{zhang2025veteranad,
 title={Perception in Plan: Coupled Perception and Planning for End-to-End Autonomous Driving},
 author={Zhang, Bozhou and Li, Jingyu and Song, Nan and Zhang, Li},
 booktitle={AAAI},
 year={2026},
}
```
