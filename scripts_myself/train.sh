export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/SSD_DISK/datasets/openscene/maps"
export OPENSCENE_DATA_ROOT="/SSD_DISK/datasets/openscene"
export NAVSIM_EXP_ROOT="/SSD_DISK/users/zhangbozhou/VeteranAD/exp"
export NAVSIM_DEVKIT_ROOT="/SSD_DISK/users/zhangbozhou/VeteranAD"
export PYTHONPATH="/SSD_DISK/users/zhangbozhou/VeteranAD"
export HF_ENDPOINT=https://hf-mirror.com
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

# bs=32 for 48G GPU, bs=64 for 80G GPU
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent.is_training=True \
    experiment_name=train \
    dataloader.params.batch_size=32
