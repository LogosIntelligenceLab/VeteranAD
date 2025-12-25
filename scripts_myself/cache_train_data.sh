export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/SSD_DISK/datasets/openscene/maps"
export OPENSCENE_DATA_ROOT="/SSD_DISK/datasets/openscene"
export NAVSIM_EXP_ROOT="/SSD_DISK/users/zhangbozhou/VeteranAD/exp"
export NAVSIM_DEVKIT_ROOT="/SSD_DISK/users/zhangbozhou/VeteranAD"
export PYTHONPATH="/SSD_DISK/users/zhangbozhou/VeteranAD"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py
