import os
from os.path import join
import gzip, json, pickle
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
from vis_utils import calculate_cube_vertices, calculate_occlusion_stats, edges, DIS_CAR_SAVE
import cv2
import multiprocessing
import argparse

# All data in the Bench2Drive dataset are in the left-handed coordinate system.
# This code converts all coordinate systems (world coordinate system, vehicle coordinate system,
# camera coordinate system, and lidar coordinate system) to the right-handed coordinate system
# consistent with the nuscenes dataset.

DATAROOT = '../../data/bench2drive'
MAP_ROOT = '../../data/bench2drive/maps'
OUT_DIR = '../../data/infos'

MAX_DISTANCE = 75  # Filter bounding boxes that are too far from the vehicle
FILTER_Z_SHRESHOLD = 10  # Filter bounding boxes that are too high/low from the vehicle
FILTER_INVISINLE = True  # Filter bounding boxes based on visibility
NUM_VISIBLE_SHRESHOLD = 1  # Filter bounding boxes with fewer visible vertices than this value
NUM_OUTPOINT_SHRESHOLD = 7  # Filter bounding boxes where the number of vertices outside the frame is greater than this value in all cameras
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
CAMERA_TO_FOLDER_MAP = {'CAM_FRONT': 'rgb_front', 'CAM_FRONT_LEFT': 'rgb_front_left',
                        'CAM_FRONT_RIGHT': 'rgb_front_right', 'CAM_BACK': 'rgb_back', 'CAM_BACK_LEFT': 'rgb_back_left',
                        'CAM_BACK_RIGHT': 'rgb_back_right'}

stand_to_ue4_rotate = np.array([[0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]])

lidar_to_righthand_ego = np.array([[0, 1, 0, 0],
                                   [-1, 0, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

lefthand_ego_to_lidar = np.array([[0, 1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

left2right = np.eye(4)
left2right[1, 1] = -1



def apply_trans(vec, world2ego):
    vec = np.concatenate((vec, np.array([1])))
    t = world2ego @ vec
    return t[0:3]


def get_pose_matrix(dic):
    new_matrix = np.zeros((4, 4))
    new_matrix[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=dic['theta'] - np.pi / 2).rotation_matrix
    new_matrix[0, 3] = dic['x']
    new_matrix[1, 3] = dic['y']
    new_matrix[3, 3] = 1
    return new_matrix


def get_npc2world(npc):
    for key in ['world2vehicle', 'world2ego', 'world2sign', 'world2ped']:
        if key in npc.keys():
            npc2world = np.linalg.inv(np.array(npc[key]))
            yaw_from_matrix = np.arctan2(npc2world[1, 0], npc2world[0, 0])
            yaw = npc['rotation'][-1] / 180 * np.pi
            if abs(yaw - yaw_from_matrix) > 0.01:
                npc2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=yaw).rotation_matrix
            npc2world = left2right @ npc2world @ left2right
            return npc2world
    npc2world = np.eye(4)
    npc2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=npc['rotation'][-1] / 180 * np.pi).rotation_matrix
    npc2world[0:3, 3] = np.array(npc['location'])
    return left2right @ npc2world @ left2right


def get_global_trigger_vertex(center, extent, yaw_in_degree):
    x, y = center[0], -center[1]
    dx, dy = extent[0], extent[1]
    yaw_in_radians = -yaw_in_degree / 180 * np.pi
    vertex_in_self = np.array([[dx, dy],
                               [-dx, dy],
                               [-dx, -dy],
                               [dx, -dy]])
    rotate_matrix = np.array([[np.cos(yaw_in_radians), -np.sin(yaw_in_radians)],
                              [np.sin(yaw_in_radians), np.cos(yaw_in_radians)]])
    rotated_vertex = (rotate_matrix @ vertex_in_self.T).T
    vertex_in_global = np.array([[x, y]]).repeat(4, axis=0) + rotated_vertex
    return vertex_in_global


def get_image_point(loc, K, w2c):
    point = np.array([loc[0], loc[1], loc[2], 1])
    point_camera = np.dot(w2c, point)
    point_camera = point_camera[0:3]
    depth = point_camera[2]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2], depth


def get_action(index):
    Discrete_Actions_DICT = {
        0: (0, 0, 1, False),
        1: (0.7, -0.5, 0, False),
        2: (0.7, -0.3, 0, False),
        3: (0.7, -0.2, 0, False),
        4: (0.7, -0.1, 0, False),
        5: (0.7, 0, 0, False),
        6: (0.7, 0.1, 0, False),
        7: (0.7, 0.2, 0, False),
        8: (0.7, 0.3, 0, False),
        9: (0.7, 0.5, 0, False),
        10: (0.3, -0.7, 0, False),
        11: (0.3, -0.5, 0, False),
        12: (0.3, -0.3, 0, False),
        13: (0.3, -0.2, 0, False),
        14: (0.3, -0.1, 0, False),
        15: (0.3, 0, 0, False),
        16: (0.3, 0.1, 0, False),
        17: (0.3, 0.2, 0, False),
        18: (0.3, 0.3, 0, False),
        19: (0.3, 0.5, 0, False),
        20: (0.3, 0.7, 0, False),
        21: (0, -1, 0, False),
        22: (0, -0.6, 0, False),
        23: (0, -0.3, 0, False),
        24: (0, -0.1, 0, False),
        25: (1, 0, 0, False),
        26: (0, 0.1, 0, False),
        27: (0, 0.3, 0, False),
        28: (0, 0.6, 0, False),
        29: (0, 1.0, 0, False),
        30: (0.5, -0.5, 0, True),
        31: (0.5, -0.3, 0, True),
        32: (0.5, -0.2, 0, True),
        33: (0.5, -0.1, 0, True),
        34: (0.5, 0, 0, True),
        35: (0.5, 0.1, 0, True),
        36: (0.5, 0.2, 0, True),
        37: (0.5, 0.3, 0, True),
        38: (0.5, 0.5, 0, True),
    }
    throttle, steer, brake, reverse = Discrete_Actions_DICT[index]
    return throttle, steer, brake


def gengrate_map(map_root):
    map_infos = {}
    for file_name in os.listdir(map_root):
        # map_infos = {}
        # print(file_name)
        if '.npz' in file_name:  # and "11" not in file_name and "12" not in file_name and "13" not in file_name
            map_info = dict(np.load(join(map_root, file_name), allow_pickle=True)['arr'])
            town_name = file_name.split('_')[0]
            map_infos[town_name] = {}
            lane_points = []
            lane_types = []
            lane_sample_points = []
            trigger_volumes_points = []
            trigger_volumes_types = []
            trigger_volumes_sample_points = []
            for road_id, road in map_info.items():
                for lane_id, lane in road.items():
                    if lane_id == 'Trigger_Volumes':
                        for single_trigger_volume in lane:
                            points = np.array(single_trigger_volume['Points'])
                            points[:, 1] *= -1  # left2right
                            trigger_volumes_points.append(points)
                            trigger_volumes_sample_points.append(points.mean(axis=0))
                            trigger_volumes_types.append(single_trigger_volume['Type'])
                    else:
                        for single_lane in lane:
                            points = np.array([raw_point[0] for raw_point in single_lane['Points']])
                            points[:, 1] *= -1
                            lane_points.append(points)
                            lane_types.append(single_lane['Type'])
                            lane_lenth = points.shape[0]
                            if lane_lenth % 50 != 0:
                                devide_points = [50 * i for i in range(lane_lenth // 50 + 1)]
                            else:
                                devide_points = [50 * i for i in range(lane_lenth // 50)]
                            devide_points.append(lane_lenth - 1)
                            lane_sample_points_tmp = points[devide_points]
                            lane_sample_points.append(lane_sample_points_tmp)
            map_infos[town_name]['lane_points'] = lane_points
            map_infos[town_name]['lane_sample_points'] = lane_sample_points
            map_infos[town_name]['lane_types'] = lane_types
            map_infos[town_name]['trigger_volumes_points'] = trigger_volumes_points
            map_infos[town_name]['trigger_volumes_sample_points'] = trigger_volumes_sample_points
            map_infos[town_name]['trigger_volumes_types'] = trigger_volumes_types
    with open(join(OUT_DIR, 'b2d_map_infos.pkl'), 'wb') as f:
        pickle.dump(map_infos, f)


def preprocess(folder_list, idx, tmp_dir, train_or_val):
    data_root = DATAROOT
    cameras = CAMERAS
    final_data = []
    max_speed = 0
    if idx == 0:
        folders = tqdm(folder_list)
    else:
        folders = folder_list

    for folder_name in folders:
        folder_path = join(data_root, folder_name)
        last_position_dict = {}
        for ann_name in sorted(os.listdir(join(folder_path, 'anno')), key=lambda x: int(x.split('.')[0])):
            position_dict = {}
            frame_data = {}
            cam_gray_depth = {}
            with gzip.open(join(folder_path, 'anno', ann_name), 'rt', encoding='utf-8') as gz_file:
                anno = json.load(gz_file)
            frame_data['ego_vel'] = np.array([anno['speed'], 0, 0])
            max_speed=max(max_speed,anno['speed'])
    print(max_speed)

def generate_infos(folder_list, workers, train_or_val, tmp_dir):
    folder_num = len(folder_list)
    devide_list = [(folder_num // workers) * i for i in range(workers)]
    devide_list.append(folder_num)
    for i in range(workers):
        sub_folder_list = folder_list[devide_list[i]:devide_list[i + 1]]
        preprocess(sub_folder_list, i, tmp_dir, train_or_val)
    #     process = multiprocessing.Process(target=preprocess, args=(sub_folder_list, i, tmp_dir, train_or_val))
    #     process.start()
    #     process_list.append(process)
    # for i in range(workers):
    #     process_list[i].join()


if __name__ == "__main__":

    os.makedirs(OUT_DIR, exist_ok=True)
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--workers', type=int, default=32, help='num of workers to prepare dataset')
    argparser.add_argument('--tmp_dir', default="tmp_data", )
    args = argparser.parse_args()
    workers = args.workers
    process_list = []
    with open('../../data/splits/bench2drive_base_train_val_split.json', 'r') as f:
        train_val_split = json.load(f)

    all_folder = os.listdir(join(DATAROOT, 'v1'))
    train_list = []
    for foldername in all_folder:
        if 'Town' in foldername and 'Route' in foldername and 'Weather' in foldername and not join('v1', foldername) in \
                                                                                              train_val_split['val']:
            train_list.append(join('v1', foldername))
    print('processing train data...')
    generate_infos(train_list, workers, 'train', args.tmp_dir)