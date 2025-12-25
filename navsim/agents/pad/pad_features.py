from enum import IntEnum
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import numpy.typing as npt
import torch
from torchvision import transforms
from shapely import affinity
from shapely.geometry import Polygon, LineString
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from navsim.agents.pad.navsim_config import PadConfig
from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from .bevformer.bev_feature_build import _get_bev_feature
from navsim.agents.pad.geometry import resize_and_crop_image, update_intrinsics
from pyquaternion import Quaternion


def get_future_egomotion(agent_input, rec, idx, s):
    # 计算从 t=0 到 t=idx 的自车运动变换矩阵, 并转换成 6-DOF 位姿。
    rec_t0 = rec
    future_egomotion = np.eye(4, dtype=np.float32)
    if idx < s - 1:
        rec_t1 = agent_input.ego_statuses[idx + 1]
        egopose_t0 = convert_egopose_to_matrix_numpy(rec_t0.ego_pose[-1], rec_t0.ego_pose[:-1])
        egopose_t1 = convert_egopose_to_matrix_numpy(rec_t1.ego_pose[-1], rec_t1.ego_pose[:-1])
        future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
        future_egomotion[3, :3] = 0.0
        future_egomotion[3, 3] = 1.0
    future_egomotion = torch.Tensor(future_egomotion).float()
    future_egomotion = mat2pose_vec(future_egomotion)

    return future_egomotion.unsqueeze(0)


def convert_egopose_to_matrix_numpy(rotation, translation):
    # 将自车的旋转四元数和位移转换为 4×4 变换矩阵。
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = Quaternion(rotation).rotation_matrix
    translation = np.array(translation)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:2, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def invert_matrix_egopose_numpy(egopose):
    # 计算 4×4 变换矩阵的逆（即 T⁻¹, 用于坐标变换）。
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def mat2pose_vec(matrix: torch.Tensor):
    """
    Converts a 4x4 pose matrix into a 6-dof pose vector
    Args:
        matrix (ndarray): 4x4 pose matrix
    Returns:
        vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
        rotation components (rx, ry, rz)
    """
    # 将 4×4 变换矩阵转换为 6-DOF [tx, ty, tz, rx, ry, rz]。

    # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

    # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
    roty = torch.atan2(matrix[..., 0, 2], cosy)

    # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
    rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

    rotation = torch.stack((rotx, roty, rotz), dim=-1)

    # Extract translation params
    translation = matrix[..., :3, 3]
    return torch.cat((translation, rotation), dim=-1)


class PadFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, config: PadConfig):
        self._config = config
        self.augmentation_parameters = self.get_resizing_and_cropping_parameters()

    def get_resizing_and_cropping_parameters(self):
        original_height, original_width = self._config.IMAGE_ORIGINAL_HEIGHT, self._config.IMAGE_ORIGINAL_WIDTH
        final_height, final_width = self._config.IMAGE_FINAL_DIM

        resize_scale = self._config.IMAGE_RESIZE_SCALE
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = self._config.IMAGE_TOP_CROP
        crop_w = int(max(0, (resized_width - final_width) / 2))
        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        if resized_width != final_width:
            print('Zero padding left and right parts of the image.')
        if crop_h + final_height != resized_height:
            print('Zero padding bottom part of the image.')

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop,
                }

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "pad_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        features = _get_bev_feature(agent_input)
        #features["lidar_feature"] = self._get_lidar_feature(agent_input)

        ego_feature_list=[]

        for ego_status in agent_input.ego_statuses:
            if ego_status is None:
                continue
            pose=torch.tensor(ego_status.ego_pose, dtype=torch.float32)
            velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32)
            acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32)
            driving_command = torch.tensor(ego_status.driving_command, dtype=torch.float32)

            ego_feature=torch.cat([pose,velocity, acceleration, driving_command], dim=-1)
            ego_feature_list.append(ego_feature)

        features["ego_status"] =torch.stack(ego_feature_list)

        # features["camera_feature_for_bev"] = self._get_camera_feature(agent_input)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )
        features["intrinsics"] = self._get_intrinsics_feature(agent_input)
        features["extrinsics"] = self._get_extrinsics_feature(agent_input)
        features["future_egomotion"] = self._get_future_egomotion(agent_input)

        return features
    
    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        h, w = self._config.IMAGE_FINAL_DIM
        normalise_image = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ]
        )

        new_cameras = []
        # for i in range(s):
        group_tensors = []
        i = -1  # default 3
        # 读取8个视角图片
        camera_views = [
            agent_input.cameras[i].cam_f0.image,  # f0
            agent_input.cameras[i].cam_l0.image,  # l0
            # agent_input.cameras[i].cam_l1.image,  # l1
            # agent_input.cameras[i].cam_l2.image,  # l2
            agent_input.cameras[i].cam_r0.image,  # r0
            # agent_input.cameras[i].cam_r1.image,  # r1
            # agent_input.cameras[i].cam_r2.image,  # r2
            # agent_input.cameras[i].cam_b0.image  # b0
        ]
        for view in camera_views:
            # view = cv2.resize(view, (w, h))
            view = resize_and_crop_image(
                view, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
            )
            view_tensor = normalise_image(view)
            # resized_view = cv2.resize(view, (w, h))  # 调整到目标大小 (w, h)
            # view_tensor = torch.from_numpy(np.transpose(resized_view, (2, 0, 1)))  # 转换为 tensor，顺序变为 (C, H, W)

            group_tensors.append(view_tensor)
        group_tensors = torch.stack(group_tensors, dim=0)  # (8, C, H, W)

        new_cameras.append(group_tensors)

        new_cameras_tensor = torch.stack(new_cameras, dim=0)  # (s, 8, C, H, W)

        return new_cameras_tensor
    
    def _get_intrinsics_feature(self, agent_input: AgentInput) -> torch.Tensor:
        # b,s,n, 3, 3
        # Combine resize/cropping in the intrinsics
        top_crop = self.augmentation_parameters['crop'][1]
        left_crop = self.augmentation_parameters['crop'][0]
        camera_intrinsics = [
            agent_input.cameras[-1].cam_f0.intrinsics,  # f0
            agent_input.cameras[-1].cam_l0.intrinsics,  # l0
            # agent_input.cameras[-1].cam_l1.intrinsics,  # l1
            # agent_input.cameras[-1].cam_l2.intrinsics,  # l2
            agent_input.cameras[-1].cam_r0.intrinsics,  # r0
            # agent_input.cameras[-1].cam_r1.intrinsics,  # r1
            # agent_input.cameras[-1].cam_r2.intrinsics,  # r2
            # agent_input.cameras[-1].cam_b0.intrinsics  # b0
        ]
        intrinsics = []
        for intrinsic in camera_intrinsics:
            intrinsic = torch.Tensor(intrinsic)
            intrinsic = update_intrinsics(
                intrinsic, top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )
            # need to do Combine resize/cropping in the intrinsics
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))

        intrinsics = torch.cat(intrinsics, dim=1)
        return intrinsics

    def _get_extrinsics_feature(self, agent_input: AgentInput) -> torch.Tensor:

        camera_sensor2lidar_rotation = [
            agent_input.cameras[-1].cam_f0.sensor2lidar_rotation,  # f0
            agent_input.cameras[-1].cam_l0.sensor2lidar_rotation,  # l0
            # agent_input.cameras[-1].cam_l1.sensor2lidar_rotation,  # l1
            # agent_input.cameras[-1].cam_l2.sensor2lidar_rotation,  # l2
            agent_input.cameras[-1].cam_r0.sensor2lidar_rotation,  # r0
            # agent_input.cameras[-1].cam_r1.sensor2lidar_rotation,  # r1
            # agent_input.cameras[-1].cam_r2.sensor2lidar_rotation,  # r2
            # agent_input.cameras[-1].cam_b0.sensor2lidar_rotation  # b0
        ]
        camera_sensor2lidar_translation = [
            agent_input.cameras[-1].cam_f0.sensor2lidar_translation,  # f0
            agent_input.cameras[-1].cam_l0.sensor2lidar_translation,  # l0
            # agent_input.cameras[-1].cam_l1.sensor2lidar_translation,  # l1
            # agent_input.cameras[-1].cam_l2.sensor2lidar_translation,  # l2
            agent_input.cameras[-1].cam_r0.sensor2lidar_translation,  # r0
            # agent_input.cameras[-1].cam_r1.sensor2lidar_translation,  # r1
            # agent_input.cameras[-1].cam_r2.sensor2lidar_translation,  # r2
            # agent_input.cameras[-1].cam_b0.sensor2lidar_translation  # b0
        ]
        extrinsic_matrices = []
        for rotation, translation in zip(camera_sensor2lidar_rotation, camera_sensor2lidar_translation):
            extrinsic = torch.eye(4)
            extrinsic[:3, :3] = torch.from_numpy(rotation)
            extrinsic[:3, 3] = torch.from_numpy(translation)
            extrinsic_matrices.append(extrinsic.unsqueeze(0).unsqueeze(0))
        extrinsic_matrices = torch.cat(extrinsic_matrices, dim=1)

        return extrinsic_matrices
    
    def _get_future_egomotion(self, agent_input: AgentInput) -> torch.Tensor:
        # 计算自车的未来运动 (future egomotion), 此处为当前时刻
        s = 1
        future_egomotions = []
        for idx in range(0, 1):
            rec = agent_input.ego_statuses[s-1]
            future_egomotion = get_future_egomotion(agent_input, rec, idx, s)
            # future_egomotion = torch.from_numpy(future_egomotion)
            future_egomotions.append(future_egomotion)
        future_egomotions = torch.stack(future_egomotions, dim=0)
        return future_egomotions

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)


class PadTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config: PadConfig):
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "pad_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = torch.tensor(
            scene.get_future_trajectory(
                num_trajectory_frames=self._config.trajectory_sampling.num_poses
            ).poses
        )
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        agent_states, agent_labels = self._compute_agent_targets(annotations)
        bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)

        return {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            "bev_semantic_map": bev_semantic_map,
            "token":scene.scene_metadata.initial_token
        }

    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float, config: PadConfig) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (
                config.lidar_min_y <= y <= config.lidar_max_y
            )

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[0],
                box[1],
                box[6],
                box[3],
                box[4],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(
                    np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
                )

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., slice(0, 1 + 1)], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(
                    map_object.baseline_path.linestring, ego_pose
                )
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(
        self, annotations: Annotations, layers: TrackedObjectType
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)#128,256
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _query_map_objects(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self, layers=layers
        )
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)


class BoundingBox2DIndex(IntEnum):

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_")
            and not attribute.startswith("__")
            and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
