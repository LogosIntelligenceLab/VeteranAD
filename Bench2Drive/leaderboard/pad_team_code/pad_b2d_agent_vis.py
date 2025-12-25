import json
import cv2
import torch
import numpy as np
from PIL import Image
import copy
import matplotlib.cm as cm

from pad_team_code.pad_b2d_agent import padAgent
import math

def get_entry_point():
    return 'padvisAgent'


class padvisAgent(padAgent):
    def save(self, tick_data, ego_traj, result=None):
        frame = self.step
        imgs_with_box = {}

        imgs_with_box['bev'] = tick_data['bev']

        imgs_with_box['CAM_FRONT'] = tick_data['imgs']['CAM_FRONT']
        imgs_with_box['CAM_FRONT_LEFT']=tick_data['imgs']['CAM_FRONT_LEFT']
        imgs_with_box['CAM_FRONT_RIGHT']=tick_data['imgs']['CAM_FRONT_RIGHT']

        proposals = result['proposals'].cpu().numpy()[0, :, :, :2]
        pdm_score = result["pdm_score"].cpu().numpy()[0]

        initial_proposals = result['proposal_list'][0].cpu().numpy()[0, :, :, :2]

        if result["pred_agents_states"] is not None:
            pred_agents_states = result["pred_agents_states"][0]

            col_agents_states = pred_agents_states[:,:,:, 0]

            col_agents_conners = col_agents_states[..., :-1].reshape(-1, 6, 4, 2).cpu().numpy()
            col_agents_label = torch.sigmoid(col_agents_states[..., -1:]).reshape(-1, 6).cpu().numpy()

            ttc_agent_states = pred_agents_states[:,:,:, 1]
            ttc_agent_conners = ttc_agent_states[..., :-1].reshape(-1, 6, 4, 2).cpu().numpy()
            ttc_agent_label = torch.sigmoid(ttc_agent_states[..., -1:]).reshape(-1, 6).cpu().numpy()

        if result["pred_area_logit"] is not None:
            pred_area_prob = torch.sigmoid(result["pred_area_logit"][0].reshape(-1, 6, 2)).cpu().numpy()

            pred_road = pred_area_prob[:, :, 0]
            pred_route = pred_area_prob[:, :, 1]
            imgs_with_box['bev1'] = tick_data['bev'].copy()
            imgs_with_box['bev2'] = tick_data['bev'].copy()

        pdm_score = pdm_score * 0.7 + 0.3

        for i in range(len(proposals)):
            proposal_i = proposals[i]
            initial_proposals_i = initial_proposals[i]
            alpha = pdm_score[i]
            color = cm.Reds(alpha)
            rgb_color_tuple = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

            imgs_with_box['bev'] = self.draw_traj_bev(initial_proposals_i, imgs_with_box['bev'], color=(237, 201, 72),
                                                      alpha=0.5, thickness=1, is_ego=True)
            imgs_with_box['CAM_FRONT'] = self.draw_traj(initial_proposals_i, imgs_with_box['CAM_FRONT'],
                                                        self.lidar2img['CAM_FRONT'], color=(237, 201, 72), alpha=0.5,
                                                        thickness=1, is_ego=True)

            imgs_with_box['bev'] = self.draw_traj_bev(proposal_i, imgs_with_box['bev'], color=rgb_color_tuple, alpha=1,
                                                      thickness=1, is_ego=True)
            imgs_with_box['CAM_FRONT'] = self.draw_traj(proposal_i, imgs_with_box['CAM_FRONT'],
                                                        self.lidar2img['CAM_FRONT'], color=rgb_color_tuple, alpha=1,
                                                        thickness=1, is_ego=True)

            if "CAM_FRONT_LEFT" in imgs_with_box.keys():
                imgs_with_box['CAM_FRONT_LEFT'] = self.draw_traj(initial_proposals_i, imgs_with_box['CAM_FRONT_LEFT'],
                                                                 self.lidar2img['CAM_FRONT_LEFT'], color=(237, 201, 72),
                                                                 alpha=0.5, thickness=1, is_ego=False)
                imgs_with_box['CAM_FRONT_RIGHT'] = self.draw_traj(initial_proposals_i, imgs_with_box['CAM_FRONT_RIGHT'],
                                                                  self.lidar2img['CAM_FRONT_LEFT'],
                                                                  color=(237, 201, 72), alpha=0.5, thickness=1,
                                                                  is_ego=False)
                imgs_with_box['CAM_FRONT_LEFT'] = self.draw_traj(proposal_i, imgs_with_box['CAM_FRONT_LEFT'],
                                                                 self.lidar2img['CAM_FRONT_LEFT'],
                                                                 color=rgb_color_tuple, alpha=1, thickness=1,
                                                                 is_ego=False)
                imgs_with_box['CAM_FRONT_RIGHT'] = self.draw_traj(proposal_i, imgs_with_box['CAM_FRONT_RIGHT'],
                                                                  self.lidar2img['CAM_FRONT_LEFT'],
                                                                  color=rgb_color_tuple, alpha=1, thickness=1,
                                                                  is_ego=False)

            if result["pred_area_logit"] is not None:
                imgs_with_box['bev1'] = self.draw_point_bev(proposal_i, imgs_with_box['bev1'], prob=pred_road[i],
                                                            alpha=1, thickness=1, is_ego=True)
                imgs_with_box['bev2'] = self.draw_point_bev(proposal_i, imgs_with_box['bev2'], prob=pred_route[i],
                                                            alpha=1, thickness=1, is_ego=True)

            if result["pred_agents_states"] is not None:

                col_agents_label_i = col_agents_label[i]
                ttc_agent_label_i = ttc_agent_label[i]

                if col_agents_label_i.max() > 0.8:
                    color = cm.Blues(alpha)
                    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

                    imgs_with_box['bev'] = self.draw_lidar_bbox3d_on_img(col_agents_conners[i], imgs_with_box['bev'],
                                                                         self.coor2topdown, scores=col_agents_label_i,
                                                                         color=color, alpha=1, canvas_size=(512, 512))

                if ttc_agent_label_i.max() > 0.8:
                    color = cm.Oranges(alpha)
                    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

                    imgs_with_box['bev'] = self.draw_lidar_bbox3d_on_img(ttc_agent_conners[i], imgs_with_box['bev'],
                                                                         self.coor2topdown, scores=ttc_agent_label_i,
                                                                         color=color, alpha=1, canvas_size=(512, 512))

        imgs_with_box['bev'] = self.draw_traj_bev(ego_traj, imgs_with_box['bev'], color=(222, 112, 97), is_ego=True)
        imgs_with_box['CAM_FRONT'] = self.draw_traj(ego_traj, imgs_with_box['CAM_FRONT'], self.lidar2img['CAM_FRONT'],
                                                    color=(222, 112, 97), is_ego=True)

        if "CAM_FRONT_LEFT" in imgs_with_box.keys():
            imgs_with_box['CAM_FRONT_LEFT'] = self.draw_traj(ego_traj, imgs_with_box['CAM_FRONT_LEFT'],
                                                             self.lidar2img['CAM_FRONT_LEFT'], color=(222, 112, 97),
                                                             is_ego=False)
            imgs_with_box['CAM_FRONT_RIGHT'] = self.draw_traj(ego_traj, imgs_with_box['CAM_FRONT_RIGHT'],
                                                              self.lidar2img['CAM_FRONT_RIGHT'], color=(222, 112, 97),
                                                              is_ego=False)

        for cam, img in imgs_with_box.items():
            Image.fromarray(img).save(self.save_path / str.lower(cam).replace('cam', 'rgb') / ('%04d.png' % frame))

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()
    
    def draw_traj(self, traj, raw_img, lidar2img_rt, canvas_size=(900, 1600), thickness=3, is_ego=True, hue_start=120,
                  hue_end=80, alpha=1, color=None):
        line = traj
        img = raw_img.copy()
        pts_4d = np.stack([line[:, 0], line[:, 1], np.ones((line.shape[0])) * (-1.84), np.ones((line.shape[0]))])
        pts_2d = ((lidar2img_rt @ pts_4d).T)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        mask = (pts_2d[:, 0] > 0) & (pts_2d[:, 0] < canvas_size[1]) & (pts_2d[:, 1] > 0) & (
                    pts_2d[:, 1] < canvas_size[0])
        if not mask.any():
            return img
        pts_2d = pts_2d[mask, 0:2]

        if is_ego:
            pts_2d = np.concatenate([np.array([[800, 900]]), pts_2d], axis=0)
        smoothed_pts = pts_2d.astype(int)
        num_points = len(smoothed_pts)
        for i in range(num_points - 1):
            cv2.line(img, (smoothed_pts[i, 0], smoothed_pts[i, 1]), (smoothed_pts[i + 1, 0], smoothed_pts[i + 1, 1]),
                     color=color, thickness=thickness)
            cv2.circle(img, (smoothed_pts[i + 1, 0], smoothed_pts[i + 1, 1]), thickness + 1, color, -1)
            if thickness == 3:
                cv2.circle(img, (smoothed_pts[i + 1, 0], smoothed_pts[i + 1, 1]), thickness + 2, (0, 0, 0), 0)
        return img

    def draw_point_bev(self, traj, raw_img, canvas_size=(512, 512), thickness=3, is_ego=False, hue_start=120,
                       hue_end=80, alpha=1, prob=None):
        line = traj
        img = raw_img.copy()
        pts_4d = np.stack([line[:, 0], line[:, 1], np.zeros((line.shape[0])), np.ones((line.shape[0]))])
        pts_2d = (self.coor2topdown @ pts_4d).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        mask = (pts_2d[:, 0] > 0) & (pts_2d[:, 0] < canvas_size[1]) & (pts_2d[:, 1] > 0) & (
                    pts_2d[:, 1] < canvas_size[0])
        if not mask.any():
            return img
        # pts_2d = pts_2d[mask,0:2]

        # try:
        #     tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
        # except:
        #     return img
        # unew = np.linspace(0, 1, 100)
        # smoothed_pts = np.stack(splev(unew, tck)).astype(int).T

        smoothed_pts = pts_2d.astype(int)

        num_points = len(smoothed_pts)
        for i in range(num_points):
            if mask[i]:
                color = cm.Reds(prob[i])
                rgb_color_tuple = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

                cv2.circle(img, (smoothed_pts[i, 0], smoothed_pts[i, 1]), 1, rgb_color_tuple, -1)

        return img

    def draw_traj_bev(self, traj, raw_img, canvas_size=(512, 512), thickness=3, is_ego=False, hue_start=120, hue_end=80,
                      alpha=1, color=None):
        if is_ego:
            line = np.concatenate([np.zeros((1, 2)), traj], axis=0)
        else:
            line = traj
        img = raw_img.copy()
        pts_4d = np.stack([line[:, 0], line[:, 1], np.zeros((line.shape[0])), np.ones((line.shape[0]))])
        pts_2d = (self.coor2topdown @ pts_4d).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        mask = (pts_2d[:, 0] > 0) & (pts_2d[:, 0] < canvas_size[1]) & (pts_2d[:, 1] > 0) & (
                    pts_2d[:, 1] < canvas_size[0])
        if not mask.any():
            return img
        pts_2d = pts_2d[mask, 0:2]

        # try:
        #     tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
        # except:
        #     return img
        # unew = np.linspace(0, 1, 100)
        # smoothed_pts = np.stack(splev(unew, tck)).astype(int).T

        smoothed_pts = pts_2d.astype(int)

        num_points = len(smoothed_pts)
        for i in range(num_points - 1):
            # hue = hue_start + (hue_end - hue_start) * (i / num_points)
            # hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
            # rgb_color = cv2.cvtColor(hsv_color[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB).reshape(-1)
            # rgb_color_tuple = (float(rgb_color[0]),float(rgb_color[1]),float(rgb_color[2]))
            if smoothed_pts[i, 0] > 0 and smoothed_pts[i, 0] < canvas_size[1] and smoothed_pts[i, 1] > 0 and \
                    smoothed_pts[i, 1] < canvas_size[0]:
                cv2.line(img, (smoothed_pts[i, 0], smoothed_pts[i, 1]),
                         (smoothed_pts[i + 1, 0], smoothed_pts[i + 1, 1]), color=color, thickness=thickness)
                cv2.circle(img, (smoothed_pts[i + 1, 0], smoothed_pts[i + 1, 1]), thickness + 1, color, -1)
                if thickness == 3:
                    cv2.circle(img, (smoothed_pts[i + 1, 0], smoothed_pts[i + 1, 1]), thickness + 2, (0, 0, 0), 0)

            # elif i==0:
            #     break

        img = cv2.addWeighted(img.astype(np.uint8), alpha, raw_img, 1 - alpha, 0)

        return img

    def draw_lidar_bbox3d_on_img(self, corners_3d, raw_img, lidar2img_rt, canvas_size=(900, 1600), scores=None,
                                 labels=None, color=(0, 255, 0), alpha=1, thickness=1):

        # print(scores)
        img = raw_img.copy()
        corners_3d = np.concatenate([corners_3d, np.zeros_like(corners_3d[:, :, :1])], axis=-1)  # 6,4,3

        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate(
            [corners_3d.reshape(-1, 3),
             np.ones((num_bbox * 4, 1))], axis=-1)
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        pts_2d = (lidar2img_rt @ pts_4d.T).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 4, 2)
        depth = pts_2d[..., 2].reshape(num_bbox, 4)
        mask1 = ((imgfov_pts_2d[:, :, 0] > -1e5) & (imgfov_pts_2d[:, :, 0] < 1e5) & (imgfov_pts_2d[:, :, 1] > -1e5) & (
                    imgfov_pts_2d[:, :, 1] < 1e5) & (depth > -1)).all(-1)
        mask2 = (imgfov_pts_2d.reshape(num_bbox, 8).max(axis=-1) - imgfov_pts_2d.reshape(num_bbox, 8).min(
            axis=-1)) < 2000
        mask = mask1 & mask2
        if scores is not None:
            mask3 = (scores >= 0.5)
            mask = mask & mask3

        if not mask.any():
            return img

        scores = scores[mask] if scores is not None else None

        imgfov_pts_2d = imgfov_pts_2d[mask]
        num_bbox = mask.sum()

        self.plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, scores, labels, color, thickness,
                                bev=(canvas_size != (900, 1600)))

        # img = cv2.addWeighted(img.astype(np.uint8), alpha, raw_img, 1 - alpha, 0)

        return img

    def plot_rect3d_on_img(self, img, num_rects, rect_corners, scores=None, labels=None, color=(0, 255, 0), thickness=1,
                           bev=False):
        line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                        (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
        if bev:
            line_indices = ((0, 1), (1, 2), (2, 3), (3, 0))
        for i in range(num_rects):
            thinck = 1
            corners = rect_corners[i].astype(np.int)
            # if scores is not None:
            #     cv2.putText(img, "{:.2f}".format(scores[i]), corners[0], cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
            if scores[i] < 0.5:
                continue
                #     c=(255,255,255)
                #     thinck=1
            for start, end in line_indices:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                         (corners[end, 0], corners[end, 1]), color, thinck,
                         cv2.LINE_AA)
        return img.astype(np.uint8)


