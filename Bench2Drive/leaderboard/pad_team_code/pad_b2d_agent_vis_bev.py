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
        colors_rgb = [
            (237, 201, 72),   # #EDC948
            (242, 151, 58),   # #F2973A
            (247, 94, 44),    # #F75E2C
            (255, 36, 30)     # #FF241E
        ]

        for k in range(4):
            proposals = result['proposal_list'][k].cpu().numpy()[0, :, :, :2]
            imgs_with_box['bev'+str(k)] = tick_data['bev'].copy()

            for i in range(len(proposals)):
                proposal_i = proposals[i]

                imgs_with_box['bev'+str(k)] = self.draw_traj_bev(proposal_i, imgs_with_box['bev'+str(k)], color=colors_rgb[k],
                                                        alpha=0.5, thickness=1, is_ego=True)


        for cam, img in imgs_with_box.items():
            Image.fromarray(img).save(self.save_path / str.lower(cam).replace('cam', 'rgb') / ('%04d.png' % frame))

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
