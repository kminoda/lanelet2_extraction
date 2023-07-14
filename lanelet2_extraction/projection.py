import cv2
import copy
import numpy as np
from typing import Dict
import av2.geometry.interpolate as interp_utils

from .labeled_polylines import LabeledPolylines
from .constants import COLOR_MAPS_RGB, CLASS2LABEL


def project_lanelet2_on_image(
    img: np.ndarray,
    labeled_polylines: LabeledPolylines,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    thickness: int = 2,
    color_maps_rgb: dict = COLOR_MAPS_RGB,
    class2label: dict = CLASS2LABEL,
):
    id2category_name = {v: k for k, v in class2label.items()}
    ego2img = _get_ego2img(extrinsic, intrinsic)
    for label in labeled_polylines.get_all_labels():
        category = id2category_name[label]
        color = color_maps_rgb[category]
        for vector in labeled_polylines.get_polylines(label):
            img = np.ascontiguousarray(img)
            _draw_polyline_ego_on_img(vector, img, ego2img, color, thickness)    
    return img

def _get_ego2img(extrinsics_mat, intrinsics_mat):
    ego2cam_rt = extrinsics_mat
    viewpad = np.eye(4)
    viewpad[:intrinsics_mat.shape[0], :intrinsics_mat.shape[1]] = intrinsics_mat
    ego2cam_rt = (viewpad @ ego2cam_rt)
    return ego2cam_rt

def _draw_polyline_ego_on_img(polyline_ego, img_bgr, ego2img, color_bgr, thickness):
    if polyline_ego.shape[1] == 2:
        zeros = np.zeros((polyline_ego.shape[0], 1))
        polyline_ego = np.concatenate([polyline_ego, zeros], axis=1)
    polyline_ego = interp_utils.interp_arc(t=500, points=polyline_ego)

    # uv, depth = points_ego2img(polyline_ego, extrinsics, intrinsics)
    uv, depth = _points_ego2img_from_one_matrix(polyline_ego, ego2img)

    h, w, c = img_bgr.shape

    is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
    is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
    is_valid_z = depth > 0
    is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

    if is_valid_points.sum() == 0:
        return
    
    uv = np.round(uv[is_valid_points]).astype(np.int32)

    _draw_visible_polyline_cv2(
        copy.deepcopy(uv),
        valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
        image=img_bgr,
        color=color_bgr,
        thickness_px=thickness,
    )

def _points_ego2img_from_one_matrix(pts_ego, ego2img):
    pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
    uv = pts_ego_4d @ ego2img.T
    uv = _remove_nan_values(uv)
    depth = uv[:, 2]
    uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)
    return uv, depth

def _remove_nan_values(uv):
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    return uv_valid

def _draw_visible_polyline_cv2(line, valid_pts_bool, image, color, thickness_px):
    """Draw a polyline onto an image using given line segments.

    Args:
        line: Array of shape (K, 2) representing the coordinates of line.
        valid_pts_bool: Array of shape (K,) representing which polyline coordinates are valid for rendering.
            For example, if the coordinate is occluded, a user might specify that it is invalid.
            Line segments touching an invalid vertex will not be rendered.
        image: Array of shape (H, W, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        thickness_px: thickness (in pixels) to use when rendering the polyline.
    """
    line = np.round(line).astype(int)  # type: ignore
    for i in range(len(line) - 1):

        if (not valid_pts_bool[i]) or (not valid_pts_bool[i + 1]):
            continue

        x1 = line[i][0]
        y1 = line[i][1]
        x2 = line[i + 1][0]
        y2 = line[i + 1][1]

        # Use anti-aliasing (AA) for curves
        # print(f'({x1}, {y1}) to ({x2}, {y2})')
        image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv2.LINE_AA)