import cv2
import json
import numpy as np
from lanelet2_extraction import Lanelet2Extractor
from lanelet2_extraction import project_lanelet2_on_image

DATA_ROOT_PATH = '/root/map_tr/data/tier4_vectormap_dataset/all_data'
IMG_PATH = 'data/camera0.png'  # 1675660188245933056_camera0.png
EXTRINSICS_INV_PATH = 'data/extrinsics.json'
INTRINSICS_PATH = 'data/intrinsics.json'
POSE_PATH = 'data/pose.json'
MAP_NAME = 'odaiba-yabloc'

with open(EXTRINSICS_INV_PATH) as f:
    extrinsics_inv = np.array(json.load(f)['data'])
with open(INTRINSICS_PATH) as f:
    intrinsics = np.array(json.load(f)['data'])
with open(POSE_PATH) as f:
    pose = np.array(json.load(f)['pose'])
img = cv2.imread(IMG_PATH)

extrinsics = np.linalg.inv(extrinsics_inv)  # Need to take inverse since the original json is base_link -> map

extractor = Lanelet2Extractor(dataroot=DATA_ROOT_PATH)
extracted_polylines = extractor.extract(pose, MAP_NAME, normalize=False)

img = project_lanelet2_on_image(img, extracted_polylines, extrinsics, intrinsics)

cv2.imwrite('./sample_output.png', img)
