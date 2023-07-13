import json
import numpy as np
from pathlib import Path
import warnings
import lanelet2
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R

from .labeled_polylines import LabeledPolylines
from .utils import load_lanelet2

warnings.filterwarnings("ignore")

COLOR_MAPS_RGB = {
    'divider': (255, 0, 0),
    'boundary': (0, 255, 0),
    'ped_crossing': (0, 0, 255),
    'stop_line': (255, 255, 0),
    'centerline': (255, 183, 51),
    'drivable_area': (255, 255, 171),
    'others': (128, 128, 128),  # gray
    'contours': (51, 255, 255),  # yellow
}

LINE_TYPE_MAPPING = {
    # "WAY_TYPE/RELATIONS_SUBTYPE": ID
    # Road
    "line_thin:road": 1,
    "line_thick:road": 1,
    "pedestrian_marking:road": -1,
    "road_border:road": 2,
    "virtual:road": -1,

    # Crosswalk
    # "line_thin:crosswalk": 3,
    # "line_thick:crosswalk": 3,
    # "pedestrian_marking:crosswalk": 3,
    # "road_border:crosswalk": 3,
    # "virtual:crosswalk": 3,
    "line_thin:crosswalk": -1,
    "line_thick:crosswalk": -1,
    "pedestrian_marking:crosswalk": -1,
    "road_border:crosswalk": -1,
    "virtual:crosswalk": -1,

    # Walkway
    "line_thin:walkway": -1,
    "line_thick:walkway": -1,
    "pedestrian_marking:walkway": -1,
    "road_boader:walkway": -1,
    "virtual:walkway": -1,

    # Stop_line not supported yet, since some of them are not associated to relation, making it difficult to extract
    # Traffic light
    "stop_line:traffic_light": 0,

    # None
    "stop_line:none": 0,
    "road_border:none": 2,
    "curbstone:none": 2,
}

CLASS2LABEL = {
    'stop_line': 0,
    'divider': 1,
    'contours': 2,
    # 'ped_crossing': 3,
    'others': -1,
}

def get_possible_map_names_from_dataset(dataroot):
    metadata_path = Path(dataroot) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    map_name_set = set()
    for val in metadata.values():
        map_name_set.add(val['map_name'])
    return map_name_set

class Lanelet2Extractor(object):
    def __init__(self,
                 maproot='/mount_hdd/data/maps/',
                 dataroot='',
                 extract_distance=250,
                 roi_size=(60, 30),
                 line_type_mapping=LINE_TYPE_MAPPING,
                 class2label=CLASS2LABEL):
        super().__init__()

        map_name_list = get_possible_map_names_from_dataset(dataroot)
        self.lanelet2_maps = {}
        for map_name in map_name_list:
            map_path = Path(maproot) / map_name / 'lanelet2_map.osm'
            self.lanelet2_maps[map_name] = load_lanelet2(map_path)
            print(f'Finished loading map {map_path}')

        self.roi_size = roi_size
        self.size = np.array([self.roi_size[0], self.roi_size[1]]) + 2
        self.line_type_mapping = line_type_mapping
        self.label2class = {v:k for k, v in class2label.items()}

    def extract(self, pose: List[float], map_name: str, normalize: bool) -> LabeledPolylines:
        assert len(pose) == 7, "pose should be [px, py, pz, qx, qy, qz, qw]"

        lanelet2_map = self.lanelet2_maps[map_name]

        labeled_polylines = self.extract_lanelets(lanelet2_map, self.roi_size, pose, normalize=normalize)

        if labeled_polylines.empty():
            print(f'No lanelet2 found at pose = ({pose[0]}, {pose[1]})')
        return labeled_polylines

    def extract_lanelets(self, lanelet2_map, roi_size, pose, normalize) -> LabeledPolylines:
        distance = np.max(roi_size) / 2
        labeled_polylines = LabeledPolylines()

        for line_string in lanelet2_map.lineStringLayer:
            # Check if the line type is of interest
            way_type = line_string.attributes['type']
            relation_subtype = get_relation_subtype(lanelet2_map, line_string)
            line_type = f'{way_type}:{relation_subtype}'
            if line_type in self.line_type_mapping.keys():
                if self.line_type_mapping[line_type] == -1:
                    continue
            else:
                continue
            label = self.line_type_mapping[line_type]
            polyline_all = line_string2list3d(line_string)

            polylines = extract_near_polylines(pose, polyline_all, distance)

            for polyline in polylines:
                polyline_local = global2local(polyline, pose)
                polylines_local_filtered = filter_polyline_by_roi_size(polyline_local, roi_size)
                if normalize:
                    polylines_local_filtered = [normalize_line(p, roi_size) for p in polylines_local_filtered]
                for polyline_local_filtered in polylines_local_filtered:

                    labeled_polylines.add_polyline(label, np.array(polyline_local_filtered))
        return labeled_polylines


def normalize_line(line, roi_size):
    '''
        prevent extrime pts such as 0 or 1. 
    '''
    size = np.array([roi_size[0], roi_size[1]]) + 2
    
    origin = -np.array([roi_size[0]/2, roi_size[1]/2])
    # for better learning
    line = np.array(line)
    line[:, :2] = line[:, :2] - origin
    line[:, :2] = line[:, :2] / size
    line = line.tolist()

    return line

def filter_polyline_by_roi_size(polyline: List[List[float]], roi_size: Tuple[int, int], debug=False) -> List[List[List[float]]]:
    polylines_filtered = []
    polyline_tmp = []
    if debug:
        print(roi_size)
    for i in range(len(polyline)):
        point_within_area = abs(polyline[i][0]) < roi_size[0] / 2 and abs(polyline[i][1]) < roi_size[1] / 2
        if i == 0:
            prev_point_within_area = point_within_area

        cur_point = polyline[i]
        prev_point = None if i == 0 else polyline[i - 1]

        if point_within_area and prev_point_within_area: # o-o -> add current point
            polyline_tmp.append(cur_point)
            if debug:
                print(f'{cur_point}: o-o')
        elif (not point_within_area) and prev_point_within_area and prev_point is not None: # o-x -> add interval point
            interval_point = rectangle_line_segment_intersection(prev_point, cur_point, roi_size)[0]
            polyline_tmp.append(interval_point)
            polylines_filtered.append(polyline_tmp)
            if debug:
                print('here!1', len(polyline_tmp))
            polyline_tmp = []
            if debug:
                print(f'{cur_point}: o-x')
        elif point_within_area and (not prev_point_within_area) and prev_point is not None: # x-o -> add interval point and current point
            interval_point = rectangle_line_segment_intersection(prev_point, cur_point, roi_size)[0]
            polyline_tmp.append(interval_point)
            polyline_tmp.append(cur_point)
            if debug:
                print(f'{cur_point}: x-o')
        elif (not point_within_area) and (not prev_point_within_area) and prev_point is not None: # x-x -> add nothing
            interval_points = rectangle_line_segment_intersection(prev_point, cur_point, roi_size)
            if len(interval_points) == 2:
                polylines_filtered.append(interval_points)
            if debug:
                print(f'{cur_point}: x-x')

        if i == len(polyline) - 1 and point_within_area and len(polyline_tmp) > 0:
            polylines_filtered.append(polyline_tmp)
            if debug:
                print('here!3', len(polyline_tmp))
            polyline_tmp = []
        prev_point_within_area = point_within_area
    return polylines_filtered

def line_segment_intersection(p0, p1, q0, q1):
    d = (p1[0] - p0[0]) * (q1[1] - q0[1]) - (p1[1] - p0[1]) * (q1[0] - q0[0])
    if d == 0:
        return None

    t = ((q0[0] - p0[0]) * (q1[1] - q0[1]) - (q0[1] - p0[1]) * (q1[0] - q0[0])) / d
    u = ((q0[0] - p0[0]) * (p1[1] - p0[1]) - (q0[1] - p0[1]) * (p1[0] - p0[0])) / d

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection = np.array(p0) + t * (np.array(p1) - np.array(p0))
        return intersection
    else:
        return None

def rectangle_line_segment_intersection(p0, p1, roi_size):
    A, B = roi_size[0] / 2, roi_size[1] / 2
    rectangle_edges = [
        (np.array([-A, -B]), np.array([A, -B])),
        (np.array([A, -B]), np.array([A, B])),
        (np.array([A, B]), np.array([-A, B])),
        (np.array([-A, B]), np.array([-A, -B])),
    ]

    intersections = []
    for q0, q1 in rectangle_edges:
        intersection = line_segment_intersection(p0, p1, q0, q1)
        if intersection is not None:
            intersections.append(intersection)

    # Sort intersections by distance to p0
    intersections.sort(key=lambda point: np.linalg.norm(point - np.array(p0)))
    return intersections

def extract_near_polylines(pose, polyline_all, distance):
    polylines = []
    polyline_to_add = []
    prev_within_area = False
    for i, p in enumerate(polyline_all):
        within_area = (p[0] - pose[0]) ** 2 + (p[1] - pose[1]) ** 2 < distance ** 2
        if within_area:
            polylines.append(polyline_all)
            break
    return polylines

def global2local(polyline: List[List[float]], pose: List[float]) -> List[List[float]]:
    """Transform a polyline from global coordinates to local coordinates based on a given pose.
    
    Args:
        polyline: A list of 3D points in global coordinates. E.g. [[x, y, z], [x, y, z], ...]
        pose: A list containing the position [px, py, pz] and orientation [qx, qy, qz, qw] of the local frame in global coordinates.
    
    Returns:
        polyline_local: A list of 3D points in local coordinates. E.g. [[x, y, z], [x, y, z], ...]
    """
    position = np.array(pose[:3])
    orientation = R.from_quat(pose[3:])

    polyline_local = []
    for point in polyline:
        point_global = np.array(point)
        point_local = orientation.inv().apply(point_global - position)
        polyline_local.append(point_local.tolist())

    return polyline_local

def get_relation_subtype(lmap, line_string):
    usages = [
        lanelet.attributes['subtype'] for lanelet in lmap.laneletLayer.findUsages(line_string)
    ]
    if 'crosswalk' in usages:
        return 'crosswalk'
    elif 'road' in usages:
        return 'road'
    elif 'walkway' in usages:
        return 'walkway'
    elif 'road_shoulder' in usages:
        return 'road_shoulder'
    elif len(usages) == 0:
        return 'none'
    else:
        print(f"ERROR! Usage {usages} not found")
        raise NotImplementedError

def line_string2list3d(line_string: lanelet2.core.LineString3d) -> List[List[float]]:
    result = []
    for p in line_string:
        result.append([p.x, p.y, p.z])
    return result
