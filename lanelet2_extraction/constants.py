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
