import lanelet2
import xml.etree.ElementTree as ET

def extractNode(child):
    x,y = 0,0
    for tag in child:
        if tag.attrib['k'] == 'local_x':
            x = tag.attrib['v']
        elif tag.attrib['k'] == 'local_y':
            y = tag.attrib['v']
        elif tag.attrib['k'] == 'ele':
            z = tag.attrib['v']
    return int(child.attrib["id"]), float(x), float(y), float(z)

def extractWay(child):
    refs = []
    tags = {}
    for tag in child:
        if tag.tag == "nd":
            refs.append(int(tag.attrib["ref"]))
        elif tag.tag == "tag":
            tags[tag.attrib["k"]] = tag.attrib["v"]
    return int(child.attrib["id"]), refs, tags

def extractRelation(child):
    road = {}
    road_tags = {}
    for tag in child:
        if tag.tag == "member":
            road[tag.attrib["role"]] = int(tag.attrib["ref"])         
        if tag.tag == "tag":
            if tag.attrib["k"] == "type":
                if not tag.attrib["v"] == "lanelet":
                    return None, None
            road_tags[tag.attrib["k"]] = tag.attrib["v"]
    return road, road_tags

def dict2list(dic, key_list):
    output = []
    for key in key_list:
        output.append(dic[key])
    return output

def load_lanelet2(rfile: str) -> lanelet2.core.LaneletMap:
    tree = ET.parse(rfile)
    root = tree.getroot()

    nodes = {}
    ways = {}
    relations = {}
    ways_tags = {}
    relations_tags = {}
    stop_line_way_ids = []
    road_border_way_ids = []

    for child in root:
        if child.tag == "node":
            ref_id, x, y, z = extractNode(child)
            nodes[ref_id] = [x, y, z]
        elif child.tag == "way":
            idx, refs, tags = extractWay(child)
            ways[idx] = refs
            ways_tags[idx] = tags

            ### Exception for stop line ###
            if "type" in tags.keys():
                if tags["type"] == "stop_line":
                    stop_line_way_ids.append(idx)
                if tags["type"] == "road_border":
                    road_border_way_ids.append(idx)
            ### Exception for stop line ###
        elif child.tag == "relation":
            idx = int(child.attrib["id"])
            road, road_tags = extractRelation(child)
            if road:
                direction = road_tags["turn_direction"] if "turn_direction" in road_tags.keys() else None
                relations[idx] = [ road["left"], road["right"], direction]
                relations_tags[idx] = road_tags
    points = {}
    for idx in nodes.keys():
        points[idx] = lanelet2.core.Point3d(idx, *nodes[idx])    

    linestrings = {}
    for idx in ways.keys():
        point_list = ways[idx]
        attribute = lanelet2.core.AttributeMap(ways_tags[idx])
        linestrings[idx] = lanelet2.core.LineString3d(idx, dict2list(points, point_list), attribute)

    lanelets = {}
    lmap = lanelet2.core.LaneletMap()

    for idx in relations.keys():
        left, right, direction = relations[idx]
        attribute = lanelet2.core.AttributeMap(relations_tags[idx])
        lanelet = lanelet2.core.Lanelet(idx, linestrings[left], linestrings[right], attribute)
        lmap.add(lanelet)

    ### Exception for stop line ###
    for stop_line_way_id in stop_line_way_ids:
        lmap.add(linestrings[stop_line_way_id])
    ### Exception for stop line ###
    ### Exception for road border ###
    for road_border_way_id in road_border_way_ids:
        if lmap.lineStringLayer.exists(road_border_way_id):
            continue
        lmap.add(linestrings[road_border_way_id])
    ### Exception for road border ###
    return lmap
