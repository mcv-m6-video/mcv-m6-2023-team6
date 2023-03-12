from collections import defaultdict
import xml.etree.ElementTree as ET
import os


def load_from_xml(path):
    frame_dict = defaultdict(list)
    for event, elem in ET.iterparse(path, events=('start',)):
        if elem.tag == 'track' and elem.attrib.get('label') == 'car':
            for x in (child.attrib for child in elem):
                frame = f"f_{x['frame']}"
                frame_dict[frame].append([float(x['xtl']), float(x['ytl']),
                                          float(x['xbr']), float(x['ybr'])])
    return frame_dict

if __name__ == '__main__':
    # Set the parent directory of your current directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

    # Set the relative path to the XML file
    relative_path = 'datasets/ai_challenge_s03_c010-full_annotation.xml'

    # Get the absolute path of the XML file
    path = os.path.join(parent_dir, relative_path)

    # Print the absolute path
    print(path)
    frame_dict = load_from_xml(path)
    print(frame_dict)
