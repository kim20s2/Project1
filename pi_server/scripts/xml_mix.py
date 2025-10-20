import os
import xml.etree.ElementTree as ET

def parse_frames(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    frames = []
    for frame in root.findall("frame"):
        elapsed = float(frame.attrib["elapsed_seconds"])
        frames.append((elapsed, frame))
    return frames

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xml_mixed = os.path.join(root, "srv_tmp", "xml", "log.xml")
xml1 = os.path.join(root,"srv_tmp", "xml" ,"f_log.xml")
xml2 = os.path.join(root,"srv_tmp", "xml", "p_log.xml")

# 두 파일에서 frame 추출
frames1 = parse_frames(xml1)
frames2 = parse_frames(xml2)

# 합치고 elapsed_seconds 기준 정렬
all_frames = frames1 + frames2
all_frames.sort(key=lambda x: x[0])

# 새로운 루트 생성
root = ET.Element("merged_session")

for elapsed, frame in all_frames:
    root.append(frame)

# XML 트리 저장
tree = ET.ElementTree(root)
tree.write(xml_mixed, encoding="utf-8", xml_declaration=True)
