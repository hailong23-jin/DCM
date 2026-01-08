import json

def read_annotation(ann_path):
    with open(ann_path) as f:
        ann_info = json.load(f)
    return ann_info