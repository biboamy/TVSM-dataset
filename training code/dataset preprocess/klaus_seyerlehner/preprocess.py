import glob
import os
import json
import numpy as np


# process the original labels to our csv version
def process_label(dir_path, des_path):
    if not os.path.exists(des_path):
        os.mkdir(des_path)

    with open('../../data/OpenBMAT/annotations/json/MD_mapping.json') as f:
        data = json.load(f)

    print(data)

#process_label(None, '~/data/OpenBMAT/labels/')