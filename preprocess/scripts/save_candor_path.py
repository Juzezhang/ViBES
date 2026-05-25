import os
from os.path import join as pjoin
import json


candor_structure = {}
seq_list = os.listdir(pjoin('/path/to/CANDOR/', 'FLAME_coeffs'))
for seq in seq_list:
    video_list = os.listdir(pjoin('/path/to/CANDOR/', 'FLAME_coeffs', seq))
    candor_structure[seq] = []
    for video in video_list:
        if video.endswith('.npz'):
            candor_structure[seq].append(video)
            continue
        else:
            print(video)


with open(pjoin('/path/to/CANDOR/', 'candor_structure.json'), 'w') as f:
    json.dump(candor_structure, f)

pass
