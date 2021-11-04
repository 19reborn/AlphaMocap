import json
import os
from pickle import FALSE
from networkx.readwrite.pajek import parse_pajek
from torch.nn.modules.module import T
from tqdm import tqdm
from easymocap.mytools.vis_base import draw3d,drawRepro, drawSmpl3d, drawSmplrepro
from easymocap.mytools.camera_utils import read_camera
from easymocap.mytools import Timer
from easymocap.smplmodel import check_keypoints, load_model, select_nf
from easymocap.dataset import CONFIG
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#out = 'E:\study\EasyMocap\datasets\Alphapose/output-track-8.11'
out = 'E:\study\EasyMocap\datasets/test/output'
image_dir = 'E:\study\EasyMocap\datasets\Alphapose\images'
#out = 'E:\study\EasyMocap\datasets/neuralbody/output'
#image_dir = 'E:\study\EasyMocap\datasets/neuralbody\images'
#out = 'E:\study\EasyMocap\datasets\mvmp\output-track'
#image_dir = 'E:\study\EasyMocap\datasets\mvmp\images'

file_path = os.path.join(out,'keypoints3d')
pose3d_path = os.path.join(out,'pose3d')
smpl_repro_path = os.path.join(out,'smpl_repro')
smpl_3d_path = os.path.join(out,'smpl_3d')
os.makedirs(pose3d_path, exist_ok=True)
os.makedirs(smpl_repro_path, exist_ok=True)
os.makedirs(smpl_3d_path, exist_ok=True)


sub_vis = [20]
#sub_vis = None
frames = range(0,150)
draw_3d = False
draw_repro = False
draw_smpl = True
draw_smpl_repro = True
# 读入相机参数
intri_name = os.path.join(image_dir, '..', 'intri.yml')
extri_name = os.path.join(image_dir, '..', 'extri.yml')
if os.path.exists(intri_name) and os.path.exists(extri_name):
    cameras = read_camera(intri_name, extri_name)
    cameras.pop('basenames')
config = CONFIG['body25']
with Timer('Loading {}, {}'.format('smpl', 'nuetral')):
    body_model = load_model(gender='neutral', model_type='smpl')
for frame in tqdm(frames, desc='draw'):
    #results = []
    path = os.path.join(file_path,str(frame).rjust(6,'0')+'.json')
    assert os.path.exists(path)
    with open(path ,'r') as json_file:
        data = json.load(json_file)
    #for people in data:
    #    results.append(people["keypoints3d"])
    if draw_3d:
        draw3d(data, os.path.join(pose3d_path,str(frame).rjust(6,'0')+'.jpg'))
    if draw_repro:
        drawRepro(data, out, frame, image_dir, cameras, config, sub_vis)
    if draw_smpl:
        path = os.path.join(out,'smpl',str(frame).rjust(6,'0')+'.json')
        assert os.path.exists(path)
        with open(path ,'r') as json_file:
            datas = json.load(json_file)    
            outputs = []
            for data in datas:
                for key in ['Rh', 'Th', 'poses', 'shapes', 'expression']:
                    if key in data.keys():
                        data[key] = np.array(data[key])
        # for smplx results
                outputs.append(data)    
        vertices = {}
        for param in outputs:
            params = param.copy()
            params.pop('id')
            vertice = body_model(return_verts=True, return_tensor=False, **params)
            vertices[param['id']] = vertice
        scale = 1.6
        faces = body_model.faces
        render_data = {}
        for pid, data in vertices.items():
            render_data[pid] = {
                'vertices': data[0]*scale, 'faces': faces, 
                'vid': pid, 'name': 'human_{}_{}'.format(frame, pid)}
        drawSmplrepro(render_data, out, frame, image_dir , cameras , sub_vis = sub_vis)


