'''
  @ Date: 2021-04-21 15:19:21
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-28 11:55:27
  @ FilePath: /EasyMocapRelease/easymocap/mytools/reader.py
'''
# function to read data
"""
    This class provides:
                  |  write  | vis
    - keypoints2d |    x    |  o
    - keypoints3d |    x    |  o
    - smpl        |    x    |  o
"""
import numpy as np
import os
from os.path import join
from glob import glob
from .file_utils import read_json, read_annot

def read_keypoints2d(filename, mode):
    return read_annot(filename, mode)

def read_keypoints3d(filename):
    data = read_json(filename)
    res_ = []
    for d in data:
        pid = d['id'] if 'id' in d.keys() else d['personID']
        pose3d = np.array(d['keypoints3d'])
        if pose3d.shape[0] > 25:
            # 对于有手的情况，把手的根节点赋值成body25上的点
            pose3d[25, :] = pose3d[7, :]
            pose3d[46, :] = pose3d[4, :]
        if pose3d.shape[1] == 3:
            pose3d = np.hstack([pose3d, np.ones((pose3d.shape[0], 1))])
        res_.append({
            'id': pid,
            'keypoints3d': pose3d
        })
    return res_

def read_smpl(filename):
    datas = read_json(filename)
    outputs = []
    for data in datas:
        for key in ['Rh', 'Th', 'poses', 'shapes', 'expression']:
            if key in data.keys():
                data[key] = np.array(data[key])
        # for smplx results
        outputs.append(data)
    return outputs

def read_keypoints3d_a4d(outname):
    res_ = []
    with open(outname, "r") as file:
        lines = file.readlines()
        if len(lines) < 2:
            return res_
        nPerson, nJoints = int(lines[0]), int(lines[1])
        # 只包含每个人的结果
        lines = lines[1:]
        # 每个人的都写了关键点数量
        line_per_person = 1 + 1 + nJoints
        for i in range(nPerson):
            trackId = int(lines[i*line_per_person+1])
            content = ''.join(lines[i*line_per_person+2:i*line_per_person+2+nJoints])
            pose3d = np.fromstring(content, dtype=float, sep=' ').reshape((nJoints, 4))
            # association4d 的关节顺序和正常的定义不一样
            pose3d = pose3d[[4, 1, 5, 9, 13, 6, 10, 14, 0, 2, 7, 11, 3, 8, 12], :]
            res_.append({'id':trackId, 'keypoints3d':np.array(pose3d)})
    return res_

def read_keypoints3d_all(path, key='keypoints3d', pids=[]):
    assert os.path.exists(path), '{} not exists!'.format(path)
    results = {}
    filenames = sorted(glob(join(path, '*.json')))
    for filename in filenames:
        nf = int(os.path.basename(filename).replace('.json', ''))
        datas = read_keypoints3d(filename)
        for data in datas:
            pid = data['id']
            if len(pids) > 0 and pid not in pids:
                continue
            # 注意 这里没有考虑从哪开始的
            if pid not in results.keys():
                results[pid] = {key: [], 'frames': []}
            results[pid][key].append(data[key])
            results[pid]['frames'].append(nf)
    if key == 'keypoints3d':
        for pid, result in results.items():
            result[key] = np.stack(result[key])
    return results, filenames