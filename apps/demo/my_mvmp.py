'''
  @ Date: 2021-06-23 16:13:53
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-25 11:52:49
  @ FilePath: /EasyMocapRelease/apps/demo/mvmp.py
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from easymocap.dataset import CONFIG
from easymocap.dataset import CONFIG
from easymocap.affinity.affinity import ComposedAffinity
from easymocap.assignment.associate import simple_associate
from easymocap.assignment.group import PeopleGroup
from easymocap.mytools.vis_base import draw3d
import json
import numpy as np
from easymocap.mytools import Timer
from tqdm import tqdm

def distance_pl(pose2d, pose3d, camera):

    # 将pose3d转换到相机坐标系下
    pose3d = pose3d[:,:3]
    pose3d = (camera['R'].dot(pose3d.T)+camera['T']).T
    focal_x = camera['K'][0][0]
    focal_y = camera['K'][1][1]
    center_x = camera['K'][0][2]
    center_y = camera['K'][1][2]
    dis = 0
    for i in range(len(pose2d)):
        line = np.array([(pose2d[i][0]-center_x)/focal_x,(pose2d[i][1]-center_y)/focal_y, 1.0])
        joint3d = pose3d[i]
        dis += np.linalg.norm(np.cross(line,joint3d))
    dis /= len(pose2d)
    return dis

def all_distance_pl(all_pose2d, pose3d, camera):

    all_dis = np.zeros((len(all_pose2d)))
    all_line = np.ones((len(pose3d),len(all_pose2d),3))
    # 将pose3d转换到相机坐标系下
    all_pose2d = all_pose2d.transpose(1,0,2)
    pose3d = pose3d[:,:3]
    pose3d = (camera['R'].dot(pose3d.T)+camera['T']).T
    focal_x = camera['K'][0][0]
    focal_y = camera['K'][1][1]
    center_x = camera['K'][0][2]
    center_y = camera['K'][1][2]

    all_line[:,:,0] = (all_pose2d[:,:,0]-center_x)/focal_x
    all_line[:,:,1] = (all_pose2d[:,:,1]-center_y)/focal_y
    for i in range(len(pose3d)):
        all_dis += np.linalg.norm(np.cross(all_line[i],pose3d[i]), axis=1)
    all_dis /= len(pose3d)
    return all_dis

def temporalMatch(annots, cameras, last_3d , max_id, images = None):
    imageGroups = []
    annotGroups = []
    for people in last_3d:
        annotGroups.append({'id':people['id'],'groups':[]})
        for views in range(len(annots)):
            annotGroups[-1]['groups'].append([])
        #imageGroups.append([])
    annotGroups.append({'id':-1,'groups':[]})
    for views in range(len(annots)):
        annotGroups[-1]['groups'].append([])

    for view in range(len(annots)):
        annot = annots[view]
        if np.array(annot).size == 0:
            continue
        all_pose2d = []
        all_dis_min = []
        all_id = []
        for person in annot:
            all_pose2d.append(person['keypoints']) 
            all_dis_min.append(1e6)
            all_id.append(-1)
        all_dis_min = np.array(all_dis_min)
        all_id = np.array(all_id)
        all_pose2d = np.array(all_pose2d)

        #all_pose2d = np.reshape(all_pose2d,(-1,-1,3))
        thres = 0.5
        for i in range(len(last_3d)):
            pose3d = last_3d[i]['keypoints3d']
            dis = all_distance_pl(all_pose2d.copy(),pose3d.copy(),cameras[str(view)])
            
            update = np.where(dis < all_dis_min)
            all_id[update] = i
            all_dis_min[update] = dis[update]
        for i in range(len(annot)):
            id = all_id[i]
            pose2d = all_pose2d[i]
            if all_dis_min[i] <= thres:
                annotGroups[id]['groups'][view].append({'keypoints':pose2d,'bbox':annot[i]['bbox'],'id':last_3d[id]['id']})
            else:
                annotGroups[-1]['groups'][view].append({'keypoints':pose2d,'bbox':annot[i]['bbox'],'id':-1})

        # for person in annot:
        #     pose2d = person['keypoints']
        #     d_min = 1e6
        #     thres = 1.0
        #     id = -1
        #     for i in range(len(last_3d)):
        #         pose3d = last_3d[i]['keypoints3d']
        #         dis = distance_pl(pose2d,pose3d.copy(),cameras[str(view)])
        #         if dis < d_min:
        #             d_min = dis
        #             id = i       
        #     # if last_3d != []:
        #     #     import pdb
        #     #     pdb.set_trace()
        #     if d_min <= thres:
        #         annotGroups[id]['groups'][view].append({'keypoints':pose2d,'bbox':person['bbox'],'id':last_3d[id]['id']})
        #     else:
        #         annotGroups[-1]['groups'][view].append({'keypoints':pose2d,'bbox':person['bbox'],'id':-1})

    return imageGroups, annotGroups

def mvposev1(dataset, args, cfg):
    dataset.no_img = not (args.vis_det or args.vis_match or args.vis_repro or args.ret_crop)
    start, end = args.start, min(args.end, len(dataset))
    affinity_model = ComposedAffinity(cameras=dataset.cameras, basenames=dataset.cams, cfg=cfg.affinity)
    group = PeopleGroup(Pall=dataset.Pall, cfg=cfg.group)
    #if args.vis3d:
    #    from easymocap.socket.base_client import BaseSocketClient
    #    vis3d = BaseSocketClient(args.host, args.port)
    last_3d = []
    for nf in tqdm(range(start, end), desc='reconstruction'):
        max_id = group.maxid
        group.clear()
        group.maxid = max_id
        with Timer('load data', not args.time):
            images, annots = dataset[nf]
        if args.vis_det:
            dataset.vis_detections(images, annots, nf, sub_vis=args.sub_vis)
        # 计算不同视角的检测结果的affinity
        # 将各个视角的detection分配到上一帧的3d Pose        
        if args.vis_match:
            dataset.vis_detections(images, annots, nf, mode='match', sub_vis=args.sub_vis)
        dataset.write_keypoints2d(annots, nf)
        with Timer('2D-3D association', not args.time):
            imageGroups, annotGroups = temporalMatch(annots, dataset.cameras, last_3d , max_id, images = images)
        #assert len(imageGroups) == len(annotGroups)
        #results = []
        with Timer('affinity + associate', not args.time):
            for i in range(len(annotGroups)):

                #images = imageGroups[i]
                asso_annots = annotGroups[i]['groups']
                with Timer('compute affinity', not args.time):
                    affinity, dimGroups = affinity_model(asso_annots, images=images)
                with Timer('associate', not args.time):
                    group = simple_associate(asso_annots, affinity, dimGroups, dataset.Pall, group, cfg=cfg.associate)
                    #results = group
                    #results.append(group)
        results = group         
        if args.vis_repro:
            dataset.vis_repro(images, results, nf, sub_vis=args.sub_vis)
        dataset.write_keypoints3d(results, nf)    
        ## draw 3D image
        if args.vis3d:
            dir = os.path.join(args.path,'output','pose3d',str(nf))
            os.makedirs(dir, exist_ok = True)
            draw3d(results.results,dir)
        last_3d = results.results
    
    Timer.report()

if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    parser = load_parser()
    parser.add_argument('--vis_match', action='store_true')
    parser.add_argument('--time', action='store_true')
    parser.add_argument('--vis3d', action='store_true')
    parser.add_argument('--ret_crop', action='store_true')
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument("--host", type=str, default='127.0.0.1')  # cn0314000675l
    parser.add_argument("--port", type=int, default=9999)
    args = parse_parser(parser)
    from easymocap.config.mvmp1f import Config
    cfg = Config.load(args.cfg, args.cfg_opts)
    # Define dataset
    #args.sub = [str(i) for i in range(42)]
    help="""
  Demo code for multiple views and one person:

    - Input : {} => {}
    - Output: {}
    - Body  : {}
""".format(args.path, ', '.join(args.sub), args.out, 
    args.body)
    print(help)
    from easymocap.dataset import MVMPMF
    dataset = MVMPMF(args.path, cams=args.sub, annot_root=args.annot,
        config=CONFIG[args.body], kpts_type=args.body,
        undis=args.undis, no_img=True, out=args.out, filter2d=cfg.dataset)
    mvposev1(dataset, args, cfg)


