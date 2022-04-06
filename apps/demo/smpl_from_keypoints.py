'''
  @ Date: 2021-06-14 22:27:05
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-28 10:33:26
  @ FilePath: /EasyMocapRelease/apps/demo/smpl_from_keypoints.py
'''
# This is the script of fitting SMPL to 3d(+2d) keypoints
from pickle import FALSE
from cv2 import threshold
from easymocap.dataset import CONFIG
from easymocap.mytools import Timer
from easymocap.smplmodel import load_model, select_nf
from easymocap.mytools.reader import read_keypoints3d_all
from easymocap.mytools.file_utils import write_smpl
from easymocap.mytools.camera_utils import read_camera
from easymocap.pipeline.weight import load_weight_pose, load_weight_shape, load_weight_pose2d
from easymocap.pipeline import smpl_from_keypoints3d, smpl_from_keypoints3d2d, smpl_from_keypoints2d
from easymocap.mytools.file_utils import get_bbox_from_pose
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from os.path import join
from tqdm import tqdm
import json

fit_3d = True

def smpl_from_skel(path, sub, out, skel3d, args):
    config = CONFIG[args.body]
    results3d, filenames = read_keypoints3d_all(skel3d)

    ## Apply 2d
    # keypoints2d, bboxes = [], []
    # for frame in tqdm(range(0, 60), desc='loading'):
    #     for view in range(0,42):
    #         with open(os.path.join('E:\study\EasyMocap\datasets\dongao/annots',str(view),str(frame).rjust(3,'0')+'.json'),'r') as json_load:
    #             annots = json.load(json_load)['annots']
    #         for i in annots:
    #             keypoints2d.append(np.array(i['keypoints']))
    #             bboxes.append(np.array(i['bbox']))
    # keypoints2d = np.stack(keypoints2d)
    # bboxes = np.stack(bboxes)

    # image_dir = '/home/wangyiming/AlphaMocap/dataset/jhd_1min'
    image_dir = os.path.join(args.path, 'images')
    intri_name = os.path.join(image_dir, '..', 'intri.yml')
    extri_name = os.path.join(image_dir, '..', 'extri.yml')
    if os.path.exists(intri_name) and os.path.exists(extri_name):
        cameras = read_camera(intri_name, extri_name)
        cameras.pop('basenames')
    Pall = np.stack([cam['P'] for id, cam in cameras.items()])


    img = np.zeros((1920,1080))

    if fit_3d == False:    
        weight_shape = load_weight_shape(args.model, args.opts)
        weight_pose = load_weight_pose2d(args.model, args.opts)
        with Timer('Loading {}, {}'.format(args.model, args.gender)):
            body_model = load_model(args.gender, model_type=args.model)
        for pid, result in results3d.items():
            if pid!=0:
                continue
            kp2ds = []
            bboxes = []
        # use repro 2D keypoints        
        # pose3d = result['keypoints3d']/scale
        # for view in range(42):
        #     cam = cameras[str(view)]
        #     pose2d = (pose3d[:,:,:3] @ cam['R'].T + cam['T'].T) @ cam['K'].T
        #     pose2d = pose2d[:,:,:2] / pose2d [:,:,2:]
        #     pose2d = np.concatenate((pose2d,pose3d[:,:,-1:]),axis = 2)
        #     kp2ds.append(pose2d)
        #     bboxs = []
        #     for person in pose2d:
        #         bbox = get_bbox_from_pose(person,img)
        #         bboxs.append(bbox)
        #     bboxes.append(bboxs)
        # kp2ds = np.array(kp2ds).transpose(1,0,2,3)
        # bboxes = np.array(bboxes).transpose(1,0,2)
        ## use 2D detection
            frames = result['frames']
            pose3ds = result['keypoints3d']
            assert len(frames) == len(pose3ds)
            for frame_id , frame in enumerate(frames):
                pose3d = pose3ds[frame_id].copy()
                pose3d[:,:3] = pose3d[:,:3]
                kp2ds_view = []
                bboxes_view = []
                for view in range(42):
                    cam = cameras[str(view)]

                    repro_pose2d = (pose3d[:,:3] @ cam['R'].T + cam['T'].T) @ cam['K'].T
                    repro_pose2d = repro_pose2d[:, :2 ] / repro_pose2d [:, 2:]
                    repro_pose2d = np.concatenate((repro_pose2d,pose3d[:,-1:]),axis = 1)
                    detection_file = os.path.join('E:\study\EasyMocap\datasets\\test\output\keypoints2d',str(view),str(frame).rjust(3,'0')+'.json')
                    with open(detection_file,'r') as detection:
                        data = json.load(detection)['annots']
                        min_dis = 1e6
                        id = -1
                        for i in range(len(data)):

                            detec2d = np.array(data[i]['keypoints'])
                            dis = (((repro_pose2d[:,2] * detec2d[:,2])> 0.) * np.linalg.norm(repro_pose2d[:,:2]-detec2d[:,:2],axis=1))     
                            #repro_bbox = get_bbox_from_pose(repro_pose2d,img)
                            detec_bbox = np.array(data[i]['bbox'])
                            #size = max(repro_bbox[2]-repro_bbox[0],repro_bbox[3]-repro_bbox[1])                       
                            size = max(detec_bbox[2]-detec_bbox[0],detec_bbox[3]-detec_bbox[1])          
                            dis = dis/size
                            dis = dis.sum()/( (dis>0.).sum())
                            #dis /= ((repro_pose2d[:,2] * detec2d[:,2])> 0.).sum()

                            #kptsRepro[:, :, 2]*keypoints2d[:, :, 2]) > 0.)
                            if dis < min_dis:
                                min_dis = dis
                                id = i
                        threshold = 0.1
                        if min_dis <= threshold and id!=-1:
                            kp2ds_view.append(data[id]['keypoints'])
                            bboxes_view.append(data[id]['bbox'])
                        else:
                            kp2ds_view.append(repro_pose2d)
                            bboxes_view.append(get_bbox_from_pose(repro_pose2d,img))
                kp2ds.append(kp2ds_view)
                bboxes.append(bboxes_view)
            kp2ds = np.array(kp2ds)
            bboxes = np.array(bboxes)    

            pose3d = result['keypoints3d'].copy()
            #pose3d[:,:,:3] = pose3d[:,:,:3]/scale
            #pose3d[:,:,:3] = result['keypoints3d'][:,:,:3]/scale
            #kp2ds[:,:,:,:2] = kp2ds[:,:,:,:2]/scale
            #bboxes[:,:,:4] = bboxes[:,:,:4]/scale

            body_params = smpl_from_keypoints2d(body_model, pose3d, kp2ds, bboxes, Pall, config, args,
                weight_shape=weight_shape, weight_pose=weight_pose)
            result['body_params'] = body_params
            if pid == 0:
                break
    else:
        weight_shape = load_weight_shape(args.model, args.opts)
        weight_pose = load_weight_pose(args.model, args.opts)            
        scale = 1.6
        # Pall = []
        # for id,cam in cameras.items():
        #     T = cam['T'] /scale
        #     R = cam['R']
        #     RT = np.hstack((R, T))
        #     Pall.append(cam['K'] @ RT)
        # Pall = np.stack(Pall)
        with Timer('Loading {}, {}'.format(args.model, args.gender)):
            body_model = load_model(args.gender, model_type=args.model)

        for pid, result in results3d.items():
            print(f'****** optimize_person:{pid}')
            #if pid != 0:
            #    continue
            # kp2ds = []
            # bboxes = []

            # frames = result['frames']
            # pose3ds = result['keypoints3d']
            # assert len(frames) == len(pose3ds)
            # for frame_id , frame in enumerate(frames):
            #     pose3d = pose3ds[frame_id].copy()
            #     #pose3d[:,:3] = pose3d[:,:3]
            #     kp2ds_view = []
            #     bboxes_view = []
            #     for view in range(42):
            #         cam = cameras[str(view)]

            #         repro_pose2d = (pose3d[:,:3] @ cam['R'].T + cam['T'].T) @ cam['K'].T
            #         repro_pose2d = repro_pose2d[:, :2 ] / repro_pose2d [:, 2:]
            #         repro_pose2d = np.concatenate((repro_pose2d,pose3d[:,-1:]),axis = 1)
            #         detection_file = os.path.join('E:\study\EasyMocap\datasets\\test\output\keypoints2d',str(view),str(frame).rjust(3,'0')+'.json')
            #         with open(detection_file,'r') as detection:
            #             data = json.load(detection)['annots']
            #             min_dis = 1e6
            #             id = -1
            #             for i in range(len(data)):

            #                 detec2d = np.array(data[i]['keypoints'])
            #                 dis = (((repro_pose2d[:,2] * detec2d[:,2])> 0.) * np.linalg.norm(repro_pose2d[:,:2]-detec2d[:,:2],axis=1))     
            #                 #repro_bbox = get_bbox_from_pose(repro_pose2d,img)
            #                 detec_bbox = np.array(data[i]['bbox'])
            #                 #size = max(repro_bbox[2]-repro_bbox[0],repro_bbox[3]-repro_bbox[1])                       
            #                 size = max(detec_bbox[2]-detec_bbox[0],detec_bbox[3]-detec_bbox[1])          
            #                 dis = dis/size
            #                 dis = dis.sum()/( (dis>0.).sum())
            #                 #dis /= ((repro_pose2d[:,2] * detec2d[:,2])> 0.).sum()

            #                 #kptsRepro[:, :, 2]*keypoints2d[:, :, 2]) > 0.)
            #                 if dis < min_dis:
            #                     min_dis = dis
            #                     id = i

            #             threshold = 0.1
            #             if min_dis <= threshold and id!=-1:
            #                 kp2ds_view.append(data[id]['keypoints'])
            #                 bboxes_view.append(data[id]['bbox'])
            #             else:
            #                 kp2ds_view.append(repro_pose2d)
            #                 bboxes_view.append(get_bbox_from_pose(repro_pose2d,img))
            #     kp2ds.append(kp2ds_view)
            #     bboxes.append(bboxes_view)
            # kp2ds = np.array(kp2ds)
            # bboxes = np.array(bboxes)    

            pose3d = result['keypoints3d'].copy()
            pose3d[:,:,:3] = pose3d[:,:,:3]/scale
            #pose3d[:,:,:3] = result['keypoints3d'][:,:,:3]/scale
            #kp2ds[:,:,:,:2] = kp2ds[:,:,:,:2]/scale
            #bboxes[:,:,:4] = bboxes[:,:,:4]/scale

            body_params = smpl_from_keypoints3d(body_model, pose3d, config, args,
                weight_shape=weight_shape, weight_pose=weight_pose)
            result['body_params'] = body_params
            #if pid == 0:
            #    break
    # scale = 1.4
    # pids = list(results3d.keys())
    # weight_shape = load_weight_shape(args.model, args.opts)
    # weight_pose = load_weight_pose(args.model, args.opts)
    # with Timer('Loading {}, {}'.format(args.model, args.gender)):
    #     body_model = load_model(args.gender, model_type=args.model)
    # for pid, result in results3d.items():
    #     body_params = smpl_from_keypoints3d(body_model, result['keypoints3d']/scale, config, args,
    #         weight_shape=weight_shape, weight_pose=weight_pose)
    #     result['body_params'] = body_params
    
    # write for each frame
    for nf, skelname in enumerate(tqdm(filenames, desc='writing')):
        basename = os.path.basename(skelname)
        outname = join(out, basename)
        res = []
        for pid, result in results3d.items():
            #if pid != 0:
            #    continue
            frames = result['frames']
            if nf in frames:
                nnf = frames.index(nf)
                val = {'id': pid}
                params = select_nf(result['body_params'], nnf)
                val.update(params)
                res.append(val)
            #if pid==0:
            #    break
        write_smpl(outname, res)

if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    parser = load_parser()
    parser.add_argument('--skel3d', type=str, required=True)
    args = parse_parser(parser)
    help="""
  Demo code for fitting SMPL to 3d(+2d) skeletons:

    - Input : {} => {}
    - Output: {}
    - Body  : {}=>{}, {}
""".format(args.path, args.skel3d, args.out, 
    args.model, args.gender, args.body)
    print(help)
    smpl_from_skel(args.path, args.sub, args.out, args.skel3d, args)