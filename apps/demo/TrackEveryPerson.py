
from easymocap.mytools.file_utils import get_bbox_from_pose
from easymocap.mytools.reader import read_keypoints3d_all
from easymocap.mytools.camera_utils import read_camera
import os
import numpy as np
import json
import cv2

skel3d_path = 'E:\study\EasyMocap\datasets/test/output/keypoints3d' 
results3d, filenames = read_keypoints3d_all(skel3d_path)

image_dir = 'E:\study\EasyMocap\datasets\\Alphapose\images'
output_path = 'E:\study\EasyMocap\datasets\\nueralbody\padd_image'
intri_name = os.path.join(image_dir, '..', 'intri.yml')
extri_name = os.path.join(image_dir, '..', 'extri.yml')
if os.path.exists(intri_name) and os.path.exists(extri_name):
    cameras = read_camera(intri_name, extri_name)
    cameras.pop('basenames')
    
img = np.zeros((1920,1080))
bbox_with_id = {}
for pid, result in results3d.items():
    if pid !=4 :
        continue
    #if pid <10 :
    #    continue
    #kp2ds = []
    bboxes = []
    ## use 2D detection
    frames = result['frames']
    pose3ds = result['keypoints3d']
    assert len(frames) == len(pose3ds)
    for frame_id , frame in enumerate(frames):
        if frame != 6  :
            continue
        pose3d = pose3ds[frame_id]
        #kp2ds_view = []
        bboxes_view = []
        for view in range(42):
            if view !=0 and view != 10 and view!=23 and view!=37:
                continue
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
                    dis = np.linalg.norm(repro_pose2d[:,:2]-detec2d[:,:2],axis=0).mean()
                    if dis < min_dis:
                        min_dis = dis
                        id = i
                threshold = 20
                if min_dis <= threshold and id!=-1:
                    keypoints2d = np.array(data[id]['keypoints'])
                    #kp2ds_view.append(data[id]['keypoints'])
                    #bboxes_view.append(data[id]['bbox'])
                    #bboxes_view.append(get_bbox_from_pose(data[id]['keypoints'],img,0.5))
                else:
                    keypoints2d = np.array(repro_pose2d)
                    #kp2ds_view.append(repro_pose2d)
                    #bboxes_view.append(get_bbox_from_pose(repro_pose2d,img,0.5))
                bboxes_view.append(get_bbox_from_pose(keypoints2d,img,0.3))
        #kp2ds.append(kp2ds_view)
        bboxes.append(bboxes_view)
    #kp2ds = np.array(kp2ds)
    bboxes = np.array(bboxes)    
    image_pid_path = os.path.join(output_path,str(pid))
    #os.mkdir(image_path,exist_ok =True)
    for view in range(bboxes.shape[1]):
        image_view_path = os.path.join(image_pid_path,str(view))
        os.makedirs(image_view_path,exist_ok =True)
        for frame in range(bboxes.shape[0]):
            bbox = bboxes[frame][view]
            ori_image = cv2.imread(os.path.join(image_dir,str(view),str(frame).rjust(3,'0')+'.jpg'))
            if ori_image.size != 0:
                #crop_image = ori_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
                pad_image = np.zeros((ori_image.shape[0],ori_image.shape[1],3))
                pad_image[:,:,:] = 255
                pad_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:] = ori_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
                print(540-((int(bbox[1])+int(bbox[3]))/2-150), 960-((int(bbox[0])+int(bbox[2]))/2-150))
                #cv2.imwrite(os.path.join(image_view_path,str(frame).rjust(3,'0')+'.jpg'), pad_image)
    #bbox_with_id[pid] = bboxes

# print("######## begin crop image########")
# for pid, bboxes in bbox_with_id.items():
#     bboxes = np.array(bboxes)
#     image_path = os.path.join(output_path,str(pid))
#     os.mkdir(image_path,exist_ok =True)
#     for view in range(bboxes.shape[1]):
#         image_path = os.path.join(image_path,str(view))
#         os.mkdir(image_path,exists_ok =True)
#         for frame in range(bboxes.shape[1]):
#             bbox = bboxes[frame][view]
#             ori_image = cv2.imread(os.path.join(image_path,view,str(frame).rjust(6,'0')+'.jpg'))
#             crop_image = ori_image[bbox[0]:bbox[2],bbox[1]:bbox[3]]
#             cv2.imwrite(os.path.join(image_path,str(frame).rjust(6,'0')+'.jpg'))
