@echo off
powershell python D:\Study\2021_Research\animation\experiments\EasyMocap-master/scripts/preprocess/extract_video.py E:\study\EasyMocap\datasets\Alphapose --mode alphapose
powershell python D:\Study\2021_Research\animation\experiments\EasyMocap-master/apps/demo/mvmp.py E:\study\EasyMocap\datasets\Alphapose --out E:\study\EasyMocap\datasets\Alphapose/output --annot annots --cfg config/exp/mvmp1f_alphapose.yml --undis --vis_det --vis_repro --vis3d --vis_match --sub_vis 1 15 25 40 
powershell python D:\Study\2021_Research\animation\experiments\EasyMocap-master/apps/demo/auto_track.py E:\study\EasyMocap\datasets\Alphapose/output E:\study\EasyMocap\datasets\Alphapose/output-track --track3d
powershell python D:\Study\2021_Research\animation\experiments\EasyMocap-master/apps/demo/smpl_from_keypoints.py E:\study\EasyMocap\datasets\Alphapose --skel E:\study\EasyMocap\datasets\Alphapose/output-track/keypoints3d --out E:\study\EasyMocap\datasets\Alphapose/output-track/smpl --verbose --opts smooth_poses 1e1

pause   