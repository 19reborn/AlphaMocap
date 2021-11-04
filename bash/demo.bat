
data=D:/Study/2021_Research/animation/experiments/EasyMocap-master/datasets/demo 
# 0. extract the video to images
 python scripts/preprocess/extract_video.py E:\study\EasyMocap\datasets\demo --handface 
# 2.1 example for SMPL reconstruction
# python apps/demo/mv1p.py E:\study\EasyMocap\datasets\demo --out E:\study\EasyMocap\datasets\demo/output/smpl --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --vis_smpl 
# 2.2 example for SMPL-X reconstruction
python apps/demo/mv1p.py D:/Study/2021_Research/animation/experiments/EasyMocap-master/datasets/demo --out D:/Study/2021_Research/animation/experiments/EasyMocap-master/datasets/demo/output/smplx --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --body bodyhandface --model smplx --gender male --vis_smpl 
# 2.3 example for MANO reconstruction
#     MANO model is required for this part
python apps/demo/mv1p.py D:/Study/2021_Research/animation/experiments/EasyMocap-master/datasets/demo --out D:/Study/2021_Research/animation/experiments/EasyMocap-master/datasets/demo/output/manol --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --body handl --model manol --gender male --vis_smpl 
python apps/demo/mv1p.py D:/Study/2021_Research/animation/experiments/EasyMocap-master/datasets/demo --out D:/Study/2021_Research/animation/experiments/EasyMocap-master/datasets/demo/output/manor --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --body handr --model manor --gender male --vis_smpl 