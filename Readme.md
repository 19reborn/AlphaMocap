# AlphaMocap
My implementation version of [EasyMocap](https://github.com/zju3dv/EasyMocap). With the use of my implementation version of [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) and some new tricks, this repo is targeted to achieve human motion capture in crowded scenes. 

<!-- Here supposed to be a demo gif -->

## Data Formats
The data is organized as follows:
```
<case_name>
|-- videos
    |-- 01.mp4
    |-- 02.mp4
    ...
|-- images
    |-- {camera_id}
        |-- {frames}.jpg
        ...
    ...
|--extri.yml
|--intri.yml
```

## install


###  SMPL models

This step is the same as [smplx](https://github.com/vchoutas/smplx#model-loading).

To download the *SMPL* model go to [this](http://smpl.is.tue.mpg.de) (male and female models, version 1.0.0, 10 shape PCs) and [this](http://smplify.is.tue.mpg.de) (gender neutral model) project website and register to get access to the downloads section. 

To download the *SMPL+H* model go to [this project website](http://mano.is.tue.mpg.de) and register to get access to the downloads section. 

To download the *SMPL-X* model go to [this project website](https://smpl-x.is.tue.mpg.de) and register to get access to the downloads section. 

**Place them as following:**

```bash
data
└── smplx
    ├── J_regressor_body25.npy
    ├── J_regressor_body25_smplh.txt
    ├── J_regressor_body25_smplx.txt
    ├── J_regressor_mano_LEFT.txt
    ├── J_regressor_mano_RIGHT.txt
    ├── smpl
    │   ├── SMPL_FEMALE.pkl
    │   ├── SMPL_MALE.pkl
    │   └── SMPL_NEUTRAL.pkl
    ├── smplh
    │   ├── MANO_LEFT.pkl
    │   ├── MANO_RIGHT.pkl
    │   ├── SMPLH_FEMALE.pkl
    │   └── SMPLH_MALE.pkl
    └── smplx
        ├── SMPLX_FEMALE.pkl
        ├── SMPLX_MALE.pkl
        └── SMPLX_NEUTRAL.pkl
```

### Requirements
- [MyAlphaPose](https://github.com/19reborn/MyAlphaPose)
- conda create -n alphamocap python==3.7.0
- conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
- pip install -r requirements.txt
- python3 setup.py develop --user



## Running

```
data=/home/wangyiming/AlphaMocap/dataset/jhd_1min/

conda activate alphapose
python scripts/preprocess/extract_video.py ${data} --mode alphapose --use-video

conda activate alphamocap
python apps/demo/my_mvmp.py ${data} --out ${data}/output --annot annots --cfg config/exp/mvmp1f_test.yml --undis --vis_det --vis_repro --vis3d --vis_match --sub_vis 01 15 25 40

python apps/demo/smpl_from_keypoints.py ${data} --skel ${data}/output/keypoints3d --out ${data}/output/smpl --verbose --opts smooth_poses 1e-1
```