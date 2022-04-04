'''
  @ Date: 2020-11-28 17:23:04
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-03 22:31:31
  @ FilePath: /EasyMocap/easymocap/mytools/vis_base.py
'''
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from os.path import join
from easymocap.mytools.file_utils import get_bbox_from_pose, save_json

nviews = 42

BODY25 = [[ 1,  0],
    [ 2,  1],
    [ 3,  2],
    [ 4,  3],
    [ 5,  1],
    [ 6,  5],
    [ 7,  6],
    [ 8,  1],
    [ 9,  8],
    [10,  9],
    [11, 10],
    [12,  8],
    [13, 12],
    [14, 13],
    [15,  0],
    [16,  0],
    [17, 15],
    [18, 16],
    [19, 14],
    [20, 19],
    [21, 14],
    [22, 11],
    [23, 22],
    [24, 11]]

def generate_colorbar(N = 20, cmap = 'jet'):
    bar = ((np.arange(N)/(N-1))*255).astype(np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(bar, cv2.COLORMAP_JET).squeeze()
    if False:
        colorbar = np.clip(colorbar + 64, 0, 255)
    import random
    random.seed(666)
    index = [i for i in range(N)]
    random.shuffle(index)
    rgb = colorbar[index, :]
    rgb = rgb.tolist()
    return rgb

colors_bar_rgb = generate_colorbar(cmap='hsv')

colors_table = {
    'b': [0.65098039, 0.74117647, 0.85882353],
    '_pink': [.9, .7, .7],
    '_mint': [ 166/255.,  229/255.,  204/255.],
    '_mint2': [ 202/255.,  229/255.,  223/255.],
    '_green': [ 153/255.,  216/255.,  201/255.],
    '_green2': [ 171/255.,  221/255.,  164/255.],
    'r': [ 251/255.,  128/255.,  114/255.],
    '_orange': [ 253/255.,  174/255.,  97/255.],
    'y': [ 250/255.,  230/255.,  154/255.],
    '_r':[255/255,0,0],
    'g':[0,255/255,0],
    '_b':[0,0,255/255],
    'k':[0,0,0],
    '_y':[255/255,255/255,0],
    'purple':[128/255,0,128/255],
    'smap_b':[51/255,153/255,255/255],
    'smap_r':[255/255,51/255,153/255],
    'smap_b':[51/255,255/255,153/255],
}

# 这个顺序是BGR的。
colors = [
    (0.5, 0.2, 0.2, 1.),  # Defalut BGR
    (.5, .5, .7, 1.),  # Pink BGR
    (.44, .50, .98, 1.), # Red
    (.7, .7, .6, 1.),  # Neutral
    (.5, .5, .7, 1.),  # Blue
    (.5, .55, .3, 1.),  # capsule
    (.6, .6, .6, 1.), # gray
    (0.95, 0.74, 0.65, 1.),
    (.9, .7, .7, 1.),
    
]
def get_my_rgb(index):
    # if isinstance(index, int):
    #     if index == -1:
    #         return (255, 255, 255)
    #     if index < -1:
    #         return (0, 0, 0)
    #     col = colors_bar_rgb[index%len(colors_bar_rgb)]
    # else:
    #     col = colors_table.get(index, (1, 0, 0))
    #     col = tuple([int(c*255) for c in col[::-1]])
    #colors = list(colors_table.values())
    color = colors[index%len(colors)]
    col = [color[2],color[1],color[0]]
    return col

def get_rgb(index):
    if isinstance(index, int):
        if index == -1:
            return (255, 255, 255)
        if index < -1:
            return (0, 0, 0)
        col = colors_bar_rgb[index%len(colors_bar_rgb)]
    else:
        col = colors_table.get(index, (1, 0, 0))
        col = tuple([int(c*255) for c in col[::-1]])
    return col

def get_rgb_01(index):
    col = get_rgb(index)
    return [i*1./255 for i in col[:3]]

def plot_point(img, x, y, r, col, pid=-1, font_scale=-1, circle_type=-1):
    cv2.circle(img, (int(x+0.5), int(y+0.5)), r, col, circle_type)
    if font_scale == -1:
        font_scale = img.shape[0]/4000
    if pid != -1:
        cv2.putText(img, '{}'.format(pid), (int(x+0.5), int(y+0.5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 1)


def plot_line(img, pt1, pt2, lw, col):
    cv2.line(img, (int(pt1[0]+0.5), int(pt1[1]+0.5)), (int(pt2[0]+0.5), int(pt2[1]+0.5)),
        col, lw)

def plot_cross(img, x, y, col, width=-1, lw=-1):
    if lw == -1:
        lw = int(round(img.shape[0]/1000))
        width = lw * 5
    cv2.line(img, (int(x-width), int(y)), (int(x+width), int(y)), col, lw)
    cv2.line(img, (int(x), int(y-width)), (int(x), int(y+width)), col, lw)
    
def plot_bbox(img, bbox, pid, vis_id=True):
    # 画bbox: (l, t, r, b)
    x1, y1, x2, y2 = bbox[:4]
    x1 = int(round(x1))
    x2 = int(round(x2))
    y1 = int(round(y1))
    y2 = int(round(y2))
    color = get_rgb(pid)
    lw = max(img.shape[0]//300, 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
    if vis_id:
        font_scale = img.shape[0]/1000
        cv2.putText(img, '{}'.format(pid), (x1, y1+int(25*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

def plot_keypoints(img, points, pid, config, vis_conf=False, use_limb_color=True, lw=2):
    for ii, (i, j) in enumerate(config['kintree']):
        if i >= len(points) or j >= len(points):
            continue
        pt1, pt2 = points[i], points[j]
        if use_limb_color:
            col = get_rgb(config['colors'][ii])
        else:
            col = get_rgb(pid)
        if pt1[-1] > 0.01 and pt2[-1] > 0.01:
            image = cv2.line(
                img, (int(pt1[0]+0.5), int(pt1[1]+0.5)), (int(pt2[0]+0.5), int(pt2[1]+0.5)),
                col, lw)
    for i in range(len(points)):
        x, y = points[i][0], points[i][1]
        c = points[i][-1]
        if c > 0.01:
            col = get_rgb(pid)
            cv2.circle(img, (int(x+0.5), int(y+0.5)), lw*2, col, -1)
            if vis_conf:
                cv2.putText(img, '{:.1f}'.format(c), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

def plot_points2d(img, points2d, lines, lw=4, col=(0, 255, 0), putText=True):
    # 将2d点画上去
    if points2d.shape[1] == 2:
        points2d = np.hstack([points2d, np.ones((points2d.shape[0], 1))])
    for i, (x, y, v) in enumerate(points2d):
        if v < 0.01:
            continue
        c = col
        plot_cross(img, x, y, width=10, col=c, lw=lw)
        if putText:
            font_scale = img.shape[0]/2000
            cv2.putText(img, '{}'.format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, c, 2)
    for i, j in lines:
        if points2d[i][2] < 0.01 or points2d[j][2] < 0.01:
            continue
        plot_line(img, points2d[i], points2d[j], 2, col)

row_col_ = {
    2: (2, 1),
    7: (2, 4),
    8: (2, 4),
    9: (3, 3),
    26: (4, 7)
}
def get_row_col(l):
    if l in row_col_.keys():
        return row_col_[l]
    else:
        from math import sqrt
        row = int(sqrt(l) + 0.5)
        col = int(l/ row + 0.5)
        if row*col<l:
            col = col + 1
        if row > col:
            row, col = col, row
        return row, col

def merge(images, row=-1, col=-1, resize=False, ret_range=False, **kwargs):
    if row == -1 and col == -1:
        row, col = get_row_col(len(images))
    height = images[0].shape[0]
    width = images[0].shape[1]
    ret_img = np.zeros((height * row, width * col, images[0].shape[2]), dtype=np.uint8) + 255
    ranges = []
    for i in range(row):
        for j in range(col):
            if i*col + j >= len(images):
                break
            img = images[i * col + j]
            # resize the image size
            img = cv2.resize(img, (width, height))
            ret_img[height * i: height * (i+1), width * j: width * (j+1)] = img
            ranges.append((width*j, height*i, width*(j+1), height*(i+1)))
    if resize:
        min_height = 3000
        if ret_img.shape[0] > min_height:
            scale = min_height/ret_img.shape[0]
            ret_img = cv2.resize(ret_img, None, fx=scale, fy=scale)
    if ret_range:
        return ret_img, ranges
    return ret_img

def draw3d(results,dir):
    pos3d = {}

    for i in results:
        pos = np.array(i['keypoints3d'].copy())
        #pos[..., 1] = -pos[..., 1]
        pos3d[i['id']]= pos
    #pos3d = [i / 1000 for i in pos3d]
    #pos3d = np.array(pos3d)

    #for i in pos3d:
    #    i[..., 1] = -i[..., 1]
    fig = plt.figure()
    if len(pos3d) == 0 :
        return
    def calc_lim():
        pos = np.array(list(pos3d.values()))
        tmp = np.concatenate(pos[:,:,:3], axis=0).reshape(-1, 3)
        bound = np.array([np.min(tmp, axis=0), np.max(tmp, axis=0)])
        bound[1, :] -= bound[0, :]
        bound[1, :] = np.max(bound[1, :])
        bound[1, :] += bound[0, :]
        return bound
    
    bound = calc_lim()
    def render(color):
        # plot1.set_zsticks()
        plot1.set_xlabel('x')
        plot1.set_ylabel('y')
        plot1.set_zlabel('z')


        #plot1.auto_scale_xyz(bound[:, 0], bound[:, 2], -bound[:, 1])
        #plot1.set_ylim(range(-65,-40,5))
        #plot1.set_zlim(range(-5, 3, 1))
        #plot1.set_xlim(range(-10, 15, 5))
        #plot1.set_ylim(-40,-30)
        #plot1.set_zlim(-10,-5)
        #plot1.set_zlim(-5,5)
        #plot1.set_xlim(-70,-20)
        #plot1.set_xlim(-30,20)
        #plot1.set_ylim(-50,30)
        #plot1.set_zlim(-4,0)
        #plot1.view_init(elev=10., azim=11)
        plot1.view_init(elev=30, azim=15)
        plot1.set_xlim(-25,30)
        plot1.set_ylim(-25,20)
        plot1.set_zlim(0,8)
        for bone_info in BODY25:
            jpos = person_pos[bone_info]  # (2, 4)
            if jpos[0][3] >= 0.4 and jpos[1][3] >= 0.4:
                plot1.plot(jpos[:, 0], jpos[:, 1], jpos[:, 2], color = color)
    
    plot1 = plot1 = fig.add_subplot(111, projection='3d')
    for person_idx, person_pos in pos3d.items():
        render(get_my_rgb(person_idx))
    
    plt.rcParams['savefig.dpi'] = 500 #图片像素
    plt.rcParams['figure.dpi'] = 500 #分辨率
    plt.savefig(dir)
    plt.tight_layout()
    #plt.show()
    plt.close(0)

def drawRepro(infos, out , frame , images_dir, cameras, config, sub_vis = None):
    out = join(out,'repro')
    os.makedirs(out, exist_ok=True)
    # cameras: (K, R, T)
    #images_all = os.listdir(images_dir)
    images = {}
    if sub_vis ==None:
        sub_vis = range(nviews)
    for view in sub_vis:
        images[view]=join(images_dir,str(view),'%03d.jpg' %frame)
    images_vis = []
    for nv, image in images.items():
        img = cv2.imread(image)
        #img = image.copy()
        K, R, T = cameras[str(nv)]['K'], cameras[str(nv)]['R'], cameras[str(nv)]['T']
        P = K @ np.hstack([R, T])
        for info in infos:
            pid = info['id']
            keypoints3d = np.array(info['keypoints3d'])
            # 重投影
            kcam = np.hstack([keypoints3d[:, :3], np.ones((keypoints3d.shape[0], 1))]) @ P.T
            kcam = kcam[:, :2]/kcam[:, 2:]
            k2d = np.hstack((kcam, keypoints3d[:, -1:]))
            bbox = get_bbox_from_pose(k2d, img)
            plot_bbox(img, bbox, pid=pid, vis_id=pid)
            plot_keypoints(img, k2d, pid=pid, config=config, use_limb_color=False, lw=2)
        images_vis.append(img)
    savename = join(out, '{:06d}.jpg'.format(frame))
    image_vis = merge(images_vis, resize=False)
    cv2.imwrite(savename, image_vis)


def drawSmplrepro(infos, out, frame, images_dir, cameras, sub_vis = None):
    out = join(out,'smpl_repro')
    os.makedirs(out, exist_ok=True)
    images = {}
    if sub_vis == None:
        sub_vis = range(nviews)
    for view in sub_vis:
        images[view] = (cv2.imread(join(images_dir,f'{view:03}','%06d.jpg' %frame)))
    from ..visualize.renderer import Renderer
    render = Renderer(height=1024, width=1024, faces=None)
    render_results = render.render(infos, cameras, images)
    image_vis = merge(render_results, resize=False)
    savename = join(out,'{:06d}.jpg'.format(frame))
    cv2.imwrite(savename, image_vis)
    return image_vis

def drawSmpl3d(infos, out, frame, images_dir, cameras):
    return