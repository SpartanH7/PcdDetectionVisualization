from multiprocessing import Pipe, Process, Manager
import multiprocessing
import pickle
import json
from .open3d_arrow import get_arrow
import concurrent.futures as futures
from collections import Iterable
from tqdm import tqdm
import copy
import torch
import math
import cv2
import matplotlib.pyplot as plt
import open3d
import numpy as np
import os
import random
import string
from pathlib import Path
from easydict import EasyDict
ROOT = Path(__file__).absolute().parent.parent


buildInViewFile = {
    'follow': 'view_follow.json',
    'birdeye': 'view_birdeye.json',
    'birdeye_inv': 'view_birdeye_inv.json',
}

colorMap = {
    'black': np.zeros(3, dtype=float),
    'white': np.ones(3, dtype=float),
    'red': np.array([1,0,0], dtype=float),
    'green': np.array([0,1,0], dtype=float),
    'blue': np.array([0,0,1], dtype=float),
}

def defaultColorBank(maxColorType):
    level=2
    while math.pow(level,3)-level<maxColorType:
        level+=1
    stride=256//(level)
    stages=[i*stride for i in range(1,level)]+[255]
    colors=[[r,g,b] for b in stages for r in stages for g in stages if not (b==g and b==r)]
    colors=np.array(colors, dtype=np.uint8)
    return colors

def CheckDataType(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        pass
    else:
        raise NotImplementedError("Only support numpy array or torch tensor")
    return obj


def PrepareData(points=None,
                    pointLabels=None,
                    pointNames=None,
                    pointColors=None,
                    gtBoxes=None,
                    gtLabels=None,
                    gtNames=None,
                    refBoxes=None,
                    refLabels=None,
                    refNames=None,
                    refScores=None,
                    photos=None,
                    scoreThresh=0.0,
                    ):
    points=copy.deepcopy(points)
    pointLabels=copy.deepcopy(pointLabels)
    pointNames=copy.deepcopy(pointNames)
    pointColors=copy.deepcopy(pointColors)
    gtBoxes=copy.deepcopy(gtBoxes)
    gtLabels=copy.deepcopy(gtLabels)
    gtNames=copy.deepcopy(gtNames)
    refBoxes=copy.deepcopy(refBoxes)
    refLabels=copy.deepcopy(refLabels)
    refNames=copy.deepcopy(refNames)
    refScores=copy.deepcopy(refScores)
    photos=copy.deepcopy(photos)
    if points is not None:
        if isinstance(points, list):
            for i, frame in enumerate(points):
                frame = CheckDataType(frame)
                assert len(frame.shape) == 2
                assert frame.shape[1] in [3, 4, 6], "Point shape should be (x,y,z) or (x,y,z,intensity) or (x,y,z,r,g,b)"
                points[i] = frame
        else:
            points = CheckDataType(points)
            assert points.shape[-1] in [3, 4, 6], "Point shape should be (x,y,z) or (x,y,z,intensity) or (x,y,z,r,g,b)"
            if len(points.shape) == 3:
                points = [frame for frame in points]
            elif len(points.shape) == 2:
                points = [points]
            else:
                raise ValueError(
                    "Points data should has a shape with a length of 2(single pointcloud) or 3(pointcloud batch)")
        SEQ_LEN = len(points)
        point_numbers = [frame.shape[0] for frame in points]
    else:
        SEQ_LEN = 1
    assert SEQ_LEN>=1, "There should be at last 1 frame"
    DATAS=[
        EasyDict(dict(
            points=points[i] if points is not None else None,
            point_labels=None,
            point_names=None,
            point_colors=None,
            gt_boxes=None,
            gt_labels=None,
            gt_names=None,
            ref_boxes=None,
            ref_labels=None,
            ref_names=None,
            ref_scores=None,
            photos=None,
        )) for i in range(SEQ_LEN)
    ]
    
    if SEQ_LEN == 1:
        pointColors = [pointColors] if pointColors is not None else None
        pointLabels = [pointLabels] if pointLabels is not None else None
        pointNames = [pointNames] if pointNames is not None else None
        gtBoxes = [gtBoxes] if gtBoxes is not None else None
        gtLabels = [gtLabels] if gtLabels is not None else None
        gtNames = [gtNames] if gtNames is not None else None
        refBoxes = [refBoxes] if refBoxes is not None else None
        refLabels = [refLabels] if refLabels is not None else None
        refNames = [refNames] if refNames is not None else None
        refScores = [refScores] if refScores is not None else None
        photos=[photos] if photos is not None else None
    for index in range(SEQ_LEN):
        if points is not None:
            if pointLabels is not None:
                point_labels = pointLabels[index]
                assert len(point_labels) == point_numbers[index]
                DATAS[index].point_labels=point_labels
            if pointNames is not None:
                point_names = pointNames[index]
                assert len(point_names) == point_numbers[index]
                DATAS[index].point_names=point_names
            if pointColors is not None:
                point_colors = pointColors[index]
                point_colors = CheckDataType(point_colors)
                assert point_colors.shape[0] == point_numbers[index]
                if len(point_colors.shape) == 1 or point_colors.shape[1] == 1:
                    point_colors = point_colors.reshape(-1, 1)[:, [0, 0, 0]]
                elif point_colors.shape[1] == 3:
                    pass
                else:
                    raise NotImplementedError("pointColors shape not support")
                DATAS[index].point_colors = point_colors
        if gtBoxes is not None:
            gt_boxes = gtBoxes[index]
            gt_boxes = CheckDataType(gt_boxes)
            assert gt_boxes.shape[1] in [
                7, 9], "box shape should be (cx, cy, cz, dx, dy, dz, rz, [vx, vy])"
            gt_box_num = gt_boxes.shape[0]
            DATAS[index].gt_boxes = gt_boxes
        if gtLabels is not None:
            gt_labels = gtLabels[index]
            assert len(gt_labels) == gt_box_num
            DATAS[index].gt_labels=gt_labels
        if gtNames is not None:
            gt_names = gtNames[index]
            assert len(gt_names) == gt_box_num
            DATAS[index].gt_names=gt_names
        if refBoxes is not None:
            ref_boxes = refBoxes[index]
            ref_boxes = CheckDataType(ref_boxes)
            assert ref_boxes.shape[1] in [
                7, 9], "box shape should be (cx, cy, cz, dx, dy, dz, rz, [vx, vy])"
            ref_box_num = ref_boxes.shape[0]
            DATAS[index].ref_boxes=ref_boxes
        if refLabels is not None:
            ref_labels = refLabels[index]
            assert len(ref_labels) == ref_box_num
            DATAS[index].ref_labels=ref_labels
        if refNames is not None:
            ref_names = refNames[index]
            assert len(ref_names) == ref_box_num
            DATAS[index].ref_names=ref_names
        if refScores is not None:
            ref_scores = refScores[index]
            assert len(ref_scores) == ref_box_num
            DATAS[index].ref_scores=ref_scores

            ref_scores = np.array(ref_scores)
            ref_score_thresh_mask = ref_scores > scoreThresh
            ref_boxes = ref_boxes[ref_score_thresh_mask]
            DATAS[index].ref_boxes=ref_boxes
            if ref_labels is not None:
                ref_labels = np.array(ref_labels)
                ref_labels = ref_labels[ref_score_thresh_mask]
                DATAS[index].ref_labels=ref_labels
            if ref_names is not None:
                ref_names = np.array(ref_names)
                ref_names = ref_names[ref_score_thresh_mask]
                DATAS[index].ref_names=ref_names
        if photos is not None:
            DATAS[index].photos=photos[index]
    return DATAS

def getFormattedData(data, index):
    points, pointColors, gtBoxes, gtLabels, gtNames, refBoxes, refLabels, refNames, refScores =data
    cur_points = points[index]
    point_colors = pointColors[index] if pointColors is not None else None
    gt_boxes = gtBoxes[index] if gtBoxes is not None else None
    gt_labels = gtLabels[index] if gtLabels is not None else None
    gt_names = gtNames[index] if gtNames is not None else None
    ref_boxes = refBoxes[index] if refBoxes is not None else None
    ref_labels = refLabels[index] if refLabels is not None else None
    ref_names = refNames[index] if refNames is not None else None
    return cur_points, point_colors, gt_boxes, gt_labels, gt_names, ref_boxes, ref_labels, ref_names

def GetBuildInViewFile(name):
    filename = buildInViewFile[name]
    abspath = ROOT / 'data' / filename
    return str(abspath)


def get_o3d_pcd(points, point_colors=None):
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    if point_colors is None:
        if points.shape[1] > 3:
            color = plt.get_cmap("terrain")(points[:, 3])[:, :3]
            pts.colors = open3d.utility.Vector3dVector(color)
        else:
            pts.colors = open3d.utility.Vector3dVector(
                np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)
    return pts


def getLabelColor(labels, colormap=None, labelStartFromZero=True):
    labels = np.array(labels, dtype=int)
    if not labelStartFromZero: labels -= 1
    if colormap is None:
        colormap = defaultColorBank(np.max(labels)+1)
    else:
        colormap = np.array(colormap)
    colormap=np.array(colormap, dtype=float)
    if np.max(colormap) > 1:
        colormap /= np.max(colormap)
    label_colors = colormap[labels]
    return label_colors


def get_box(boxes, defaultColor=(0, 1, 0), labels=None, colormap=None, withArrow=False, labelStartFromZero=True):
    if not isinstance(boxes, np.ndarray):
        boxes = np.asarray(boxes)
    objs = []
    arrows = []
    colors = []
    if labels is not None:
        labelColors = getLabelColor(labels, colormap, labelStartFromZero)
    for i in range(boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(boxes[i])
        if labels is None:
            line_set.paint_uniform_color(defaultColor)
        else:
            line_set.paint_uniform_color(labelColors[i])
            colors.append(labelColors[i])
        objs.append(line_set)
        if withArrow and boxes.shape[1] == 9:
            _, speedarrow, begin, _ = get_arrow(
                boxes[i][0:3], [boxes[i][7], boxes[i][8], 0])
            arrows.append([speedarrow,begin])
    return objs, arrows, colors


def translate_boxes_to_open3d_instance(gt_boxes, front_mark=True):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0.0, 0.0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    box3d.color = [0, 0, 0]
    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    if front_mark:
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
        line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d

def getaxis(size=1.0, x=0, y=0, z=0):
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=[x, y, z])
    line_set, box3d = translate_boxes_to_open3d_instance([x,y,z,1,1,1,0])
    line_set.paint_uniform_color((0, 0, 1))
    return axis_pcd, line_set

def img_open3d_to_opencv(img):
    img = np.asarray(copy.deepcopy(img))
    img *= 255
    img = img.astype(np.uint8)
    newimg = img[...,[2,1,0]]
    return np.ascontiguousarray(newimg)

def loadJson(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def saveJson(data: dict, json_path: str, indent=1):
    with open(json_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def loadPickle(path):
    with open(path,'rb') as f:
        data=pickle.load(f)
    return data

def savePickle(data, path):
    with open(path,'wb') as f:
        pickle.dump(data,f)

def getScaleSize(originHeight, originWidth, targetHeight=None, targetWidth=None):
    h=originHeight
    w=originWidth
    if targetWidth is None and targetHeight is not None:
        targetWidth=w*(targetHeight/h)
    if targetHeight is None and targetWidth is not None:
        targetHeight=h*(targetWidth/w)
    if targetWidth is None and targetHeight is None:
        targetWidth=w
        targetHeight=h
    return targetHeight, targetWidth

def showImgCV2(img, height=None, width=None, title='Image', waitKey=True):
    h=img.shape[0]
    w=img.shape[1]
    height, width=getScaleSize(h, w, height,width)
    img = cv2.resize(img, (int(width), int(height)))
    cv2.imshow(title, img)
    if waitKey: cv2.waitKey(0)

def saveImgCV2(img, savePath='image.jpg',height=None, width=None):
    h=img.shape[0]
    w=img.shape[1]
    height, width=getScaleSize(h, w, height,width)
    img = cv2.resize(img, (int(width), int(height)))
    cv2.imwrite(str(savePath), img)

def addLabelcolorLegend(img, labelcolor, lineheight=20, colortype='opencv'):
    posx = posy = 20
    lineheight = lineheight
    margin = int(lineheight/4)
    for label in labelcolor:
        text = str(label)
        color = labelcolor[label]
        if colortype in ['open3d']:
            color = img_open3d_to_opencv(color)
        cv2.rectangle(img, (posx, posy), (posx+lineheight, posy+lineheight), color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
        cv2.putText(img=img, text=text, org=(posx+lineheight+margin, posy+lineheight),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=float(lineheight/20), color=(0, 255, 0), thickness=2)
        posy += lineheight+margin
    return img

def cvImread(file_path, flag=-1):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),flag)
    return cv_img

def generateVideo(videoPath,imgFiles,fps=None,width=None,height=None,pbar=False):
    videoPath=str(Path(videoPath).absolute())
    fps=4 if fps is None else int(fps)
    img0=cvImread(imgFiles[0])
    height, width=getScaleSize(img0.shape[0], img0.shape[1], height,width)
    video=cv2.VideoWriter(videoPath,cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,height))
    frameNum=len(imgFiles)
    if pbar:
        filename=videoPath
        filetqdm=tqdm(total=frameNum, desc=f"Generating video {limitLenStr(filename)}")
    for file in imgFiles:
        img=cvImread(file)
        img=cv2.resize(img,(width,height))
        video.write(img)
        if pbar: filetqdm.update()
    video.release()
    return videoPath

def limitLenStr(text, maxLen=20):
    limitText=text
    if len(limitText) > maxLen: limitText=f"...{limitText[(len(limitText)-maxLen):]}"
    return limitText

def randomString(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (N, 3 + C)
        angle: angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros=0.0
    ones=1.0
    rot_matrix = np.array(
        [cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones]
    ).reshape(3, 3)
    points_rot = np.dot(points[..., 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[..., 3:]), axis=-1)
    return points_rot

def points_in_box(xyzs,box,MARGIN=0.05):
    origin_xyz=xyzs[:,:3]
    cx,cy,cz,dx,dy,dz,rz=box
    local_xyz=origin_xyz-box[:3]
    local_xyz=rotate_points_along_z(local_xyz, -rz)
    local_xyz=np.abs(local_xyz)
    in_box_flag=(local_xyz[:,0]<(dx/2+MARGIN)) & (local_xyz[:,1]<(dy/2+MARGIN)) & (local_xyz[:,2]<(dz/2+MARGIN))
    return in_box_flag

def points_in_boxes(points,boxes,pbar=False):
    if pbar: boxes=tqdm(boxes, desc="Caculating foreground points", leave=False)
    flag_list=[
        points_in_box(points, box) for box in boxes
    ]
    return np.stack(flag_list,axis=0)

def segPointsByBoxes(points, boxes, boxLabels=None, boxColors=None, lableColorMap=None, labelStartFromZero=True, overlapMode:str="first"):
    inBoxFlag=points_in_boxes(points,boxes[:,:7])   #(box_num, point_num)
    fgMask=inBoxFlag.sum(axis=0).astype(bool).reshape(-1,1)
    points_color=None
    if boxColors is None:
        if boxLabels is not None:
            boxColors=getLabelColor(boxLabels, lableColorMap, labelStartFromZero=labelStartFromZero)
        else:
            boxColors=np.ones([len(boxes), 3], dtype=float) * np.array([0,1,0], dtype=float)
    if overlapMode.lower() == 'first':
        point_box=np.argmax(inBoxFlag,axis=0)
        points_color=boxColors[point_box]
    elif overlapMode.lower() == 'average':
        points_color=np.dot(inBoxFlag.T,boxColors)
        in_sum=np.sum(inBoxFlag, axis=0).reshape(-1,1)
        in_sum[in_sum<1]=1
        points_color/=in_sum
    else: raise NotImplementedError("Overlap mode can only be [first|average]")
    return points_color, fgMask, boxColors

def capture_screen(vis,savepath=None, depth = False):
    img=None
    if savepath is not None:
        img=savepath
        savepath=str(savepath)
        if depth:
            vis.capture_depth_image(savepath, False)
        else:
            vis.capture_screen_image(savepath, False)
    else:
        if depth:
            img=vis.capture_depth_float_buffer()
        else:
            img=vis.capture_screen_float_buffer()
    return img


def pointsInView(xyzs,angle,offset=0):
    # z > |kx| + b
    rad=math.radians(angle)
    return xyzs[:,2]>= math.tan((math.radians(180)-rad)/2)*abs(xyzs[:,0])+offset

def inImgArea(x,y,width,height):
    if x>=0 and x<width and y>=0 and y<height:
        return True
    return False

def applyDistortionOnPoints(points,camera):
    u=points[:,0]
    v=points[:,1]
    other=points[:,2:]
    fx=camera.intrinsic[0,0]
    fy=camera.intrinsic[1,1]
    cx=camera.intrinsic[0,2]
    cy=camera.intrinsic[1,2]
    k1=camera.distortion.k1
    k2=camera.distortion.k2
    k3=camera.distortion.k3
    p1=camera.distortion.p1
    p2=camera.distortion.p2
    x=(u-cx)/fx
    y=(v-cy)/fy
    r2=x**2+y**2
    x_=x*(1+k1*r2+k2*(r2**2)+k3*(r2**3))+2*p1*x*y+p2*(r2+2*(x**2))
    y_=y*(1+k1*r2+k2*(r2**2)+k3*(r2**3))+p1*(r2+2*(y**2))+2*p2*x*y
    x_=x_*fx+cx
    y_=y_*fy+cy
    x_=x_.reshape(-1,1)
    y_=y_.reshape(-1,1)
    points=np.concatenate([x_,y_,other], axis=-1)
    return points

def applyDistortionOnO3dObject(object, camera):
    xyz=np.asarray(object.points)[:,:3]
    xyz=applyDistortionOnPoints(xyz, camera)
    object.points=open3d.utility.Vector3dVector(xyz[:, :3])
    return object

def distortionCorrection(img,camera):
    k1=camera.distortion.k1
    k2=camera.distortion.k2
    k3=camera.distortion.k3
    p1=camera.distortion.p1
    p2=camera.distortion.p2
    intrinsic=camera.intrinsic[:3,:3]
    distortions=np.array([k1, k2, p1, p2, k3])
    h, w = img.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(intrinsic, distortions, None,intrinsic, (w, h), 5)
    newimg=cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return newimg

def getIou(rec_1,rec_2):
    '''
    rec_1:左上角(rec_1[0],rec_1[1])    右下角：(rec_1[2],rec_1[3])
    rec_2:左上角(rec_2[0],rec_2[1])    右下角：(rec_2[2],rec_2[3])

    (rec_1)
    1--------1
    1   1----1------1
    1---1----1      1
        1           1
        1-----------1 (rec_2)
    '''
    s_rec1=(rec_1[2]-rec_1[0])*(rec_1[3]-rec_1[1])   #第一个bbox面积 = 长×宽
    s_rec2=(rec_2[2]-rec_2[0])*(rec_2[3]-rec_2[1])   #第二个bbox面积 = 长×宽
    sum_s=s_rec1+s_rec2                              #总面积
    left=max(rec_1[0],rec_2[0])                      #并集左上角顶点横坐标
    right=min(rec_1[2],rec_2[2])                     #并集右下角顶点横坐标
    bottom=max(rec_1[1],rec_2[1])                    #并集左上角顶点纵坐标
    top=min(rec_1[3],rec_2[3])                       #并集右下角顶点纵坐标
    iou=0.0
    if left >= right or top <= bottom:               #不存在并集的情况
        return 0.0
    else:
        inter=(right-left)*(top-bottom)              #求并集面积
        iou=(inter/(sum_s-inter))*1.0                #计算IOU
    return iou

def getBoundInImage(points,imgw=None,imgh=None):
    left=np.min(points[:,0])
    right=np.max(points[:,0])
    top=np.min(points[:,1])
    bottom=np.max(points[:,1])
    if imgw is not None:
        left=np.clip(left, 0, imgw-1)
        right=np.clip(right, 0, imgw-1)
    if imgh is not None:
        top=np.clip(top, 0, imgh-1)
        bottom=np.clip(bottom, 0, imgh-1)
    return left,top,right,bottom

def getviewbox(viewbox):
    line_set, box3d = translate_boxes_to_open3d_instance(viewbox)
    line_set.paint_uniform_color((0,1,0))
    return line_set

def getImage(images, key):
    blackImage=np.zeros([1000,1000,3], dtype=np.uint8)
    if key is None: return blackImage
    if isinstance(key, int):
        if key<len(images) and key >=0:
            return images[key]['image'][...,:3]
        else:
            return blackImage
    elif isinstance(key, str):
        for index, image in enumerate(images):
            if image['name']==key:
                return image['image'][...,:3]
        return blackImage
    else:
        return blackImage

def applyImageLayout(images, layout):
    result=None
    imageHs=[]
    imageWs=[]
    for img in images:
        h, w=img['image'].shape[:2]
        imageHs.append(h)
        imageWs.append(w)
    H=int(np.average(imageHs))
    W=int(np.average(imageWs))
    rowShapes=[]
    for i,row in enumerate(layout):
        if isinstance(row, list):
            rowShapes.append(len(row))
        else:
            rowShapes.append(1)
    maxRowLen=max(rowShapes)

    if maxRowLen == 1:
        rows=[cv2.resize(getImage(images,key), (W, H)) for key in layout]
        result=np.hstack(rows)
    else:
        rowW=maxRowLen*W
        rows=[]
        for row in layout:
            rowImage=None
            if isinstance(row, list):
                rowImage=[cv2.resize(getImage(images,key), (W, H)) for key in row]
                rowImage=np.hstack(rowImage)
            else:
                rowImage=cv2.resize(getImage(images,row), (W, H))
            rowImage=cv2.resize(rowImage, (rowW, H))
            rows.append(rowImage)
        result=np.vstack(rows)
    return result


################################################################################################################################################