
from pathlib import Path
ROOT = Path(__file__).absolute().parent.parent
import sys
sys.path.append(str(ROOT))
from easydict import EasyDict
from tqdm import tqdm
from visualizer import draw, Visualizer, Camera
from utils.tools import *

nusc_map = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

def demo_draw_single(datas):
    for data in tqdm(datas):
        data = EasyDict(data)
        points = data.points
        gt_boxes = data.gt_boxes
        gt_names = data.gt_names
        pred_boxes = data.pred_boxes
        pred_scores = data.pred_scores
        pred_labels = data.pred_labels
        pred_names = data.pred_names
        draw(points, gtBoxes=gt_boxes, gtNames=gt_names, refBoxes=pred_boxes, refLabels=pred_labels,
             refScores=pred_scores, refNames=pred_names, labelName=nusc_map, backgroundColor='black')

def demo_draw_sequence(datas):
    points = []
    gt_boxes = []
    gt_names = []
    pred_boxes = []
    pred_scores = []
    pred_labels = []
    pred_names = []
    for data in datas:
        data = EasyDict(data)
        points.append(data.points)
        gt_boxes.append(data.gt_boxes)
        gt_names.append(data.gt_names)
        pred_boxes.append(data.pred_boxes)
        pred_scores.append(data.pred_scores)
        pred_labels.append(data.pred_labels)
        pred_names.append(data.pred_names)
    draw(points, gtBoxes=gt_boxes, gtNames=gt_names, refBoxes=pred_boxes, refLabels=pred_labels,
         refScores=pred_scores, refNames=pred_names, labelName=nusc_map, backgroundColor='black', camera3dLock=True, showProgressBar=True, scoreThresh=0.3)

def demo_draw_sequence_update(datas):
    V=Visualizer()
    V.labelName=nusc_map
    V.backgroundColor='black'
    V.camera3dLock=True
    V.showProgressBar=True
    V.scoreThresh=0.3
    V.labelStartFromZero=False
    V.segPointsBylabel=True
    V.pointSize=2.0
    V.workDir="./temptest"
    for data in datas:
        data = EasyDict(data)
        points = data.points
        point_labels = data.point_labels
        gt_boxes = data.gt_boxes
        gt_names = data.gt_names
        pred_boxes = data.pred_boxes
        pred_scores = data.pred_scores
        pred_labels = data.pred_labels
        pred_names = data.pred_names
        V.draw3d(points[:,:3], pointLabels=point_labels,gtBoxes=gt_boxes, gtNames=gt_names, refBoxes=pred_boxes, refLabels=pred_labels,
             refScores=pred_scores, refNames=pred_names)
    V.saveVideo()
    V.saveLastCameraParam()
    V.saveConfig()

def demo_draw_2d(datas):
    V=Visualizer()
    V.labelName=nusc_map
    V.camera3dLock=True
    V.labelStartFromZero=False
    V.workDir="./temptest"
    V.scoreThresh=0.5
    V.imagesLayout=[
        [None, 0, 'front'],
        [-1, '3d', -1]
    ]
    V.draw2dWith3d=True
    V.saveOnly2d=False
    V.boxType2d='2d'
    cameraFront=Camera('front')
    cameraFront.extrinsic.addTranslation(0,-1,0.5)
    cameraFront.extrinsic.addRotation(rx=np.pi/2)
    cameraFront.intrinsic.matrix=[
     [1950, 0, 1950],
     [0, 1950, 1150],
     [0, 0, 1]
    ]
    cameraFront.width=3840
    cameraFront.height=2160
    cameraFront.viewangle=120
    V.addCamera2d(cameraFront)
    for data in tqdm(datas):
        data = EasyDict(data)
        points = data.points
        point_labels = data.point_labels
        gt_boxes = data.gt_boxes
        gt_names = data.gt_names
        pred_boxes = data.pred_boxes
        pred_scores = data.pred_scores
        pred_labels = data.pred_labels
        pred_names = data.pred_names
        V.draw2d(points,
         pointLabels=point_labels,gtBoxes=gt_boxes, gtNames=gt_names, refBoxes=pred_boxes, refLabels=pred_labels,refScores=pred_scores, refNames=pred_names,
         )
    V.saveVideo()
    V.saveConfig()

def main():
    datapath = ROOT / 'data' / 'demo.pkl'
    datas = loadPickle(datapath)[:5]
    demo_draw_single(datas)
    demo_draw_sequence(datas)
    demo_draw_sequence_update(datas)
    demo_draw_2d(datas)


if __name__ == "__main__":
    main()
