from easydict import EasyDict
import open3d
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
# from utils.tools import GetBuildInViewFile, CheckDataFormat, getFormattedData, get_o3d_pcd, get_box, getaxis, colorMap, loadJson, saveJson, showImgCV2, saveImgCV2, addLabelcolorLegend, generateVideo,getLabelColor, segPointsByBoxes
from utils.tools import *
from utils.open3d_visualizer import draw as draw_single
from utils.open3d_visualizer import open3dSequenceVisualizer, projectionOpenCV
from utils.camera import Camera


def draw(points,
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
         savePath=None,
         windowWidth=1280,
         windowHeight=720,
         windowName='Open3D',
         drawMode='det',
         backgroundColor='black',
         viewFile='$birdeye',
         pointSize=1.0,
         camera3dLock=False,
         drawOriginAxis=True,
         drawSpeedArrow=False,
         drawLabelLegend=True,
         segPointsBy=None,
         scoreThresh=0.0,
         labelName=None,
         lableColorMap=None,
         labelStartFromZero=True,
         showProgressBar=False,
         workDir=None,
):
    """
    :param points: Point cloud data, shape like (x,y,z) or (x,y,z,intensity) or (x,y,z,r,g,b)
    :param pointLabels: Labels for points, numbers
    :param pointNames: Names for points
    :param pointColors: Colors for points, when this data is given, segPointsBy will be disabled
    :param gtBoxes: Ground truth boxes, shape like (cx, cy, cz, dx, dy, dz, rz), won't be colored when drawMode is "detection"
    :param gtLabels: Labels for ground truth boxes
    :param gtNames: Names for ground truth boxes labels
    :param refBoxes: Reference boxes, shape like (cx, cy, cz, dx, dy, dz, rz, [vx, vy]), will be colored by their labels when drawMode is "detection"
    :param refLabels: Labels for reference boxes
    :param refNames: Names for reference boxes labels
    :param refScores: Scores for reference boxes
    :param savePath: Save path for visualization results, should be a specific document when there is only one frame of data, or it should be a folder path
    :param windowWidth: Width of visualization window (px)
    :param windowHeight: Height of visualization window (px)
    :param windowName: Name of visualization window
    :param drawMode: The way to draw objects, choosed in [detection|segmentation]
    :param backgroundColor: The background color of visualization window, rgb(0,0,0)-(1,1,1)
    :param viewFile: Camera params file for visualization, use $ to access built-in files like "$birdeye"
    :param pointSize: Point size when draw point cloud, default 1.0
    :param camera3dLock: Lock the camera of visualization progress, used for automatically visualizing a sequence of data
    :param drawOriginAxis: Place an xyz axis at the origin point
    :param drawSpeedArrow: Draw a speed arrow for every reference box if their shape is 9 (cx, cy, cz, dx, dy, dz, rz, [vx, vy])
    :param drawLabelLegend: Draw legend for colored labels(only in the saved image)
    :param segPointsBy: Paint point cloud with [None|label|gtbox|refbox], when this option is not None, the drawMode will be set to "segmentation"
    :param scoreThresh: Score threshhold if your reference data has scores
    :param labelName: Names for labels, should be correspondence in numerical order
    :param lableColorMap: Colors for labels, should be correspondence in numerical order, rgb(0,0,0)-(1,1,1)
    :param labelStartFromZero: Whether the labels is started from 0, should be checked or it may cause incorrect label and name correspondence
    :param showProgressBar: Whether to display a progress bar when more than 1 frame of data are inputted
    :param workDir: The default save path of visualization results
    """
    vis = Visualizer(windowWidth=windowWidth,
                     windowHeight=windowHeight,
                     windowName=windowName,
                     drawMode=drawMode,
                     backgroundColor=backgroundColor,
                     viewFile=viewFile,
                     pointSize=pointSize,
                     camera3dLock=camera3dLock,
                     drawOriginAxis=drawOriginAxis,
                     drawSpeedArrow=drawSpeedArrow,
                     drawLabelLegend=drawLabelLegend,
                     segPointsBy=segPointsBy,
                     scoreThresh=scoreThresh,
                     labelName=labelName,
                     lableColorMap=lableColorMap,
                     labelStartFromZero=labelStartFromZero,
                     showProgressBar=showProgressBar,
                     workDir=workDir,)
    value = vis.draw3d(points=points,
                       pointLabels=pointLabels,
                       pointNames=pointNames,
                       pointColors=pointColors,
                       gtBoxes=gtBoxes,
                       gtLabels=gtLabels,
                       gtNames=gtNames,
                       refBoxes=refBoxes,
                       refLabels=refLabels,
                       refNames=refNames,
                       refScores=refScores,
                       savePath=savePath)
    return value


class Visualizer():
    def __init__(self, *kargs, **kwargs):
        """
        :param windowWidth: Width of visualization window (px)
        :param windowHeight: Height of visualization window (px)
        :param windowName: Name of visualization window
        :param drawMode: The way to draw objects, choosed in [detection|segmentation]
        :param backgroundColor: The background color of visualization window, rgb(0,0,0)-(1,1,1)
        :param viewFile: Camera params file for visualization, use $ to access built-in files like "$birdeye"
        :param pointSize: Point size when draw point cloud, default 1.0
        :param camera3dLock: Lock the camera of visualization progress, used for automatically visualizing a sequence of data
        :param drawOriginAxis: Place an xyz axis at the origin point
        :param drawSpeedArrow: Draw a speed arrow for every reference box if their shape is 9 (cx, cy, cz, dx, dy, dz, rz, [vx, vy])
        :param drawLabelLegend: Draw legend for colored labels(only in the saved image)
        :param segPointsBy: Paint point cloud with [None|label|gtbox|refbox], when this option is not None, the drawMode will be set to "segmentation"
        :param scoreThresh: Score threshhold if your reference data has scores
        :param labelName: Names for labels, should be correspondence in numerical order
        :param lableColorMap: Colors for labels, should be correspondence in numerical order, rgb(0,0,0)-(1,1,1)
        :param labelStartFromZero: Whether the labels is started from 0, should be checked or it may cause incorrect label and name correspondence
        :param showProgressBar: Whether to display a progress bar when more than 1 frame of data are inputted
        :param workDir: The default save path of visualization results
        """
        self.buildInParams = ('config', 'sequenceVis',
                              'frameNum', 'savedImages', 'lastCameraParam')
        self.config = EasyDict()
        self.sequenceVis = None
        self.frameNum = 0
        self.savedImages = []
        self.lastCameraParam = None
        self.initConfig(*kargs, **kwargs)

    def initConfig(self,
                   windowWidth=1280,
                   windowHeight=720,
                   windowName='Open3D',
                   drawMode='det',
                   backgroundColor='black',
                   viewFile='$birdeye',
                   pointSize=1.0,
                   camera3dLock=False,
                   drawOriginAxis=True,
                   drawSpeedArrow=False,
                   drawLabelLegend=True,
                   segPointsBy=None,
                   scoreThresh=0.0,
                   labelName=None,
                   lableColorMap=None,
                   labelStartFromZero=True,
                   showProgressBar=False,
                   workDir=None,
                   cameras2d=[],
                   imagesLayout=None,
                   draw2dWith3d=False,
                   saveOnly2d=False,
                   boxType2d='3d',
                   coordinateBase2d='3d',
                   **kwargs):
        """
        :param windowWidth: Width of visualization window (px)
        :param windowHeight: Height of visualization window (px)
        :param windowName: Name of visualization window
        :param drawMode: The way to draw objects, choosed in [detection|segmentation]
        :param backgroundColor: The background color of visualization window, rgb(0,0,0)-(1,1,1)
        :param viewFile: Camera params file for visualization, use $ to access built-in files like "$birdeye"
        :param pointSize: Point size when draw point cloud, default 1.0
        :param camera3dLock: Lock the camera of visualization progress, used for automatically visualizing a sequence of data
        :param drawOriginAxis: Place an xyz axis at the origin point
        :param drawSpeedArrow: Draw a speed arrow for every reference box if their shape is 9 (cx, cy, cz, dx, dy, dz, rz, [vx, vy])
        :param drawLabelLegend: Draw legend for colored labels(only in the saved image)
        :param segPointsBy: Paint point cloud with [None|label|gtbox|refbox], when this option is not None, the drawMode will be set to "segmentation"
        :param scoreThresh: Score threshhold if your reference data has scores
        :param labelName: Names for labels, should be correspondence in numerical order
        :param lableColorMap: Colors for labels, should be correspondence in numerical order, rgb(0,0,0)-(1,1,1)
        :param labelStartFromZero: Whether the labels is started from 0, should be checked or it may cause incorrect label and name correspondence
        :param showProgressBar: Whether to display a progress bar when more than one frame of data are inputted
        :param workDir: The default save path of visualization results
        """
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.windowName = windowName
        self.drawMode=drawMode
        self.viewFile = viewFile
        self.pointSize = pointSize
        self.drawOriginAxis = drawOriginAxis
        self.drawSpeedArrow = drawSpeedArrow
        self.drawLabelLegend = drawLabelLegend
        self.segPointsBy = segPointsBy
        self.scoreThresh = scoreThresh
        self.labelName = labelName
        self.lableColorMap = lableColorMap
        self.labelStartFromZero = labelStartFromZero
        self.showProgressBar = showProgressBar
        self.backgroundColor = backgroundColor
        self.camera3dLock = camera3dLock
        self.cameras2d = cameras2d
        self.workDir = workDir
        self.imagesLayout = imagesLayout
        self.draw2dWith3d=draw2dWith3d
        self.saveOnly2d=saveOnly2d
        self.boxType2d=boxType2d
        self.coordinateBase2d=coordinateBase2d

    def __setitem__(self, key: str, value):
        # print(f"__setitem__ {key} {value}")
        if key in ['backgroundColor', 'foregroundColor']:
            if isinstance(value, str):
                if value in colorMap:
                    self.config[key] = colorMap[value]
                else:
                    print("Unsupported color name, set color to black")
                    self.config[key] = colorMap['black']
            else:
                assert len(value) == 3
                color = np.array(value, dtype=float)
                if np.max(color) > 1:
                    color /= np.max(color)
                self.config[key] = color
        elif key == 'workDir' and value is not None:
            workDir = Path(value).absolute()
            workDir.mkdir(exist_ok=True, parents=True)
            self.config[key] = value
        elif key.startswith('segPointsBy'):
            paintKey = key.removeprefix('segPointsBy')
            if len(paintKey) == 0:
                if value is not None:
                    paintKey = value
                    value = True
                else:
                    self.config.segPointsBy = None
                    if not 'detection'.startswith(self.drawMode):
                        self.config.drawMode = 'detection'
                    return
            paintKey = paintKey.lower()
            assert paintKey in [
                'label', 'gtbox', 'refbox'], "You can only paint points by [label|gtbox|refbox]"
            if value:
                self.config.segPointsBy = paintKey
                if not 'segmentation'.startswith(self.drawMode):
                    self.config.drawMode = 'segmentation'
            elif self.config.get('segPointsBy', None) == paintKey:
                self.config.segPointsBy = None
        elif key == 'drawMode':
            value=value.lower()
            if 'detection'.startswith(value):
                self.config.drawMode = 'detection'
                self.config.segPointsBy = None
            elif 'segmentation'.startswith(value):
                self.config.drawMode = 'segmentation'
                if self.config.get('segPointsBy', None) is None:
                    self.config.segPointsBy = 'label'
            else: raise NotImplementedError("Draw mode should be [detection|segmentation]")
        elif key == 'cameras2d':
            self.config[key] = []
            if value is not None:
                if isinstance(value, list):
                    for camera in value:
                        self.addCamera2d(camera)
                elif isinstance(value, Camera) or isinstance(value, dict):
                    self.addCamera2d(value)
                else: raise ValueError("You can directly add a camera only by Camera object or params dict")
        else:
            self.config[key] = value

    def __getattribute__(self, name):
        # print(f"__getattribute__ {name}")
        if name == 'buildInParams' or name in object.__getattribute__(self, 'buildInParams'):
            return object.__getattribute__(self, name)
        elif name in ['viewFile']:
            viewFile = self.config.viewFile
            if viewFile.startswith('$'):
                viewFile = GetBuildInViewFile(
                    viewFile.removeprefix('$'))
            return viewFile
        elif name in ['workDir']:
            workDir=self.config.get('workDir',None)
            if workDir is None: return workDir
            workDir = Path(workDir).absolute()
            workDir.mkdir(exist_ok=True, parents=True)
            return workDir
        elif name in object.__getattribute__(self, 'config').keys():
            return self.config[name]
        elif name in ['foregroundColor']:
            return 1 - self.config.backgroundColor
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        # print(f"__setattr__ {name} {value}")
        if name == 'buildInParams' or name in self.buildInParams:
            object.__setattr__(self, name, value)
        else:
            self.__setitem__(name, value)

    def initSequenceVis(self):
        if self.camera3dLock:
            self.sequenceVis = open3dSequenceVisualizer(
                windowName=self.windowName,
                windowHeight=self.windowHeight,
                windowWidth=self.windowWidth,
                backgroundColor=self.backgroundColor,
                viewFile=self.viewFile,
                pointSize=self.pointSize,
            )

    def loadConfig(self, configPath):
        configPath = Path(configPath).absolute()
        config = loadJson(configPath)
        self.initConfig(**config)
        print(f"Config loaded from {configPath}")

    def saveConfig(self, savePath=None):
        if self.workDir is not None and savePath is None:
            savePath = self.workDir
        savePath = Path(savePath).absolute()
        if savePath.exists() and savePath.is_dir():
            savePath = savePath / 'config.json'
        config = EasyDict(self.config.copy())
        config.backgroundColor = tuple(config.backgroundColor)
        if 'foregroundColor' in config:
            config.foregroundColor = tuple(config.foregroundColor)
        if 'cameras2d' in config:
            for i in range(len(config.cameras2d)):
                camera:Camera=config.cameras2d[i]
                camera=camera.jsonObject()
                config.cameras2d[i]=camera
        saveJson(config, savePath)
        print(f"Config saved to {savePath}")

    def _getO3dGeometries(self,
                         points=None,
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
                         **kwargs):
        pcd_o3d = None
        gt_boxes_o3d = []
        ref_boxes_o3d = []
        other_geometries = []
        ref_label_color_dict = None
        legendLabels=None
        legendNames=None
        legendColors=None
        cur_points = points

        if gt_boxes is not None:
            if self.labelName is not None:
                offset = 0 if self.labelStartFromZero else 1
                if gt_labels is None and gt_names is not None:
                    gt_labels = [self.labelName.index(
                        name)+offset for name in gt_names]
                if gt_names is None and gt_labels is not None:
                    gt_names = [self.labelName[int(
                        label)-offset] for label in gt_labels]
            if 'detection'.startswith(self.drawMode):
                gt_boxes_o3d, _, _ = get_box(gt_boxes, self.foregroundColor)
        if ref_boxes is not None:
            if self.labelName is not None:
                offset = 0 if self.labelStartFromZero else 1
                if ref_labels is None and ref_names is not None:
                    ref_labels = [self.labelName.index(
                        name)+offset for name in ref_names]
                if ref_names is None and ref_labels is not None:
                    ref_names = [self.labelName[int(
                        label)-offset] for label in ref_labels]
            if 'detection'.startswith(self.drawMode):
                ref_boxes_o3d, ref_arrows_o3d, ref_colors = get_box(
                    ref_boxes, (0, 1, 0), labels=ref_labels, colormap=self.lableColorMap, withArrow=self.drawSpeedArrow, labelStartFromZero=self.labelStartFromZero)
                if ref_labels is not None:
                    legendLabels=ref_labels
                    legendNames=ref_names
                    legendColors=ref_colors
                other_geometries.extend(ref_arrows_o3d)

        if self.labelName is not None:
            offset = 0 if self.labelStartFromZero else 1
            if point_labels is None and point_names is not None:
                point_labels = [self.labelName.index(
                    name)+offset for name in point_names]
            if point_names is None and point_labels is not None:
                point_names = [self.labelName[int(
                    label)-offset] for label in point_labels]
        if cur_points is not None:
            xyz = cur_points[:, :3]
            pointNum, pointDim = cur_points.shape
            if point_colors is None:
                if pointDim == 3:
                    point_colors = np.repeat(
                        self.foregroundColor, pointNum).reshape(pointNum, -1)
                elif pointDim == 4:
                    intencity = cur_points[:, 3]
                    point_colors = plt.get_cmap("terrain")(intencity)[:, :3]
                elif pointDim == 6:
                    point_colors = cur_points[:, 3:6]
                else:
                    raise ValueError("points shape error")
                if 'segmentation'.startswith(self.drawMode):
                    fgMask=None
                    if self.segPointsBy == 'label' and point_labels is not None:
                        bg_mask=(point_labels<0)
                        point_labels[bg_mask]=0
                        point_colors=getLabelColor(point_labels,colormap=self.lableColorMap, labelStartFromZero=self.labelStartFromZero)
                        point_colors[bg_mask]=self.foregroundColor
                        point_names=np.array(point_names)
                        point_labels[bg_mask]=-1
                        point_names[bg_mask]='backgroud'
                        legendLabels=point_labels
                        legendNames=point_names
                        legendColors=point_colors
                    if self.segPointsBy == 'gtbox' and gt_boxes is not None:
                        fgColors, fgMask, boxColors=segPointsByBoxes(points,boxes=gt_boxes,boxLabels=gt_labels,lableColorMap=self.lableColorMap, labelStartFromZero=self.labelStartFromZero)
                        if gt_labels is not None:
                            legendLabels=gt_labels
                            legendNames=gt_names
                            legendColors=boxColors
                    if self.segPointsBy == 'refbox' and ref_boxes is not None:
                        fgColors, fgMask, boxColors=segPointsByBoxes(points,boxes=ref_boxes,boxLabels=ref_labels,lableColorMap=self.lableColorMap, labelStartFromZero=self.labelStartFromZero)
                        if ref_labels is not None:
                            legendLabels=ref_labels
                            legendNames=ref_names
                            legendColors=boxColors
                    if fgMask is not None:
                        point_colors=fgColors*fgMask+point_colors*(~fgMask)
            if np.max(point_colors) > 1:
                point_colors /= np.max(point_colors)
            pcd_o3d = get_o3d_pcd(xyz, point_colors)


        if legendLabels is not None and legendColors is not None:
            ref_label_color_dict = {}
            for i in range(len(legendLabels)):
                name = legendLabels[i]
                if legendNames is not None:
                    name = legendNames[i]
                name = str(name)
                if name not in ref_label_color_dict:
                    ref_label_color_dict[name] = legendColors[i]

        if self.drawOriginAxis:
            axis_pcd, _ = getaxis()
            other_geometries.append(axis_pcd)
        return pcd_o3d, list(gt_boxes_o3d), list(ref_boxes_o3d), other_geometries, ref_label_color_dict

    def _draw3dSingle(self, formattedData):
        pcd_o3d, gt_boxes_o3d, ref_boxes_o3d, other_geometries, ref_label_color_dict = self._getO3dGeometries(
            **formattedData)
        boxes_o3d = gt_boxes_o3d+ref_boxes_o3d
        if not self.camera3dLock:
            screen_shot_3d, camera_param = draw_single(pcd_o3d, boxes_o3d, other_geometries,
                                                       view_file=self.viewFile, pointSize=self.pointSize, backgroundColor=self.backgroundColor)
        else:
            if self.sequenceVis is None:
                self.initSequenceVis()
            screen_shot_3d, camera_param = self.sequenceVis.update(
                pcd_o3d, boxes_o3d, other_geometries)
        self.lastCameraParam = camera_param
        screen_shot_3d_h, screen_shot_3d_w, _ = screen_shot_3d.shape
        if self.drawLabelLegend and ref_label_color_dict is not None:
            screen_shot_3d = addLabelcolorLegend(screen_shot_3d, ref_label_color_dict, colortype='open3d', lineheight=(
                (screen_shot_3d_h//1000+1)*20))
        return screen_shot_3d

    def draw3d(self,
               points=None,
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
               savePath=None,
               ):
        """
        :param points: Point cloud data, shape like (x,y,z) or (x,y,z,intensity) or (x,y,z,r,g,b)
        :param pointLabels: Labels for points, numbers less than 0 will be seen as backgourd points
        :param pointNames: Names for points
        :param pointColors: Colors for points, when this data is given, segPointsBy will be disabled
        :param gtBoxes: Ground truth boxes, shape like (cx, cy, cz, dx, dy, dz, rz), won't be colored when drawMode is "detection"
        :param gtLabels: Labels for ground truth boxes
        :param gtNames: Names for ground truth boxes labels
        :param refBoxes: Reference boxes, shape like (cx, cy, cz, dx, dy, dz, rz, [vx, vy]), will be colored by their labels when drawMode is "detection"
        :param refLabels: Labels for reference boxes
        :param refNames: Names for reference boxes labels
        :param refScores: Scores for reference boxes
        :param savePath: Save path for visualization results, should be a specific document when there is only one frame of data, or it should be a folder path
        """
        DATAS = PrepareData(
                points=points,
                pointLabels=pointLabels,
                pointNames=pointNames,
                pointColors=pointColors,
                gtBoxes=gtBoxes,
                gtLabels=gtLabels,
                gtNames=gtNames,
                refBoxes=refBoxes,
                refLabels=refLabels,
                refNames=refNames,
                refScores=refScores,
                scoreThresh=self.scoreThresh)
        SEQ_LEN = len(DATAS)
        pbar = None
        images=[]
        if self.showProgressBar and SEQ_LEN > 1:
            pbar = tqdm(total=SEQ_LEN)
        if self.workDir is not None and savePath is None:
            savePath = self.workDir
        if savePath is not None:
            savePath = Path(savePath).absolute()
            if SEQ_LEN > 1 or self.camera3dLock:
                savePath.mkdir(exist_ok=True, parents=True)
            elif SEQ_LEN == 1:
                savePath.parent.mkdir(exist_ok=True, parents=True)
        for index in range(SEQ_LEN):
            formattedData = DATAS[index]
            screen_shot_3d = self._draw3dSingle(formattedData)
            if savePath is not None:
                filePath = savePath
                if savePath.exists() and savePath.is_dir():
                    filePath = savePath / f"{self.frameNum}.png"
                saveImgCV2(screen_shot_3d, filePath)
                self.savedImages.append(filePath)
            self.frameNum += 1
            if pbar is not None:
                pbar.update()
            images.append(screen_shot_3d)
        if len(images)==1:
            images=images[0]
        return images

    def saveLastCameraParam(self, savePath=None):
        if self.workDir is not None and savePath is None:
            savePath = self.workDir
        savePath = Path(savePath).absolute()
        if savePath.exists() and savePath.is_dir():
            savePath = savePath / 'view.json'
        open3d.io.write_pinhole_camera_parameters(
            str(savePath), self.lastCameraParam)
        print(f"Last camera param saved to {savePath}")

    def saveVideo(self, savePath=None, fps=None, videoWidth=None, videoHeight=None, pbar=True):
        if self.workDir is not None and savePath is None:
            savePath = self.workDir
        savePath = Path(savePath).absolute()
        if savePath.exists() and savePath.is_dir():
            savePath = savePath / 'video.avi'
        videoPath = generateVideo(
            savePath, self.savedImages, fps, videoWidth, videoHeight, pbar=pbar)
        print(f"Video saved to {videoPath}")
        return videoPath

    def getCamera2d(self, key):
        if isinstance(key, int):
            assert key>=0
            if key<len(self.cameras2d):
                return self.cameras2d[key], key
            else:
                return None
        elif isinstance(key, str):
            for index, camera in enumerate(self.cameras2d):
                if camera.name==key:
                    return camera, index
            return None
        else:
            raise KeyError("Camera can only be indexed by name or order number")

    def addCamera2d(self, camera=None, **cameraParams):
        newCamera=None
        if camera is not None:
            if isinstance(camera, Camera):
                newCamera=camera
            elif isinstance(camera, dict):
                newCamera=Camera.createFromDict(camera)
            else: raise ValueError("You can directly add a camera only by Camera object or params dict")
        else:
            newCamera=Camera(**cameraParams)
        name=newCamera.name
        if self.getCamera2d(name) is not None:
            index=1
            newname=f"camera{index}"
            while self.getCamera2d(name) is not None:
                index+=1
                newname=f"camera{index}"
            newCamera.name=newname
        self.cameras2d.append(newCamera)

    def _draw2dSingle(self, formattedData):
        pcd_o3d, gt_boxes_o3d, ref_boxes_o3d, other_geometries, ref_label_color_dict = self._getO3dGeometries(
            **formattedData)
        images=[]
        for i in range(len(self.cameras2d)):
            camera:Camera=EasyDict(self.cameras2d[i].data())
            _pcd_o3d=copy.deepcopy(pcd_o3d)
            point_labels=formattedData['point_labels']
            point_names=formattedData['point_names']
            _ref_boxes_o3d:list=copy.deepcopy(ref_boxes_o3d)
            ref_labels=formattedData['ref_labels']
            ref_names=formattedData['ref_names']
            # extrinsic transform
            if _pcd_o3d is not None:
                _pcd_o3d.transform(camera.extrinsic)
                points=np.asarray(_pcd_o3d.points)
                colors=np.asarray(_pcd_o3d.colors)
                validMask=pointsInView(points,camera.viewangle)
                points=points[validMask]
                colors=colors[validMask]
                _pcd_o3d.points = open3d.utility.Vector3dVector(points)
                _pcd_o3d.colors = open3d.utility.Vector3dVector(colors)
            if _ref_boxes_o3d is not None:
                for i in range(len(_ref_boxes_o3d)-1,-1,-1):
                    box=_ref_boxes_o3d[i]
                    box.transform(camera.extrinsic)
                    points=np.asarray(box.points)
                    validMask=pointsInView(points,camera.viewangle)
                    validPtNum=np.sum(validMask)
                    if validPtNum <=1:
                        _ref_boxes_o3d.pop(i)
                        continue
                    box.points = open3d.utility.Vector3dVector(points)
            # *******DEBUG CODE********
            # draw_single(_pcd_o3d, _ref_boxes_o3d, other_geometries)
            # *******DEBUG CODE********
            # intrinsic transform
            if _pcd_o3d is not None:
                points=np.asarray(_pcd_o3d.points)
                colors=np.asarray(_pcd_o3d.colors)
                points[points[:,2]==0][:,2]=1
                points/=points[:,None,2]
                _pcd_o3d.points = open3d.utility.Vector3dVector(points)
                _pcd_o3d.colors = open3d.utility.Vector3dVector(colors)
                _pcd_o3d.transform(camera.intrinsic)
            if _ref_boxes_o3d is not None:
                for i in range(len(_ref_boxes_o3d)-1,-1,-1):
                    box=_ref_boxes_o3d[i]
                    points=np.asarray(box.points)
                    points[points[:,2]==0][:,2]=1
                    points/=points[:,None,2]
                    box.points = open3d.utility.Vector3dVector(points)
                    box.transform(camera.intrinsic)
            # *******DEBUG CODE********
            # width=camera.width
            # height=camera.height
            # viewbox2d=getviewbox([width/2,height/2,0,width,height,0,0,0,0])
            # imgaxis,_=getaxis(size=height/2)
            # draw_single(_pcd_o3d, _ref_boxes_o3d, other_geometries=[viewbox2d, imgaxis])
            # *******DEBUG CODE********
            photo=None
            if formattedData['photos'] is not None:
                if isinstance(formattedData['photos'], list) and i<len(formattedData['photos']):
                    photo=formattedData['photos'][i]
                elif isinstance(formattedData['photos'], dict):
                    photo=formattedData['photos'].get(camera.name,None) 
            newimg=projectionOpenCV(photo,pcd=_pcd_o3d, boxes3d=_ref_boxes_o3d, camera=camera, coordinateBase=self.coordinateBase2d,boxType=self.boxType2d)
            images.append(dict(
                name=camera.name,
                image=newimg,
            ))
        return images

    def draw2d(self,
               points=None,
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
               savePath=None,
               photos=None,):
        DATAS = PrepareData(
                points=points,
                pointLabels=pointLabels,
                pointNames=pointNames,
                pointColors=pointColors,
                gtBoxes=gtBoxes,
                gtLabels=gtLabels,
                gtNames=gtNames,
                refBoxes=refBoxes,
                refLabels=refLabels,
                refNames=refNames,
                refScores=refScores,
                photos=photos,
                scoreThresh=self.scoreThresh)
        SEQ_LEN = len(DATAS)
        pbar = None
        images=[]
        if self.showProgressBar and SEQ_LEN > 1:
            pbar = tqdm(total=SEQ_LEN)
        if self.workDir is not None and savePath is None:
            savePath = self.workDir
        if savePath is not None:
            savePath = Path(savePath).absolute()
            if SEQ_LEN > 1 or self.camera3dLock:
                savePath.mkdir(exist_ok=True, parents=True)
            elif SEQ_LEN == 1:
                savePath.parent.mkdir(exist_ok=True, parents=True)
        for index in range(SEQ_LEN):
            formattedData = DATAS[index]
            images2d=self._draw2dSingle(formattedData)
            if self.draw2dWith3d:
                screen_shot_3d = self._draw3dSingle(formattedData)
                images2d.append(dict(
                    name='3d',
                    image=screen_shot_3d
                ))
            if self.imagesLayout is not None:
                finalImage=applyImageLayout(images2d, self.imagesLayout)
                if not self.saveOnly2d:
                    showImgCV2(finalImage, width=self.windowWidth, title=self.windowName)
                if savePath is not None:
                    filePath = savePath
                    if savePath.exists() and savePath.is_dir():
                        filePath = savePath / f"{self.frameNum}.png"
                    saveImgCV2(finalImage, filePath)
                    self.savedImages.append(filePath)
            else:
                
                for item in images2d:
                    image=item['image']
                    name=item['name']
                    if not self.saveOnly2d: showImgCV2(image, self.windowHeight, self.windowWidth, name)
                    if savePath is not None:
                        if savePath.exists() and savePath.is_dir():
                            filePath = savePath / f"{self.frameNum}_{name}.png"
                        else:
                            filePath = savePath.parent / f"{savePath.stem}_{name}{savePath.suffix}"
                        saveImgCV2(image, filePath)
                        self.savedImages.append(filePath)
                finalImage=images2d
            self.frameNum+=1
            if pbar is not None:
                pbar.update()
            images.append(finalImage)
        return images