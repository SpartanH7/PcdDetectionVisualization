
import os
import open3d
import matplotlib
import numpy as np
from .tools import *


def draw(points=None, boxes=[], other_geometries=[],
         backgroundColor=(0, 0, 0), pointSize=1.0,
         view_file=None, width=1280, height=720, window_name='Open3D'):
    vis = open3d.visualization.Visualizer()
    if view_file is not None and os.path.exists(view_file):
        view_cfg = loadJson(view_file)
        width = view_cfg['intrinsic']['width']
        height = view_cfg['intrinsic']['height']
    vis.create_window(width=width, height=height, window_name=window_name)

    vis.get_render_option().point_size = pointSize
    vis.get_render_option().background_color = backgroundColor

    if points is not None: vis.add_geometry(points)

    for i, box in enumerate(boxes):
        vis.add_geometry(box)

    for i, item in enumerate(other_geometries):
        vis.add_geometry(item)

    if view_file is not None and os.path.exists(view_file):
        ctr = vis.get_view_control()
        param = open3d.io.read_pinhole_camera_parameters(view_file)
        ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    img = img_open3d_to_opencv(np.asarray(capture_screen(vis), dtype=float))
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    vis.destroy_window()
    return img, param


class open3dSequenceVisualizer():
    def __init__(self, windowName='Open3D', windowWidth=1280, windowHeight=720, backgroundColor=(0, 0, 0), viewFile='./view.json', pointSize=1.0):
        self.vis = open3d.visualization.Visualizer()
        if viewFile is not None and os.path.exists(viewFile):
            view_cfg = loadJson(viewFile)
            windowWidth = view_cfg['intrinsic']['width']
            windowHeight = view_cfg['intrinsic']['height']
        self.vis.create_window(window_name=windowName, width=windowWidth, height=windowHeight)
        opt = self.vis.get_render_option()
        opt.background_color = backgroundColor
        opt.point_size = pointSize
        self.param = open3d.io.read_pinhole_camera_parameters(viewFile)
        self.ctr = self.vis.get_view_control()

        self.framenum = 0
        self.geometry_list = []

    def __del__(self):
        self.vis.destroy_window()
        self.vis.close()

    def add_geometry(self,item):
        self.vis.add_geometry(item)
        self.geometry_list.append(item)        

    def update(self, points, boxes, other_geometries):

        for item in self.geometry_list:
            self.vis.remove_geometry(item)
        if points is not None: self.add_geometry(points)
        for i, box in enumerate(boxes):
            self.add_geometry(box)

        for i, item in enumerate(other_geometries):
            self.add_geometry(item)
        self.ctr.convert_from_pinhole_camera_parameters(self.param)
        self.vis.poll_events()
        self.vis.update_renderer()
        img = img_open3d_to_opencv(np.asarray(
            capture_screen(self.vis), dtype=float))
        self.framenum += 1
        return img, self.param


def projectionOpenCV(img=None, pcd=None, boxes3d=None, camera=None, coordinateBase='2d', boxType='2d'):
    assert camera is not None
    if img is None:
        img=np.zeros([
            camera.get('height', 1080),
            camera.get('width', 1920),
            4,
            ], dtype=np.uint8)
    else:
        img=copy.deepcopy(img)
    imgH,imgW,C=img.shape
    base=(imgH//1500+1)*2
    coordinateBase=coordinateBase.lower()
    boxType=boxType.lower()
    if coordinateBase == '2d':
        if pcd is not None:
            pcd=applyDistortionOnO3dObject(pcd,camera)
        if boxes3d is not None:
            for i in range(len(boxes3d)):
                boxes3d[i]=applyDistortionOnO3dObject(boxes3d[i],camera)
    elif coordinateBase == '3d':
        img=distortionCorrection(img, camera)
    else: raise NotImplementedError("CoordinateBase should be 2d or 3d")

    if pcd is not None:
        points=np.asarray(pcd.points)
        colors=np.asarray(pcd.colors)
        colors=img_open3d_to_opencv(colors)
        for i,point in enumerate(points):
            x=int(point[0])
            y=int(point[1])
            color=colors[i]
            if C==4: color=list(color)+[255]
            color=tuple((int(a) for a in color))
            if inImgArea(x,y,img.shape[1],img.shape[0]):
                xy=(x,y)
                cv2.circle(img, xy, int(base/2), color, thickness=base)
    
    if boxes3d is not None:
        for i,box3d in enumerate(boxes3d):
            points=np.asarray(box3d.points)
            color=np.asarray(box3d.colors)[0]
            color=img_open3d_to_opencv(color)
            if C==4: color=list(color)+[255]
            color=tuple((int(a) for a in color))
            points=(np.rint(points)).astype(int)[:,:2]
            if boxType == '2d':
                bound=getBoundInImage(points, imgW, imgH)
                cv2.rectangle(img,(int(bound[0]),int(bound[1])),(int(bound[2]),int(bound[3])),color=color,thickness=base)
            elif boxType == '3d':
                lines=np.asarray(box3d.lines)
                for i,line in enumerate(lines):
                    cv2.line(img,points[line[0]],points[line[1]], color,thickness=base)
            else: raise NotImplementedError("Boxtype2d should be 2d or 3d")
    return img

