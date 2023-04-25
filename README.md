# PcdDetectionVisualizetion
Simple visualization tools for point cloud detection tasks, based on open3d.
## Project Layout

```text
│  visualizer.py        The main entrance to the functions
|        draw()         Simple draw function, based on class Visualizer
|        Visualizer     Core class for visualization
├─data                  Build-in view files and data for demo
├─demo
│      demo.py
└─utils
        camera.py               Camera class
        open3d_arrow.py         Draw arrows in open3d
        open3d_visualizer.py    Draw object with open3d
        tools.py
```
## Parameters for Visualizer
| Parameter | Format | Action |
|-----------|--------|-------|
windowWidth|int| Width of visualization window (px)
windowHeight|int| Height of visualization window (px)
windowName|str| Name of visualization window
drawMode|detection/segmentation or det/seg is ok| The way to draw objects, detection mode will draw boxes, segmentation mode will paint the point cloud
backgroundColor|rgb(0,0,0)-(1,1,1)| The background color of 3d visualization window
viewFile|filepath or \$key| Camera params file for visualization, use $ to access built-in files like "$birdeye"
pointSize|float| Point size when draw point cloud, default 1.0
camera3dLock|bool| Lock the camera of visualization progress, used for automatically visualizing a sequence of data
drawOriginAxis|bool| Place an xyz axis at the origin point
drawSpeedArrow|bool| Draw a speed arrow for every reference box if their shape is 9 (cx, cy, cz, dx, dy, dz, rz, [vx, vy])
drawLabelLegend|bool| Draw legend for colored labels(only in the saved image)
segPointsBy|segPointsBy=key or segPointsBy[key]=bool, key can be None/label/gtbox/refbox| Paint point cloud with key, when this option is not None, the drawMode will be set to "segmentation"
scoreThresh|float| Score threshhold if your reference data has scores
labelName|['label1','label2',...]| Names for labels, should be correspondence in numerical order
lableColorMap|[(r1,g1,b1), (r2,g2,b2), ...], rgb(0,0,0)-(1,1,1)| Colors for labels, should be correspondence in numerical order
labelStartFromZero|bool| Whether the labels is started from 0, should be checked or it may cause incorrect label and name correspondence
showProgressBar|bool| Whether to display a progress bar when more than one frame of data are inputted
workDir|dirpath| The default save path of visualization results
cameras2d|[camera1, camera2, ...]  camera can be Camera object or camera param dict|Cameras for draw2d
imagesLayout|[key1, key2, ...] or [[key11, key12], [key21,key22],...] key can be camera name or camera index(>=0), any invalid key will bereplaced by a black image|If this param is enabled, images drawn in draw2d will be pieced together into a large picture based on the layout
draw2dWith3d|bool|Draw a 3d lidar image in draw2d, whose camera name is '3d'
saveOnly2d|bool|Not preview, directly save the images in draw2d
boxType2d|2d/3d|Boxtype in images drawn in draw2d, 3d type is just the projection of 3d boxes, with 8 vertices, 2d type is rectangles like detetion in images
coordinateBase2d|2d/3d|When this param is 3d, the photo input will get a distortion correction, while it is 2d, the projection of point cloud and boxes will be adjusted through the distortion

## Main functions in Visualizer

### draw3d
Draw your data in 3d visualizer
| Parameter | Format | Action |
|-----------|--------|-------|
points|(batch, points, data) or (points, data), data should be (x,y,z) or (x,y,z,intensity) or (x,y,z,r,g,b)| Point cloud data, shape like 
pointLabels|Shape like points except the data dimension, int| Labels for points, numbers less than 0 will be seen as backgourd points
pointNames|Shape like points except the data dimension, str| Names for points
pointColors|Shape like points except the data dimension, data should be rgb(0,0,0)-(1,1,1)| Colors for points, when this data is given, segPointsBy will be disabled
gtBoxes|(batch, boxes, data) or (boxes, data), data like (cx, cy, cz, dx, dy, dz, rz)| Ground truth boxes, won't be colored when drawMode is "detection"
gtLabels|Shape like gtBoxes except the data dimension, int| Labels for ground truth boxes
gtNames|Shape like gtBoxes except the data dimension, str| Names for ground truth boxes labels
refBoxes|(batch, boxes, data) or (boxes, data), data like (cx, cy, cz, dx, dy, dz, rz, [vx, vy])| Reference boxes, will be colored by their labels when drawMode is "detection"
refLabels|Shape like refBoxes except the data dimension, int| Labels for reference boxes
refNames|Shape like refBoxes except the data dimension, str| Names for reference boxes labels
refScores|Shape like refBoxes except the data dimension, float| Scores for reference boxes
savePath|a specific document when there is only one frame of data, or it should be a folder path| Save path for visualization results

### draw2d
Draw your data to images, all input parameters same like draw3d, except:
| Parameter | Format | Action |
|-----------|--------|-------|
photos|(batch, data) or data, data like [photo1, photo2, ...] or {camera1: photo1, camera2: photo2, ...}, photo should be ndarray, opencv format|Photos for cameras

### addCamera2d
Add a camera for Visualizer
| Parameter | Format | Action |
|-----------|--------|-------|
camera|Camera object or camera params dict, see [`Class Camera`](#class-camera)|Directly add a formatted camera
**cameraParams|Camera params, see [`Class Camera`](#class-camera)|Create a Camera object through params

### getCamera2d
Get Camera object from Visualizer through camera name or index

### saveVideo
Generate a video from the images saved when your workdir or savePath in draw2d/draw3d is not None
| Parameter | Format | Action |
|-----------|--------|-------|
savePath|file path|Save path of video
fps|int|Frames per second of video
videoWidth|int|Width of the video(px)
videoHeight|int|Height of the video(px)
pbar|bool|Show a progress bar when generating the video

### saveLastCameraParam
Save the 3d visualizer camera params of last frame to a json file (open3d format) so you can use it next time

### loadConfig
Load the config you saved

### saveConfig
Save the config of Visualizer to a json file so you can use it next time

## Class Camera
Camera object for draw2d
| Parameter | Format | Action |
|-----------|--------|--------|
name|str|Camera name, should be unique
extrinsic|[`Extrinsic`](#extrinsic) object or extrinsic matrix|Extrinsic transform from lidar coordinate system to camera coordinate system
intrinsic|[`Intrinsic`](#intrinsic) object or intrinsic matrix|Ixtrinsic transform from camera coordinate system to image coordinate system
distortion|[`Distortion`](#distortion) object or distortion params dict|Distortion params of camera
width|int|Default width for blank photo when there is no photo input
height|int|Default height for blank photo when there is no photo input
viewangle|int(0-180)|View angle of camera

### Extrinsic
| Parameter | Format | Action |
|-----------|--------|--------|
|matirx|ndarray(4,4)|Extrinsic transform matrix
### Extrinsic.addaddTranslation
Add a tanslation operation shape like (x, y, z) to transform matrix
### Extrinsic.addRotation
Add a rotation operation shape like (rx, ry, rz)(Euler angle) with a order like 'xyz' to transform matrix


### Intrinsic
Pinhole Camera Model intrinsic params
| Parameter | Format | Action |
|-----------|--------|--------|
fx|float|Focal length of axis x (px)
fy|float|Focal length of axis y (px)
cx|float|Principal point x (px)
cy|float|Principal point y (px)
matrix|ndarray(4,4)|Ixtrinsic transform matrix

### Distortion

| Parameter | Format | Action |
|-----------|--------|--------|
k1|float|Radial distortion param k1
k2|float|Radial distortion param k2
k3|float|Radial distortion param k3
p1|float|Tangential distortion param p1
p2|float|Tangential distortion param p2

When you init Distortion with a array, it will be seen as [k1,k2,p1,p2,k3] (opencv format)
