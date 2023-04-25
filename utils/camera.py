
from easydict import EasyDict
import open3d
from .tools import *
class Camera():
    def __init__(self, name='camera',extrinsic=None, intrinsic=None, distortion=None, width=None, height=None, viewangle=180):
        self._buildInParams = ('name','viewangle','width','height')
        self.name=name
        self.viewangle=float(viewangle)
        self.width=width
        self.height=height
        self.extrinsic=Extrinsic() if extrinsic is None else extrinsic
        self.intrinsic=Intrinsic() if intrinsic is None else intrinsic
        self.distortion=Distortion() if distortion is None else distortion

    def __setattr__(self, name, value):
        if name == '_buildInParams' or name in self._buildInParams:
            object.__setattr__(self, name, value)
        elif name == 'extrinsic':
            if isinstance(value, Extrinsic):
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, Extrinsic(value))
        elif name == 'intrinsic':
            if isinstance(value, Intrinsic):
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, Intrinsic(value))
        elif name == 'distortion':
            if isinstance(value, Distortion):
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, Distortion(value))
        else: raise KeyError(f"Unsupported parameter for Camera: {name}")
    
    def data(self):
        data=dict(
            name=self.name,
            viewangle=self.viewangle,
            extrinsic=self.extrinsic.data(),
            intrinsic=self.intrinsic.data(),
            distortion=self.distortion.data(),
        )
        if self.width is not None:
            data['width']=self.width
        if self.height is not None:
            data['height']=self.height
        return data
    
    def jsonObject(self):
        data=self.data()
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key]=data[key].tolist()
        return data
    
    @staticmethod
    def createFromDict(data):
        return Camera(**data)
    


class Intrinsic():
    def __init__(self, matrix=None, fx=None, fy=None, cx=None, cy=None, **kwargs):
        self._buildInParams = ('params',)
        self.params=EasyDict()
        self.params.fx=1.0
        self.params.fy=1.0
        self.params.cx=0.0
        self.params.cy=0.0
        self.matrix=np.identity(4)
        if matrix is not None:
            self.matrix = matrix
        else:
            if fx is not None: self.fx=fx
            if fy is not None: self.fy=fy
            if cx is not None: self.cx=cx
            if cy is not None: self.cy=cy
    
    def data(self):
        return self.matrix

    def __setattr__(self, name, value):
        if name == '_buildInParams' or name in self._buildInParams:
            object.__setattr__(self, name, value)
        elif name in self.params.keys():
            value=float(value)
            self.params[name]=value
            matrix=self.getMatrixFromParams(**self.params)
            object.__setattr__(self, 'matrix', matrix)
        elif name == 'matrix':
            value=np.array(value).reshape(-1)
            if len(value)==9:
                value=value.reshape(3,3)
                idmatrix=np.identity(4)
                idmatrix[:3,:3]=value
                value=idmatrix
            elif len(value)==16:
                value=value.reshape(4,4)
            else:
                raise ValueError("Intrinsic matrix shape should be (3,3) or (4,4)")
            object.__setattr__(self, name, value)
            params=self.getParamsFromMatrix(value)
            self.params.update(params)
        else: raise KeyError(f"Unsupported parameter for Intrinsic: {name}")

    def __getattribute__(self, name):
        if name == '_buildInParams' or name in object.__getattribute__(self, '_buildInParams'):
            return object.__getattribute__(self, name)
        elif name in self.params.keys():
            return self.params[name]
        elif name=='matrix':
            return object.__getattribute__(self, name)
        else: 
            return object.__getattribute__(self, name)

    @staticmethod
    def getMatrixFromParams(fx=1.0, fy=1.0, cx=0.0, cy=0.0):
        return np.array([
            fx, 0, cx, 0,
            0, fy, cy, 0,
            0,  0,  1, 0,
            0,  0,  0, 1,
        ], dtype=float).reshape(4,4)
    
    @staticmethod
    def getParamsFromMatrix(matrix):
        return EasyDict(
            fx=matrix[0,0],
            fy=matrix[1,1],
            cx=matrix[0,2],
            cy=matrix[1,2],
        )

class Extrinsic():
    def __init__(self, matrix=None, **kwargs):
        self.matrix=np.identity(4) if matrix is None else matrix

    def data(self):
        return self.matrix

    def __setattr__(self, name, value):
        if name == 'matrix':
            value=np.array(value).reshape(-1)
            if len(value)==9:
                value=value.reshape(3,3)
                idmatrix=np.identity(4)
                idmatrix[:3,:3]=value
                value=idmatrix
            elif len(value)==16:
                value=value.reshape(4,4)
            else:
                raise ValueError("Extrinsic matrix shape should be (3,3) or (4,4)")
            object.__setattr__(self, name, value)
    
    def addTranslation(self,x=0.0, y=0.0, z=0.0):
        translationMatrix=np.array([
            1,  0,  0,  x,
            0,  1,  0,  y,
            0,  0,  1,  z,
            0,  0,  0,  1,
        ], dtype=float).reshape(4,4)
        self.matrix=np.dot(translationMatrix, self.matrix)
    
    def addRotation(self, rx=0.0, ry=0.0, rz=0.0, order='xyz'):
        order=order.lower()
        index=['xyz'.index(c) for c in order]
        rot=[rx, ry, rz]
        rotationMatrix=np.identity(3)
        for i in index:
            axis_angle=[0,0,0]
            axis_angle[i]=rot[i]
            axisMatrix=open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
            rotationMatrix=np.dot(axisMatrix, rotationMatrix)
        newMatrix=np.identity(4)
        newMatrix[:3,:3]=rotationMatrix
        self.matrix=np.dot(newMatrix, self.matrix)

class Distortion():
    def __init__(self, params=None, k1=None, k2=None, p1=None, p2=None, k3=None, **kwargs):
        self._buildInParams = ('params')
        self.params=EasyDict()
        self.params.k1=0.0
        self.params.k2=0.0
        self.params.p1=0.0
        self.params.p2=0.0
        self.params.k3=0.0
        if params is not None:
            if isinstance(params, dict):
                for key in params:
                    self.__setattr__(key, params[key])
            elif len(params) == 5:
                names=['k1','k2','p1','p2','k3']
                for i in range(5):
                    self.params[names[i]]=float(params[i])
            else: raise ValueError("Incorrect format of distortion parameters")
        else:
            if k1 is not None: self.k1=k1
            if k2 is not None: self.k2=k2
            if p1 is not None: self.p1=p1
            if p2 is not None: self.p2=p2
            if k3 is not None: self.k3=k3

    def data(self):
        return dict(self.params)

    def __setattr__(self, name, value):
        if name == '_buildInParams' or name in self._buildInParams:
            object.__setattr__(self, name, value)
        elif name in self.params.keys():
            value=float(value)
            self.params[name]=value
        else: raise KeyError(f"Unsupported parameter for Distortion: {name}")

    def __getattribute__(self, name):
        if name == '_buildInParams' or name in object.__getattribute__(self, '_buildInParams'):
            return object.__getattribute__(self, name)
        elif name in self.params.keys():
            return self.params[name]
        else: 
            return object.__getattribute__(self, name)