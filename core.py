import numpy as np
import cv2
import copy
from matrix import *
from objects import *

class Obj3D:
    """3D物体
    n (int): 三角面的个数
    v (np.ndarray):
    t:三角面 p:点 如t1p1x即第一个三角面第一个点的x坐标
    [
        [t1p1x, t1p2x, t1p3x, t2p1x, ...],
        [t1p1y, t1p2y, t1p3y, t2p1y, ...],
        [t1p1z, t1p2z, t1p3z, t2p1z, ...],
        [1    , 1    , 1    , 1    , ...]
    ]
    v.shape = (4, 3*n)
    color (np.ndarray): 每个顶点的RGB颜色 3*n
    [
        [t1p1R, t1p2R, t1p3R, t2p1R, ...],
        [t1p1G, t1p2G, t1p3G, t2p1G, ...],
        [t1p1B, t1p2B, t1p3B, t2p1B, ...]
    ]
    color.shape = (3, 3*n)
    """    
    def __init__(self, Triangles):
        self.n = len(Triangles)
        self.v = np.zeros((4, 3*self.n))
        self.color = np.zeros((3, 3*self.n))
        for i in range(self.n):
            self.v[:, i:i+3] = Triangles[i].v
        for i in range(self.n):
            self.color[:, i:i+3] = Triangles[i].color
    
    def getTriangle(self, index):
        index = index*3
        return Triangle(self.v[:,index:index+3], self.color[:,index:index+3])
    
    def mutiTransform(self, matrixs):
        for m in matrixs:
            if m.shape != (4,4):
                raise ValueError()
        m = np.eye(4)
        for i in range(len(matrixs)):
            m = np.dot(matrixs[i], m)
        self.v = np.dot(m, self.v)

    def transform(self, matrix):
        if matrix.shape == (4,4):
            self.v = np.dot(matrix, self.v)
        else:
            raise ValueError()
        
    def normalize(self):
        self.v = self.v/self.v[3]
    
class Triangle(Obj3D):
    """_summary_

    v (np.ndarray):
    p:点
    [
        [p1x, p2x, p3x],
        [p1y, p2y, p3y],
        [p1z, p2z, p3z],
        [1  , 1  , 1  ]
    ]
    v.shape = (4,3)
    color (np.ndarray): 顶点的RGB颜色
    [
        [p1R, p2R, p3R],
        [p1G, p2G, p3G],
        [p1B, p2B, p3B]
    ]
    color.shape = (3, 3)
    """
    def __init__(self,v, color):
        self.v = v
        self.color = color
    
    def avgColor(self):
        return np.sum(self.color, axis=1)/3
    


class Cinema(Obj3D):
    """_summary_

    v : 
    [
        [px,ux,lx],
        [py,uy,ly],
        [pz,uz,lz],
        [1 ,0 ,0 ]
    ]
    Args:
        Obj3D (_type_): _description_
    """
    def __init__(self, pos, up, lookAt, FovY, aspect, n, f):
        """_summary_

        Args:
            pos (_type_): [x,y,z,1]
            up (_type_): [x,y,z,0]
            lookAt (_type_): [x,y,z,0]
            FovY Field of Y : tan(FovY/2)=t/|n| 
            n: near 近平面的z坐标(为负值)
            f: far 远平面的z坐标(为负值)
            aspect = r/t = 宽除以高
        """
        self.v = np.column_stack((pos, up, lookAt))
        self.FovY = FovY
        self.aspect = aspect
        t = np.tan(FovY/2)*(-n)
        b = -t
        r = aspect*t
        l = -r
        self.viewBox = {"t":t,"b":b,"l":l,"r":r, "n":n,"f":f} # [t,b,l,r,n,f]

    def unzip(self):
        cinema = self.v.T
        viewPoint = np.array(cinema[0]).T
        up = np.array(cinema[1]).T
        lookAt = np.array(cinema[2]).T
        return viewPoint, up, lookAt

class Render:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.img = np.zeros((width, height, 3), dtype=np.uint8)


    def render(self, cinema, model, config):
        """_summary_

        Args:
            cinema (_type_): _description_
            model (_type_): _description_
            config (dict): 配置
                mapper:投影方法
                可选值：
                ["透视", "toushi", "persp"] 使用透视投影
                ["正交", "zhengjiao", "ortho"] 使用正交投影
                fill:是否填充(bool)
                line:是否有轮廓线(bool)

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """        
        #投影
        width, height = self.width, self.height
        trans = []
        trans.append(CinemaTransform(cinema))
        if config["mapper"] in ["透视", "toushi", "persp"]:
            #透视投影
            trans.append(M_persp2ortho(cinema.viewBox))
        elif config["mapper"] in ["正交", "zhengjiao", "ortho"]:
            #正交投影
            pass
        else:
            raise ValueError()
        trans.append(M_ortho(cinema.viewBox))
        trans.append(M_viewport(self.width, self.height))
        model.mutiTransform(trans)
        model.normalize()
        for i in range(model.n):
            t = model.getTriangle(i)
            img = self.renderTriangle(self.img, t, config)
        
        return img

    def renderTriangle(self, img, triangle, config):
        avgc = triangle.avgColor()
        avgc_tup = (int(avgc[0]), int(avgc[1]), int(avgc[2]))
        t = triangle.v[:2,:].T #2*3
        t = t.astype(np.int64)
        try:
            # for x in range(int(np.min(t,axis=1)[0]), int(np.max(t,axis=1)[0])):
            #     for y in range(int(np.min(t,axis=1)[1]), int(np.max(t,axis=1)[1])):
            #         if self.pointInTriangle(triangle, x+0.5, y+0.5):
            #             img[y][x] = avgc
            if config["fill"]:
                img = cv2.fillConvexPoly(img, t, avgc_tup)
            if config["line"]:
                p1 = (t[0][0], t[0][1])
                p2 = (t[1][0], t[1][1])
                p3 = (t[2][0], t[2][1])
                cv2.line(img, p1, p2, avgc_tup)
                cv2.line(img, p2, p3, avgc_tup)
                cv2.line(img, p3, p1, avgc_tup)

        except ValueError:
            return img
        return img
    
    def pointInTriangle(self, triangle, x, y):
        t = triangle.v[:2,:]
        p1,p2,p3 = t.T[0],t.T[1],t.T[2]
        p = np.array([x,y])
        u1,u2,u3 = p1-p,p2-p,p3-p
        v1,v2,v3 = p2-p1,p3-p2,p1-p3
        if np.cross(u1,v1)*np.cross(u2,v2)>0 and np.cross(u2,v2)*np.cross(u3,v3)>0:
            return True
        else:
            return False



if __name__ == '__main__':
    SCREEN_WIDTH = 720
    SCREEN_HEIGHT = 720
    RENDER_CONFIG = {"mapper":"toushi", "fill":True, "line":False}

    t = Triangle(triangleMesh, np.ones((3,3))*255)
    model = Obj3D([t])
    cinema = Cinema(
        pos = np.array([0,0,0,1]),
        up = np.array([0,1,0,0]),
        lookAt = np.array([0,0,-1,0]),
        FovY = 45/180*np.pi,
        aspect = 1,
        n = -1,
        f = -5
    )
    render = Render(SCREEN_WIDTH, SCREEN_HEIGHT)
    render.render(cinema, model, RENDER_CONFIG)
    cv2.imshow("img", render.img)
    cv2.waitKey(0)