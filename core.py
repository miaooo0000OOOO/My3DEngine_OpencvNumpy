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
        """构造函数

        Args:
            Triangles (List): List of Triangles
        """
        self.n = len(Triangles)
        self.v = np.zeros((4, 3*self.n))
        self.color = np.zeros((3, 3*self.n))
        for i in range(self.n):
            self.v[:, 3*i:3*i+3] = Triangles[i].v
        for i in range(self.n):
            self.color[:, 3*i:3*i+3] = Triangles[i].color
    
    def getTriangle(self, index):
        """返回三角面

        Args:
            index (int): 三角面的索引

        Returns:
            Triangle : 三角面
        """
        if not (0<=index<self.n and int(index)==index):
            raise ValueError("index必须是0到self.n-1之间的整数")
        index = index*3
        return Triangle(self.v[:,index:index+3], self.color[:,index:index+3])
    
    def mutiTransform(self, matrixs):
        """复合变换
        按列表中顺序应用矩阵变换

        Args:
            matrixs (List): 矩阵变换列表

        Raises:
            ValueError: 矩阵的形状必须是(4,4)
        """
        for m in matrixs:
            if m.shape != (4,4):
                raise ValueError("矩阵{}的形状必须为(4,4)".format(m))
        m = np.eye(4)
        for i in range(len(matrixs)):
            m = np.dot(matrixs[i], m)
        self.v = np.dot(m, self.v)

    def transform(self, matrix):
        """矩阵变换

        Args:
            matrix (np.ndarray): 矩阵

        Raises:
            ValueError: 矩阵的形状必须是(4,4)
        """
        if matrix.shape == (4,4):
            self.v = np.dot(matrix, self.v)
        else:
            raise ValueError("矩阵{}的形状必须为(4,4)".format(matrix))
        
    def normalize(self):
        self.v = self.v/self.v[3]
    
class Triangle(Obj3D):
    """三角面

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
    def __init__(self,v, color = None):
        if v.shape == (3, 3):
            v = np.row_stack((v, np.array((1,1,1))))
        self.v = v
        if color is None:
            color = np.ones((3,3), dtype=np.uint8)*255
        self.color = color
    
    def avgColor(self):
        """三个顶点的平均颜色

        Returns:
            np.ndarray(3,)
        """        
        return np.sum(self.color, axis=1)/3
    


class Cinema(Obj3D):
    """摄像机

    v : p:pos, u:up, l:lookAt
    [
        [px,ux,lx],
        [py,uy,ly],
        [pz,uz,lz],
        [1 ,0 ,0 ]
    ]
    """
    def __init__(self, pos, up, lookAt, FovY, aspect, n, f):
        """初始化摄像机

        Args:
            pos (np.ndarray): [x,y,z,1]
            up (np.ndarray): [x,y,z,0]
            lookAt (np.ndarray): [x,y,z,0]
            FovY (float): Field of Y  tan(FovY/2)=t/|n| 
            n (float) : near 近平面的z坐标(为负值)
            f (float) : far 远平面的z坐标(为负值)
            aspect (float): r/t = 宽除以高
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
        """解压Cinema

        Returns:
            tuple : (viewPoint, up, lookAt)
        """
        cinema = self.v.T
        viewPoint = np.array(cinema[0]).T
        up = np.array(cinema[1]).T
        lookAt = np.array(cinema[2]).T
        return viewPoint, up, lookAt
    
    def transform(self, matrix):
        v = self.v
        self.v = np.dot(matrix, v)
    
    def getPos(self):
        return self.unzip()[0]

    def getUp(self):
        return self.unzip()[1]
    
    def getLookAt(self):
        return self.unzip()[2]
    
    def normalize(self):
        def length(v):
            return np.sqrt(v[0]**2+v[1]**2+v[2]**2)
        v = self.getPos()
        v = v/v[3]
        self.v[:,0] = v
        self.v[:,1] = self.v[:,1]/length(self.v[:,1])
        self.v[:,2] = self.v[:,2]/length(self.v[:,2])
        self.v[3,1] = 0
        self.v[3,2] = 0

class Render:
    def __init__(self, width, height):
        """初始化渲染器

        Args:
            width (int): 宽(px)
            height (int): 高(px)
        """        
        self.width = width
        self.height = height
        self.img = np.zeros((width, height, 3), dtype=np.uint8)
    
    def clear(self):
        self.img = np.zeros((self.width, self.height, 3), dtype=np.uint8)

    def renderTriangles(self, cinema, triangles, config):
        """渲染三角形

        Args:
            cinema (Cinema): 摄像机
            model (Obj3D): 3D模型
            config (dict): 配置
                mapper:投影方法
                可选值：
                ["透视", "toushi", "persp"] 使用透视投影
                ["正交", "zhengjiao", "ortho"] 使用正交投影
                fill:是否填充(bool)
                line:是否有轮廓线(bool)

        Raises:
            ValueError: config错误

        Returns:
            np.ndarray: 图片
        """        
        model = copy.copy(triangles)
        if "mapper" not in config or "fill" not in config or "line" not in config:
            raise ValueError("config缺少键")
        trans = []
        trans.append(CinemaTransform(cinema))
        if config["mapper"] in ["透视", "toushi", "persp"]:
            #透视投影
            trans.append(M_persp2ortho(cinema.viewBox))
        elif config["mapper"] in ["正交", "zhengjiao", "ortho"]:
            #正交投影
            pass
        else:
            raise ValueError('config的键"mapper"可选:["透视", "toushi", "persp"] 使用透视投影;["正交", "zhengjiao", "ortho"] 使用正交投影')
        trans.append(M_ortho(cinema.viewBox))
        trans.append(M_viewport(self.width, self.height))
        model.mutiTransform(trans)
        model.normalize()
        for i in range(model.n):
            t = model.getTriangle(i)
            self.renderOneTriangle(t, config)
        
        return self.img

    def renderOneTriangle(self, triangle, config):
        """渲染单个三角形

        Args:
            triangle (Triangle): 三角面
            config (dict): 配置
        """
        avgc = triangle.avgColor()
        avgc_tup = (int(avgc[0]), int(avgc[1]), int(avgc[2]))
        t = triangle.v[:2,:].T #2*3
        t = t.astype(np.int64)
        # for x in range(int(np.min(t,axis=1)[0]), int(np.max(t,axis=1)[0])):
        #     for y in range(int(np.min(t,axis=1)[1]), int(np.max(t,axis=1)[1])):
        #         if self.pointInTriangle(triangle, x+0.5, y+0.5):
        #             img[y][x] = avgc
        if config["fill"]:
            self.img = cv2.fillConvexPoly(self.img, t, avgc_tup)
        if config["line"]:
            p1 = (t[0][0], t[0][1])
            p2 = (t[1][0], t[1][1])
            p3 = (t[2][0], t[2][1])
            cv2.line(self.img, p1, p2, avgc_tup)
            cv2.line(self.img, p2, p3, avgc_tup)
            cv2.line(self.img, p3, p1, avgc_tup)
    
    def pointInTriangle(self, triangle, x, y):
        """判断点是否在三角形内(已弃用)

        Args:
            triangle (Triangle): 三角面
            x (float): 点x
            y (float): 点y

        Returns:
            bool
        """
        t = triangle.v[:2,:]
        p1,p2,p3 = t.T[0],t.T[1],t.T[2]
        p = np.array([x,y])
        u1,u2,u3 = p1-p,p2-p,p3-p
        v1,v2,v3 = p2-p1,p3-p2,p1-p3
        if np.cross(u1,v1)*np.cross(u2,v2)>0 and np.cross(u2,v2)*np.cross(u3,v3)>0:
            return True
        else:
            return False

class Controller():
    def __init__(self, render, cinema, model, renderConfig):
        self.render = Render(renderConfig["width"], renderConfig["height"])
        self.model = model
        self.cinema = cinema
        self.renderConfig = renderConfig
        self.waitTime = int(1/renderConfig["fps"]*1000)
    
    def mainloop(self):
        def inputKeyIs(inputKey, value):
            if type(value)== type(""):
                return inputKey & 0xFF == ord(value)
            else:
                return inputKey & 0xFF == value
        def roundAngle(angle):
            if angle >= 2*np.pi:
                angle = 0
            if angle < 0:
                angle += 2*np.pi
                return angle
        azimuth = 0
        elevation = 0
        while True:
            self.render.renderTriangles(self.cinema, self.model, self.renderConfig)
            cv2.imshow("Engine3D", self.render.img)
            k = cv2.waitKey(self.waitTime)

            left = np.dot(CrossProduct(self.cinema.getUp()), self.cinema.getLookAt())
            right = -np.dot(CrossProduct(self.cinema.getUp()), self.cinema.getLookAt())
            if inputKeyIs(k, "w"):
                print("w")
                self.cinema.transform(T(self.cinema.getLookAt()))
            elif inputKeyIs(k, "s"):
                print("s")
                self.cinema.transform(T(-self.cinema.getLookAt()))
            elif inputKeyIs(k, "a"):
                print("a")
                self.cinema.transform(T(left))
            elif inputKeyIs(k, "d"):
                print("d")
                self.cinema.transform(T(right))
            elif inputKeyIs(k, "q"):       # space
                print("q")
                self.cinema.transform(T(self.cinema.getUp()))
            elif inputKeyIs(k, "e"):
                print("e")
                self.cinema.transform(T(-self.cinema.getUp()))
            elif inputKeyIs(k, "i"):
                print("i")

                self.cinema.transform(Rodrigues(right, 5/180*np.pi))
            elif inputKeyIs(k, "k"):
                print("k")
                self.cinema.transform(Rodrigues(left, 5/180*np.pi))
            elif inputKeyIs(k, "j"):
                print("j")
                self.cinema.transform(Rodrigues(self.cinema.getUp(), 5/180*np.pi))
                self.cinema.normalize()
            elif inputKeyIs(k, "l"):
                print("l")
                self.cinema.transform(Rodrigues(-self.cinema.getUp(), 5/180*np.pi))
                self.cinema.normalize()
            elif inputKeyIs(k, 27):
                print("esc")
                break
            self.render.clear()
            #print(k)
        
    def __str__(self):
        return "{}\n{}".format(self.cinema.v, self.model.v)



if __name__ == '__main__':
    SCREEN_WIDTH = 720
    SCREEN_HEIGHT = 720
    RENDER_CONFIG = {"mapper":"toushi", "fill":True, "line":False, "width":SCREEN_WIDTH, "height":SCREEN_HEIGHT, "fps":10}

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
    window = Controller(render, cinema, model, RENDER_CONFIG)
    window.mainloop()