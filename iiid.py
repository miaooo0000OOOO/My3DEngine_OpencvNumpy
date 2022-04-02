import numpy as np
import cv2
import time

SCREEN_WIDTH = 720
SCREEN_HEIGHT = 720
FPS = 30
DEBUG = True

RED = [0,0,255]
GREEN = [0,255,0]
BLUE = [255,0,0]


def unzip_cinema(cinema):
    """返回使摄影机矩阵分解为3个向量(viewPoint,up,lookAt)

    Args:
        cinema (np.ndarray): 摄影机(viewPoint,up,lookAt)分别为(摄影机坐标，表示摄影机正上方的单位向量，表示摄影机正前方的单位向量)
        以下简记为(v,u,l)
    cinema = [
        vx,ux,lx,0;
        vy,uy,ly,0;
        vz,uz,lz,0;
        1 ,0 ,0 ,1
    ]
    """    
    cinema = cinema.T
    viewPoint = np.array(cinema[0]).T
    up = np.array(cinema[1]).T
    lookAt = np.array(cinema[2]).T
    return viewPoint, up, lookAt

def CrossProduct(v):
    """返回叉乘(外积)的矩阵表示
    axb = np.dot(CrossProduct(a),b)

    Args:
        v (np.ndarray): [vx,vy,vz[,any_number]]
    """    
    M = np.array([
        [0,-v[2],v[1],0],
        [v[2],0,-v[0],0],
        [-v[1],v[0],0,0],
        [0,0,0,1]
    ])
    return M

def S(x):
    """返回缩放矩阵

    Args:
        x (np.ndarray): [s_x, s_y, s_z]
        s_x:x轴方向的伸缩系数
        以此类推
    """    
    return np.array([
        [x[0],0,0,0],
        [0,x[1],0,0],
        [0,0,x[2],0],
        [0,0,0,1]
    ])

def T(x):
    """返回平移矩阵

    Args:
        x (np.ndarray): [t_x, t_y, t_z]
        t_x:x轴方向平移t_x
        以此类推
    """    
    return np.array([
        [1,0,0,x[0]],
        [0,1,0,x[1]],
        [0,0,1,x[2]],
        [0,0,0,1]
    ])

def RotateVector(x,y,z):
    """返回基坐标旋转矩阵, 基坐标分别变为x,y,z

    Args:
        x (np.ndarray): [x,y,z[,0]]
        y (np.ndarray): [x,y,z[,0]]
        z (np.ndarray): [x,y,z[,0]]
    """    
    Trans = np.array([
        [x[0],y[0],z[0],0],
        [x[1],y[1],z[1],0],
        [x[2],y[2],z[2],0],
        [0,0,0,1]
    ])
    return Trans


def RotateAngle(x):
    """返回以原点为中心, 绕x,y,z轴逆时针旋转的矩阵

    Args:
        x (np.ndarray): [r_x,r_y,r_z]
        r_x:绕x轴逆时针旋转r_x弧度
        以此类推
    """    
    Rx = np.array([
        [1              ,0              ,0              ,0],
        [0              ,np.cos(x[0])   ,-np.sin(x[0])  ,0],
        [0              ,np.sin(x[0])   ,np.cos(x[0])   ,0],
        [0              ,0              ,0              ,1]
    ])
    Ry = np.array([
        [np.cos(x[1])   ,0              ,np.sin(x[1])   ,0],
        [0              ,1              ,0              ,0],
        [-np.sin(x[1])  ,0              ,np.cos(x[1])   ,0],
        [0              ,0              ,0              ,1]
    ])
    Rz = np.array([
        [np.cos(x[2])   ,-np.sin(x[2])  ,0              ,0],
        [np.sin(x[2])   ,np.cos(x[2])   ,0              ,0],
        [0              ,0              ,1              ,0],
        [0              ,0              ,0              ,1]
    ])
    return np.dot(Rx,np.dot(Ry,Rz))

def Rodrigues(n, a):
    """罗德里格斯(Rodrigues)旋转方程, 返回以n为单位轴向量, 逆时针旋转a弧度的矩阵

    Args:
        n (np.ndarray): [n_x,n_y,n_z,1]
        n为轴向量
        a (float): 旋转弧度
    """    
    return np.cos(a)*np.eye(4)+(1-np.cos(a))*n*n.T+np.sin(a)*np.array([
        [0      ,-n[2]  ,n[1]   ,0],
        [n[2]   ,0      ,-n[0]  ,0],
        [-n[1]  ,n[0]   ,0      ,0],
        [0      ,0      ,0      ,1]
    ])

def persp2ortho(viewBox):
    """perspective2orthogonal

    Args:
        viewBox (_type_): _description_

    Returns:
        _type_: _description_
    """    
    n,f = viewBox['n'],viewBox['f']
    M = np.array([
        [n,0,0,0],
        [0,n,0,0],
        [0,0,n+f,-n*f],
        [0,0,1,0]
    ])
    return M


def CinemaTransform(cinema):
    """返回使摄影机坐落于坐标原点, 面向-z方向, 其正上方为y轴的矩阵

    Args:
        cinema (np.ndarray): 摄影机(viewPoint,up,lookAt)分别为(摄影机坐标，表示摄影机正上方的单位向量，表示摄影机正前方的单位向量)
        以下简记为(v,u,l)
    cinema = [
        vx,ux,lx,0;
        vy,uy,ly,0;
        vz,uz,lz,0;
        1 ,0 ,0 ,1
    ]
    """    
    viewPoint,up,lookAt = unzip_cinema(cinema)
    R_view = RotateVector(np.dot(CrossProduct(lookAt),up),up,-lookAt).T
    T_view = T(-viewPoint)
    return np.dot(R_view,T_view)

def generateCinema(viewPoint,up,lookAt):
    """返回摄像机矩阵

    Args:
        viewPoint (np.ndarray): xyz
        up (np.ndarray): ...
        lookAt (np.ndarray): ...
    cinema = [
        vx,ux,lx,0;
        vy,uy,ly,0;
        vz,uz,lz,0;
        1 ,0 ,0 ,1
    ]
    """    
    v,u,l = viewPoint,up,lookAt
    cinema = np.array([
        [v[0],u[0],l[0],0],
        [v[1],u[1],l[1],0],
        [v[2],u[2],l[2],0],
        [1 ,0 ,0 ,1]
    ])
    return cinema

def Viewing(cinema, model, viewBox):
    model = np.dot(CinemaTransform(cinema), model)
    xT = np.array([
        -(viewBox["t"]+viewBox["b"])/2,
        -(viewBox["l"]+viewBox["r"])/2,
        -(viewBox["n"]+viewBox["f"])/2])
    xS1 = np.array([
        1/(viewBox["t"]-viewBox["b"]),
        1/(viewBox["l"]-viewBox["r"]),
        1/(viewBox["n"]-viewBox["f"])])
    xS2 = np.array([SCREEN_HEIGHT, SCREEN_WIDTH, 1])
    model = np.dot(S(xS2), np.dot(S(xS1), model))
    return model

def perspViewing(cinema, model, viewBox):
    model = np.dot(CinemaTransform(cinema), model)
    xT = np.array([
        -(viewBox["t"]+viewBox["b"])/2,
        -(viewBox["l"]+viewBox["r"])/2,
        -(viewBox["n"]+viewBox["f"])/2])
    xS1 = np.array([
        1/(viewBox["t"]-viewBox["b"]),
        1/(viewBox["l"]-viewBox["r"]),
        1/(viewBox["n"]-viewBox["f"])])
    p2o = persp2ortho(viewBox)

    model = np.dot(S(xS1), model)
    model = np.dot(p2o, model)
    model = model/model[3]
    return model

def isinViewBox(point, viewBox):
    t,b,l,r,n,f = viewBox["t"],viewBox["b"],viewBox["l"],viewBox["r"],viewBox["n"],viewBox["f"]

    l *= SCREEN_WIDTH
    r *= SCREEN_WIDTH
    t *= SCREEN_HEIGHT
    b *= SCREEN_HEIGHT
    return (b<=point[0]<=t and r<=point[1]<=l and f<=point[2]<=n)
        


if __name__ == '__main__':
    VIDEO = True
    if VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter('output.avi',fourcc, FPS, (SCREEN_WIDTH,SCREEN_HEIGHT), True)
    img = np.zeros((SCREEN_HEIGHT,SCREEN_WIDTH,3), dtype=np.uint8)

    viewBox = {
                "t":10,
                "b":-10,
                "l":10,
                "r":-10, 
                "n":-1.5,
                "f":-20} # [t,b,l,r,n,f]
    triangleMesh = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [0,0,0]
    ])
    cinema = generateCinema([0,0,0],[0,1,0],[0,0,-1])

    cubePoints = np.array([
        [1,-1,-1,1],
        [1,1,-1,1],
        [-1,1,-1,1],
        [-1,-1,-1,1],
        [1,-1,1,1],
        [1,1,1,1],
        [-1,1,1,1],
        [-1,-1,1,1]])
    cubeList = []
    for i in range(len(cubePoints)):
        for j in range(len(cubePoints)):
            if i > j:
                cubeList.extend(np.linspace(cubePoints[i], cubePoints[j], 200))
    cubeLines = np.array(cubeList).T
    cubePoints = cubePoints.T

    C = 1
    xAxis = np.linspace([0,0,0,1], [C,0,0,1], 100).T
    yAxis = np.linspace([0,0,0,1], [0,C,0,1], 100).T
    zAxis = np.linspace([0,0,0,1], [0,0,C,1], 100).T

    da = np.pi/180
    a = 0
    start_time = time.time()
    while True:
        img = np.zeros((SCREEN_HEIGHT,SCREEN_WIDTH,3), dtype=np.uint8)
        a += da
        if a >= 2*np.pi and VIDEO:
            video.release()
            break
        
        # 模型变换
        model = np.hstack((xAxis,yAxis,zAxis,cubeLines))
        #model = cubePoints
        
        n = np.array([1,1,1,0])/(3**0.5)
        model = np.dot(S([1]*3),model)
        # model = np.dot(Rodrigues(n,a),model)
        model = np.dot(RotateAngle([a,a,a]), model)
        model = np.dot(T([0,0,-5]), model)
        
        # 视图变换  
        
        model = perspViewing(cinema, model, viewBox)
        xS2 = np.array([SCREEN_HEIGHT, SCREEN_WIDTH, 1])
        model = np.dot(S(xS2),model)
        # print(model.T[1])
        # 投影
        pointsXY = model[:2][:3].T
        for i in range(len(model.T)):
            if not isinViewBox(model.T[i], viewBox) and False:
                continue
            x = pointsXY[i][1]
            y = pointsXY[i][0]
            z = model.T[i][2]
            x += SCREEN_HEIGHT/2
            y += SCREEN_WIDTH/2
            if 0<int(x)<SCREEN_WIDTH and 0<int(y)<SCREEN_HEIGHT:
                # if 0<=i<len(xAxis.T):
                #     color = RED
                # elif len(xAxis.T)<=i<len(xAxis.T)+len(yAxis.T):
                #     color = GREEN
                # else:
                #     color = BLUE
                img[SCREEN_HEIGHT-int(x)][int(y)] = [255,255,255]
        if VIDEO:
            video.write(img)
        else:
            cv2.imshow("img", img)
            # cv2.waitKey(int(1/FPS*1000))
            cv2.waitKey(0)
    end_time = time.time()
    print("耗时{}秒".format(end_time-start_time))
    print("每帧计算{}个点".format(len(cubeLines.T)))
    print("FPS:{}".format(FPS))