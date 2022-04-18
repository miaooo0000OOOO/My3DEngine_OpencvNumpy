import numpy as np
import cv2
import time
from matrix import *

SCREEN_WIDTH = 720
SCREEN_HEIGHT = 720
FPS = 30
DEBUG = False

RED = [0,0,255]
GREEN = [0,255,0]
BLUE = [255,0,0]

# 相机和视图框
###########################

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

def isinViewBox(point, viewBox):
    """点是否在视图框内

    Args:
        point (np.ndarray): [x,y,z[,1]]
        viewBox (viewBox): 视图框
    """    
    t,b,l,r,n,f = viewBox["t"],viewBox["b"],viewBox["l"],viewBox["r"],viewBox["n"],viewBox["f"]

    l *= SCREEN_WIDTH
    r *= SCREEN_WIDTH
    t *= SCREEN_HEIGHT
    b *= SCREEN_HEIGHT
    return (b<=point[0]<=t and r<=point[1]<=l and f<=point[2]<=n)



# 视图变换
###########################

def Viewing(cinema, model, viewBox):
    """返回模型的正交投影

    Args:
        cinema (np.ndarray): 摄影机(viewPoint,up,lookAt)
        model (np.ndarray): shape=(4,n)
        viewBox (Dict): 视图框
    """    
    model = np.dot(M_ortho(viewBox),model)/model[3]
    return model

def perspViewing(cinema, model, viewBox):
    """返回模型的透视投影

    Args:
        cinema (np.ndarray): 摄影机(viewPoint,up,lookAt)
        model (np.ndarray): shape=(4,n)
        viewBox (Dict): 视图框
    """    
    # M_persp = np.dot(M_ortho(viewBox),M_persp2ortho(viewBox))
    # if (model[3] == 0).any():
    #     raise ValueError("任意z坐标应不等于0")
    # model = np.dot(M_persp,model)/model[3]
    model = np.dot(M_persp2ortho(viewBox),model)
    if (model[3] == 0).any():
        raise ValueError("任意z坐标应不等于0")
    # model = model/model[3]
    model = np.dot(M_ortho(viewBox),model)
    return model

# 渲染
###########################

def render(img, model):
    pointsXY = model[:2][:3].T
    pointsXY = pointsXY + np.array([SCREEN_HEIGHT/2,SCREEN_WIDTH/2])
    for i in range(len(model.T)):
        x = pointsXY[i][1]
        y = pointsXY[i][0]
        z = model.T[i][2]
        if 0<int(x)<SCREEN_WIDTH and 0<int(y)<SCREEN_HEIGHT:
            color = np.array([255,255,255])
            z = model.T[i][2]
            # color = [-z*1.5]*3
            if 0<=color[0]<=255 and 0<=color[1]<=255 and 0<=color[2]<=255:
                img[int(x)][int(y)] = color
    return img 


###########################

if __name__ == '__main__':
    from objects import *
    
    VIDEO = False
    if VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter('output.avi',fourcc, FPS, (SCREEN_WIDTH,SCREEN_HEIGHT), True)
    
    img = np.zeros((SCREEN_HEIGHT,SCREEN_WIDTH,3), dtype=np.uint8)

    cinema = generateCinema([0,0,0],[0,1,0],[0,0,-1])

    da = np.pi/180
    a = 0
    start_time = time.time()
    while True:
        img = np.zeros((SCREEN_HEIGHT,SCREEN_WIDTH,3), dtype=np.uint8)
        if a >= 2*np.pi and VIDEO:
            video.release()
            break  
        # 模型变换
        model = np.hstack((xAxis,yAxis,zAxis,cubeLines,front))
        #model = cubePoints
        
        model = np.dot(S([1]*3),model)
        # n = np.array([1,1,1,0]).T/(3**0.5)
        model = np.dot(RotateAngle([a,a,a]), model)
        model = np.dot(T([0,0,-3]), model)
        
        # 视图变换  
        model = np.dot(CinemaTransform(cinema),model)
        model = perspViewing(cinema, model, viewBox)
        xS2 = np.array([SCREEN_HEIGHT, SCREEN_WIDTH, 1])
        model = np.dot(S(xS2),model)
        model = model/model[3]
        # 投影
        img = render(img, model)
        if VIDEO:
            video.write(img)
        else:
            cv2.imshow("img", img)
            if DEBUG:
                cv2.waitKey(0)
            else:
                cv2.waitKey(int(1/FPS*1000))
        a += da
    end_time = time.time()
    if VIDEO:
        print("耗时{:.2f}秒".format(end_time-start_time))
        print("每帧计算{}个点".format(len(cubeLines.T)))
        print("FPS:{}".format(FPS))