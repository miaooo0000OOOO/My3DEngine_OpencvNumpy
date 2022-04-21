import numpy as np

def CrossProduct(v):
    """返回叉乘(外积)的矩阵表示
    axb = np.dot(CrossProduct(a),b)

    Args:
        v (np.ndarray): [vx,vy,vz[,any_number]]
    """    
    M = np.array([
        [0      ,-v[2]  ,v[1]   ,0],
        [v[2]   ,0      ,-v[0]  ,0],
        [-v[1]  ,v[0]   ,0      ,0],
        [0      ,0      ,0      ,1]
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
        [x[0]   ,0      ,0      ,0],
        [0      ,x[1]   ,0      ,0],
        [0      ,0      ,x[2]   ,0],
        [0      ,0      ,0      ,1]
    ])

def T(x):
    """返回平移矩阵

    Args:
        x (np.ndarray): [t_x, t_y, t_z]
        t_x:x轴方向平移t_x
        以此类推
    """    
    return np.array([
        [1      ,0      ,0      ,x[0]],
        [0      ,1      ,0      ,x[1]],
        [0      ,0      ,1      ,x[2]],
        [0      ,0      ,0      ,1]
    ])

def RotateVector(x,y,z):
    """返回基坐标旋转矩阵, 基坐标分别变为x,y,z

    Args:
        x (np.ndarray): [x,y,z[,0]]
        y (np.ndarray): [x,y,z[,0]]
        z (np.ndarray): [x,y,z[,0]]
    """    
    Trans = np.array([
        [x[0]   ,y[0]   ,z[0]   ,0],
        [x[1]   ,y[1]   ,z[1]   ,0],
        [x[2]   ,y[2]   ,z[2]   ,0],
        [0      ,0      ,0      ,1]
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
    """(可能有Bug / Bug Maybe)
    罗德里格斯(Rodrigues)旋转方程, 返回以n为单位轴向量, 逆时针旋转a弧度的矩阵

    Args:
        n (np.ndarray): [n_x,n_y,n_z,0]
        n为轴向量
        a (float): 旋转弧度
    """    
    return np.cos(a)*np.eye(4)+(1-np.cos(a))*n*n.T+np.sin(a)*CrossProduct(n)

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
    viewPoint,up,lookAt = cinema.unzip()
    R_view = RotateVector(np.dot(CrossProduct(lookAt),up),up,-lookAt).T
    T_view = T(-viewPoint)
    return np.dot(R_view,T_view)


def M_persp2ortho(viewBox):
    """perspective2orthogonal
    返回透视投影到正交投影的矩阵

    Args:
        viewBox (np.ndarray): 视图框
    """    
    n,f = viewBox['n'],viewBox['f']
    return np.array([
        [n,0,0,0],
        [0,n,0,0],
        [0,0,n+f,-n*f],
        [0,0,1,0]
    ])

def M_ortho(viewBox):
    """返回正交投影矩阵,将viewBox中的点压缩到[-1,1]^3的立方体中

    Args:
        viewBox (dict): 视图框

    """
    xT = np.array([
        -(viewBox["r"]+viewBox["l"])/2,
        -(viewBox["t"]+viewBox["b"])/2,
        -(viewBox["n"]+viewBox["f"])/2])
    xS = np.array([
        2/(viewBox["r"]-viewBox["l"]),
        2/(viewBox["t"]-viewBox["b"]),
        2/(viewBox["n"]-viewBox["f"])])
    return np.dot(S(xS),T(xT))   

def M_viewport(width, height):
    """返回视口变换矩阵

    Args:
        width (int): 宽
        height (int): 高
    """
    w,h = width, height
    return np.array([
        [w/2,0,0,w/2],
        [0,h/2,0,h/2],
        [0,0,1,0],
        [0,0,0,1]
    ])