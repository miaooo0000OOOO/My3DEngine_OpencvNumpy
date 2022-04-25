import numpy as np
import struct
import core
import cv2
# import stl
"""
- 参考

https://zhuanlan.zhihu.com/p/398208443
"""

def stlGetFormat(fileName):
    fid = open(fileName,'rb')
    fid.seek(0,2)                # Go to the end of the file
    fidSIZE = fid.tell()         # Check the size of the file
    if (fidSIZE-84)%50 > 0:
        stlFORMAT = 'ascii'
    else:
        fid.seek(0,0)            # go to the beginning of the file
        header  = fid.read(80).decode() 
        isSolid = header[0:5]=='solid'
        fid.seek(-80,2)          # go to the end of the file minus 80 characters
        tail       = fid.read(80)
        isEndSolid = tail.find(b'endsolid')+1

        if isSolid & isEndSolid:
            stlFORMAT = 'ascii'
        else:
            stlFORMAT = 'binary'
    fid.close()
    return stlFORMAT

def READ_stlbinary(stlFILENAME):
    # Open the binary STL file 
    fidIN = open(stlFILENAME,'rb')
    # Read the header
    fidIN.seek(80,0)                                   # Move to the last 4 bytes of the header
    facetcount = struct.unpack('I',fidIN.read(4))[0]   # Read the number of facets (uint32:'I',4 bytes) 读取三角面数

    # Initialise arrays into which the STL data will be loaded:  初始化数组
    coordNORMALS  = np.zeros((facetcount,3))        #法向量
    coordVERTICES = np.zeros((facetcount,3,3))      #三角面顶点
    # Read the data for each facet:
    for loopF in np.arange(0,facetcount):
        tempIN = struct.unpack(12*'f',fidIN.read(4*12))# Read the data of each facet (float:'f',4 bytes)
        coordNORMALS[loopF,:]    = tempIN[0:3]   # x,y,z components of the facet's normal vector    三角面法向量的xyz分量
        coordVERTICES[loopF,:,0] = tempIN[3:6]   # x,y,z coordinates of vertex 1                    顶点1的坐标
        coordVERTICES[loopF,:,1] = tempIN[6:9]   # x,y,z coordinates of vertex 2                    ...
        coordVERTICES[loopF,:,2] = tempIN[9:12]  # x,y,z coordinates of vertex 3 
        fidIN.read(2);   # Move to the start of the next facet.  Using file.read is much quicker than using seek 
    
    fidIN.close()
    return [coordVERTICES,coordNORMALS]

def READ_stlascii(stlFILENAME):
    # Read the ascii STL file
    fidIN = open(stlFILENAME,'r')
    fidCONTENTlist = [line.strip() for line in fidIN.readlines() if line.strip()]     #Read all the lines and Remove all blank lines
    fidCONTENT = np.array(fidCONTENTlist)
    fidIN.close()

    # Read the STL name
    line1 = fidCONTENT[0]
    if (len(line1) >= 7):
        stlNAME = line1[6:]
    else:
        stlNAME = 'unnamed_object'; 

    # Read the vector normals
    stringNORMALS = fidCONTENT[np.char.find(fidCONTENT,'facet normal')+1 > 0]
    coordNORMALS  = np.array(np.char.split(stringNORMALS).tolist())[:,2:].astype(float)

    # Read the vertex coordinates
    facetTOTAL       = stringNORMALS.size
    stringVERTICES   = fidCONTENT[np.char.find(fidCONTENT,'vertex')+1 > 0]
    coordVERTICESall = np.array(np.char.split(stringVERTICES).tolist())[:,1:].astype(float)
    cotemp           = coordVERTICESall.reshape((3,facetTOTAL,3),order='F')
    coordVERTICES    = cotemp.transpose(1,2,0)

    return [coordVERTICES,coordNORMALS,stlNAME]

def Read_stl(stlFILENAME):
    stlFORMAT = stlGetFormat(stlFILENAME)

    if stlFORMAT=='ascii':
        [coordVERTICES,coordNORMALS,stlNAME] = READ_stlascii(stlFILENAME)
    elif stlFORMAT=='binary':
        [coordVERTICES,coordNORMALS] = READ_stlbinary(stlFILENAME)
        stlNAME = 'unnamed_object'
    return [coordVERTICES,coordNORMALS,stlNAME]

def getModelFromSTL(stlFILENAME, stlFORMAT = "binary"):
    if stlFORMAT == "binary":
        V, N = READ_stlbinary(stlFILENAME)
    if format == 'acsii':
        res = READ_stlascii(stlFILENAME)
        if len(res) == 2:
            V, N = res
        else:
            V, N, _ = res
    for i in range(len(V)):
        triangleList.append(core.Triangle(V[i]))
    model = core.Obj3D(triangleList)
    return model
    

if __name__ == '__main__':
    FILENAME = "models/wings.stl"
    V, N, _ = Read_stl(FILENAME)
    # V = np.zeros((3, 3*len(V_)))
    # a_ = np.array([[[1,1,1],[2,2,2],[3,3,3]],[[4,4,4],[5,5,5],[6,6,6]]])
    # a = np.zeros((3, 3*len(a_)))
    # for i in range(len(a_)):
    #     for j in range(3):
    #         a[:, 3*i+j] = a_[i,j,:]
    # print(a)
    # for i in range(len(V_)):
    #     for j in range(3):
    #         V[:, 3*i+j] = V_[i,j,:]
    triangleList = []
    for i in range(len(V)):
        triangleList.append(core.Triangle(V[i]))
    model = core.Obj3D(triangleList)

    SCREEN_WIDTH = 720
    SCREEN_HEIGHT = 720
    RENDER_CONFIG = {"mapper":"toushi", "fill":False, "line":True, "width":SCREEN_WIDTH, "height":SCREEN_HEIGHT, "fps":10}

    cinema = core.Cinema(
        pos = np.array([0,0,0,1]),
        up = np.array([0,1,0,0]),
        lookAt = np.array([0,0,-1,0]),
        FovY = 80/180*np.pi,
        aspect = SCREEN_WIDTH/SCREEN_HEIGHT,
        n = -1,
        f = -100
    )
    render = core.Render(SCREEN_WIDTH, SCREEN_HEIGHT)
    window = core.Controller(render, cinema, model, RENDER_CONFIG)
    window.mainloop()