import numpy as np

C = 1

viewBox = {
    "t":1.5,
    "b":-1.5,
    "l":1.5,
    "r":-1.5, 
    "n":-1,
    "f":-10} # [t,b,l,r,n,f]

triangleMesh = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,1,1]
])
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
            cubeList.extend(np.linspace(cubePoints[i], cubePoints[j], 50))
cubeLines = np.array(cubeList).T
cubePoints = cubePoints.T

xAxis = np.linspace([0,0,0,1], [C/2,0,0,1], 50).T
yAxis = np.linspace([0,0,0,1], [0,C,0,1], 50).T
zAxis = np.linspace([0,0,0,1], [0,0,C*2,1], 50).T

front  = np.linspace([-1,0,1,1], [1,0,1,1], 50).T