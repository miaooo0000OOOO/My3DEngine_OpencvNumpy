class Vector():
    def __init__(self, x, y, z):
        """空间向量

        Args:
            x (float): 。。。
            y (float): 。。。
            z (float): 。。。
        """        
        self.x, self.y, self.z = x, y, z
        self.length = (self.x**2+self.y**2+self.z**2)**0.5
    
    def __str__(self):
        return str([self.x,self.y,self.z])
    
    def __add__(self, other):
        """向量加法

        Args:
            other (Vector): 加数
        """        
        return Vector(self.x+other.x, self.y+other.y, self.z+other.z)
    
    def __sub__(self, other):
        """向量减法

        Args:
            other (Vector): 减数
        """        
        return Vector(self.x-other.x, self.y-other.y, self.z-other.z)

    def __mul__(self, other):
        """向量点乘，或向量与数字相乘

        Args:
            other (Vector or float): 乘数
        """
        if type(other)==Vector:
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            return Vector(self.x*other, self.y*other, self.z*other)

class Plane():
    def __init__(self, point, normalVector, i=None, j=None):
        """空间平面，点法式

        Args:
            point (Vector): 平面经过的点
            normalVector (Vector): 平面的法向量
            i (Vector): 与平面平行的单位向量
            j (Vector): 与平面平行的单位向量
            * 要求i与j垂直
        """
        self.point = point
        self.normalVector = normalVector
        if i!=None and j!=None:
            # i和j有默认值
            if i.length!=1 or j.length!=1:
                raise ValueError("i向量或j向量必须是单位向量")
            if i*self.normalVector==0:
                self.i = i
            else:
                raise ValueError("i向量必须与平面的法向量垂直")
            if j*self.normalVector==0 and j*i==0:
                self.j = j
            else:
                if j*i!=0:
                    raise ValueError("j向量必须与i向量垂直")
                else:
                    raise ValueError("j向量必须与平面的法向量垂直")
        else:
            i = self.project(Vector(0,0,0))
            if i.length == 0:
                i = self.project(Vector(0,0,1))
                if i.length == 0:
                    i = self.project(Vector(0,1,0))
            i = i*(1/i.length)
            """
            解二元一次方程组
            设j=(a,b,c), i=(x,y,z), n=(x,y,z)
            1.令a=1
            解
            {x_i+b*y_i+c*z_i=0
            {x_n+b*y_n+c*z_n=0
            """
            n = self.normalVector
            if i.z*n.y-n.z*i.y != 0 and i.y != 0:
                b = (i.x*n.y*i.z - n.x*i.y*i.z - i.x*n.y*i.z + i.x*i.y*n.z) / i.y
                c = (n.x*i.y - i.x*n.y) / (i.z*n.y - n.z*i.y)
                j = Vector(1, b, c)
            else:
                raise ValueError("出了问题，不知道怎么搞")
            j = j*(1/j.length)

            self.i = i
            self.j = j        
        
    def map2d(self, point):
        """将空间中一点(P)投影到面(过A点,法向量为n)上,并求出以平面上向量ij为基的线性组合
        将P投影到面上,为AP~
        AP~=ai+bj
        求(a,b)

        Args:
            point (Vector): 点
        """         
        point = self.project(point)
        


    def project(self, point):
        """点到平面的投影
        若点为P(point), 平面过点O(self.point), 法向量为n(self.normalVector), P在平面上的投影为P~
        则返回向量OP~

        Args:
            point (Vector): 点
        """
        d = self.distance(point)
        return (point-self.point) - self.normalVector*d
    
    def distance(self, point):
        """点到平面的距离

        Args:
            point (Vector): 点
        """
        return abs((point-self.point)*self.normalVector)/self.normalVector.length

    
    def pointIsOnThePlane(self, point):
        """点是否在平面上

        Args:
            point (Vector): 点
        """
        if self.normalVector*(point-self.point) == 0:
            return True
        else:
            return False
    
if __name__ == '__main__':
    p = Vector(0,0,1)
    af = Plane(Vector(1,1,0), Vector(0,0,1))
    print(af.distance(p))