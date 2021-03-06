from skimage import io
from skimage import draw
import numpy as np

class Tomograf:
    def __init__(self, n, l, a):
        self.dalfa = a
        self.number = n
        self.length = l
        self.height = 0
        self.width = 0
        self.center = np.array([0, 0])
        self.r = np.array([0, 0])
        self.sinogram = np.empty([0, n])

    def loadImg(self, name):
        self.img = io.imread(name, as_gray=True)
        self.height = len(self.img)
        self.width = len(self.img[0])
        self.center = np.array([(self.height-1)/2, (self.width-1)/2])
        self.r = np.sqrt(self.center[0]**2+self.center[1]**2)


    def getSinogram(self):
        for i in np.linspace(0, 2*np.pi, num=360//self.dalfa):
            tmp = np.empty([0])
            for j in range(self.number):
                tab = np.array([np.sin(i+np.pi-(self.length/2)+j*(self.length/(self.number-1))),
                                np.cos(i+np.pi-(self.length/2)+j*(self.length/(self.number-1)))])
                x, y = draw.line_nd(self.center+self.r*tab, 2*self.center-(self.center+self.r*tab))
                #print("od {} do {}".format(self.center+self.r*tab, 2*self.center-(self.center+self.r*tab)))
                tmp2 = []
                for k in range(len(y)):
                    if 0 <= x[k] < self.width and 0 <= y[k] < self.height:
                        if self.img[y[k]][x[k]] > 0:
                            tmp2.append(self.img[y[k]][x[k]])
                if len(tmp2)>0:
                    sum = 0
                    for z in tmp2:
                        sum += z*0.1
                    if sum > 1:
                        sum = 1
                    tmp = np.concatenate((tmp, [sum]))
                else:
                    tmp = np.concatenate((tmp, [0.0]))
            self.sinogram = np.concatenate((self.sinogram, [tmp]))
        io.imshow(self.sinogram)
        io.show()


t = Tomograf(180, np.pi/4, 4)
t.loadImg("photos/Kropka.jpg")
t.getSinogram()