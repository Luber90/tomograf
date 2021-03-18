from skimage import io
from skimage import draw
import numpy as np

class Tomograf:
    def __init__(self, n, l, a):
        self.dalfa = a #kat o jaki obraca sie tomograf
        self.number = n #ilosc detektorow/emiterow
        self.length = l #rozpietosc detektorow w radianach
        self.height = 0 #wysokosc obrazka
        self.width = 0 #szereokosc obrazka
        self.center = np.array([0, 0])
        self.r = np.array([0, 0])
        self.sinogram = np.empty([0, n])

    def loadImg(self, name):
        self.img = io.imread(name, as_gray=True)
        self.height = len(self.img)
        self.width = len(self.img[0])
        self.center = np.array([(self.height-1)/2, (self.width-1)/2]) #srodek obrazka
        self.r = np.sqrt(self.center[0]**2+self.center[1]**2) #dlugosc polowy przekatnej, czyli promien kola opisanego
                                                              # na zdjeciu


    def getSinogram(self):
        for i in np.linspace(0, 2*np.pi, num=360//self.dalfa): #kazda iteracja to jeden obrot
            tmp = np.empty([0])
            detectorsTab = [] #tu beda wspolrzedne detektorow
            emitersTab = [] #jak wyzej tylko emitery
            for j in range(self.number): #iteracja po kazdym detektorze
                tab = np.array([np.sin(i + np.pi - (self.length / 2) + j * (self.length / (self.number - 1))),
                                np.cos(i + np.pi - (self.length / 2) + j * (self.length / (self.number - 1)))])
                #to wyzej ze wzoru co on dal na kolejne punkty na luku
                detectorsTab.append(self.center+self.r*tab)
                emitersTab.insert(0, 2*self.center-(self.center+self.r*tab))
            for j in range(self.number):
                x, y = draw.line_nd(detectorsTab[j], emitersTab[j]) #to znajduje te punkty calkowite miedzy emiterem
                                                                    #a detektorem
                #print("od {} do {}".format(self.center+self.r*tab, 2*self.center-(self.center+self.r*tab)))
                tmp2 = []
                for k in range(len(y)):
                    if 0 <= x[k] < self.width and 0 <= y[k] < self.height: #jesli punkt jest na obrazie to idzie dalej
                        if self.img[y[k]][x[k]] > 0: #dodajemy tylko jesli piksel nie jest czarny
                            tmp2.append(self.img[y[k]][x[k]])
                if len(tmp2)>0: #jak cos jest to srednia z tych pikseli
                    tmp = np.concatenate((tmp, [np.average(tmp2)]))
                else: #jak nie to zero
                    tmp = np.concatenate((tmp, [0.0]))
            self.sinogram = np.concatenate((self.sinogram, [tmp]))
        io.imshow(self.sinogram)
        io.show()


t = Tomograf(180, np.pi, 4)
t.loadImg("photos/Kropka.jpg")
t.getSinogram()