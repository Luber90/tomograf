from skimage import io
from skimage import draw
from skimage import exposure
import numpy as np

class Tomograf:
    def __init__(self, n, l, a):
        self.dalfa = a #kat o jaki obraca sie tomograf
        self.number = n #ilosc detektorow/emiterow
        self.length = l #rozpietosc detektorow w radianach
        self.height = 0 #wysokosc obrazka
        self.width = 0 #szereokosc obrazka
        self.center = np.array([0, 0])
        self.sinogram = np.empty([0, n])


    def loadImg(self, name):
        self.img = io.imread(name, as_gray=True)
        self.img2 = np.copy(self.img)
        self.height = len(self.img)
        self.width = len(self.img[0])
        self.result = np.zeros((self.height, self.width))
        self.center = np.array([(self.height-1)/2, (self.width-1)/2]) #srodek obrazka
        self.r = np.sqrt(self.center[0]**2+self.center[1]**2) #dlugosc polowy przekatnej, czyli promien kola opisanego
                                                              # na zdjeciu


    def run(self):
        all = [] #wszystkei wspolrzedne emiterow i detektorow tu beda
        for i in np.linspace(0, 2*np.pi-(np.pi*2*(self.dalfa/360)), num=360//self.dalfa): #kazda iteracja to jeden obrot
            tmp = np.empty([0])
            detectorsTab = [] #tu beda wspolrzedne detektorow
            emitersTab = [] #jak wyzej tylko emitery
            for j in range(self.number): #iteracja po kazdym detektorze
                tab = np.array([np.sin(i + np.pi - (self.length / 2) + j * (self.length / (self.number - 1))),
                                np.cos(i + np.pi - (self.length / 2) + j * (self.length / (self.number - 1)))])
                #to wyzej ze wzoru co on dal na kolejne punkty na luku

                #bierzemy punk i tworzymy jego punkt na ukos wzgledem srodka, ale on bedzie emiterem dla innego detektora
                #po drugiej stronie tablicy, dlatego detektory maja append a emitery insert
                detectorsTab.append(self.center+self.r*tab)
                emitersTab.insert(0, 2*self.center-(self.center+self.r*tab))
            all.append([detectorsTab, emitersTab])
            for j in range(self.number):
                x, y = draw.line_nd(detectorsTab[j], emitersTab[j]) #to znajduje te punkty calkowite miedzy emiterem                                            #a detektorem
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
            
        # fourier
        freq = np.fft.fftfreq(len(self.sinogram)).reshape(-1, 1)
        f_filter = 2 * np.abs(freq)
        projection = np.fft.fft(self.sinogram, axis=0) * f_filter
        self.sinogram = np.real(np.fft.ifft(projection, axis=0))

        kernel_len = 21  # 21?
        kernel = np.zeros(kernel_len)

        # filtr
        for i in range(0, kernel_len):
            if (i - kernel_len // 2) % 2 != 0:
                kernel[i] = -4 / (np.pi ** 2) / ((i - kernel_len // 2) ** 2)  # -4/pi^2/k^2
            kernel[kernel_len // 2] = 1

        # splot
        for i in range(360 // self.dalfa):
            self.sinogram[i, :] = np.convolve(self.sinogram[i, :], kernel, mode='same')

        #odtwarzanie obrazka
        for i in range(360//self.dalfa):  # kazda iteracja to jeden obrot
            for j in range(self.number):
                x, y = draw.line_nd(all[i][1][j], all[i][0][j])
                for k in range(len(y)):
                    if 0 <= x[k] < self.width and 0 <= y[k] < self.height: #jesli punkt jest na obrazie to idzie dalej
                        if self.sinogram[i][j] > 0:
                            #if (self.result[y[k]][x[k]] + self.sinogram[i][j] <=1.0):
                            self.result[y[k]][x[k]] += self.sinogram[i][j] #dodajemy na całej długości wartość sinogramy
                            #else:
                                #self.result[y[k]][x[k]] = 1
        '''p3, p98 = np.percentile(self.result, [3, 98])
        for i in range(len(self.result)):
            for j in range(len(self.result[0])):
                if self.result[i][j] >= p98:
                    self.result[i][j] = np.max(self.result)
                elif self.result[i][j] <= p3:
                    self.result[i][j] = 0'''
        self.result = exposure.rescale_intensity(self.result) #poprawia

    def showSinogram(self):
        io.imshow(self.sinogram)
        io.show()

    def showResult(self):
        io.imshow(self.result)
        io.show()


t = Tomograf(180, np.pi/2, 4)
t.loadImg("photos/Paski2.jpg")
t.run()
t.showResult()
