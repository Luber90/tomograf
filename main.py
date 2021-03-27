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
        #self.sinogram = np.empty([0, n])
        self.sinogram = np.zeros((360//a, n))


    def loadImg(self, name):
        self.img = io.imread(name, as_gray=True)
        self.img2 = np.copy(self.img)
        self.height = len(self.img)
        self.width = len(self.img[0])
        self.result = np.zeros((self.height, self.width))
        self.center = np.array([(self.width-1)/2, (self.height-1)/2])#srodek obrazka
        #self.r = np.sqrt(self.center[0]**2+self.center[1]**2) #dlugosc polowy przekatnej, czyli promien kola opisanego
                                                             # na zdjeciu
        self.r = min(self.height, self.width)/2


    def run(self, four=False, filtr=True, n=180, l=np.pi, a=4):
        self.number = n
        self.length = l
        self.dalfa = a
        self.sinogram = np.zeros((360 // a, n))
        self.showPicture()
        all = [] #wszystkie wspolrzedne emiterow i detektorow tu beda
        for z, i in enumerate(np.linspace(0, 2*np.pi-(np.pi*2*(self.dalfa/360)), num=360//self.dalfa)): #kazda iteracja to jeden obrot
            detectorsTab = [] #tu beda wspolrzedne detektorow
            emitersTab = [] #jak wyzej tylko emitery
            for j in range(self.number): #iteracja po kazdym detektorze
                tab = np.array([np.cos(i + np.pi - (self.length / 2) + j * (self.length / (self.number - 1))),
                                np.sin(i + np.pi - (self.length / 2) + j * (self.length / (self.number - 1)))])
                #to wyzej ze wzoru co on dal na kolejne punkty na luku

                #bierzemy punk i tworzymy jego punkt na ukos wzgledem srodka, ale on bedzie emiterem dla innego detektora
                #po drugiej stronie tablicy, dlatego detektory maja append a emitery insert
                detectorsTab.append(self.center+self.r*tab)
                emitersTab.insert(0, 2*self.center-(self.center+self.r*tab))
            all.append([detectorsTab.copy(), emitersTab.copy()])
            for j in range(self.number):
                x, y = draw.line_nd(detectorsTab[j], emitersTab[j]) #to znajduje te punkty calkowite miedzy emiterem
                # #a detektorem
                #print("od {} do {}".format(self.center+self.r*tab, 2*self.center-(self.center+self.r*tab)))
                tmp2 = []

                for k in range(len(y)):
                    if 0 <= x[k] < self.width and 0 <= y[k] < self.height: #jesli punkt jest na obrazie to idzie dalej
                        if self.img[y[k]][x[k]] > 0: #dodajemy tylko jesli piksel nie jest czarny
                            tmp2.append(self.img[y[k]][x[k]])
                if len(tmp2)>0: #jak cos jest to srednia z tych pikseli
                    #tmp = np.concatenate((tmp, [np.average(tmp2)]))
                    self.sinogram[z][j] = np.average(tmp2)
                #else: #jak nie to zero
                    #tmp = np.concatenate((tmp, [0.0]))
            #self.sinogram = np.concatenate((self.sinogram, [tmp]))

        # fourier
        if four:
            freq = np.fft.fftfreq(len(self.sinogram)).reshape(-1, 1)
            f_filter = 2 * np.abs(freq)
            projection = np.fft.fft(self.sinogram, axis=0) * f_filter
            self.sinogram = np.real(np.fft.ifft(projection, axis=0))

        kernel_len = 21  # 21?
        kernel = np.zeros(kernel_len)

        if filtr:
            # filtr
            kernel[kernel_len // 2] = 1
            for i in range(0, kernel_len//2):
                if i % 2 != 0:
                    kernel[kernel_len // 2 + i] = -4 / (np.pi ** 2) / (i** 2)  # -4/pi^2/k^2
                    kernel[kernel_len // 2 - i] = -4 / (np.pi ** 2) / ( i** 2)
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
                            self.result[y[k]][x[k]] += self.sinogram[i][j] #dodajemy na całej długości wartość sinogramy
        self.showSinogram()
        #p_start, p_end = np.percentile(self.result, [30, 70])
        #self.result = exposure.rescale_intensity(self.result, in_range=(p_start, p_end), out_range=(0, 1)) #poprawia

    def showPicture(self):
        io.imshow(self.img)
        io.show()

    def showSinogram(self):
        cos = exposure.rescale_intensity(self.sinogram, out_range=(0,1))
        io.imshow(cos)
        io.show()

    def showResult(self, p1=3, p2=98):
        p_start, p_end = np.percentile(self.result, [p1, p2])
        cos = exposure.rescale_intensity(self.result, in_range=(p_start, p_end), out_range=(0, 1))  # poprawia
        io.imshow(cos)
        io.show()

if __name__ == "__main__":
    t = Tomograf(180, np.pi, 4)
    t.loadImg("photos/Kropka.jpg")
    t.run(False, True, 180, np.pi, 4)
    t.showResult(3, 98)
