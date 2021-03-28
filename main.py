from skimage import io
from skimage import draw
from skimage import exposure
import numpy as np
import os
import datetime
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset

class Tomograf:
    def __init__(self, n, l, a):
        self.dalfa = a #kat o jaki obraca sie tomograf
        self.number = n #ilosc detektorow/emiterow
        self.length = l #rozpietosc detektorow w radianach
        self.height = 0 #wysokosc obrazka
        self.width = 0 #szereokosc obrazka
        self.center = np.array([0, 0])
        #self.sinogram = np.empty([0, n])
        self.p1 = 3
        self.p2 = 98
        self.sinogram = np.zeros((int(360//a), n))


    def loadImg(self, name):
        self.img = io.imread(name, as_gray=True)
        self.img2 = np.copy(self.img)
        self.height = len(self.img)
        self.width = len(self.img[0])
        self.result = np.zeros((self.height, self.width))
        self.center = np.array([(self.width-1)/2, (self.height-1)/2])#srodek obrazka
        #self.r = np.sqrt(self.center[0]**2+self.center[1]**2) #dlugosc polowy przekatnej, czyli promien kola opisanego
                                                             # na zdjeciu
        self.r = max(self.height, self.width)/2


    def run(self, four=False, filtr=True, n=180, l=np.pi, a=4):
        self.number = n
        self.length = l
        self.dalfa = a
        self.sinogram = np.zeros((int(360//a), n))
        #self.showPicture()
        all = [] #wszystkie wspolrzedne emiterow i detektorow tu beda
        for z, i in enumerate(np.linspace(0, 2*np.pi-(np.pi*2*(self.dalfa/360)), num=int(360//a))): #kazda iteracja to jeden obrot
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
                        #if self.img[y[k]][x[k]] > 0: #dodajemy tylko jesli piksel nie jest czarny
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
            for i in range(int(360//a)):
                self.sinogram[i, :] = np.convolve(self.sinogram[i, :], kernel, mode='same')
        #odtwarzanie obrazka
        for i in range(int(360//a)):  # kazda iteracja to jeden obrot
            for j in range(self.number):
                x, y = draw.line_nd(all[i][1][j], all[i][0][j])
                for k in range(len(y)):
                    if 0 <= x[k] < self.width and 0 <= y[k] < self.height: #jesli punkt jest na obrazie to idzie dalej
                        self.result[y[k]][x[k]] += self.sinogram[i][j] #dodajemy na całej długości wartość sinogramy
        #self.showSinogram()
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
        self.p1 = p1
        self.p2 = p2
        p_start, p_end = np.percentile(self.result, [p1, p2])
        cos = exposure.rescale_intensity(self.result, in_range=(p_start, p_end), out_range=(0, 1))  # poprawia
        io.imshow(cos)
        io.show()

    def saveResult(self, p1=3, p2=98):
        p_start, p_end = np.percentile(self.result, [p1, p2])
        cos = exposure.rescale_intensity(self.result, in_range=(p_start, p_end), out_range=(0, 1))  # poprawia
        io.imsave("res.jpg", cos)

    def currRes(self):
        p_start, p_end = np.percentile(self.result, [self.p1, self.p2])
        return exposure.rescale_intensity(self.result, in_range=(p_start, p_end), out_range=(0, 1))

    def rmse(self):
        return np.sqrt(np.mean((self.currRes()-self.img)**2))


    def saveDicom(self, name, idd, age, comment, date, time):
        image = self.currRes() * 255
        image = image.astype(np.uint16)


        filename = os.path.dirname(os.path.abspath(__file__))
        filename += "\dicom.dcm"

        # Populate required values for file meta information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        file_meta.ImplementationClassUID = "1.2.3.4"

        # Create the FileDataset instance (initially no data elements, but file_meta
        # supplied)
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
        # Add the data elements
        ds.PatientName = name
        ds.PatientID = idd
        ds.PatientAge = age
        ds.PatientComments = comment

        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.HighBit = 15
        ds.Columns = image.shape[0]
        ds.Rows = image.shape[1]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.SmallestImagePixelValue = b'\\x00\\x00'
        ds.LargestImagePixelValue = b'\\xff\\xff'

        # Set the transfer syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = True

        # Set creation date/time
        dt = datetime.datetime.now()
        #ds.ContentDate = dt.strftime('%Y%m%d')
        ds.ContentDate = date
        #timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
        ds.ContentTime = time

        ds.PixelData = image.tobytes()

        ds.save_as(filename, write_like_original=False)
        print("File saved")

        
def readDicom(filename):
    path = os.path.dirname(os.path.abspath(__file__))
    file = path + "\\" + filename
    ds = pydicom.dcmread(file)
    image = ds.pixel_array
    image = image.astype(np.uint16) * 255

    return ds, image
        
    
if __name__ == "__main__":
    # t = Tomograf(180, np.pi, 2)
    # t.loadImg("photos/Kropka.jpg")
    # t.run(True, True, 360, 3 * np.pi / 4, 1)
    # t.showResult(20, 98)
    # print(t.rmse())
    # t.saveResult(20, 98)
    # t.saveDicom()
    dicom_ds, image = readDicom("dicom.dcm")
    io.imshow(image)
    io.show()
    print(dicom_ds.data_element("ContentDate").value)
    print(dicom_ds.data_element("ContentTime").value)
    print(dicom_ds.data_element("PatientName").value)
    print(dicom_ds.data_element("PatientID").value)
    print(dicom_ds.data_element("PatientAge").value)
    print(dicom_ds.data_element("PatientComments").value)
