#Imports Block
from skimage import data
from pylab import * #uint8, float64
import numpy as np
from skimage import img_as_ubyte
from skimage import img_as_float
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import color
from skimage.filter import threshold_otsu, sobel #Binarization
from skimage.restoration import denoise_tv_chambolle #Suavization Filters
from skimage import img_as_ubyte
from skimage.morphology import watershed
from skimage import io
from skimage import exposure #histograma
from PIL import Image, ImageOps
from matplotlib import pyplot
import copy

matplotlib.rcParams['font.size'] = 12

#Functions Block
    #CanalExtration:
    #Parametros: Img= Imagem a ser processada, name= nome do colorspace
def canalExtration(img, name):
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.imshow(img,)
    ax0.set_title("Original image " + name)

    ax1.imshow(img[:, :, 0])
    ax1.set_title("Channel 1 " + name)

    ax2.imshow(img[:, :, 1])
    ax2.set_title("Channel 2 "  + name)

    ax3.imshow(img[:, :, 2])
    ax3.set_title("Channel 3 "  + name)

    for ax in axes.ravel(): #Removing Axis
        ax.axis('off')

    fig.subplots_adjust(hspace=0.3)

    return plt

    #HistogramPlot:
    #Parametros: Img= Imagem a ser processada, name= nome do colorspace, min=valor minimo das camadas, max= valor maximo das camadas
def histogramPlot(img, name, min, max):

    img1 = copy.copy(img[:,:,:])#Teste: Trocar numeros por ":"
    img2 = copy.copy(img[:,:,:])
    img3 = copy.copy(img[:,:,:])

    #Teste: Removendo os outros canais.
    img1[:,:,1] = 0
    img1[:,:,2] = 0
    img2[:,:,0] = 0
    img2[:,:,2] = 0
    img3[:,:,0] = 0
    img3[:,:,1] = 0
    

    hist1 = np.histogram(img1, bins=np.arange(min, max))
    hist2 = np.histogram(img2, bins=np.arange(min, max))
    hist3 = np.histogram(img3, bins=np.arange(min, max))

    fig, axes = plt.subplots(2, 3, figsize=(7, 6))
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.ravel()
    ax0.imshow(img1)
    ax0.set_title("Channel 1 " + name)

    ax1.imshow(img2)
    ax1.set_title("Channel 2 " + name)

    ax2.imshow(img3)
    ax2.set_title("Channel 3 " + name)

    ax3.plot(hist1[1][:-1], hist1[0], lw=2)
    ax4.plot(hist2[1][:-1], hist2[0], lw=2)
    ax5.plot(hist3[1][:-1], hist3[0], lw=2)

    #Removing axis
    ax0.axis("off")
    ax1.axis("off")
    ax2.axis("off")

    return plt

#Input's Block

    #Single Reader
img = data.imread('img/nor.jpg', False,)
    #Set Reader

#Convert Block
img_rgb = color.convert_colorspace(img, 'RGB', 'RGB') #No need
img_hsv = color.convert_colorspace(img_rgb, 'RGB', 'HSV')
img_lab = color.rgb2lab(img_rgb)
img_hed = color.rgb2hed(img_rgb)
img_luv = color.rgb2luv(img_rgb)
img_rgb_cie = color.convert_colorspace(img_rgb, 'RGB', 'RGB CIE')
img_xyz = color.rgb2xyz(img_rgb)

#Save Test Block
"""io.imsave("image_hsv.jpg", img_hsv, )
io.imsave("image_lab.jpg", img_lab, )
io.imsave("image_hed.jpg", img_hed, )
io.imsave("image_luv.jpg", img_luv, )
io.imsave("image_rgb_cie.jpg", img_rgb_cie, )
io.imsave("image_xyz.jpg", img_xyz, )
"""
#Layers Block
"""
canalExtration(img_rgb, "RGB").show()
canalExtration(img_hsv, "HSV").show()
canalExtration(img_lab, "LAB").show()
canalExtration(img_hed, "HED").show()
canalExtration(img_luv, "LUV").show()
canalExtration(img_rgb_cie, "RGB_CIE").show()
canalExtration(img_xyz, "XYZ").show()
"""

# B&W Convert Block


# Histogram Generation Block

histogramPlot(img_rgb, "RGB", 0,256).show()
histogramPlot(img_hsv, "HSV",-10,10).show()
histogramPlot(img_lab, "LAB",-50,100).show()
histogramPlot(img_hed, "HED",-10,10).show()
histogramPlot(img_luv, "LUV",-50,100).show()
histogramPlot(img_rgb_cie, "RGB_CIE",-10,10).show()
histogramPlot(img_xyz, "XYZ",-10,10).show()

# Classification Block

