#!/usr/bin/python

# Librerias
from PIL import Image, ImageTk
from math import floor, sqrt
from random import random
import numpy
from time import time

image = None
lastImage = None


def slicing(l, n):
    return [l[a:a+n] for a in range(0, len(l), n)]

def de_slicing(p):
    pixels = list()
    for a in p:
        pixels += a
    return pixels

def getNeighbours(pixels, a, b):
    neighbours = list()
    try: neighbours.append(pixels[a-1][b-1])
    except IndexError: pass
    try: neighbours.append(pixels[a-1][b]) 
    except IndexError: pass
    try: neighbours.append(pixels[a-1][b+1])
    except IndexError: pass
    try: neighbours.append(pixels[a+1][b-1])
    except IndexError: pass       
    try: neighbours.append(pixels[a+1][b])
    except IndexError: pass
    try: neighbours.append(pixels[a+1][b+1])
    except IndexError: pass
    try: neighbours.append(pixels[a][b-1])
    except IndexError: pass
    try: neighbours.append(pixels[a][b+1])
    except IndexError: pass
    return neighbours

def grayscale(pixels, lmin=0, lmax=255):
    for a, pixel in enumerate(pixels): 
        color = sum(pixel)/3
        color = 255 if(color >= lmax) else color
        color = 0 if(color <= lmin) else color
        pixels[a] = (color, color, color)
    return pixels

def blurPixel(pixels):
    newPixel = (sum([pixel[0] for pixel in pixels])/len(pixels),\
                sum([pixel[1] for pixel in pixels])/len(pixels),\
                sum([pixel[2] for pixel in pixels])/len(pixels))
    return newPixel

def blur(pixels, width, height):
    newPixels = list()
    pixels = slicing(pixels, width)
    for a, pLine in enumerate(pixels):
        print str(round(float(a*100.0/height),2))
        for b, pixel in enumerate(pLine):
            pNeighbours = getNeighbours(pixels, a, b)
            pNeighbours.append(pixel)
            newPixel = blurPixel(pNeighbours)
            newPixels.append(newPixel)
    return newPixels

def negative(pixels, cMax=255):
    for a, pixel in enumerate(pixels):
        R = cMax - pixel[0]
        G = cMax - pixel[1]
        B = cMax - pixel[2]
        pixels[a] = (R, G, B)
    return pixels 

def sepia(pixels):
    pixels = grayscale(pixels)
    for a, pixel in enumerate(pixels):
        R = int(pixel[1] / 1.0)
        G = int(pixel[1] / 1.2)
        B = int(pixel[2] / 2.0)
        pixels[a] = (R, G, B)
    return pixels 

def noise(pixels, level, intensity):
    level *= 0.01
    intensity *= 13
    print intensity
    for a, pixel in enumerate(pixels):
        if(random() < level):
            color = (0+intensity) if random() < 0.5 else (255-intensity)
            pixel = (color, color, color)
        pixels[a] = pixel
    return pixels

def removeNoise(pixels, width,height):
    newPixels = list()
    pixels = slicing(pixels, width)
    for a, pLine in enumerate(pixels):
        print str(round(float(a*100.0/height),2))
        for b, pixel in enumerate(pLine):
            pNeighbours = getNeighbours(pixels, a, b)
            newPixel = blurPixel(pNeighbours)         
            a1 = abs(newPixel[0] - pixel[0])
            a2 = abs(newPixel[1] - pixel[1])
            a3 = abs(newPixel[2] - pixel[2])
            if(a1>35 and a2>35 and a3>35): 
                newPixels.append(newPixel)
            else:
                newPixels.append(pixel)
    return newPixels

def difference(pixels, width, height):
    pixelsOr = grayscale(pixels)
    pixelsBG = grayscale(pixels)
    pixelsBG = blur(pixelsBG, width, height)
    newPixels = list()
    for a, pixel in enumerate(pixelsOr):
        newPixel = tuple([p1-p2 for p1,p2 in zip(pixelsOr[a], pixelsBG[a])])
        newPixels.append(newPixel)
    return grayscale(newPixels, lmin=10, lmax=11)

#Convolucion 2D con listas
#def convolution2D(f,h):
#    F = list()
#    for x, fLine in enumerate(f):
#        for y, pixel in enumerate(fLine):
#            mSum = 0
#            for i, hLine in enumerate(h):
#                i1 = i-(len(h)/2)
#                for j, k in enumerate(hLine):
#                    i2 = j-(len(h)/2)
#                    try:
#                        mSum += f[x+i1][y+i2][0]*h[i][j]
#                    except IndexError: pass
#            mSum = int(floor(mSum))
#            F.append(tuple([mSum,mSum,mSum]))
#    return F

def convolution2D(f,h):                              # Convolucion discreta usando numpy
    fS, hS = f.shape, h.shape                        # Obtenemos el tamano de la mascara y la imagen
    F = numpy.zeros(shape=fS)                        # Creamos el arreglo donde se guardaran los calculos
    for x in range(fS[0]):                           # Recorremos la imagen a lo alto
        print str(round(float(x*100.0/fS[0]),2))     # Imprimimos el progreso de la rutina
        for y in range(fS[1]):                       # Recorremos la imagen a lo ancho
            mSum = numpy.array([0.0, 0.0, 0.0])      # Inicializamos la sumatoria en cero   
            for i in range(hS[0]):                   # Recorremos la mascara a lo alto
                i1 = i-(hS[0]/2)                     # Centramos la mascara a lo alto
                for j in range(hS[1]):               # Recorremos la mascara a lo ancho
                    j2 = j-(hS[0]/2)                 # Centramos la mascara a lo ancho  
                    try:                             # Realizamos la sumatoria de los valores
                        mSum += f[x+i1,y+j2]*h[i,j]  # Los bloques try, catch ayudan en a evitar errores
                    except IndexError: pass          # cuando estamos en los pixeles de las orillas
            F[x,y] = mSum                            # Agregamos el nuevo valor al arreglo de la gradiente
    return F      # Regresamos los valores de la gradiente calculados
    
def applyMask(i, pixels, width):
    mask = [[1.0, 2.0, 1.0],[2.0, 4.0, 2.0],[1.0, 2.0, 1.0]]
    print mask
    mask = numpy.array(mask)
    print mask
    #mask = [[x*(1.0/16.0) for x in line] for line in mask]
    mask = (1.0/16.0)*mask
    print mask
    #pi = slicing(pixels, width)
    i = numpy.asarray(i)
    raw_input("Enter para seguir")
    return convolution2D(i, mask)

def borderDetection(pixels, width):
    start = time()
    pixels = grayscale(pixels)      # Convertir la imagen a escala de grises
    pixels = slicing(pixels, width) # Pasar los pixeles a un arreglo compatible con numpy
    pixels = numpy.array(pixels)    # Pasar los pixeles a un arreglo numpy
 
    newPixels = list() # Lista que almacenara los nuevos pixeles de la nueva imagen
    iS = pixels.shape  # Obtenermos el tamano del arreglo (tamano de la imagen)

    n = 1.0/1.0        # Multiplicador de las mascaras
    # Usaremos 4 mascaras de Prewitt
    mask1 = numpy.array([[-1,0,1],[-1,0,1],[-1,0,1]]) * n # Prewitt simetrica vertical
    mask2 = numpy.array([[1,1,1],[0,0,0],[-1,-1,-1]]) * n # Prewitt simetrica horizontal
    mask3 = numpy.array([[-1,1,1],[-1,-2,1],[-1,1,1]])* n # Prewitt 0 grados
    mask4 = numpy.array([[1,1,1],[-1,-2,1],[-1,-1,1]])* n # Prewitt 45 grados

    g1 = convolution2D(pixels, mask1) # Llamamos a la rutina de convolucion discreta
    g2 = convolution2D(pixels, mask2) # para aplicar las mascaras
    g3 = convolution2D(pixels, mask3) # una por una
    g4 = convolution2D(pixels, mask4)

    for x in range(iS[0]):            # Recorremos los gradientes que hemos obtenido de aplicar
        for y in range(iS[1]):        # las mascaras a la imagen
            pixel = (g1[x,y]**2) + (g2[x,y]**2) + (g3[x,y]**2) + (g4[x,y]**2) # Buscamos los cambios de direcciones
            pixel = tuple([int(floor(sqrt(p))) for p in pixel])               # aplicando un acoplamiento
            newPixels.append(pixel)   # Agregamos el nuevo pixel a la lista para armar la nueva imagen
    
    newPixels = grayscale(newPixels)  # Binarizamos la imagen aplicando umbrales
    end = time()
    print (end - start)
    return newPixels                  # Regresamos la lista de nuevos pixeles

    
        

