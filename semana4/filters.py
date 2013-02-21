#!/usr/bin/python

# Libraries
from PIL import Image, ImageTk

from math import floor, sqrt
from random import random, randint
import numpy
from time import time

# ======= solid colors =======
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
GRAY = (128,128,128)
# ============================

def percentage(x,y):
    return round(float(y*100.0/x),2)

def slicing(l, n):
    return [l[a:a+n] for a in xrange(0, len(l), n)]

def de_slicing(p):
    pixels = list()
    for a in p:
        pixels += a
    return pixels


TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)
def turn(p1, p2, p3):
    return cmp((p2[0] - p1[0])*(p3[1] - p1[1]) - (p3[0] - p1[0])*(p2[1] - p1[1]), 0)

def graham_scan(points):
    u = list()
    l = list()
    
    points.sort()

    for p in points:
        while len(u) > 1 and turn(u[-2], u[-1], p) <= 0:
            u.pop()
        while len(l) > 1 and turn(l[-2], l[-1], p) >= 0:
            l.pop()
        u.append(p)
        l.append(p)
    
    l = l[1:-1][::-1]
    l += u
    return l


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
        print str(percentage(height,a))
        for b, pixel in enumerate(pLine):
            pNeighbours = getNeighbours(pixels, a, b)
            pNeighbours.append(pixel)
            newPixel = blurPixel(pNeighbours)
            newPixels.append(newPixel)
    return newPixels


def negative(pixels, cMax=255):
    for a, pixel in enumerate(pixels):
        pixels[a] = tuple([cMax-p for p in pixel])
    return pixels 


def sepia(pixels):
    pixels = grayscale(pixels)
    values = (1.0, 1.2, 2.0)
    for a, pixel in enumerate(pixels):
        pixels[a] = tuple([int(p/v) for p,v in zip(pixel, values)])
    return pixels 


def noise(pixels, level, intensity):
    level *= 0.01
    intensity *= 13
    for a, pixel in enumerate(pixels):
        if(random() < level):
            color = (0+intensity) if random() < 0.5 else (255-intensity)
            pixel = (color, color, color)
        pixels[a] = pixel
    return pixels


def removeNoise(pixels, width, height, aggressiveness):
    aggressiveness *= 10
    newPixels = list()
    pixels = slicing(pixels, width)
    for a, pLine in enumerate(pixels):
        print str(percentage(height,a))
        for b, pixel in enumerate(pLine):
            pNeighbours = getNeighbours(pixels, a, b)
            newPixel = blurPixel(pNeighbours)
            a1 = abs(newPixel[0] - pixel[0])
            a2 = abs(newPixel[1] - pixel[1])
            a3 = abs(newPixel[2] - pixel[2])
            if(a1>aggressiveness and a2>aggressiveness and a3>aggressiveness):
                newPixels.append(newPixel)
            else:
                newPixels.append(pixel)
    return newPixels


def difference(pixels, width, height):
    pixelsOr = grayscale(pixels)
    pixelsBG = blur(pixelsOr, width, height)
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


def convolution2D(f,h):
    fS, hS = f.shape, h.shape
    F = numpy.zeros(shape=fS)
    for x in xrange(fS[0]):
        print str(percentage(fS[0],x))
        for y in xrange(fS[1]):
            mSum = numpy.array([0.0, 0.0, 0.0])
            for i in xrange(hS[0]):
                i1 = i-(hS[0]/2)
                for j in xrange(hS[1]):
                    j2 = j-(hS[0]/2)
                    try:
                        mSum += f[x+i1,y+j2]*h[i,j]
                    except IndexError: pass
            F[x,y] = mSum
    return F
    

def applyMask(pixels, width, gray=True):
    newPixels = list()

    n = 1.0/9.0
    mask = numpy.array([[0.0, 1.0, 0.0],[1.0, -6.0, 1.0],[0.0, 1.0, 0.0]]) * n
    
    if(gray): pixels = grayscale(pixels)
    pixels = slicing(pixels, width)
    pixels = numpy.array(pixels)
    pixels = convolution2D(pixels, mask)
    
    pS = pixels.shape
    for x in xrange(pS[0]):
        for y in xrange(pS[1]):
            newPixels.append(tuple([int(v) for v in pixels[x,y]]))
               
    return newPixels


def borderDetection(pixels, width):
    #start = time()
    pixels = grayscale(pixels)
    pixels = slicing(pixels, width)
    pixels = numpy.array(pixels)
 
    newPixels = list()
    iS = pixels.shape

    n = 1.0/1.0
    mask1 = numpy.array([[-1,0,1],[-1,0,1],[-1,0,1]]) * n
    mask2 = numpy.array([[1,1,1],[0,0,0],[-1,-1,-1]]) * n
    mask3 = numpy.array([[-1,1,1],[-1,-2,1],[-1,1,1]])* n
    mask4 = numpy.array([[1,1,1],[-1,-2,1],[-1,-1,1]])* n

    g1 = convolution2D(pixels, mask1)
    g2 = convolution2D(pixels, mask2)
    g3 = convolution2D(pixels, mask3)
    g4 = convolution2D(pixels, mask4)

    for x in xrange(iS[0]):
        for y in xrange(iS[1]):
            pixel = (g1[x,y]**2) + (g2[x,y]**2) + (g3[x,y]**2) + (g4[x,y]**2)
            pixel = tuple([int(floor(sqrt(p))) for p in pixel])
            newPixels.append(pixel)
    
    newPixels = grayscale(newPixels)
    #end = time()
    #print (end - start)
    return newPixels


def bfs(pixels, visited, coordinates, newColor, width, height):
    queue = [coordinates]
    original = pixels[coordinates[1]][coordinates[0]]
    localPixels = list()
    while(len(queue) > 0):
        (x,y) = queue.pop(0)
        pColor = pixels[y][x]
        if(pColor == original or pColor == newColor):
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    (i,j) = (x + dx, y + dy)
                    if(i >= 0 and i < width and j >= 0 and j < height):
                        contenido = pixels[j][i]
                        if(contenido == original):
                            pixels[j][i] = newColor
                            queue.append((i,j))
                            visited[j][i] = 1
                            localPixels.append((i,j))
    return pixels, visited, localPixels


def objectDetection(pixels, width, height, coordinates, color):
    pixels = slicing(pixels, width)
    visited = [[0 for b in xrange(width)] for a in xrange(height)]
    pixels, visited, objPixels = bfs(pixels, visited, coordinates, color, width, height)
    return de_slicing(pixels), visited, objPixels


def objectClassification(pixels, width, height, color=BLACK):
    pixels = slicing(pixels, width)
    visited = [[0 for b in xrange(width)] for a in xrange(height)]
    objects = list()
    objID = 1
    for x in xrange(height):
        print str(percentage(height,x))
        for y in xrange(width):
            if(not visited[x][y] and pixels[x][y] == color):
                objColor = BLUE#(randint(0,255), randint(0,255), randint(0,255))
                pixels, visited, objPixels = bfs(pixels, visited, (y,x), objColor, width, height)
                objSize = len(objPixels)
                objPrcnt = percentage(width*height, objSize)
                if(objPrcnt > 0.1):
                    ySum = sum(i for i,j in objPixels)
                    xSum = sum(j for i,j in objPixels)
                    objCenter = tuple([ySum/len(objPixels), xSum/len(objPixels)])
                    mObject = {"id":objID, "size":objSize, "percentage":objPrcnt, "center":objCenter, "pixels":objPixels}
                    objects.append(mObject)
                    objID += 1
    biggestObject = max(objects, key=lambda x:x["percentage"])
    for p in biggestObject["pixels"]:
        pixels[p[1]][p[0]] = BLUE
    return de_slicing(pixels), objects

