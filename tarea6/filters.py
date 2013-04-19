#!/usr/bin/python

# Libraries
from PIL import Image, ImageTk

from math import sqrt, floor, ceil, pi, sin, cos, tan, atan, atan2, radians, degrees
from random import random, randint, choice
import numpy
from time import time
from sys import maxint

# ======= solid colors =======
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
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


def array2list(a):
    aS = a.shape
    newPixels = list()
    for y in xrange(aS[0]):
        for x in xrange(aS[1]):
            newPixels.append(tuple([int(v) for v in a[y,x]]))
    return newPixels


def turn(p1, p2, p3):
    return cmp((p2[0] - p1[0])*(p3[1] - p1[1]) - (p3[0] - p1[0])*(p2[1] - p1[1]), 0)
TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)
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


def getNeighbours(pixels, y, x):
    neighbours = list()
    try: neighbours.append(pixels[y-1][x-1])
    except IndexError: pass
    try: neighbours.append(pixels[y-1][x]) 
    except IndexError: pass
    try: neighbours.append(pixels[y-1][x+1])
    except IndexError: pass
    try: neighbours.append(pixels[y+1][x-1])
    except IndexError: pass       
    try: neighbours.append(pixels[y+1][x])
    except IndexError: pass
    try: neighbours.append(pixels[y+1][x+1])
    except IndexError: pass
    try: neighbours.append(pixels[y][x-1])
    except IndexError: pass
    try: neighbours.append(pixels[y][x+1])
    except IndexError: pass
    return neighbours


def normalize(pixels):
    newPixels = list()
    maximum = map(max, zip(*pixels))
    minimum = map(min, zip(*pixels))
    div = tuple([a-b for a,b in zip(maximum, minimum)])
    for pixel in pixels:
        newPixels.append(tuple([(p-m)/d for p,m,d in zip(pixel, minimum, div)]))
    return newPixels


def euclidean(values, shape):
    newValues = numpy.zeros(shape=shape)
    for y in xrange(shape[0]):
        for x in xrange(shape[1]):
            pixel = sum([(value[y,x]**2) for value in values])
            pixel = tuple([int(floor(sqrt(p))) for p in pixel])
            newValues[y,x] = pixel
    return newValues


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
    for y, pLine in enumerate(pixels):
        print str(percentage(height,y))
        for x, pixel in enumerate(pLine):
            pNeighbours = getNeighbours(pixels, y, x)
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
    for y, pLine in enumerate(pixels):
        print str(percentage(height,y))
        for x, pixel in enumerate(pLine):
            pNeighbours = getNeighbours(pixels, y, x)
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
    return grayscale(newPixels, lmin=10, lmax=10)


def convolution2D(f,h):
    fS, hS = f.shape, h.shape
    F = numpy.zeros(shape=fS)
    for y in xrange(fS[0]):
        print str(percentage(fS[0],y))
        for x in xrange(fS[1]):
            mSum = numpy.array([0.0, 0.0, 0.0])
            for j in xrange(hS[0]):
                j2 = j-(hS[0]/2)
                for i in xrange(hS[1]):
                    i2 = i-(hS[0]/2)
                    try:
                        mSum += f[y+j2,x+i2]*h[j,i]
                    except IndexError: pass
            F[y,x] = mSum
    return F
    

def applyMask(pixels, width, gray=True):
    if(gray): pixels = grayscale(pixels)
    pixels = slicing(pixels, width)
    pixels = numpy.array(pixels)
    pS = pixels.shape

    n = 1.0/1.0
    mask = numpy.array([[0.0, 1.0, 0.0],[1.0, -6.0, 1.0],[0.0, 1.0, 0.0]]) * n
    
    newPixels = array2list(convolution2D(pixels, mask))
               
    return newPixels


def borderDetection(pixels, width):
    #start = time()
    pixels = grayscale(pixels)
    pixels = slicing(pixels, width)
    pixels = numpy.array(pixels)
    pS = pixels.shape

    n = 1.0/1.0	
    mask1 = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * n
    mask2 = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) * n

    g1 = convolution2D(pixels, mask1)
    g2 = convolution2D(pixels, mask2)

    newPixels = grayscale(array2list(euclidean([g1,g2], pS)))
    #end = time()
    #print (end - start)
    return newPixels


def bfs(pixels, visited, coordinates, newColor, width, height):
    queue = [coordinates]
    original = pixels[coordinates[1]][coordinates[0]]

    massPixels = list()
    while(len(queue) > 0):
        (x,y) = queue.pop(0)
        pColor = pixels[y][x]
        if(pColor == original or pColor == newColor):
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    (i,j) = (x + dx, y + dy)
                    if(i >= 0 and i < width and j >= 0 and j < height):
                        contenido = pixels[j][i]
                        if(contenido == original):
                            pixels[j][i] = newColor
                            queue.append((i,j))
                            visited[j][i] = 1
                            massPixels.append((i,j))
    return pixels, visited, massPixels


def objectDetection(pixels, width, height, coordinates):
    pixels = slicing(pixels, width)
    visited = [[0 for b in xrange(width)] for a in xrange(height)]
    color = (randint(0,255), randint(0,255), randint(0,255))
    pixels, visited, objPixels = bfs(pixels, visited, coordinates, color, width, height)
    return de_slicing(pixels), visited, objPixels


def objectClassification(pixels, width, height, color=BLACK):
    pixels = slicing(pixels, width)
    visited = [[0 for b in xrange(width)] for a in xrange(height)]
    objects = list()
    objID = 1
    for y in xrange(height):
        print str(percentage(height,y))
        for x in xrange(width):
            if(not visited[y][x] and pixels[y][x] == color):
                detectionPoint = (x,y)
                objColor = (randint(0,255), randint(0,255), randint(0,255))
                pixels, visited, objPixels = bfs(pixels, visited, (x,y), objColor, width, height)
                objSize = len(objPixels)
                objPrcnt = percentage(width*height, objSize)
                if(objPrcnt > 0.1):
                    ySum = sum(i for i,j in objPixels)
                    xSum = sum(j for i,j in objPixels)
                    objCenter = tuple([ySum/len(objPixels), xSum/len(objPixels)])
                    mObject = {"id":objID, "size":objSize, "percentage":objPrcnt, "center":objCenter, "pixels":objPixels, "dp":detectionPoint}
                    objects.append(mObject)
                    objID += 1
    biggestObject = max(objects, key=lambda x:x["percentage"])
    for p in biggestObject["pixels"]:
        pixels[p[1]][p[0]] = GRAY
    return de_slicing(pixels), objects


def houghLines(pixels, width, height):
    newPixels = list()
    results = [[None for a in xrange(width)] for b in xrange(height)]
    combinations = dict()

    pixelsOr = slicing(pixels, width)
    pixels = slicing(pixels, width)
    pixels = numpy.array(pixels)
    pS = pixels.shape

    maskX = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * 1.0/8.0
    maskY = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) * 1.0/8.0

    Gx = convolution2D(pixels, maskX)
    Gy = convolution2D(pixels, maskY)

    for y in xrange(height):
        for x in xrange(width):
            gx = Gx[y,x][0]
            gy = Gy[y,x][0]

            if(abs(gx) + abs(gy) <= 0.0):
                theta = None
            else:
                theta = atan2(gy,gx)
            
            if theta is not None:
                rho = ceil((x - width/2) * cos(theta) + (height/2 - y) * sin(theta))
                theta = int(degrees(theta))/18
                combination = ("%d"%(theta), "%d"%rho)
                results[y][x] = combination

                if x > 0 and y > 0 and x < width-1 and y < height-1: 
                    if combination in combinations:
                        combinations[combination] += 1
                    else:
                        combinations[combination] = 1
            else:
                results[y][x] = (None, None)

    frec = sorted(combinations, key=combinations.get, reverse=True)
    frec = frec[:int(ceil(len(combinations) * 1.0))]
    
    for y in xrange(height):
        for x in xrange(width):
            (ang, rho) = results[y][x]
            if(ang, rho) in frec:
                ang = int(ang)
                if(ang == -10 or ang == 0 or ang == 10):
                    newPixels.append(RED)
                elif(ang == -5 or ang == 5):
                    newPixels.append(BLUE)
                else:
                    newPixels.append(GREEN)
            else:
                newPixels.append(GRAY)
    return newPixels


def areCircles(radius, centers, borders):
    sensibility = 25
    newCenters = list()
    angles = [round(radians(a),2) for a in range(0,361,9)]
    for center in centers:
        count = 0
        for theta in angles:
            x = int(radius * cos(theta)) + center[0]
            y = int(radius * sin(theta)) + center[1]
            s = (x,y)
            s1 = (x,y+1)
            s2 = (x+1,y)
            s3 = (x+1,y+1)
            s4 = (x-1,y-1)
            s5 = (x,y-1)
            s6 = (x-1,y)
            if s in borders or s1 in borders or s2 in borders or s3 in borders or s4 in borders or s5 in borders or s6 in borders:
                count = count + 1
        if(count >= sensibility):
            print "[!] Possible circle found ",center
            newCenters.append(center)
    return newCenters


def groupCircles(centers):
    newCenters = list()
    mainCenter = list()
    rThreshold = 5
    x, y = centers[0]
    for a, center in enumerate(centers):
        if(abs(center[0]-x) <= rThreshold and abs(center[1]-y) <= rThreshold):
            mainCenter.append(center)
        else:
            newCenter = (sum([ce[0] for ce in mainCenter])/len(mainCenter),\
                         sum([ce[1] for ce in mainCenter])/len(mainCenter))
            mainCenter = [center]
            newCenters.append(newCenter)
            #print "[!] Possible circle center found ", newCenter
            try: x, y = centers[a]
            except IndexError: return newCenters
        if(a == len(centers)-1):
            newCenter = (sum([ce[0] for ce in mainCenter])/len(mainCenter),\
                         sum([ce[1] for ce in mainCenter])/len(mainCenter))
            newCenters.append(newCenter)
            #print "[!] Possible circle center found ", newCenter
            try: x, y = centers[a]
            except IndexError: return newCenters
    return newCenters


def groupRadius(circles):
    newCircles = dict()
    allRadius = sorted([int(r) for r in circles])
    rThreshold = 3
    pRadius = allRadius[0]
    kRadius = str(pRadius)
    nCircles = circles[kRadius]
    nRadius = [pRadius]
    
    for a, radius in enumerate(allRadius):
        kRadius = str(radius)
        if(abs(radius-pRadius) <= rThreshold):
            nRadius.append(radius)
            nCircles += circles[kRadius]
        else:
            nCircles = groupCircles(sorted(nCircles))
            nRadius = sum(nRadius)/len(nRadius)
            newCircles[str(nRadius)] = nCircles
            try:
                pRadius = allRadius[a]
                kRadius = str(pRadius)
                nCircles = circles[kRadius]
                nRadius = [pRadius]
            except IndexError:
                return newCircles
        if(a == len(allRadius)-1):
            nCircles = groupCircles(sorted(nCircles))
            nRadius = sum(nRadius)/len(nRadius)
            newCircles[str(nRadius)] = nCircles
            try:
                pRadius = allRadius[a]
                kRadius = str(pRadius)
                nCircles = circles[kRadius]
                nRadius = [pRadius]
            except IndexError:
                return newCircles
    return newCircles


def groupVotes(votes, width, height):
    for threshold in xrange(1, int(round(width * 0.1)),2):
        agregado = True
        while agregado:
            agregado = False
            for y in xrange(height):
                for x in xrange(width):
                    v = votes[y][x]
                    if v > 1:
                        for dx in xrange(-threshold, threshold):
                            for dy in xrange(-threshold, threshold):
                                if not (dx == 0 and dy == 0):
                                    if y + dy >= 0 and y + dy < height and x + dx >= 0 and x + dx < width:
                                        w = votes[y + dy][x + dx]
                                        if w > 0:
                                            if v - threshold >= w:
                                                votes[y][x] = v + w 
                                                votes[y + dy][x + dx] = 0
                                                agregado = True

    vMax = max(max(v) for v in votes)
    vSum = sum(sum(v) for v in votes)
					 
    vAverage = vSum / (width * height)
    threshold = (vMax + vAverage) / 2.0

    groupedVotes = [(x,y) for x in xrange(width) for y in xrange(height) if votes[y][x] > threshold]

    return groupedVotes


def houghCircles(pixels, Gx, Gy, width, height, radius):
    votes = [[0 for a in xrange(width)] for b in xrange(height)]

    for ym in xrange(height):
        y = height /2- ym
        for xm in xrange(width):    
            if(pixels[ym][xm] == WHITE):
                x = xm - width / 2
                gx = Gx[ym,xm][0]
                gy = Gy[ym,xm][0]
                g = sqrt(gx**2 + gy**2)

                if(abs(g) > 0.0):
                    cosTheta = gx / g
                    sinTheta = gy / g
                    xc = int(round(x - radius * cosTheta))
                    yc = int(round(y - radius * sinTheta))
                    xcm = xc + width / 2
                    ycm = height / 2 - yc
                    if xcm >= 0 and xcm < width and ycm >= 0 and ycm < height:
                        votes[ycm][xcm] += 1
    newPixels = groupVotes(votes, width, height)

    return groupCircles(newPixels)


def circleDetection(pixels, width, height, radius=0):
    circles = dict()
    lPixels = pixels
    pixelsOr = slicing(pixels, width)
    pixels = slicing(pixels, width)
    pixels = numpy.array(pixels)
    pS = pixels.shape

    n = 1.0/1.0
    maskX = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * n
    maskY = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) * n

    Gx = convolution2D(pixels, maskX)
    Gy = convolution2D(pixels, maskY)

    maxDiag = int(floor(sqrt(width**2 + height**2)))
    if(radius == 0):
        maxRadius = int(ceil(maxDiag * 0.3))
        minRadius = int(ceil(maxDiag * 0.01))
        borders = list()

        p, objects = objectClassification(lPixels, width, height, color=WHITE)
        for o in objects:
            borders+=o["pixels"]

        print "[O] Circle detection global advance >> %s%%"%(str(percentage(maxRadius,0)))
        for r in range(minRadius, maxRadius):
            lastCenters = houghCircles(pixelsOr, Gx, Gy, width, height, r)
            if(len(lastCenters) > 0):
                lastCircles = areCircles(r, lastCenters, borders)
                if(len(lastCircles) > 0):
                    circles[str(r)] = lastCircles
            print "[O] Circle detection global advance [r=%d] >> %s%%"%(r, str(percentage(maxRadius,r)))
        circles = groupRadius(circles)
    else:
        circles[str(radius)] = houghCircles(pixelsOr, Gx, Gy, width, height, radius)

    return circles


def calcTheta(gx,gy):
    if(abs(gx) + abs(gy) <= 0.0):
        theta = None
    else:
        theta = atan2(gy,gx)
        while(theta < 0.0): theta += 2*pi
        while(theta > 2*pi): theta -= 2*pi
        theta = (theta - pi/2) * -1
    return theta


def calcTangentLine(x, y, theta, l):
    x1, y1 = x + (l * cos(theta)), y + (l * sin(theta))
    x2, y2 = x + (l * cos(theta-pi)), y + (l * sin(theta-pi))
    return (x1,y1), (x2,y2)


def itIntersect(line1, line2):
    intersection = (0,0)
    (x11,y11), (x12,y12) = line1["start"], line1["end"]
    (x21,y21), (x22,y22) = line2["start"], line2["end"]

    if(max(x11,x12) < min(x21,x22)):
        return False, intersection
    else:
        a1 = (y11-y12)/(x11-x12)
        a2 = (y21-y22)/(x21-x22)
        if(a1 == a2):
            return False, intersection
        else:
            b1 = y11 - a1 * x11
            b2 = y21 - a2 * x21
            xA = (b2-b1)/(a1-a2)
            if(xA < max(min(x11,x12),min(x21,x22)) or xA > min(max(x11,x12),max(x21,x22))):
                return False, intersection
            else:
                yA = a1 * xA + b1
                return True, (xA, yA)


def calcEllipse(c, pixels): 
    center = c[0]
    minD, maxD = maxint, 0
    ellipse = {"center":center}
    for pixel in pixels:
        d = sqrt((center[0]-pixel[0])**2 + (center[1]-pixel[1])**2)
        if(d < minD):
            minD = d
            ellipse["minD"] = pixel
            ellipse["minSemiD"] = d*2
            ellipse["minTheta"] = calcTheta(pixel[0], pixel[1])
        if(d > maxD):
            maxD = d
            ellipse["maxD"] = pixel
            ellipse["maxSemiD"] = d*2
            ellipse["maxTheta"] = calcTheta(pixel[0], pixel[1])
    dx, dy = (center[0] - ellipse["minD"][0]), (center[1] - ellipse["minD"][1])
    ellipse["minD"] = (ellipse["minD"], (center[0]+(1*dx),center[1]+(1*dy)))
    dx, dy = (center[0] - ellipse["maxD"][0]), (center[1] - ellipse["maxD"][1])
    ellipse["maxD"] = (ellipse["maxD"], (center[0]+(1*dx),center[1]+(1*dy)))
    b = ellipse["minD"] + ellipse["maxD"]
    minX, minY = min(p[0] for p in b), min(p[1] for p in b)
    maxX, maxY = max(p[0] for p in b), max(p[1] for p in b)
    ellipse["box"] = (minX, minY, maxX, maxY)
    return ellipse


def houghEllipses(pixels, borders, Gx, Gy, width, height):
    minX, minY = min(p[0] for p in borders), min(p[1] for p in borders)
    maxX, maxY = max(p[0] for p in borders), max(p[1] for p in borders)
    maxDiag = int(floor(sqrt(width**2 + height**2)))
    threshold = 20

    results = [[0 for a in xrange(width)] for b in xrange(height)]
    lines, origins, points = list(), list(), list()

    for a, p1 in enumerate(borders):
        x1, y1 = p1[0], p1[1]
        gx1 = Gx[y1,x1][0]
        gy1 = Gy[y1,x1][0]
        theta1 = calcTheta(gx1, gy1)

        for b, p2 in enumerate(borders):
            dx = abs(p1[0]-p2[0])
            dy = abs(p1[1]-p2[1])
            if(p1 != p2 and dx > threshold and dy > threshold and b%2 == 0):
                x2, y2 = p2[0], p2[1]
                gx2 = Gx[y2,x2][0]
                gy2 = Gy[y2,x2][0]
                theta2 = calcTheta(gx2, gy2)

                if(theta1 is not None and theta2 is not None):
                    line1 = {"origin":(x1,y1), "theta":theta1}
                    line2 = {"origin":(x2,y2), "theta":theta2}
                    line1["start"], line1["end"] = calcTangentLine(x1, y1, theta1, maxDiag)
                    line2["start"], line2["end"] = calcTangentLine(x2, y2, theta2, maxDiag)
                    flag, T = itIntersect(line1, line2)
                    if(flag):
                        M = ((x1+x2)/2, (y1+y2)/2)
                        xT, yT = T
                        xM, yM = M
                        if(xT != xM and yT != yM):
                            p = list()
                            dx, dy = (xM - xT), (yM - yT)
                            mD = min(dx, dy)
                            dx, dy = dx/mD, dy/mD
                            #print "dx = %f, dy = %f, mD = %f"%(dx,dy,mD)
                            m = dy/dx
                            theta3 = tan(dy/dx)
                            line3 = {"origin":T, "start":T, "end":M, "theta":theta3}
                            line4 = {"origin":M, "start":M, "theta":theta3}
                            line4["end"] = (M[0] - 10, (m * ((M[0]-10) - xM)) + yM)
                            line4["end"] = (M[0]+(-100*dx),M[1]+(-100*dy))
                            for c in range(maxDiag):
                                #xV = xT#xM + c
                                #yV = yT#(m * (xV - xM)) + yM
                                xV, yV = int(ceil(M[0]+(-c*dx))), int(ceil(M[1]+(-c*dy)))
                                if((xV,yV) in borders or xV <= minX or yV <= minY or xV >= maxX or yV >= maxY):
                                    break
                                else:
                                    p.append((xV, yV))
                                    results[yV][xV] += 1
                        
                            lines.append((line1["start"], line1["end"]))
                            lines.append((line2["start"], line2["end"]))
                            lines.append((line3["start"], line3["end"]))
                            lines.append((line4["start"], line4["end"]))
                            
                            points.append(p)
                            origins.append(line1["origin"])
                            origins.append(line2["origin"])
                            origins.append(line3["origin"])
                            origins.append(line4["origin"])
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                continue
        print "[O] Ellipse detection partial advance >> %s%%"%(str(percentage(len(borders),a)))
    votes = groupVotes(results, width, height)
    return lines, points, origins, votes


def houghEllipses2(pixels, borders, Gx, Gy, width, height):
    minX, minY = min(p[0] for p in borders), min(p[1] for p in borders)
    maxX, maxY = max(p[0] for p in borders), max(p[1] for p in borders)
    maxDiag = int(floor(sqrt(width**2 + height**2)))
    threshold = int(ceil(len(borders)*0.15))

    results = [[0 for a in xrange(width)] for b in xrange(height)]
    lines, origins, points = list(), list(), list()

    b1 = sorted(borders, key=lambda x: x[1])
    b2 = sorted(borders, key=lambda x: x[1], reverse=True)
    #b1 = b[:len(b)/2]
    #b2 = b[len(b)/2:]

    for a in range(1000):
        p1, p2 = choice(borders), choice(borders)
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        gx1 = Gx[y1,x1][0]
        gy1 = Gy[y1,x1][0]
        gx2 = Gx[y2,x2][0]
        gy2 = Gy[y2,x2][0]
        theta1 = calcTheta(gx1, gy1)
        theta2 = calcTheta(gx2, gy2)

        if(theta1 is not None and theta2 is not None):
            line1 = {"origin":(x1,y1), "theta":theta1}
            line2 = {"origin":(x2,y2), "theta":theta2}
            line1["start"], line1["end"] = calcTangentLine(x1, y1, theta1, maxDiag)
            line2["start"], line2["end"] = calcTangentLine(x2, y2, theta2, maxDiag)
            flag, T = itIntersect(line1, line2)
            if(flag):
                M = ((x1+x2)/2, (y1+y2)/2)
                xT, yT = T
                xM, yM = M
                if(xT != xM and yT != yM):
                    p = list()
                    dx, dy = (xM - xT), (yM - yT)
                    mD = min(dx, dy)
                    dx, dy = dx/mD, dy/mD
                    #print "dx = %f, dy = %f, mD = %f"%(dx,dy,mD)
                    m = dy/dx
                    theta3 = tan(dy/dx)
                    line3 = {"origin":T, "start":T, "end":M, "theta":theta3}
                    line4 = {"origin":M, "start":M, "theta":theta3}
                    line4["end"] = (M[0] - 10, (m * ((M[0]-10) - xM)) + yM)
                    line4["end"] = (M[0]+(-100*dx),M[1]+(-100*dy))
                    for c in range(maxDiag/6):
                        #xV = xT#xM + c
                        #yV = yT#(m * (xV - xM)) + yM
                        xV, yV = int(ceil(M[0]+(-c*dx))), int(ceil(M[1]+(-c*dy)))
                        if((xV,yV) in borders or xV <= minX or yV <= minY or xV >= maxX or yV >= maxY):
                            break
                        else:
                            p.append((xV, yV))
                            results[yV][xV] += 1
                
                    lines.append((line1["start"], line1["end"]))
                    lines.append((line2["start"], line2["end"]))
                    lines.append((line3["start"], line3["end"]))
                    lines.append((line4["start"], line4["end"]))
                    
                    points.append(p)
                    origins.append(line1["origin"])
                    origins.append(line2["origin"])
                    origins.append(line3["origin"])
                    origins.append(line4["origin"])
                else:
                    continue
            else:
                continue
        else:
            continue
        print "[O] Ellipse detection partial advance >> %s%%"%(str(percentage(5000,a)))
    votes = groupVotes(results, width, height)
    return lines, points, origins, votes


def ellipseDetection(pixels, width, height):
    lines, points, origins, ellipses = list(), list(), list(), list()

    p, objects = objectClassification(pixels, width, height, color=WHITE)
    pixelsOr = slicing(pixels, width)
    pixels = slicing(pixels, width)
    pixels = numpy.array(pixels)
    pS = pixels.shape

    n = 1.0/1.0
    maskX = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * n
    maskY = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) * n

    Gx = convolution2D(pixels, maskX)
    Gy = convolution2D(pixels, maskY)

    for a, ob in enumerate(objects):
        print "%d : %d"%(len(objects), a+1)
        pix = graham_scan(ob["pixels"])
        l, p, o, v = houghEllipses(pixelsOr, pix, Gx, Gy, width, height)
        lines += l
        points += p
        origins += o
        ellipse = calcEllipse(v, ob["pixels"])
        ellipses.append(ellipse)

    visited = [[0 for b in xrange(width)] for a in xrange(height)]
    objID = 1

    for ellipse in ellipses:
        center = ellipse["center"]
        objColor = (randint(0,255), randint(0,255), randint(0,255))
        pixels, visited, objPixels = bfs(pixelsOr, visited, center, objColor, width, height)
        objSize = len(objPixels)
        objPrcnt = percentage(width*height, objSize)
        ellipse["size"], ellipse["percentage"], ellipse["pixels"] = objSize, objPrcnt, objPixels

    return de_slicing(pixels), lines, points, origins, ellipses
