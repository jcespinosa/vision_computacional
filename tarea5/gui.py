#!/usr/bin/python

# Librerias
from Tkinter import *
from tkFileDialog import askopenfilename, asksaveasfilename
from tkMessageBox import showerror
from PIL import Image, ImageTk, ImageDraw

from sys import argv
from time import time

import filters


class App(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.pack_propagate(1)
        self.buildUI() 
        self.parent.config(menu=self.menubar)
        self.mouseCoords = (0,0)
        self.outputFilename = ""
        self.filetypes = [('Portable Network Graphics','*.png'),('JPEG / JFIF','*.jpg')]
        return
        
    def buildUI(self):
        self.parent.title("Vision computacional")
        self.pack()

        self.menubar = Menu(root)

        self.filemenu = Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Open", command=self.loadFile)
        self.filemenu.add_command(label="Save", command=self.saveAll)
        self.filemenu.add_command(label="Save As", command=self.saveAs)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Configuration", command=self.configBox)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.parent.quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.filtersmenu = Menu(self.menubar, tearoff=0)
        self.filtersmenu.add_command(label="Grayscale", command=lambda:applyFilter("g"))
        self.filtersmenu.add_command(label="Blur", command=lambda:applyFilter("b"))
        self.filtersmenu.add_command(label="Negative", command=lambda:applyFilter("n"))
        self.filtersmenu.add_command(label="Threshold", command=lambda:applyFilter("t"))
        self.filtersmenu.add_command(label="Sepia", command=lambda:applyFilter("s"))
        self.menubar.add_cascade(label="Filters", menu=self.filtersmenu)

        self.imagemenu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Image", menu=self.imagemenu)

        self.advancedmenu = Menu(self.menubar, tearoff=0)
        self.advancedmenu.add_command(label="Add noise", command=lambda:applyFilter("no"))
        self.advancedmenu.add_command(label="Difference", command=lambda:applyFilter("d"))
        self.advancedmenu.add_command(label="2D Convolution", command=lambda:applyFilter("c"))
        self.advancedmenu.add_separator()
        self.advancedmenu.add_command(label="Remove noise", command=lambda:applyFilter("rn"))
        self.menubar.add_cascade(label="Advanced", menu=self.advancedmenu)

        self.machinevisionmenu = Menu(self.menubar, tearoff=0)
        self.machinevisionmenu.add_command(label="Border detection", command=lambda:applyFilter("bd"))
        self.machinevisionmenu.add_command(label="Convex hull", command=convexHull)
        self.machinevisionmenu.add_command(label="Object detection", command=lambda:applyFilter("od"))
        self.machinevisionmenu.add_command(label="Object classification", command=lambda:objectClassification((0,0,0),label=True))
        self.machinevisionmenu.add_command(label="Line detection", command=lineDetection)
        self.machinevisionmenu.add_command(label="Circle detection", command=lambda:circleDetection(radius=46))
        self.machinevisionmenu.add_command(label="Deep circle detection", command=lambda:circleDetection())
        self.menubar.add_cascade(label="Machine Vision", menu=self.machinevisionmenu)

        self.menubar.add_command(label="Reset", command=self.resetCanvas)

        canvasContainer = Frame(self.parent)
        canvasContainer.pack(side=TOP)
        self.canvas = Canvas(canvasContainer, width=200, height=20)
        self.canvas.bind('<Button-1>', callback)
        self.canvas.pack(side=LEFT, padx=5,pady=15)
        self.edit_canvas = Canvas(canvasContainer, width=200, height=20)
        self.edit_canvas.bind('<Button-1>', callback)
        self.edit_canvas.pack(side=RIGHT, padx=5,pady=15)
        
        self.configIsOpen = False
        self.configBox()
        
        return

    def configBox(self):
        if(not self.configIsOpen):
            self.configIsOpen = True
            top = self.top = Toplevel(self)

            a = Label(top, text="Threshold: ").grid(row=0, column=0)
            a = Label(top, text="Min: ").grid(row=0, column=1)
            self.minThreshold = Scale(top, from_=0, to=255, orient=HORIZONTAL)
            self.minThreshold.grid(row=0, column=2)

            a = Label(top, text="Max:").grid(row=0, column=3)
            self.maxThreshold = Scale(top, from_=0, to=255, orient=HORIZONTAL)
            self.maxThreshold.set(255)
            self.maxThreshold.grid(row=0, column=4)

            a = Label(top, text="Noise: ").grid(row=1, column=0)
            a = Label(top, text="Level: ").grid(row=1, column=1)
            self.noiseLevel = Scale(top, from_=1, to=5, orient=HORIZONTAL)
            self.noiseLevel.grid(row=1, column=2)

            a = Label(top, text="Intensity: ").grid(row=1, column=3)
            self.noiseIntensity = Scale(top, from_=0, to=10, orient=HORIZONTAL)
            self.noiseIntensity.grid(row=1, column=4)

            a = Label(top, text="Remove Noise: ").grid(row=2, column=0)
            a = Label(top, text="Aggressiveness: ").grid(row=2, column=1)
            self.noiseAggressiveness = Scale(top, from_=1, to=10, orient=HORIZONTAL)
            self.noiseAggressiveness.set(3)
            self.noiseAggressiveness.grid(row=2, column=2)
    
            self.top.protocol("WM_DELETE_WINDOW", self.closeConfig)
        return

    def closeConfig(self):
        self.configIsOpen = False
        self.top.destroy()
        return

    def resetCanvas(self):
        global image
        self.edit_canvas.delete("all")
        self.loadFile(filename=image)
        print "Done!"
        return

    def updateCanvas(self, original=False):
        global workingImage
        w, h = workingImage["size"][0], workingImage["size"][1]
        self.editImage = ImageTk.PhotoImage(workingImage["instance"])
        self.edit_canvas.delete("all")
        self.edit_canvas.configure(width=w, height=h)
        self.edit_canvas.create_image(w/2, h/2, image=self.editImage)
        if(original):
            self.canvasImage = self.editImage
            self.canvas.delete("all")
            self.canvas.configure(width=w, height=h)
            self.canvas.create_image(w/2, h/2, image=self.canvasImage)
        return

    def loadFile(self, filename=False):
        global image, workingImage
        if(not filename):
            filename = askopenfilename(filetypes=[("PNG","*.png"),\
                                                  ("JPEG","*.jpeg"),\
                                                  ("JPEG","*.jpg")])
        if(filename):
            try:
                image = filename
                workingImage = imageToPixels(filename)
                self.updateCanvas(original=True)
            except:
                print "Error opening file or file does not exists"
                showerror("Open Source File", "Failed to read file\n'%s'" % filename)                
        return

    def saveAll(self):
        global workingImage
        if(self.outputFilename == ""):
            self.saveAs()
        else:
            print "Saving changes to %s" % self.outputFilename
            workingImage["instance"].save(self.outputFilename)
            workingImage["instance"].show()
        return

    def saveAs(self):
        global workingImage
        self.outputFilename = asksaveasfilename(parent=self.parent,filetypes=self.filetypes ,title="Save the image as...")
        if len(self.outputFilename) > 0:
            print "Saving under %s" % self.outputFilename
            workingImage["instance"].save(self.outputFilename)
            workingImage["instance"].show()
        else:
            print "Cancel save operation"
        return


def callback(event):
    global app
    app.mouseCoords = (event.x, event.y)
    print "Clic: %d, %d"%(event.x, event.y)
    return

def applyFilter(f):
    global image, workingImage, app

    if(workingImage == None):
        app.loadFile()

    width, height = workingImage["size"][0], workingImage["size"][1]
    pixels = workingImage["pixels"]

    if(f == "g"):
        pixels = filters.grayscale(pixels)
    elif(f == "t"):
        minT, maxT = app.minThreshold.get(), app.maxThreshold.get()
        pixels = filters.grayscale(pixels, lmin=minT, lmax=maxT)
    elif(f == "b"):
        pixels = filters.blur(pixels, width, height)
    elif(f == "n"):
        pixels = filters.negative(pixels)
    elif(f == "s"):
        pixels = filters.sepia(pixels)
    elif(f == "no"):
        level, intensity = app.noiseLevel.get(), app.noiseIntensity.get()
        pixels = filters.noise(pixels, level, intensity)
    elif(f == "rn"):
        aggressiveness = app.noiseAggressiveness.get()
        pixels = filters.removeNoise(pixels, width, height, aggressiveness)
    elif(f == "d"):
        pixels = filters.difference(pixels, width, height)
    elif(f == "c"):
        pixels = filters.applyMask(pixels, width)
    elif(f == "bd"):
        pixels = filters.borderDetection(pixels, width)
    elif(f == "od"):
        coordinates = app.mouseCoords
        pixels, dmp1, dmp2 = filters.objectDetection(pixels, width, height, coordinates)
    else:
        pass
    saveImageChanges(pixels)
    app.updateCanvas()
    print "Done!"
    return


def objectClassification(color, label=False):
    global image, workingImage, app

    width, height = workingImage["size"][0], workingImage["size"][1]
    pixels = workingImage["pixels"]

    pixels, objects = filters.objectClassification(pixels, width, height, color=color)

    if(label):
        mask = Image.new("RGBA", workingImage["size"])
        draw = ImageDraw.Draw(mask)
        saveImageChanges(pixels)
        for mObject in objects:
            x, y = mObject["center"][0], mObject["center"][1]
            draw.ellipse((mObject["center"][0]-3, mObject["center"][1]-3, mObject["center"][0]+3, mObject["center"][1]+3), fill="black")
            draw.text((x+2,y+2), "Ob%d"%(mObject["id"]), fill="white")
            print "Object ID: %d, Size (pixels): %s, Size (percentage): %s%%, Detection Point: %s"%(mObject["id"], len(mObject["pixels"]), mObject["percentage"], mObject["dp"])
        saveImageDraw(mask)
        app.updateCanvas()
    print "Done!"
    return objects


def convexHull():
    global app, workingImage
    mask = Image.new("RGBA", workingImage["size"])
    draw = ImageDraw.Draw(mask)
    #start = time()
    objects = objectClassification((255,255,255))
    for mObject in objects:
        hulls= filters.graham_scan(mObject["pixels"])
        points = list()
        for hull in hulls:
            points += list(hull)
            draw.ellipse((hull[0]-2, hull[1]-2, hull[0]+2, hull[1]+2), fill="red")
        draw.polygon(points, outline="red", fill=None)
    saveImageDraw(mask)
    app.updateCanvas()
    #print (time() - start)
    return


def lineDetection():
    global app, workingImage
    pixels = workingImage["pixels"]
    width, height = workingImage["size"][0], workingImage["size"][1]
    lines = filters.houghTransform(pixels, width, height)
    saveImageChanges(lines)
    app.updateCanvas()
    return


def circleDetection(radius=0):
    global app, workingImage

    mask = Image.new("RGBA", workingImage["size"])
    draw = ImageDraw.Draw(mask)
    pixels = workingImage["pixels"]
    width, height = workingImage["size"][0], workingImage["size"][1]

    allCircles, maxD = filters.circleDetection(pixels, width, height, radius=radius)
    #monedas100=72, #monedas5=46, #aros=41
    app.resetCanvas()
    cId = 0
    for radius in allCircles:
        r = int(radius)
        d = r*2
        dP = d*100/maxD
        for c in allCircles[radius]:
            draw.ellipse((c[0]-r, c[1]-r, c[0]+r, c[1]+r), fill=None, outline="yellow")
            draw.ellipse((c[0]-3, c[1]-3, c[0]+2, c[1]+2), fill="green", outline="green")
            draw.text((c[0]+2,c[1]+2), "C%d"%(cId), fill="white")
            print "Circle [%d]\tRadius = %d\t Diameter = %d\t Diam.Percent = %d%%"%(cId, r, d, dP)
            cId += 1
    saveImageDraw(mask)
    app.updateCanvas()    
    return


def imageToPixels(inputImage):
    image = dict()
    i = Image.open(inputImage)
    pixels = i.load()
    w, h = i.size
    pixelsRGB = list()
    for y in xrange(h):
        for x in xrange(w):
            pixel = pixels[x,y][:3]
            pixelsRGB.append(pixel)
    image["instance"] = i
    image["size"] = tuple([w, h])
    image["pixels"] = pixelsRGB
    return image

def saveImageChanges(pixels):
    global workingImage
    workingImage["instance"].putdata(pixels)
    workingImage["pixels"] = pixels
    return

def saveImageDraw(mask):
    global workingImage
    workingImage["instance"].paste(mask, mask=mask)
    i = workingImage["instance"]
    pixels = i.load()
    w, h = i.size
    pixelsRGB = list()
    for y in xrange(h):
        for x in xrange(w):
            pixel = pixels[x,y][:3]
            pixelsRGB.append(pixel)
    workingImage["instance"] = i
    workingImage["size"] = tuple([w, h])
    workingImage["pixels"] = pixelsRGB
    return

root = Tk()
app = App(root)
try:
    image = argv[1]
    app.loadFile(filename=image)
    try:
        mFilter = sys.argv[2]
        applyFilter(mFilter)
    except:
        pass
except:
    image, workingImage = None, None
root.mainloop() 
