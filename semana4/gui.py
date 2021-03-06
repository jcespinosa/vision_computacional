#!/usr/bin/python

# Librerias
from Tkinter import *
from tkFileDialog import askopenfilename
from tkMessageBox import showerror
from PIL import Image, ImageTk

from sys import argv
from time import time

import filters


class App(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.buildUI() 
        self.parent.config(menu=self.menubar)
        self.mouseCoords = (0,0)
        return
        

    def buildUI(self):
        self.parent.title("Vision computacional")
        self.pack()

        self.menubar = Menu(root)

        self.filemenu = Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Open", command=self.loadFile)
        self.filemenu.add_command(label="Save All", command=saveAllChanges)
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
        self.menubar.add_cascade(label="Machine Vision", menu=self.machinevisionmenu)

        self.menubar.add_command(label="Undo", command=self.resetCanvas)
        self.menubar.add_command(label="Reset", command=self.resetCanvas)

        canvasContainer = Frame(self.parent)
        canvasContainer.pack(side=TOP)
        self.canvas = Canvas(canvasContainer, width=50, height=50)
        self.canvas.bind('<Button-1>', callback)
        self.canvas.pack(side=LEFT, padx=5,pady=15)
        self.edit_canvas = Canvas(canvasContainer, width=50, height=50)
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
        global image, lastImage
        self.edit_canvas.delete("all")
        self.updateCanvas(image)
        lastImage = image
        print "Done!"
        return

    def updateCanvas(self, i, original=False):
        i = Image.open(i)
        w, h = i.size
        self.editImage = ImageTk.PhotoImage(i)
        self.edit_canvas.configure(width=w, height=h)
        self.edit_canvas.create_image(w/2, h/2, image=self.editImage)
        if(original):
            self.canvasImage = self.editImage
            self.canvas.configure(width=w, height=h)
            self.canvas.create_image(w/2, h/2, image=self.canvasImage)
        return

    def setCanvas(c):
        self.edit_canvas = c
        return

    def loadFile(self):
        global image, lastImage
        filename = askopenfilename(filetypes=[("PNG","*.png"),\
                                              ("JPEG","*.jpeg"),\
                                              ("JPEG","*.jpg")])
        if(filename):
            try:
                image = filename
                lastImage = image
                self.updateCanvas(image, original=True)
            except:
                print "Error opening file or file does not exists"
                showerror("Open Source File", "Failed to read file\n'%s'" % filename)                
        return
    
def callback(event):
    global app
    app.mouseCoords = (event.x, event.y)
    print "Clic: %d, %d"%(event.x, event.y)
    return

def applyFilter(f):
    global image, lastImage, app
    if(f == "o"):
        i = image
    else:        
        i = "tmp.png"
        a, width, height, pixels = imageToPixels(lastImage)
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
            color = (0,0,255)
            pixels, dmp1, dmp2 = filters.objectDetection(pixels, width, height, coordinates, color)
        else:
            pass
        saveImage((width, height), pixels, i)
    lastImage = i
    app.updateCanvas(i)
    print "Done!"
    return


def objectClassification(color, label=False):
	global image, lastImage, app
	i = "tmp.png"
	a, width, height, pixels = imageToPixels(lastImage)
	pixels, objects = filters.objectClassification(pixels, width, height, color=color)
	saveImage((width, height), pixels, i)
	lastImage = i
	a.putdata(pixels)
	app.editImage = ImageTk.PhotoImage(a)
	app.edit_canvas.configure(width=width, height=height)
	app.edit_canvas.create_image(width/2, height/2, image=app.editImage)
	if(label):
		for mObject in objects:
			#x,y = mObject["center"][0], mObject["center"][1]
			#app.edit_canvas.create_oval(mObject["center"][0]-3, mObject["center"][1]-3, mObject["center"][0]+3, mObject["center"][1]+3, fill="black", outline="black")
			#l = Label(app.edit_canvas, text="Ob%d"%(mObject["id"])).place(x=x,y=y)
			print "Object ID: %d, Size (pixels): %s, Size (percentage): %s%%"%(mObject["id"], len(mObject["pixels"]), mObject["percentage"])
	print "Done!"
	return objects


def convexHull():
    global app
    start = time()
    objects = objectClassification((255,255,255))
    for mObject in objects:
        hulls= filters.graham_scan(mObject["pixels"])
        points = list()
        for hull in hulls:
            points += list(hull)
            #app.edit_canvas.create_oval(hull[0]-3, hull[1]-3, hull[0]+3, hull[1]+3, fill="black", outline="red")
        app.edit_canvas.create_polygon(points, outline="red", fill="", width=3.0)
    print (time() - start)
    return
        

def imageToPixels(inputImage):
    i = Image.open(inputImage)
    pixels = i.load()
    w, h = i.size
    pixelsRGB = list()
    for x in xrange(h):
        for y in xrange(w):
            pixel = pixels[y,x]
            pixelsRGB.append(pixel)
    return i, w, h, pixelsRGB

def saveImage(size, pixels, outputName, show=False):
    im = Image.new('RGB', size)
    im.putdata(pixels)
    im.save(outputName)
    if(show): im.show()
    return

def saveAllChanges():
    global lastImage
    i = "output.png"
    a, width, height, pixels = imageToPixels(lastImage) 
    saveImage((width, height), pixels, i, show=True)
    return

root = Tk()
app = App(root)
try:
    image = argv[1]
    lastImage = image
    app.updateCanvas(image, original=True)
except:
    image, lastImage = None, None
root.mainloop() 
