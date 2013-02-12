#!/usr/bin/python

# Librerias
from Tkinter import *
from tkFileDialog import askopenfilename
from tkMessageBox import showerror

from PIL import Image, ImageTk
from sys import argv

import filters


class App(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.buildUI() 
        self.parent.config(menu=self.menubar)

    def buildUI(self):
        self.parent.title("Semana 3: Cosas avanzadas")
        self.pack()

        self.menubar = Menu(root)

        self.filemenu = Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Open", command=self.loadFile)
        self.filemenu.add_command(label="Save All", command=saveAllChanges)
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
        self.menubar.add_cascade(label="Machine Vision", menu=self.machinevisionmenu)

        self.menubar.add_command(label="Undo", command=lambda:applyFilter("o"))

        canvasContainer = Frame(self.parent)
        canvasContainer.pack(side=TOP)
        self.canvas = Canvas(canvasContainer, width=50, height=50)
        self.canvas.pack(side=LEFT, padx=5,pady=15)
        self.edit_canvas = Canvas(canvasContainer, width=50, height=50)
        self.edit_canvas.pack(side=RIGHT, padx=5,pady=15)

        scaleContainer = Frame(self.parent)
        scaleContainer.pack(side=BOTTOM)
        a = Label(scaleContainer, text="Threshold: ")
        a.pack(side=LEFT, padx=10)
        a = Label(scaleContainer, text="Min: ")
        a.pack(side=LEFT)
        self.minThreshold = Scale(scaleContainer, from_=0, to=255, orient=HORIZONTAL)
        self.minThreshold.pack(side=LEFT, padx=10)
        a = Label(scaleContainer, text="Max:")
        a.pack(side=LEFT)
        self.maxThreshold = Scale(scaleContainer, from_=0, to=255, orient=HORIZONTAL)
        self.maxThreshold.set(255)
        self.maxThreshold.pack(side=LEFT)

        scaleContainer = Frame(self.parent)
        scaleContainer.pack(side=BOTTOM)
        a = Label(scaleContainer, text="Noise: ")
        a.pack(side=LEFT, padx=10)
        a = Label(scaleContainer, text="Level: ")
        a.pack(side=LEFT)
        self.noiseLevel = Scale(scaleContainer, from_=1, to=5, orient=HORIZONTAL)
        self.noiseLevel.pack(side=LEFT, padx=10)
        a = Label(scaleContainer, text="Intensity: ")
        a.pack(side=LEFT)
        self.noiseIntensity = Scale(scaleContainer, from_=0, to=10, orient=HORIZONTAL)
        self.noiseIntensity.pack(side=LEFT)

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
            pixels = filters.removeNoise(pixels, width, height)
        elif(f == "d"):
            pixels = filters.difference(pixels, width, height)
        elif(f == "c"):
            pixels = filters.applyMask(a, pixels, width)
        elif(f == "bd"):
            pixels = filters.borderDetection(pixels, width)
        else:
            pass
        saveImage((width, height), pixels, i)
    lastImage = i
    app.updateCanvas(i)
    print "Done!"
    return

def imageToPixels(inputImage):
    i = Image.open(inputImage)
    pixels = i.load()
    w, h = i.size
    pixelsRGB = list()
    for x in range(h):
        for y in range(w):
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
