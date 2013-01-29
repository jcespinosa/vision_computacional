#!/usr/bin/python

from Tkinter import *
from PIL import Image, ImageTk
from sys import argv

def filterG(pixels):
    newPixels = list()
    for pixel in pixels:
        color = sum(pixel)/3
        newPixel = (color, color, color)
        newPixels.append(newPixel)
    return newPixels

def imageToPixels(inputImage):
    i = Image.open(inputImage)
    pixels = i.load()
    w, h = i.size
    pixelsRGB = []
    for x in range(h):
        for y in range(w):
            pixel = pixels[y,x]
            pixelsRGB.append(pixel)
    return i, w, h, pixelsRGB

def saveImage(size, pixels, outputName):
    im = Image.new('RGB', size)
    im.putdata(pixels)
    im.save(outputName)
    return

def actionGrayFilter():
    pass

def main():
    image = argv[1]
    outputName = "gray_" + image
    i, width, height, pixelsRGB = imageToPixels(image)
    pixelsGray = filterG(pixelsRGB)
    saveImage((width, height), pixelsGray, outputName)
    
    master = Tk()
    master.title("Semana 2: Filtros")
    btnGrayFilter = Button(master, text="Gray Filter")
    btnGrayFilter.pack()
    canvas = Canvas(master, width=width, height=height)
    canvas.pack()
    tkImage = ImageTk.PhotoImage(i)
    canvas.create_image(width/2, height/2, image=tkImage)
    frame = Frame(master)
    
    master.mainloop()

    return

if(__name__=="__main__"):
    main()
