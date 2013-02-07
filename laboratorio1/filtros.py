#!/usr/bin/python

# Librerias
from Tkinter import *
from PIL import Image, ImageTk
from sys import argv

# Clase para crear la interfaz grafica
class App(Frame):
    # Construccion de la ventana
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.buildUI()

    # Construccion de los widgets de la ventana
    def buildUI(self):
        # Agregamos el titulo de la ventana
        self.parent.title("Semana 2: Filtros")
        self.pack()
        
        # Agregamos el canvas donde se mostrara la imagen
        self.canvas = Canvas(self.parent, width=300, height=300)
        self.canvas.pack(side=TOP, padx=10,pady=10)

        # Agregamos cada uno de los botones para aplicar los filtros
        self.btnOriginal = Button(self.parent, width=10, text="Original", command=lambda:applyFilter("o"))
        self.btnOriginal.pack(side=LEFT)

        self.btnSave = Button(self.parent, width=10, text="Save All", command=saveAllChanges)
        self.btnSave.pack(side=LEFT)

        self.btnGrayFilter = Button(self.parent, width=10, text="Grayscale", command=lambda:applyFilter("g"))
        self.btnGrayFilter.pack(side=RIGHT)

        self.btnLimitFilter = Button(self.parent, width=10, text="Limit", command=lambda:applyFilter("l"))
        self.btnLimitFilter.pack(side=RIGHT)

        self.btnBlurFilter = Button(self.parent, width=10, text="Blur", command=lambda:applyFilter("b"))
        self.btnBlurFilter.pack(side=RIGHT)

        self.btnNegativeFilter = Button(self.parent, width=10, text="Negative", command=lambda:applyFilter("n"))
        self.btnNegativeFilter.pack(side=RIGHT)

        return

    # Funcion para actualizar el canvas con la nueva imagen
    def updateCanvas(self, image):
        i = Image.open(image) # Abrimos la imagen
        w, h = i.size         # Obtenemos el tamano de la imagen
        self.canvasImage = ImageTk.PhotoImage(i) # Se convierte a una imagen compatible con el canvas 
        self.canvas.configure(width=w, height=h) # Se configura el canvas al tamano de la imagen
        self.canvas.create_image(w/2, h/2, image=self.canvasImage) # Se actualiza el canvas
        return

# Funcion para aplicar los filtros
def applyFilter(f):
    global image, lastImage, app
    i = "tmp.png"
    if(f == "g"):      # Condicion para aplicar el filtro correspondiente
        a, width, height, pixelsRGB = imageToPixels(lastImage) # Llamada a la funcion que convierte la imagen en pixeles
        pixels = filterG(pixelsRGB)           # Aplicamos el filtro correspondiente
        saveImage((width, height), pixels, i) # Guardamos la imagen
    elif(f == "l"):
        a, width, height, pixelsRGB = imageToPixels(lastImage)
        pixels = filterG(pixelsRGB, lmin=100, lmax=150)
        saveImage((width, height), pixels, i)
    elif(f == "b"):
        a, width, height, pixelsRGB = imageToPixels(lastImage)
        pixels = filterD(pixelsRGB, width, height)
        saveImage((width, height), pixels, i)
    elif(f == "n"):
        a, width, height, pixelsRGB = imageToPixels(lastImage)
        pixels = filterN(pixelsRGB)
        saveImage((width, height), pixels, i)
    else:
        i = image
    lastImage = i       # Guardamos la referencia a la imagen modificada
    app.updateCanvas(i) # Actualizamos el canvas con la imagen actualizada
    return


# Funcion para guardar los cambios que se han hecho a la imagen
def saveAllChanges():
    global lastImage # Tomamos la referencia de la imagen modificada
    i = "output.png" # Nombre del archivo de salida
    a, width, height, pixels = imageToPixels(lastImage) # Obtenemos el tamano de la imagen y los pixeles que la componen
    saveImage((width, height), pixels, i, show=True)    # Enviamos la informacion de la imagen para guardarla
    return

# Funcion para dividir la lista de imagen en una matriz de 2 dimensiones
# La funcion se utiliza para aplicar el filtro para difuminar la imagen
def slicing(l, n):
    return [l[a:a+n] for a in range(0, len(l), n)]

# Funcion para aplicar el filtro de grises y de limites
# Recibe la lista de pixeles a aplicar el filtro
# Recibe 2 argumentos opcionales para aplicar el filtro de limites
# un limite inferior y superior
def filterG(pixels, lmin=0, lmax=255):
    for a, pixel in enumerate(pixels):           # Recorremos la lista de pixeles
        color = sum(pixel)/3                     # Sumamos los valores de los colores y los promediamos entre 2
        color = 255 if(color >= lmax) else color # Condicion para aplicar el limite superior
        color = 0 if(color <= lmin) else color   # Condicion para aplicar el limite inferior
        pixels[a] = (color, color, color)        # Modificamos el color del pixel
    print "Done!" 
    return pixels                                # Regresamos la lista de pixeles modificados

# Funcion para aplicar el filtro negativo
# Recibe la lista de pixeles a aplicar el filtro
# Como argumento opcional
def filterN(pixels, cMax=255):
    for a, pixel in enumerate(pixels): # Recorremos la lista de pixeles
        R = cMax - pixel[0]            # Aplicamos el filtro negativo, restando al valor mayor
        G = cMax - pixel[1]            # el valor de cada uno de los colores actuales del pixel
        B = cMax - pixel[2]            # R, G B
        pixels[a] = (R, G, B)          # Modificamos el color del pixel
    print "Done!"
    return pixels                      # Regresamos la lista de pixeles modificados

# Funcion para aplicar el filtro difuminado a una lista de pixeles
# Recibe la lista de pixeles a promediar
def blurColor(pixels):
    # Se realiza la sumatoria de los colores correspondientes
    # y se promedia su valor entre la cantidad de pixeles
    # en la lista
    newPixel = (sum([pixel[0] for pixel in pixels])/len(pixels),\
                sum([pixel[1] for pixel in pixels])/len(pixels),\
                sum([pixel[2] for pixel in pixels])/len(pixels))
    return newPixel

# Funcion para aplicar el filtro difuminado a la imagen
# Recibe el tamano de la imagen largo y alto.
def filterD(pixels, width, height):
    newPixels = list() # Creamos una nueva secuencia de pixeles
    pixels = slicing(pixels, width)    # Creamos una matriz bidimensional del tamano de la imagen (largo * alto)
    for a, pLine in enumerate(pixels): # Recorremos cada linea de pixeles de la magen
        print str(round(float(a*100.0/height),2))     # Impresion solo para conocer el avance de la aplicacion del filtro
        for b, pixel in enumerate(pLine):             # Recorremos cada pixel de cada linea
            pNeighbours = list()                      # Y creamos una lista para los vecinos del pixel seleccionado
            try: pNeighbours.append(pixels[a-1][b-1]) # Vamos agregando los pixeles vecinos al pixel seleccionado
            except IndexError: pass                   # Si no existe el vecino, lanzamos una excepcion
            try: pNeighbours.append(pixels[a-1][b]) 
            except IndexError: pass
            try: pNeighbours.append(pixels[a-1][b+1])
            except IndexError: pass
            try: pNeighbours.append(pixels[a+1][b-1])
            except IndexError: pass       
            try: pNeighbours.append(pixels[a+1][b])
            except IndexError: pass
            try: pNeighbours.append(pixels[a+1][b+1])
            except IndexError: pass
            try: pNeighbours.append(pLine[b-1])
            except IndexError: pass
            try: pNeighbours.append(pLine[b+1])
            except IndexError: pass
            try: pNeighbours.append(pixel)
            except IndexError: pass
            newPixel = blurColor(pNeighbours) # Enviamos la lista de vecinos a la funcion que crea el efecto de difuminado
            newPixels.append(newPixel)        # Agregamos el pixel a la lista de pixeles modificados
    print "Done!"
    return newPixels     # Regresamos la lista de pixeles modificados por el filtro

# Funcion para cargar la imagen y obtener el tamano y la lista de sus pixeles
# Recibe el nombre de la imagen a cargar
def imageToPixels(inputImage):
    i = Image.open(inputImage) # Abrimos la imagen
    pixels = i.load()          # Cargamos los pixeles de la imagen
    w, h = i.size              # Obtenemos el tamano de la imagen
    pixelsRGB = list()         # Creamos una lista de para guardar los pixeles
    for x in range(h):         # Recorremos las lineas de los pixeles
        for y in range(w):     # Recorremos los pixeles de cada linea de pixeles de la imagen
            pixel = pixels[y,x]     # Tomamos un pixel de la imagen
            pixelsRGB.append(pixel) # Lo agregamos a la lista de pixeles
    return i, w, h, pixelsRGB       # Regresamos la referencia a la imagen, su tamano y la lista de pixeles que la forman
# Es necesario recorrer la imagen a la inversa para tener la orientacion correcta
# Asi mismo, para guardar la imagen es necesario tener una sola secuencia de pixeles y no una matriz

# Funcion para guardar la imagen
# Recibe el tamano, los pixeles y el nombre del archivo de salida
# El argumento "show" es para mostrar en una ventana popup la imagen recien guardada.
def saveImage(size, pixels, outputName, show=False):
    im = Image.new('RGB', size) # Creamos una nueva imagen RGB del tamano deseado
    im.putdata(pixels)          # Colocamos los datos de la imagen (pixeles)
    im.save(outputName)         # Guardamos la imagen
    if(show): im.show()                  # Mostamos la imagen de salida
    return

# Rutina para crear la ventana de la aplicacion
image = argv[1]         # Argumento de la imagen a modificar
lastImage = image       # Se utiliza como referencia a la imagen que se ha modificado
root = Tk()             # Instanciamos la interfaz grafica
app = App(root)         # Creamos la ventana de la aplicacion
app.updateCanvas(image) # Actualizamos la ventana con la imagen
root.mainloop()         # Inicializamos la interfaz grafica hasta que el usuario cierre la aplicacion
