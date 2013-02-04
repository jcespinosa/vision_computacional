#!/usr/bin/python

from sys import argv
from math import sqrt, floor

base = 1024

def getTotal(total, mag):
    global base
    magnitudes = ['k','m','g','t']
    while(len(magnitudes) > 0):
        total *= base
        magnitudes.pop()
    pass

def resolucion(mp, x, y):
    global Mpxl
    totalPixeles = mp * Mpxl
    r = sqrt(totalPixeles/(x*y))
    return (floor(16*r), floor(9*r))

def main():
    megapixeles = int(argv[1])
    magnitud = int(argv[2])
    relx = int(argv[3])
    rely = int(argv[4])

    print resolucion(megapixeles, relx, rely)

if(__name__=="__main__"):
    main()
