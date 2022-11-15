#!/usr/bin/env python3

import numpy             as np
import cv2               as cv
from umucv.stream import autoStream

# Parseo de argumentos para pasarlos por terminal
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, help='ruta hacia la imagen de fondo')
args = parser.parse_args()

# Función que no hace nada
def nothing(x): pass

# Leer la imagen de fondo
bg  = cv.imread(args.img)

# Crear la ventana del chroma con un trackbar para modificar el umbral en tiempo real
cv.namedWindow("CHRO")
cv.createTrackbar("u", "CHRO", 0, 255, nothing)

# Definir el flag de primer frame y el kernel
first = True
kernel = np.ones((3,3),np.uint8)

# Bucle principal de entrada de vídeo
for key, frame in autoStream():
    # Para el primer frame se selecciona este como fondo capturado por defecto
    if first:
        prev = frame
        first = False
    
    # Cuando se pulsa 'c' se captura el fondo
    if key == ord('c'): prev = frame
    
    # Actualizar el umbral con el trackbar
    u = cv.getTrackbarPos('u','CHRO')
    
    # Diferencia de imágenes en el espacio RGB
    diff = np.sum(cv.absdiff(prev,frame), axis=2)
    
    # Crear la máscara a partir del umbral
    mask = diff > u

    # Ajustamos el tamaño del fondo elegido para que encaje con la entrada de vídeo
    r,c = mask.shape
    result = cv.resize(bg,(c,r))
    
    # Expandir la máscara a 3 canales para poder copiar RGB
    mask3 = np.expand_dims(mask,axis=2)
    np.copyto(result, frame, where = mask3)

    # Mostrar imagen resultante
    cv.imshow('CHRO',result)

# Salir del programa
cv.destroyAllWindows()








