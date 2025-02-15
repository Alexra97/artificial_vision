#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream

# Crear ventana 
cv.namedWindow("MSAT", cv.WINDOW_NORMAL)

# Definir una función para no hacer nada
def nothing(x): pass

# Crear los trackbars para controlar las características del espacio de color
cv.createTrackbar("h", "MSAT", 0, 179, nothing)
cv.createTrackbar("s", "MSAT", 0, 255, nothing)
cv.createTrackbar("v", "MSAT", 0, 255, nothing)

# Bucle principal de entrada de vídeo
for key, frame in autoStream():
    # Transformar el espacio BGR en HSV que facilita el acceso a ciertas características
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Obtener la posición de los trackbars
    h = cv.getTrackbarPos('h','MSAT')
    s = cv.getTrackbarPos('s','MSAT')
    v = cv.getTrackbarPos('v','MSAT')
    
    # Modificar las características de la imagen con el trackbar
    hsv[:,:,0] = hsv[:,:,0]+h # Matiz
    hsv[:,:,1] = hsv[:,:,1]+s # Saturación
    hsv[:,:,2] = hsv[:,:,2]+v # Valor (Luminosidad)

    # Convertir el espacio a BGR de nuevo para mostrarlo
    out = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('MSAT',out)
    
# Al salir del bucle cerrar las ventanas
cv.destroyAllWindows()

    









