#!/usr/bin/env python3

import numpy             as np
import cv2               as cv
from collections import deque
from umucv.util import putText

# Parseo de argumentos para pasarlos por terminal
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, help='ruta hacia la imagen')
args = parser.parse_args()

# Función auxiliar para añadir puntos a las colas de las dos ventanas
def encolar(event, x, y, flags, queue):
    if event == cv.EVENT_LBUTTONDOWN:
        if queue == 0: pointsR.append((x,y)) 
        elif queue == 1: pointsL.append((x,y)) 

# Definir la ventana para la imagen original
cv.namedWindow("Original")
cv.setMouseCallback("Original", encolar, 0)

# Leer la imagen pasada por parámetro
imagen = cv.imread(args.img, 1)

# Colas de los puntos
pointsR = deque(maxlen=4)  # Rectángulo
pointsL = deque(maxlen=2)  # Línea
hCard = 5.5                # Altura en cm de una tarjeta estándar
wCard = 8.5                # Ancho en cm de una tarjeta estándar
proportion = wCard/hCard   # Proporción entre los lados
factor = 20                # Factor en 20 = 10(cm a mm)*2 píxeles
hReal = hCard * factor     # Calcular la altura en píxeles con el factor
wReal = hReal * proportion # Calcular el ancho en píxeles con la proporción
x0 = 400.                  # Coordenadas de origen elegidas por prueba y error
y0 = 300.
flag = 0                   # Flag para detectar la primera pulsación de 'r'

# Bucle infinito donde se mostrarán las ventanas hasta pulsar 'esc'
while (1):
    # Copiar la imagen original
    img = imagen.copy()
    
    # Dibujar los puntos sobre la ventana original
    for pf in pointsR:
        cv.circle(img, pf, 2, (0,255,255), -1) 
    
    # Si hay cuatro puntos se dibuja el rectángulo
    if len(pointsR) == 4:
        pts = np.array([pointsR[0],pointsR[1],pointsR[2],pointsR[3]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv.polylines(img,[pts],True,(0,255,255), 2)
         
    # Si está seleccionado el rectángulo y se pulsa 'r'
    if ((cv.waitKey(10) == ord('r')) & (len(pointsR) == 4)):
        # Si el flag estaba desactivado se crea una nueva ventana
        if flag == 0:
            cv.namedWindow("Rectificacion")
            cv.setMouseCallback("Rectificacion", encolar, 1)
            flag = 1
           
        # Crear las coordenadas sobre las que se colocará el rectangulo rectificado
        real = np.array([[x0, y0], [x0+hReal, y0],
                         [x0+hReal, y0+wReal], [x0, y0+wReal]])
        
        # Obtener la homografía
        H,_ = cv.findHomography(pts, real)
        
        # Crear la rectificación
        rectif = cv.warpPerspective(imagen.copy(),H,(650,650))
            
    # Si el flag está activado
    if flag == 1:
        # Copiar la rectificación
        rec = rectif.copy()
        
        # Dibujar los puntos sobre la ventana de rectificación
        for pf in pointsL:
            cv.circle(rec, pf, 2, (255,0,0), -1) 
        
        # Si hay dos puntos se dibuja una línea entre ellos y se calculan los cm que ocupa
        if len(pointsL) == 2:
            cv.line(rec, pointsL[0], pointsL[1], (255,0,0)) 
            c = np.mean(pointsL, axis = 0).astype(int) 
            pix = np.linalg.norm(np.array(pointsL[0]) - pointsL[1])
            cm = pix/factor
            putText(rec,f'{cm:.1f} cm',orig=c)
            
        # Mostrar la imagen rectificada
        cv.imshow('Rectificacion',rec)

    # Mostrar la imagen original
    cv.imshow('Original',img)
    
    # Si se pulsa 'esc' salir del bucle
    if (cv.waitKey(10) == 27): break

# Salir del programa
cv.destroyAllWindows()








