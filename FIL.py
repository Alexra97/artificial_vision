#!/usr/bin/env python3

import cv2 as cv
from umucv.stream import autoStream
from umucv.util   import ROI

# Crear la ventana y añadir una región de interés a ella
cv.namedWindow('FIL')
roi = ROI('FIL')

# Definir función para no hacer nada
def nothing(x): pass

# Crear los trackbars para elegir los valores de los filtros
cv.createTrackbar("Gaussian Blur", "FIL", 1, 30, nothing)
cv.createTrackbar("Laplace", "FIL", 0, 50, nothing)
cv.createTrackbar("Threshold", "FIL", 0, 255, nothing)

# Flags para activar los filtros
gaussian = False
laplace = False
threshold = False

# Bucle principal de entrada de vídeo
for key, frame in autoStream():
    
    # Si se ha seleccionado una región
    if roi.roi:
        # Obtener sus coordenadas
        [x1,y1,x2,y2] = roi.roi
        
        # Recortar esa sección de la imagen
        region = frame[y1:y2+1, x1:x2+1]
        
        # Si se pulsa la tecla 'g' activar el filtro gaussiano
        if key == ord('g'):
            gaussian = True
            laplace = False
            threshold = False
            
        # Si se pulsa la tecla 'l' activar el filtro laplaciano
        if key == ord('l'):
            laplace = True
            gaussian = False
            threshold = False
            
        # Si se pulsa la tecla 't' activar el filtro de threshold
        if key == ord('t'):
            threshold = True
            gaussian = False
            laplace = False
        
        # Si se ha activado el filtro gaussiano
        if gaussian:
            # Obtener el valor del trackbar
            g = cv.getTrackbarPos('Gaussian Blur','FIL')

            # Asegurar que g es mayor que cero e impar
            if (g == 0): g = g+1
            elif (g % 2 == 0): g = g-1
            
            # Aplicar el suavizado
            region = cv.GaussianBlur(region,(g,g),cv.BORDER_DEFAULT)
         
            # Actualizar la región
            frame[y1:y2+1, x1:x2+1] = region
            
        # Si se ha activado el filtro laplaciano
        if laplace:
            # Obtener el valor del trackbar
            l = cv.getTrackbarPos('Laplace','FIL')
            
            # Aplicar el ruido laplaciano
            region = cv.Laplacian(region,-1,scale=l)
         
            # Actualizar la región
            frame[y1:y2+1, x1:x2+1] = region
        
        # Si se ha activado el filtro de threshold
        if threshold:
            # Obtener el valor del trackbar
            t = cv.getTrackbarPos('Threshold','FIL')
            
            # Aplicar el umbralizado
            _,region = cv.threshold(region,t,255,cv.THRESH_BINARY)
         
            # Actualizar la región
            frame[y1:y2+1, x1:x2+1] = region
       
        # Dibujar el rectangulo de la roi
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
    
    # Mostrar la imagen resultante
    cv.imshow('FIL',frame)
    
# Salir del programa 
cv.destroyAllWindows()