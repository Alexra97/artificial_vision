#!/usr/bin/env python

import dlib
import cv2          as cv
import numpy        as np
from umucv.stream import autoStream
from umucv.util import putText

# Parseo de argumentos para pasarlos por terminal
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--predictor', type=str, help='ruta hacia el predictor de la cara')
args = parser.parse_args()

# Crear el detector y el predictor de la cara
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.predictor)

# Flags para activar/desactivar las opciones e inicialización del factor
puntos = efectos = False
f = 0

# Bucle principal de entrada de vídeo
for key,frame in autoStream():    
    # Imprimir las instrucciones en la pantalla
    putText(frame,f'Pulsa \'p\' para mostrar los puntos de la cara',orig=(10,20))
    putText(frame,f'Pulsa \'e\' para aplicar efectos. Factor actual: {f:d}',orig=(10,40))

    # Activar/Desactivar los flags y actualizar el factor
    if key == ord('p'): 
        if puntos: puntos = False
        else: puntos = True
    if key == ord('e'): 
        if efectos & (f < 4): f = f+1
        elif efectos & (f == 4): 
            efectos = False
            f = 0
        else: 
            efectos = True
            f = 2

    # Detectar la cara de la imagen
    dets = detector(frame, 0)
    # Para cada cara detectada
    for k, d in enumerate(dets):
        # Obtener los puntos de la cara
        shape = predictor(frame, d)
        
        # Crear una lista con las coordenadas de los puntos de la cara
        L = []
        for p in range(68):
            x = shape.part(p).x
            y = shape.part(p).y
            L.append(np.array([x,y]))
        L = np.array(L)
        
        # Si el flag de puntos está activado se muestran
        if puntos:
            cv.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255,128,64) )
            for p in range(68): 
                x = shape.part(p).x
                y = shape.part(p).y
                cv.circle(frame, (x,y), 2,(255,0,0), -1)
        
        # Si el flag de efectos está activado se muestra el efecto con el factor f
        if efectos:
            # Si hay puntos de cara detectados
            if len(L) > 0:
                # Obtener los puntos de la boca
                mouth = []
                for p in range(12):
                    mouth.append(L[p+48])
                mouth = np.array(mouth, dtype=np.int32)
                    
                # Crear una máscara del tamaño de las dimensiones de la imagen
                mask = np.zeros(frame.shape[:2], np.uint8)
                
                # Rellenar un polígono en la máscara con los puntos de la boca
                cv.fillPoly(mask, [mouth], 255)
                
                # Enmascarar la imagen con la máscara de la boca
                masked_data = cv.bitwise_and(frame, frame, mask=mask)

                # Obtener el contorno de la máscara y el rectángulo que la contiene
                _,thresh = cv.threshold(mask,1,255,cv.THRESH_BINARY)
                contours = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
                x,y,w,h = cv.boundingRect(contours[0])
                
                # Calcular las nuevas dimensiones en función del factor y recortar la boca de la máscara
                neww = w*f
                newh = h*f
                crop = cv.resize(masked_data[y:y+h,x:x+w], (neww,newh))
                
                # Calcular los márgenes para centrar la imagen y obtener las dimensiones originales
                marginw = int(round((neww-w)/2))
                marginh = int(round((newh-h)/2))
                height, width = frame.shape[:2]
                
                # Calcular los nuevos puntos de inicio y fin para la imagen original
                newx = x - marginw
                newy = y - marginh
                finx = newx+neww
                finy = newy+newh
                
                # Puntos de inicio y fin de la boca recortada
                cropx = 0
                cropy = 0
                cropfinx = neww
                cropfiny = newh
                
                # Comprobaciones para los casos en los que la imagen sale del 
                # rango de la cámara (derecha y abajo)
                if (newx+neww) > width:
                    cropfinx = cropfinx - (finx - width)
                    finx = width
                if (newy+newh) > height: 
                    cropfiny = cropfiny - (finy - height)
                    finy = height
                
                # Comprobaciones para los casos en los que la imagen sale del 
                # rango de la cámara (izquierda y arriba)
                if newx < 0:
                    cropx = -newx
                if newy < 0:
                    cropy = -newy
                    
                # Bucle que muestra la boca aumentada sobre la real. Sigue las coordenadas
                # descritas e ignora los valores negros de la máscara
                for j in range(cropy,cropfiny):
                    for i in range(cropx,cropfinx):
                        if np.any(crop[j,i]): frame[newy+j,newx+i] = crop[j,i]
            
    # Mostrar la imagen
    cv.imshow("DLIB",frame)

# Salir del programa 
cv.destroyAllWindows()
