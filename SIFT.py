#!/usr/bin/env python

import cv2 as cv
import time
from umucv.stream import autoStream
from umucv.util import putText
import numpy as np

# Crear una instancia del método de selección de características, el matcher y la ventana
sift = cv.xfeatures2d.SIFT_create(nfeatures=500)
matcher = cv.BFMatcher()
cv.namedWindow('SIFT')

# Variables auxiliares
models = []         # Lista de modelos (imagenes)
kpImages = []       # Lista de keypoints de los modelos
descImages = []     # Lista de descriptores de los modelos
h = w = 100         # Alto y ancho para mostrar los modelos
first = True        # Flag para detectar el primer frame
umbral = 20         # Porcentaje umbral para aceptar una predicción

# Bucle principal de entrada de vídeo
for key, frame in autoStream():
    
    # Detectar las características del frame actual y registrar el tiempo
    t0 = time.time()
    keypoints, descriptors = sift.detectAndCompute(frame, mask=None)
    t1 = time.time()
    
    # Cuando se pulsa 'm' se captura un modelo
    if key == ord('m'):
        # Redimensionar la imagen para mostrarla
        model = cv.resize(frame,(w,h))
        if first: # Si es el primer modelo la imagen de la ventana será solo este
            modelsImg = model
            first = False
        else: # Si no es el primero se crea una imagen que combina cada modelo
            modelsImg = np.hstack((modelsImg, model))

        # En cualquier caso se añade a la lista de modelos y se registran sus características
        models.append(frame.copy())
        kpImages.append(keypoints)
        descImages.append(descriptors)
       
    # Mostrar la información en pantalla 
    putText(frame, f'Pulsa \'m\' para capturar nuevos modelos',orig=(10,20))
    putText(frame, f'TC: {len(keypoints)} pts = {1000*(t1-t0):.0f} ms',orig=(10,40))
    
    # Dibujar los keypoints
    flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    cv.drawKeypoints(frame, keypoints, frame, color=(100,150,255), flags=flag)
   
    # Para cada modelo de la lista
    bestMatch = 0
    for m in range(len(models)):
        # Usar el matcher para calcular las coincidencias
        matches = matcher.knnMatch(descriptors, descImages[m], k=2)

        # Guardar solo las coincidencias prometedoras, las que son mucho mejores que 
        # la segunda opción
        good = []
        for mat in matches:
            if len(mat) >= 2:
                best,second = mat
                if best.distance < 0.75*second.distance:
                    good.append(best)

        # Obtener el porcentaje de coincidencia (matches/keypoints totales del modelo)
        percent = (len(good)/len(kpImages[m]))*100
        
        # Si el porcentaje es mejor que el mejor que se había encontrado, se guarda
        if percent > bestMatch: 
            bestMatch = percent
            bestImg = m
     
    # Solo cuando se supere el umbral se muestra el modelo elegido (para esto debe haber modelos)       
    if bestMatch > umbral:
        putText(frame ,f'Modelo {bestImg:d} con un {bestMatch:.2f}%',orig=(10,60))
        frame[70:70+h,10:10+w] = cv.resize(models[bestImg],(w,h))
        
    # Cuando hay algún modelo registrado se muestra la ventana con los modelos registrados
    if not first: 
        cv.namedWindow('modelos')
        cv.imshow('modelos', modelsImg)
        
    # Mostrar el frame
    cv.imshow('SIFT',frame)

# Salir del programa 
cv.destroyAllWindows()
