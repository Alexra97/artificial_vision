#!/usr/bin/env python

import face_recognition
import cv2 as cv
import time
from umucv.util import putText
from umucv.stream import autoStream
from umucv.util   import ROI
import numpy as np

# Crear la ventana y añadir una región de interés a ella
cv.namedWindow('ANON')
roi = ROI('ANON')

# Variables auxiliares
h = w = 50            # Alto y ancho de los modelos a mostrar
models = []           # Lista de los modelos
names = []            # Lista de nombres de las caras  
n = 0                 # Número de nombres
flagModels = False    # Flag para saber si hay modelos
flagFaces = False     # Flag para la activación del detector de caras

# Bucle principal de entrada de vídeo
for key, frame in autoStream():

    # Si se ha seleccionado una región
    if roi.roi:
        # Obtener sus coordenadas
        [x1,y1,x2,y2] = roi.roi
        
        # Recortar esa sección de la imagen
        region = frame[y1:y2+1, x1:x2+1]
        
        # Si se pulsa la tecla 'r'
        if key == ord('r'):
            # Copiar la roi
            trozo = region.copy()
            
            # Redimensionar a la altura y ancho definida
            trozo = cv.resize(trozo,(w,h))
            models.append(trozo) 
            names.append("face"+str(n))
            n = n+1
            if not flagModels: # Si es el primer modelo se añade a la lista
                modelsImg = trozo
                flagModels = True
            else: # Si no es el primero se crea una imagen que combina cada modelo
                modelsImg = np.hstack((modelsImg, trozo))
            
            # Eliminar la roi cuando se registra
            roi.roi = []
            
        # Dibujar el rectangulo de la roi
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
    
    # Si se pulsa 'f' se activa el detector de caras
    if key == ord('f'): flagFaces = not flagFaces
    
    # Cuando hay algún modelo registrado se muestra la ventana con los modelos registrados
    if flagModels: 
        cv.namedWindow('Models')
        cv.imshow('Models', modelsImg)
        
    # Si se activó el flag de detección de caras
    if flagFaces:
        # Se inicializa una lista de encodings vacía
        encodings = []
        
        # Para cada modelo se intenta reconocer una cara en él y solo se guardan los modelos que almacenan caras
        for m in models:
            if face_recognition.face_encodings(m):
                encodings.append(face_recognition.face_encodings(m)[0])
                
        # Se registran los tiempos t0,t1 y t2 para mostrar estadísticas sobre el rendimiento del programa
        t0 = time.time()

        # Localización de las caras del frame actual
        face_locations = face_recognition.face_locations(frame)
        t1 = time.time()
    
        # Encodigns de las caras del frame actual
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        t2 = time.time()
    
        # Para cada cara obtener su posición
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Comparar la cara del frame con la del modelo
            match = face_recognition.compare_faces( encodings, face_encoding)
            
            # Para cada nombre y ls lista de matches
            name = "Unknown"
            for n, m in zip(names, match):
                # Si se encuentra a verdadero ha habido una coincidencia
                if m:
                    # Almacenar el nombre
                    name = n
                    
                    #Recortar la posición de la cara de la imagen
                    region = frame[top:bottom, left:right]
                    
                    # Obtener su forma
                    height, width = region.shape[:2]
                    
                    # Pixelar la región disminuyéndola a 16x16 píxeles
                    region = cv.resize(region, (16, 16), interpolation=cv.INTER_LINEAR)
                    
                    # Devolverla a su forma original
                    region = cv.resize(region, (width, height), interpolation=cv.INTER_NEAREST)
                    
                    # Actualizar la región en el frame
                    frame[top:bottom, left:right] = region
    
            # Dibujar el rectángulo de la cara y escribir su nombre
            cv.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            putText(frame, name, orig=(left+3,bottom+16))

        # Mostrar las estadísticas de rendimiento
        putText(frame, f'{(t1-t0)*1000:.0f} ms {(t2-t1)*1000:.0f} ms')

    # Mostrar la imagen resultante
    cv.imshow('ANON',frame)

# Salir del programa al finalizar el bucle
cv.destroyAllWindows()