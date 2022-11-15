#!/usr/bin/env python3

import numpy as np
from umucv.stream import autoStream
import cv2 as cv
from umucv.util import putText

# Utilidades de extracción de contornos (disponibles en umucv)

# area, con signo positivo si el contorno se recorre "counterclockwise"
def orientation(x):
    return cv.contourArea(x.astype(np.float32),oriented=True)

# ratio area/perímetro^2, normalizado para que 100 (el arg es %) = círculo
def redondez(c):
    p = cv.arcLength(c.astype(np.float32),closed=True)
    oa = orientation(c)
    if p>0:
        return oa, 100*4*np.pi*abs(oa)/p**2
    else:
        return 0,0
    
def boundingBox(c):
    (x1, y1), (x2, y2) = c.min(0), c.max(0)
    return (x1, y1), (x2, y2)

# comprobar que el contorno no se sale de la imagen
def internal(c,h,w):
    (x1, y1), (x2, y2) = boundingBox(c)
    return x1>1 and x2 < w-2 and y1 > 1 and y2 < h-2

# reducción de nodos
def redu(c,eps=0.5):
    red = cv.approxPolyDP(c,eps,True)
    return red.reshape(-1,2)

# intenta detectar polígonos de n lados
def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if r.shape[0] == n ]

# detecta siluetas oscuras que no sean muy pequeñas ni demasiado alargadas
def extractContours(g, minarea=10, minredon=25, reduprec=1):
    ret, gt = cv.threshold(g,189,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    contours = cv.findContours(gt, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[-2]

    h,w = g.shape
    
    tharea = (min(h,w)*minarea/100.)**2 
    
    def good(c):
        oa,r = redondez(c)
        black = oa > 0 # and positive orientation
        return black and abs(oa) >= tharea and r > minredon

    ok = [redu(c.reshape(-1,2),reduprec) for c in contours if good(c)]
    return [ c for c in ok if internal(c,h,w) ]

# añadimos una coordenada cero a todas las filas
# para convertir un polígono del plano en un polígono
# en el espacio, a altura z=0
def addzerocol(x):
    return np.hstack([x,np.zeros([len(x),1])])

# matriz de calibración sencilla dada la
# resolución de la imagen y el fov horizontal en grados
def Kfov(sz,hfovd):
    hfov = np.radians(hfovd)
    f = 1/np.tan(hfov/2)
    # print(f)
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])
 
# convierte un conjunto de puntos ordinarios (almacenados como filas de la matriz de entrada)
# en coordenas homogéneas (añadimos una columna de 1)
def homog(x):
    ax = np.array(x)
    uc = np.ones(ax.shape[:-1]+(1,))
    return np.append(ax,uc,axis=-1)

# convierte en coordenadas tradicionales
def inhomog(x):
    ax = np.array(x)
    return ax[..., :-1] / ax[...,[-1]]    
    
# aplica una transformación homogénea h a un conjunto
# de puntos ordinarios, almacenados como filas 
def htrans(h,x):
    return inhomog(homog(x) @ h.T)

# juntar columnas
def jc(*args):
    return np.hstack(args)

# mide el error de una transformación (p.ej. una cámara)
# rms = root mean squared error
# "reprojection error"
def rmsreproj(view, model, transf):
    err = view - htrans(transf,model)
    return np.sqrt(np.mean(err.flatten()**2))    
    
def pose(K, image, model):
    ok,rvec,tvec = cv.solvePnP(model, image, K, (0,0,0,0))
    if not ok:
        return 1e6, None
    R,_ = cv.Rodrigues(rvec)
    M = K @ jc(R,tvec)
    rms = rmsreproj(image,model,M)
    return rms, M

def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]

# probamos todas las asociaciones de puntos imagen con modelo
# y nos quedamos con la que produzca menos error
def bestPose(K,view,model):
    poses = [ pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p[0])[0]

# Definir la forma del marcador de referencia
ref = (np.array(
   [[0,   0  ],
    [0,   1  ],
    [0.5, 1  ],
    [0.5, 0.5],
    [1,   0.5],
    [1,   0  ],
    [0.5, 0  ]]))

ref3d = addzerocol(ref)
marker = ref3d[:-1]

# Obtener la matriz de calibración
K = Kfov((640,480), 64)

# Objeto tridimensional a mostrar
cube = np.array([
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
    [0,0,0],
    
    [0,0,1],
    [1,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
        
    [1,0,1],
    [1,0,0],
    [1,1,0],
    [1,1,1],
    [0,1,1],
    [0,1,0]
    ])

# Definir la ventana para el cálculo del FOV
cv.namedWindow("RA")
   
# Flags e inicialización de los valores de las características del objeto tridimensional 
flag_color = 0
color = (255,0,0)
grosor = 1
tam = 1.

# Bucle principal de entrada de vídeo
for key, frame in autoStream():
    # Copiar el frame antes de modificarlo
    img = frame.copy()
    
    # Obtener los contornos interesantes
    g = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    conts = extractContours(g, reduprec=3)
    good = polygons(conts,6)
    
    # Si se pulsa la tecla 'c'
    if (key == ord('c')):
        # Actualizar el color en función del flag
        if flag_color == 0: 
            color = (0,255,0)
            flag_color = 1
        elif flag_color == 1:
            color = (0,0,255)
            flag_color = 2
        elif flag_color == 2:
            color = (255,0,0)
            flag_color = 0
    
    # Si se pulsa la tecla 'g' aumentar el grosor
    if (key == ord('g')):
        if grosor <= 5: grosor = grosor+1
    
    # Si se pulsa la tecla 'f' disminuir el grosor
    if (key == ord('f')):
        if grosor > 2: grosor = grosor-1
        
    # Si se pulsa la tecla 'g' aumentar el tamaño
    if (key == ord('b')):
        if tam < 1.6: tam = tam+0.1
        
    # Si se pulsa la tecla 'g' dsimunuir el tamaño
    if (key == ord('p')):
        if tam > 0.2: tam = tam-0.1
    
    # Para cada polígono seleccionado mostrar el objeto si el error es menor que 2
    for g in good:
        err,Me = bestPose(K,g,marker)
        if err < 2:
            pts = htrans(Me,cube*tam)
            cv.polylines(img,np.int32([pts]),True,color,grosor)
    
    # Mostrar la información en pantalla
    putText(img,f'Teclas para cambiar las caracteristicas de los elementos 3D:',orig=(10,20))
    putText(img,f'-\'c\' para cambiar el color.',orig=(20,40))
    putText(img,f'-\'g\' para hacer mas grueso.',orig=(20,60))
    putText(img,f'-\'f\' para hacer mas fino.',orig=(20,80))
    putText(img,f'-\'b\' para hacer mas grande.',orig=(20,100))
    putText(img,f'-\'p\' para hacer mas pequeno.',orig=(20,120))
    
    # Mostrar la imagen resultante
    cv.imshow('RA',img)

# Salir del programa 
cv.destroyAllWindows()



