import cv2
import numpy as np
from keras.models import load_model

print("Loading model...")
modelo = load_model("./resources/digit_model.h5")
print("Model loaded")

ancho = 640
alto = 480


def imgProcesado(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgTresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return imgTresh

def getContours(img, img2):
    areaMax = 0
    cntMax = 0
    res = False
    vertices=[]
    contours, hyerarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>40000:
            p = cv2.arcLength(cnt, True)
            lados = cv2.approxPolyDP(cnt, 0.01 * p, True)
            if area > areaMax and len(lados) == 4:
                cntMax = cnt
                areaMax = area
                res=True
    if res:
        cv2.drawContours(img2, cntMax, -1, (0, 0, 255), thickness=4)
        vertices = cv2.approxPolyDP(cntMax, cv2.arcLength(cntMax, True) * 0.05, True)
    return vertices,res

def warpImagen(img, vertices):
    ptsWarp = orientarWarp(vertices)
    ptsObj = np.float32([[0, 0], [0, alto], [ancho, alto], [ancho, 0]])
    perspectiva = cv2.getPerspectiveTransform(ptsWarp, ptsObj)
    imgWarp = cv2.warpPerspective(img, perspectiva, (ancho, alto))
    return imgWarp

def orientarWarp(pts):
    r1=np.zeros(4)
    r2=np.zeros(4)
    obj=np.zeros((4,2))
    y0, y1 = alto, alto
    x0, x1 = 0, 0
    for x in range(4):
        r1[x]=np.sqrt(np.power(pts[x][0],2)+np.power(pts[x][1],2))
        r2[x]=np.sqrt(np.power(pts[x][0],2)+np.power(pts[x][1]-alto,2))
    obj[0] = pts[np.argmin(r1)]
    obj[1] = pts[np.argmin(r2)]
    obj[2] = pts[np.argmax(r1)]
    obj[3] = pts[np.argmax(r2)]
    obj = np.float32(obj)
    return obj

def leeSudoku(imgW, M):
    imgW = cv2.cvtColor(imgW, cv2.COLOR_BGR2GRAY)
    dx = round(imgW.shape[0] / 9)
    dy = round(imgW.shape[1] / 9)
    cx = round(dx / 7)
    cy = round(dy / 7)
    for x in range(9):
        for y in range(9):
            casilla = imgW[dx * x + cx:dx * (x + 1) - cx, dy * y + cy:dy * (y + 1) - cy]
            trs, casilla = cv2.threshold(casilla, 130, 255, cv2.THRESH_BINARY_INV)
            if buscaNumero(casilla):
                añadirNumero(x,y,M,casilla)

def buscaNumero(casilla):
    blanco = 0
    dx = round(casilla.shape[0] / 5)
    dy = round(casilla.shape[1] / 4)
    recorte=casilla[dx:casilla.shape[0]-dx][dy:casilla.shape[1]-dy]
    for x in range(recorte.shape[0]):
        for y in range(recorte.shape[1]):
            if recorte[x][y] > 250: blanco += 1
    if blanco < 20: return False
    return True

def añadirNumero(x,y,M,casilla):
    umbral=0.85
    entrada=casilla.copy()
    dx = round(alto/ 9)
    dy = round(ancho / 9)
    cx=round(dx/6)
    cy=round(dy/6)
    result=modelo.predict(preparaEntrada(entrada))
    if np.argmax(result, axis=-1) < umbral:
        entrada=casilla[cx:dx-cx][cy:dy-cy]
        result = modelo.predict(preparaEntrada(entrada))
    M[x][y]=np.argmax(result, axis=-1)

def preparaEntrada(num):
    num = cv2.resize(num, (28, 28))
    num = num / 255
    num = num.reshape(1, 28, 28, 1)
    return num

def pintaResultado(imgWarp,M,Mmask):
    dx = round(ancho / 9)
    dy = round(alto / 9)
    cx=round(dx/3)
    cy=round(dy/5)
    for x in range(9):
        for y in range(1,10):
            if Mmask[y-1][x] == 0:
                cv2.putText(imgWarp, str(int(M[y-1][x])), ((dx*x)+cx,(dy*y)-cy), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 0, 0), thickness=2)

def pintaLectura(imgWarp,M):
    dx = round(ancho / 9)
    dy = round(alto / 9)
    cx=round(dx/3)
    cy=round(dy/5)
    for x in range(9):
        for y in range(1,10):
            if M[y-1][x] != 0:
                cv2.putText(imgWarp, str(int(M[y-1][x])), ((dx*x),(dy*y)-cy), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), thickness=2)

def warpImagenInv(img, vertices):
    ptsWarp = orientarWarp(vertices)
    ptsObj = np.float32([[0, 0], [0, alto], [ancho, alto], [ancho, 0]])
    perspectiva = cv2.getPerspectiveTransform( ptsObj, ptsWarp)
    imgWarp = cv2.warpPerspective(img, perspectiva, (ancho, alto))
    return imgWarp

def combinaImg(img,imgWI):
    for x in range(alto):
        for y in range(ancho):
            if np.all(imgWI[x][y]==0):
                imgWI[x][y]=img[x][y]

def haySolucion(M):
    for x in range(9):
        for y in range(9):
            if M[x][y] == 0:return False
    return True

def rayas(img):
    dy=round(img.shape[0]/9)
    dx=round(img.shape[1]/9)
    for x in range(9):
        cv2.line(img,(dx*x,0),(dx*x,alto),(0, 0, 255),thickness=2)
        cv2.line(img,(0,dy*x),(ancho,dy*x),(0, 0, 255),thickness=2)

