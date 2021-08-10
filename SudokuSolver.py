import cv2
import numpy as np

from FuncionesDeteccion import *
from Solver import resolver

ancho=640
alto=480
M=np.zeros((9,9))
Msol=np.zeros((9,9))

imagenPath="./resources/example.jpg"


img = cv2.imread(imagenPath)
img = cv2.resize(img, (ancho, alto))
img2=img.copy()

imgTresh = imgProcesado(img)

vertices,res = getContours(imgTresh,img)

if res:
    vertices = vertices.reshape(4,2)
    num=1
    imgWarp=warpImagen(img,vertices)

    leeSudoku(imgWarp,M)

    resolver(M,Msol)

    if haySolucion(Msol):

        pintaResultado(imgWarp,Msol,M)
        rayas(imgWarp)
        imgWI=warpImagenInv(imgWarp,vertices)

        combinaImg(img,imgWI)

        linea=np.zeros((alto,8,3),dtype = "uint8")

        cv2.putText(img2, "Original", (round(ancho/10),alto-15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), thickness=2)
        cv2.putText(imgWI, "Solution", (round(ancho/10),alto-15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), thickness=2)

        hor=np.hstack((img2,linea,imgWI))
        cv2.imwrite("./resources/solved.jpg",hor)
        cv2.imshow("Solucion", hor)
        cv2.waitKey(0)
    else:
        print(M)
        pintaLectura(imgWarp,M)
        rayas(imgWarp)
        imgWI=warpImagenInv(imgWarp,vertices)

        combinaImg(img,imgWI)

        cv2.putText(imgWI, "Error during reading", (5, alto - 15), cv2.FONT_HERSHEY_DUPLEX,0.8, (0, 0, 255), thickness=2)
        cv2.imshow("Foto", imgWI)
        cv2.waitKey(0)
else:
    cv2.putText(img2, "No sudoku detected", (round(ancho/12),alto-15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), thickness=2)
    cv2.imshow("Foto", img2)
    cv2.waitKey(0)
