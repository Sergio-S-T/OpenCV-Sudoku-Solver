import numpy as np


def limitaCuadrado(x):
   r=0
   if x<3: r=0
   elif x<6: r=3
   else: r=6
   return r

def comprobar(sudoku,fila,col,num):

   #Comprueba Fila
   for x in range(9):
       if sudoku[fila][x]==num:
           return False
   #comprueba columna
   for y in range(9):
       if sudoku[y][col]==num:
           return False
   #comprueba cuadrado
   a=limitaCuadrado(fila)
   b=limitaCuadrado(col)
   for x in range(3):
       for y in range(3):
           if sudoku[a+x][b+y] == num:
               return False
   return True

def resolver(M,Msol):
   for x in range(9):
       for y in range(9):
           if np.int(M[x][y]) == 0:
               for n in range (1,10):
                   if comprobar(M, x, y, n):
                       M[x][y]=n
                       resolver(M,Msol)
                       M[x][y]=0
               return
   for x in range(9):
       for y in range(9):
           Msol[x][y]=M[x][y]