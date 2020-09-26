# Importamos las debidas librerias 
import cv2
import numpy as np

import time
import urllib.request


def find_marker(image):
    # convertimos la imagen a escala de grises y difuminamos, para detectar bordes
    blurred_frame = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    # definimos el rango del color que queremos que detecte
    lower_blue = np.array([38, 86, 0])
    upper_blue = np.array([121, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    con = max(contours, key=cv2.contourArea)

    return cv2.minAreaRect(con)


def distance_to_camera(anchoConocida, longitudFocal, anchoPorConocer):
    # calculamos y retornamos la distancia de la camara al objeto 
    return (anchoConocida * longitudFocal) / anchoPorConocer


# ponemos el objeto a 30 cm de la camara para calibrar 
DistanciaConocida = 30.0

# el ancho del objeto en cm 
anchoConocido = 3.0
# recordar que estamos usando el IPWebcam para la trasmisión de video 

url = 'http://192.168.1.7:8080/shot.jpg'
print("<<<---- taking calibrating Image ---->>>")
time.sleep(2)

imgResp = urllib.request.urlopen(url)
imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
c_image = cv2.imdecode(imgNp, -1)

marker = find_marker(c_image)
longitudFocal = (marker[1][0] * DistanciaConocida) / anchoConocido

time.sleep(2)

print("<<<---- Main program Staring ---->>>")

while True:

    imgResp = urllib.request.urlopen(url)
    image = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marker = find_marker(image)
    CM = distance_to_camera(anchoConocido, longitudFocal, marker[1][0])

    # escribimos la salida, en nuestro caso la distancia 
    cv2.putText(image, "%.2fcm" % CM,
                (image.shape[1] - 350, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (255, 0, 0), 3)
    cv2.imshow("image", image)
    key = cv2.waitKey(1)
    # Presioname ESC para parar la ejecución de nuestro programa 
    if key == 27:
        break

cv2.destroyAllWindows()
