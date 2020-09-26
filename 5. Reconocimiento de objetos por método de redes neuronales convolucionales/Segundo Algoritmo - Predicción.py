import numpy as np
import cv2
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
longitud, altura=100,100
modelo="D:/modelo/modelo.h5"
pesos="D:/modelo/pesos.h5"
cnn=load_model(modelo)
cnn.load_weights(pesos)

def predict(file):
    x=load_img(file, target_size=(longitud,altura))
    x=img_to_array(x)
    x=np.expand_dims(x,axis=0)
    arreglo=cnn.predict(x)
    resultado=arreglo[0]
    respuesta=np.argmax(resultado)
    if respuesta==0:
        print("boxeo")
    elif respuesta==1:
        print("formula uno")
    elif respuesta==2:
        print("futbol")
    return respuesta
#predict("D:/images.jpg")
def capturar_imagen():
    cap = cv2.VideoCapture("https://192.168.137.54:8080/video")
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLORMAP_JET)
    cv2.imwrite('D:/imagen_9.jpg',gray)
    cap.release()
capturar_imagen()
predict("D:/imagen_9.jpg")
