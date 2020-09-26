import cv2
import numpy as np
import urllib
import urllib.request
import urlopen

url = 'http://10.128.26.232:8080/photo.jpg'
cap = cv2.VideoCapture(1)
while (1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    imgResponse = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    # ---------------------------------------------
    mask = cv2.inRange(img, lower_red, upper_red)
    res = cv2.bitwise_and(img, img, mask=mask)
    # ---------------------------------------------
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    edges = cv2.Canny(img, 100, 200)
    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('sobelx', sobelx)
    cv2.imshow('sobely', sobely)
    cv2.imshow('Edges', edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
