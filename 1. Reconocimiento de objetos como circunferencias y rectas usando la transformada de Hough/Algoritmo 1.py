# import the opencv library
import cv2
import numpy as np
import urllib.request


# define a video capture object
def lines():
    url = 'http://192.168.137.61:8080/shot.jpg'
    while True:
        with urllib.request.urlopen(url) as response:
            entrada = response.read()

        imgNp = np.array(bytearray(entrada), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        cv2.imwrite('opencv.png', img)
        # time.sleep(1)
        img = cv2.imread('opencv.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, 3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is None:
            lines = []
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def circlesDetect():
    url = 'http://192.168.137.61:8080/shot.jpg'
    while True:
        with urllib.request.urlopen(url) as response:
            entrada = response.read()
        imgNp = np.array(bytearray(entrada), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        cv2.imwrite('opencv.png', img)
        img = cv2.imread('opencv.png')
        src = cv2.medianBlur(img, 5)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=20, maxRadius=100)
        if circles is None:
            circles = [[]]
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # dibujar circulo
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # dibujar centro
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.imshow('detected circles', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


# circlesDetect()
lines()

