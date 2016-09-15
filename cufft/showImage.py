import numpy as np
import cv2

f = open('lenar.f', 'r')
i = np.loadtxt(f)
cv2.imwrite('h.jpg',i)
