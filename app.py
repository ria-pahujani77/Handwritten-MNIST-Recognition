import pygame
import sys
from pygame.locals import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

WINDOWSIZEX = 630
WINDOWSIZEY = 490

BOUNDARYINC = 5
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0) 

IMAGESAVE = False
PREDICT = True  # Set to True if you want to enable digit prediction

MODEL = load_model("handwritten_digit_cnnmodel.h5")

LABELS = {0: "Zero", 1: "One", 
          2: "Two", 3: "Three", 
          4: "Four", 5: "Five", 
          6: "Six", 7: "Seven", 
          8: "Eight", 9: "Nine"}

pygame.init()

# FONT = pygame.font.Font("freesansbold.tff", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption("Digit Board")

iswriting = False

number_xcord = []
number_ycord = []

image_cnt = 1

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0]-BOUNDARYINC, 0), min(number_xcord[-1]+BOUNDARYINC, WINDOWSIZEX)
            rect_min_y, rect_max_y = max(number_ycord[0]-BOUNDARYINC, 0), min(number_ycord[-1]+BOUNDARYINC, WINDOWSIZEY)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite(f"image_{image_cnt}.png", img_arr)
                image_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, ((10, 10),(10,10)) ,'constant', constant_values=0)

                if image.shape != (28, 28):
                     print(f"Warning: Image shape is {image.shape}, resizing to (28, 28)")
                     image = cv2.resize(image, (28, 28))

                image = image.reshape(28,28,1) / 255.0

                # Add an extra dimension to match the model's input shape
                image = np.expand_dims(image, axis=0)

                label = str(LABELS[np.argmax(MODEL.predict(image))])
                font = pygame.font.Font(None, 18)
                textSurface = font.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRecObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
