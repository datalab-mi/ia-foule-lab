import cv2, os
from random import randrange
from datetime import timedelta, datetime

def random_date():
    start = datetime.strptime('1/1/2000 1:30 PM', '%m/%d/%Y %I:%M %p')
    end = datetime.strptime('1/1/2022 4:50 AM', '%m/%d/%Y %I:%M %p')

    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return (start + timedelta(seconds=random_second) ).strftime(format="%m/%d/%Y, %H:%M:%S")

def write(x, y, text, img):
    """
    x : percentage of width from bottom left corner
    y : percentage of length from bottom left corner
    """
    y = int(img.shape[0] - y * img.shape[0] / 100 )
    x =  int(x * img.shape[1] / 100 )

    scale =  img.shape[1] / 1024
    
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x,y)
    fontScale              = 1 * scale
    fontColor              = (0,0,0)
    lineType               = int(4 * scale)

    cv2.putText(img,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    fontColor              = (255,255,255)
    lineType               = int(2 * scale)

    cv2.putText(img,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    return img