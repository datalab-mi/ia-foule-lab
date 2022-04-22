import cv2, os
from random import randrange
from datetime import timedelta, datetime
from PIL import ImageFont, ImageDraw

def random_date():
    start = datetime.strptime('1/1/2000 1:30 PM', '%m/%d/%Y %I:%M %p')
    end = datetime.strptime('1/1/2022 4:50 AM', '%m/%d/%Y %I:%M %p')

    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return (start + timedelta(seconds=random_second) ).strftime(format="%d/%m/%Y %H:%M:%S")

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


def write_texts(image, text2add):
    if text2add!='':
        draw = ImageDraw.Draw(image)
        text = text2add
        x= 2
        y = 92
        x = int(x * image.size[0] / 100 )
        y =  int(y * image.size[1] / 100 )
        font_size = int(image.size[1] / 20)
        font = ImageFont.truetype('DejaVuSerif.ttf', size=font_size)
        #print(image.size,x, y, font_size, text)
        draw.text((x, y), text, fill='white', font=font,
               stroke_width=2, stroke_fill='black')

        text = random_date()
        x= 66 + (4*(image.size[0] - 1980)/512)
        y = 92
        x = int(x * image.size[0] / 100 )
        y =  int(y * image.size[1] / 100 )
        font_size = int(image.size[1] / 20)
        font = ImageFont.truetype('DejaVuSerif.ttf', size=font_size)
        #print(image.size,x, y, font_size, text)
        draw.text((x, y), text, fill='white', font=font,
               stroke_width=2, stroke_fill='black')
    return image