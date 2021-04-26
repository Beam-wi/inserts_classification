import numpy as np
from PIL import Image
import cv2
import numpy as np


def letterbox_image_pil(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    shift = [(w-nw)//2, (h-nh)//2]
    return new_image, scale, shift


def letterbox_image_cv(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.shape[1], image.shape[0]
    h, w = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC)

    new_image = np.zeros((h, w, 3), np.uint8) + 128
    new_image[(h-nh)//2:(h-nh)//2+nh, (w-nw)//2:(w-nw)//2+nw, :] = image
    shift = [(w-nw)//2, (h-nh)//2]
    return new_image, scale, shift