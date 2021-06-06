import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json
import random

img_path = "/home/saeed/Pictures"
caption_path = "/home/saeed/software/python/thirdsite/annotations/val.json"

def read_img_caption(img_path=img_path,caption_path=caption_path):

    paths = os.listdir(img_path)

    img_length,img_width = 500,500
    img_number = len(paths)


    #reading images from directory with id + image
    imgs_id = []
    for path,i in zip(paths,range(img_number)):

        img = cv2.imread(img_path + "/" + path,0)
        img = cv2.resize(img,(img_length,img_width))
        imgs_id.append({'id':path[:-4],'img':img})


    #reading captions , id + caption
    with open(caption_path) as file:
        caption_json = json.load(file)['annotations']

    captions_id = []
    for i in range(len(caption_json)):
        captions_id.append({'id':caption_json[i]['image_id'],'caption':caption_json[i]['caption']})

    random.shuffle(captions_id)

    #just image and caption without id but with sorting
    captions = []
    imgs = []
    for i in range(len(captions_id)):
        id = captions_id[i]['id']
        captions = captions_id[i]['caption']
        img = cv2.imread(img_path + "/" + str(id) + '.png', 0)
        img = cv2.resize(img, (img_length, img_width))
        imgs.append(img)


    return imgs,captions


def read_img_caption_example(img_path=img_path,caption_path=caption_path):

    img_length, img_width = 500, 500

    # reading captions , id + caption
    with open(caption_path) as file:
        caption_json = json.load(file)['annotations']

    captions_id = []
    for i in range(len(caption_json)):
        captions_id.append({'id': caption_json[i]['image_id'], 'caption': caption_json[i]['caption']})

    random.shuffle(captions_id)

    # read image and caption
    id = captions_id[1]['id']
    caption = captions_id[1]['caption']
    img = cv2.imread(img_path + "/" + str(id) + '.png', 0)
    # img = cv2.imread(img_path + "/" + "old.png",0)
    img = cv2.resize(img, (img_length, img_width))

    #showing image and caption
    print("caption : " +caption)
    plt.imshow(img)

    return 0

