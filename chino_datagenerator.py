# coding: utf-8

import os,glob,sys,json,random,cv2,threading,pprint,queue
from time import sleep
import numpy as np
import math
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter, ImageEnhance
import tensorflow as tf
graph = tf.get_default_graph()

STANDARD_SIZE_S1 = (128, 128)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def new_convert(img, mode):
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
    elif img.mode == "LA":
        bg = Image.new("L", img.size, (255,))
        bg.paste(img, mask=img.split()[1])
    else:
        bg = img
    return bg.convert(mode)

def image_arrange(path, resize=(128,128)):
    img = Image.open(path)
    img = new_convert(img, 'RGB')
    img = img.resize(resize, Image.BICUBIC)
    return img


class D_Datagenerator():
    def __init__(self, images_path, batch_size, g_model, val=0):
        # コンストラクタ
        self.images_path = images_path
        self.batch_size = batch_size
        self.images = []
        for f in os.listdir(images_path):
            self.images.append(os.path.join(images_path, f))
        random.shuffle(self.images)
        self.val = val
        self.valid_images = random.sample(self.images, len(self.images)//10)
        for d in self.valid_images:
            self.images.remove(d)
        self.g_model = g_model
        self.old_fake = {}

        # エラー回避のため、１回空振りさせる
        noise = np.random.normal(0.0,1.0,(1,64))
        # noise = np.random.uniform(-1,1,(self.batch_size,32,32))
        with graph.as_default():
            g_model.predict_on_batch(noise)
        sleep(2)

    def __getitem__(self, idx):
        # データの取得実装
        if self.val == 999:
            ytmp1 = random.sample(self.valid_images, self.batch_size//2) 
        else:
            ytmp1 = random.sample(self.images, self.batch_size//2) 
            # ytmp1 = self.images[self.batch_size*idx//2:self.batch_size*(idx+1)//2]

        x1 = np.zeros((self.batch_size//2, 128, 128, 3))
        y1 = np.zeros((self.batch_size//2, 2))
        x2 = np.zeros((self.batch_size//2, 128, 128, 3))
        y2 = np.zeros((self.batch_size//2, 2))

        if self.val == 0 or self.val == 999:
            for p, path in enumerate(ytmp1):
                #本物
                img = image_arrange(path, resize=STANDARD_SIZE_S1)
                img = (np.asarray(img)-127.5)/127.5
                x1[p] = img
                r = random.uniform(0.0, 0.2)
                y1[p] = [1.0 - r, 0.0 + r]

            x = x1
            y = y1

        elif self.val == 1 or self.val == 999:
            #偽物
            # noise = np.random.uniform(-1,1,(self.batch_size//2,32,32))
            noise = np.random.normal(0.0,1.0,(self.batch_size//2,64))
            with graph.as_default():
                x2 = self.g_model.predict_on_batch(noise)
            for i,_ in enumerate(x2):
                r = random.uniform(0.0, 0.2)
                y2[i] = [0.0 + r, 1.0 - r]

            if random.randint(0,10) == 0:
                filename = os.path.join("temp/", f"test.png")
                tmp = (x2[0]*127.5+127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(tmp).save(filename, optimize=True)

            x = x2
            y = y2

        if self.val == 999:
            x = np.concatenate([x1,x2], axis=0)
            y = np.concatenate([y1,y2], axis=0)

        return x, y

    def __len__(self):
        # 全データ数をバッチサイズで割って、何バッチになるか返すよー！
        sample_per_epoch = math.ceil(len(self.images)/self.batch_size)
        return sample_per_epoch

class Comb_Datagenerator():
    def __init__(self, batch_size, val=0):
        # コンストラクタ
        self.batch_size = batch_size
        self.val = val

    def __getitem__(self, idx):
        # データの取得実装
        # x = np.random.uniform(-1,1,(self.batch_size,32,32))
        x = np.random.normal(0.0,1.0,(self.batch_size,64))
        y = np.zeros((self.batch_size, 2))
        for i in range(self.batch_size):
            r = random.uniform(0.0, 0.2)
            y[i] = [1.0 - r, 0.0 + r]

        return x, y

    def __len__(self):
        return 999999

