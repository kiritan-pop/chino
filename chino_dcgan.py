# coding: utf-8
import tensorflow as tf
from keras.models import Model, load_model
from keras.optimizers import Adam,Adadelta,Nadam
from keras.utils import plot_model, multi_gpu_model
from keras.callbacks import EarlyStopping, LambdaCallback
from keras import backend

import os,glob,sys,json,random,cv2,threading,queue,multiprocessing
from time import sleep
import numpy as np
import math
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter
import argparse
from chino_model import build_generator, build_discriminator, build_combined, build_frozen_discriminator
from chino_datagenerator import D_Datagenerator, Comb_Datagenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True

total_epochs = 999999

img_dir = './images_face/'
g_chino_path = 'g_chino.h5'
d_chino_path = 'd_chino.h5'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default='1')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--queue_size", type=int, default=10)
    args = parser.parse_args()
    return args

def dataQ(generator, queue_size=5, MP=False):
    def datagen(generator, que):
        i=0
        lim = generator.__len__()
        while True:
            x,y = generator.__getitem__(i%lim)
            que.put((x,y))
            i+=1
    if MP:
        que  = multiprocessing.Queue(queue_size)
        multiprocessing.Process(target=datagen, args=(generator, que)).start()
    else:
        que  = queue.Queue(queue_size)
        threading.Thread(target=datagen, args=(generator, que)).start()

    return que

def gan_s1(GPUs, start_idx, batch_size):
    def g_on_epoch_end(epoch, path):
        g_chino.save(g_chino_path)
        g_chino.save(path + g_chino_path)
        generator_test(g_chino, epoch, path)

    def g_on_epoch_end_sub(epoch):
        generator_test(g_chino, epoch, None, True)

    def d_on_epoch_end(epoch, path):
        d_chino.save(d_chino_path)
        d_chino.save(path + d_chino_path)
        discrimin_test(d_chino, epoch, path, q_valid_d)

    #######################################################
    # STAGE-1
    if os.path.exists(g_chino_path):
        g_chino = load_model(g_chino_path)
    else:
        g_chino = build_generator()

    if GPUs > 1:
        g_chino_tr = multi_gpu_model(g_chino, gpus=GPUs)
    else:
        g_chino_tr = g_chino

    def summary_write(line):
        f.write(line+"\n")
    with open('g_chino.txt', 'w') as f:
        g_chino.summary(print_fn=summary_write)
    plot_model(g_chino, to_file='g_model.png')

    if os.path.exists(d_chino_path):
        d_chino = load_model(d_chino_path)
    else:
        d_chino = build_discriminator()

    if GPUs > 1:
        d_chino_tr = multi_gpu_model(d_chino, gpus=GPUs)
    else:
        d_chino_tr = d_chino
    d_chino_tr.compile(loss='categorical_crossentropy',
                    optimizer=Nadam(),
                    )
    with open('d_chino.txt', 'w') as f:
        d_chino.summary(print_fn=summary_write)

    frozen_d = build_frozen_discriminator(d_chino)

    _tmpmdl = build_combined(g_chino, frozen_d)
    if GPUs > 1:
        combined = multi_gpu_model(_tmpmdl, gpus=GPUs)
    else:
        combined = _tmpmdl

    combined.compile(loss=['categorical_crossentropy'],
                    optimizer=Nadam(),
                    )

    #
    g_chino_tr._make_predict_function()
    d_chino_tr._make_predict_function()
    d_chino_tr._make_train_function()
    combined._make_predict_function()
    combined._make_train_function()

    # discriminator用のデータジェネレータ、データを格納するキュー
    ddgens = []
    q1s = []
    for i in range(2):
        ddgen = D_Datagenerator(images_path=img_dir, batch_size=batch_size, g_model=g_chino_tr ,val=i)
        q1 = dataQ(ddgen, args.queue_size)
        ddgens.append(ddgen)
        q1s.append(q1)
        sleep(args.queue_size/4)

    ddgen_valid = D_Datagenerator(images_path=img_dir, batch_size=batch_size, g_model=g_chino_tr, val=999)
    q_valid_d = dataQ(ddgen_valid, args.queue_size)
    sleep(args.queue_size/4)

    # 結合モデル（combined）用のデータジェネレータ、データを格納するキュー
    cdgen = Comb_Datagenerator(batch_size=batch_size)
    q2 = dataQ(cdgen, args.queue_size, MP=True)
    sleep(args.queue_size/4)

    pre_d_loss = 1000.0
    pre_g_loss = 1000.0
    for epoch in range(start_idx+1, total_epochs+1):
        print(f'\repochs={epoch:6d}/{total_epochs:6d}:', end='')
        if pre_d_loss * 15 >= pre_g_loss or epoch < 10 or epoch%10 == 0:
            d_losses = []
            for i in range(2):
                x,y = q1s[i].get()
                d_loss = d_chino_tr.train_on_batch(x, y)
                d_losses.append(d_loss)

            pre_d_loss = sum(d_losses)/3
        print(f'D_loss={pre_d_loss:.3f}  ',end='')

        # if pre_d_loss <= pre_g_loss or epoch < 10:
        # 色別にバッチを分けて実施
        x,y = q2.get()
        g_loss = combined.train_on_batch(x, y)
        pre_g_loss = g_loss
        print(f'G_loss={g_loss:.3f}',end='')

        # 100epoch毎にテスト画像生成、validuetion
        if epoch%100 == 0:
            g_on_epoch_end_sub(epoch)
            # validuate
            print(f'\tValidation :', end='')
            x,y = q_valid_d.get()
            ret = d_chino_tr.test_on_batch(x, y)
            print(f'D_loss={ret:.3f},',end='')

            x,y = q2.get()
            ret = combined.test_on_batch(x, y)
            print(f'G_loss={ret:.3f}')

        # 1000epoch毎にモデルの保存、テスト実施
        if epoch%1000 == 0:
            result_path = 'results/'
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            result_path += f'{epoch:05d}/'
            if not os.path.exists(result_path):
                os.mkdir(result_path)

            g_on_epoch_end(epoch, result_path)
            d_on_epoch_end(epoch, result_path)


def generator_test(g_model, epoch, result_path, short=False):
    # noise = np.random.uniform(-1,1,(12,32,32))
    noise = np.random.normal(0.0,1.0,(12,64))
    rets = g_model.predict_on_batch(noise)

    for i, ret in enumerate(rets):
        if short:
            save_path = f'temp2/{epoch:06d}/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        else:
            save_path = result_path + f'g1_{epoch:06}/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        filename = os.path.join(save_path, f"{i:2d}.png")
        tmp = (ret*127.5+127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(tmp).save(filename, optimize=True)


def discrimin_test(d_model, epoch, result_path, queue, stage=1):
    #判定
    imgs, _ = queue.get()
    results = d_model.predict_on_batch(imgs)
    #確認用保存
    save_path = result_path + f'd{stage}_{epoch:06}/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i, result in enumerate(results):
        filename = f'd{i:02}[{result[0]:1.2f}][{result[1]:1.2f}].png'
        tmp = (imgs[i]*127.5+127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(tmp).save(save_path + filename, 'png', optimize=True)


if __name__ == "__main__":
    #パラメータ取得
    args = get_args()
    #GPU設定
    config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=False, visible_device_list=args.gpu),
                allow_soft_placement=True, 
                log_device_placement=False
                )
    session = tf.Session(config=config)
    backend.set_session(session)
    GPUs = len(args.gpu.split(','))
    gan_s1(GPUs=GPUs, start_idx=args.idx, batch_size=args.batch_size)
