# coding: utf-8
from keras.models import Sequential,Model,load_model
from keras.layers import Input, Dense, Dropout, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D,\
                        Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Reshape, Concatenate, Average, Reshape,\
                        GaussianNoise, LeakyReLU, BatchNormalization, Embedding

def build_discriminator():
    en_alpha=0.3
    stddev=0.1 #0.2でいいかな？

    input_image = Input(shape=(128, 128, 3), name="d_s1_input_main")

    model = GaussianNoise(stddev)(input_image)
    model = Conv2D(filters=32,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    model = Conv2D(filters=64,  kernel_size=4, strides=2, padding='same')(model) # >64
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=64,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    model = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(model) # >32
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=128,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    model = Conv2D(filters=256,  kernel_size=4, strides=2, padding='same')(model) # >16
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=256,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = GlobalAveragePooling2D()(model)
    model = Dense(2)(model)
    truefake = Activation('softmax', name="d_s1_out1_trfk")(model)
    return Model(inputs=[input_image], outputs=[truefake])

def build_generator():
    en_alpha=0.3
    dec_alpha=0.1

    input_tensor = Input(shape=(64,), name="g_s1_input_main")
    model = Dense(32*32)(input_tensor)
    model = LeakyReLU(alpha=en_alpha)(model)
    model = Reshape(target_shape=(32, 32, 1))(model)

    model = Conv2D(filters=128,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=256, kernel_size=4, strides=2, padding='same')(model) # 32-> 16
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=512, kernel_size=4, strides=2, padding='same')(model) # 16-> 8
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same')(model) #8->16
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=dec_alpha)(model)

    model = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(model)  #16->32
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=dec_alpha)(model)

    model = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same')(model)  #32->64
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=dec_alpha)(model)

    model = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(model)  #64->128
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=dec_alpha)(model)

    model = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=3  , kernel_size=3, strides=1, padding='same')(model)

    return Model(inputs=[input_tensor], outputs=[model])


def build_combined(generator, discriminator):
    return Model(inputs=[generator.inputs[0]], outputs=[discriminator(generator.outputs[0]) ] )


def build_frozen_discriminator(discriminator):
    frozen_d = Model(inputs=discriminator.inputs, outputs=discriminator.outputs)
    frozen_d.trainable = False
    return frozen_d