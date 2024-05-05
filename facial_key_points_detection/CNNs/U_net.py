from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model


def conv_block(inputs, n_filters):
    x = Conv2D(n_filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def encoder_block(inputs, n_filters):
    x = conv_block(inputs, n_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(inputs, skip, n_filters):
    x = Conv2DTranspose(n_filters, (2, 2), strides=2, padding='same')(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, n_filters)
    return x


def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    b1 = conv_block(p4, 1024)
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    outputs = Conv2D(n_classes, 1, padding='same', activation='softmax')(d4)
    model = Model(inputs, outputs)
    return model


if __name__ == '__main__':
    model = build_unet((512, 512, 3), 4)
    print(model.summary())