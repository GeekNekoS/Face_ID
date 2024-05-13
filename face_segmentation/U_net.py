import keras.models
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input


class ConvBlock(keras.layers.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.n_filters = n_filters
        self.conv1 = Conv2D(n_filters, 3, padding='same')
        self.batch_norm1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.conv2 = Conv2D(n_filters, 3, padding='same')
        self.batch_norm2 = BatchNormalization()
        self.act2 = Activation('relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.act2(x)
        return x

    def get_config(self):
        return {'n_filters': self.n_filters, 'conv1': self.conv1, 'batch_norm1': self.batch_norm1, 'act1': self.act1, 'conv2': self.conv2, 'batch_norm2': self.batch_norm2, 'act2': self.act2}


class EncoderBlock(keras.layers.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.n_filters = n_filters
        self.conv_block = ConvBlock(n_filters)
        self.max_pool = MaxPool2D((2, 2))

    def call(self, inputs):
        x = self.conv_block(inputs)
        p = self.max_pool(x)
        return x, p

    def get_config(self):
        return {'n_filters': self.n_filters, 'conv_block': self.conv_block, 'max_pool': self.max_pool}


class DecoderBlock(keras.layers.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.n_filters = n_filters
        self.conv_tr = Conv2DTranspose(n_filters, (2, 2), 2, 'same')
        self.concat = Concatenate()
        self.conv_block = ConvBlock(n_filters)

    def call(self, inputs, skip):
        x = self.conv_tr(inputs)
        x = self.concat([x, skip])
        x = self.conv_block(x)
        return x

    def get_config(self):
        return {'n_filters': self.n_filters, 'conv_tr': self.conv_tr, 'concat': self.concat, 'conv_block': self.conv_block}


class Unet(keras.models.Model):
    def __init__(self, input_shape, n_classes):
        super().__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.inputs = Input(input_shape)
        self.encoder_block1 = EncoderBlock(64)
        self.encoder_block2 = EncoderBlock(128)
        self.encoder_block3 = EncoderBlock(256)
        self.encoder_block4 = EncoderBlock(512)
        self.conv_block = ConvBlock(1024)
        self.decoder_block1 = DecoderBlock(512)
        self.decoder_block2 = DecoderBlock(256)
        self.decoder_block3 = DecoderBlock(128)
        self.decoder_block4 = DecoderBlock(64)
        self.outputs = Conv2D(n_classes, 1, padding='same', activation='softmax')

    def call(self, inputs):
        x = self.inputs(inputs)
        s1, p1 = self.encoder_block1(x)
        s2, p2 = self.encoder_block2(p1)
        s3, p3 = self.decoder_block3(p2)
        s4, p4 = self.encoder_block4(p3)
        b1 = self.conv_block(p4)
        d1 = self.decoder_block1([b1, s4])
        d2 = self.decoder_block2([d1, s3])
        d3 = self.decoder_block3([d2, s2])
        d4 = self.decoder_block4([d3, s1])
        output = self.outputs(d4)
        return output

    def get_config(self):
        return {'input_shape': self.input_shape, 'n_classes': self.n_classes, 'inputs': self.inputs, 'encoder_block1': self.encoder_block1, 'encoder_block2': self.encoder_block2, 'encoder_block3': self.encoder_block3, 'encoder_block4': self.encoder_block4, 'conv_block': self.conv_block, 'decoder_block1': self.decoder_block1, 'decoder_block2': self.decoder_block2, 'decoder_block3': self.decoder_block3, 'decoder_block4': self.decoder_block4, 'outputs': self.outputs}


if __name__ == '__main__':
    model = Unet((512, 512, 3), 11)
    print(model.summary())