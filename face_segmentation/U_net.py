import keras.models
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input


class ConvBlock(keras.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.units = n_filters
        self.conv1 = Conv2D(n_filters, 3, padding='same')
        self.batch_norm1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.conv2 = Conv2D(n_filters, 3, padding='same')
        self.batch_norm2 = BatchNormalization()
        self.act2 = Activation('relu')

    def compute_output_shape(self, input_shape):
        output_shape = self.conv1.compute_output_shape(input_shape)
        output_shape = self.batch_norm1.compute_output_shape(output_shape)
        output_shape = self.act1.compute_output_shape(output_shape)
        output_shape = self.conv2.compute_output_shape(output_shape)
        output_shape = self.batch_norm2.compute_output_shape(output_shape)
        output_shape = self.act2.compute_output_shape(output_shape)
        return output_shape

    def build(self, input_shape):
        self.conv1.build(input_shape)
        output_shape = self.conv1.compute_output_shape(input_shape)
        self.batch_norm1.build(output_shape)
        output_shape = self.batch_norm1.compute_output_shape(output_shape)
        self.act1.build(output_shape)
        output_shape = self.act1.compute_output_shape(output_shape)
        self.conv2.build(output_shape)
        output_shape = self.conv2.compute_output_shape(output_shape)
        self.batch_norm2.build(output_shape)
        output_shape = self.batch_norm2.compute_output_shape(output_shape)
        self.act2.build(output_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.act2(x)
        return x


class EncoderBlock(keras.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.units = n_filters
        self.conv_block = ConvBlock(n_filters)
        self.max_pool = MaxPool2D((2, 2))

    def compute_output_shape(self, input_shape):
        output_shape1 = self.conv_block.compute_output_shape(input_shape)
        output_shape2 = self.max_pool.compute_output_shape(output_shape1)
        return output_shape1, output_shape2

    def build(self, input_shape):
        self.conv_block.build(input_shape)
        output_shape = self.conv_block.compute_output_shape(input_shape)
        self.max_pool.build(output_shape)

    def call(self, inputs):
        x = self.conv_block(inputs)
        p = self.max_pool(x)
        return x, p


class DecoderBlock(keras.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.units = n_filters
        self.conv_tr = Conv2DTranspose(n_filters, (2, 2), 2, 'same')
        self.concat = Concatenate()
        self.conv_block = ConvBlock(n_filters)

    def compute_output_shape(self, input_shapes):
        output_shape = self.conv_tr.compute_output_shape(input_shapes[0])
        output_shape = self.concat.compute_output_shape((output_shape, input_shapes[1]))
        output_shape = self.conv_block.compute_output_shape(output_shape)
        return output_shape

    def build(self, input_shapes):
        self.conv_tr.build(input_shapes[0])
        output_shape = self.conv_tr.compute_output_shape(input_shapes[0])
        self.concat.build((output_shape, input_shapes[1]))
        output_shape = self.concat.compute_output_shape((output_shape, input_shapes[1]))
        self.conv_block.build(output_shape)

    def call(self, inputs):
        x = self.conv_tr(inputs[0])
        x = self.concat([x, inputs[1]])
        x = self.conv_block(x)
        return x


class Unet(keras.Model):
    def __init__(self, input_shape, n_classes, **kwargs):
        super(Unet, self).__init__(**kwargs)
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
        self.out = self.call(self.inputs)
        super(Unet, self).__init__(inputs=self.inputs, outputs=self.out, **kwargs)

    def call(self, inputs):
        s1, p1 = self.encoder_block1(inputs)
        s2, p2 = self.encoder_block2(p1)
        s3, p3 = self.encoder_block3(p2)
        s4, p4 = self.encoder_block4(p3)
        b1 = self.conv_block(p4)
        d1 = self.decoder_block1([b1, s4])
        d2 = self.decoder_block2([d1, s3])
        d3 = self.decoder_block3([d2, s2])
        d4 = self.decoder_block4([d3, s1])
        output = self.outputs(d4)
        return output


if __name__ == '__main__':
    model = Unet((512, 512, 3), 11)
    print(model.summary())