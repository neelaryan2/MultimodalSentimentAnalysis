import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Dropout, Layer, RNN
from tensorflow.keras.layers import Lambda, Multiply, TimeDistributed, LSTMCell
from tensorflow.keras.models import Model, Sequential

def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(dropped_inputs, ones, training=training) for _ in range(count)]
    return K.in_train_phase(dropped_inputs, ones, training=training)


class Mod_LSTMCELL(LSTMCell):
    def call(self, inputs, states, training=None):
        self._dropout_mask = _generate_dropout_mask(K.ones_like(inputs), 0, training=training, count=4)
        self._recurrent_dropout_mask = _generate_dropout_mask(K.ones_like(states[0]), 0, training=training, count=4)
        
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(K.ones_like(inputs), self.dropout, training=training, count=4)
        if (0 < self.recurrent_dropout < 1 and self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(K.ones_like(states[0]), self.recurrent_dropout, training=training, count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            x_i = K.dot(inputs_i, self.kernel_i)
            x_f = K.dot(inputs_f, self.kernel_f)
            x_c = K.dot(inputs_c, self.kernel_c)
            x_o = K.dot(inputs_o, self.kernel_o)
            if self.use_bias:
                x_i = K.bias_add(x_i, self.bias_i)
                x_f = K.bias_add(x_f, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)
                x_o = K.bias_add(x_o, self.bias_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            i = self.recurrent_activation(x_i + K.dot(h_tm1_i, self.recurrent_kernel_i))
            f = self.recurrent_activation(x_f + K.dot(h_tm1_f, self.recurrent_kernel_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c, self.recurrent_kernel_c))
            o = self.recurrent_activation(x_o + K.dot(h_tm1_o, self.recurrent_kernel_o))
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units:2 * self.units]
            z2 = z[:, 2 * self.units:3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)

        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return tf.concat([h ,c_tm1, c], axis=-1), [h, c]
    
def customLSTM(dim):
    regularizer = tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-3)
    cell = Mod_LSTMCELL(dim, dropout=0.5, kernel_regularizer=regularizer, recurrent_regularizer=regularizer)
    lstm_cell = RNN(cell, return_sequences=True)
    return lstm_cell


class MultiviewGatedMemory(Layer):
    def __init__(self, mem_dim, **kwargs):
        self.state_size = mem_dim
        super(MultiviewGatedMemory, self).__init__(**kwargs)

    def build(self, input_shape):
        self.chat = Sequential([Dense(128, activation='relu'),
                                Dropout(rate=0.5),
                                Dense(self.state_size, activation='tanh'),])
        self.gamma1 = Sequential([Dense(128, activation='relu'),
                                  Dropout(rate=0.5),
                                  Dense(self.state_size, activation='sigmoid'),])
        self.gamma2 = Sequential([Dense(128, activation='relu'),
                                  Dropout(rate=0.5),
                                  Dense(self.state_size, activation='sigmoid'),])

    def call(self, step_in, states):
        prev_mem = states[0]
        both = K.concatenate([step_in, prev_mem], axis=-1)
        g1 = self.gamma1(both)
        g2 = self.gamma2(both)
        cHat = self.chat(step_in)
        step_out = g1 * prev_mem + g2 * cHat
        return step_out, [step_out]
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'state_size': self.state_size,
        })
        return config


def MFN_unimodal(input_shapes, output_classes, mem_size=256):
    tot_dim = 0

    maxlen, dim0 = input_shapes[0]
    tot_dim += dim0
    input0 = Input(shape=(maxlen, dim0), dtype='float64')
    hc0 = customLSTM(dim0)(input0)
    h = Lambda(lambda x: x[:,:,:dim0])(hc0)
    cStar = Lambda(lambda x: x[:,:,dim0:])(hc0)

    attention = Sequential([
        TimeDistributed(Dense(256 ,activation='relu')),
        Dropout(rate=0.5),
        TimeDistributed(Dense(2*tot_dim ,activation='softmax'))
    ])(cStar)

    attended = Multiply()([cStar, attention])
    cell = MultiviewGatedMemory(mem_size)
    lstm_cell = RNN(cell)
    mem = lstm_cell(attended) # equation 9-11

    last_hs = Lambda(lambda x: x[:,-1,:])(h)

    final = Concatenate(axis=-1)([last_hs, mem]) # Output of MFN

    output = Sequential([
        Dense(128 ,activation='relu'),
        Dropout(rate=0.5),
        Dense(output_classes, activation='softmax')
    ], name='final_output')(final)

    model = Model(inputs=[input0], outputs=output)
    return model

def MFN_bimodal(input_shapes, output_classes, mem_size=256):
    tot_dim = 0

    maxlen, dim0 = input_shapes[0]
    tot_dim += dim0
    input0 = Input(shape=(maxlen, dim0), dtype='float64')
    hc0 = customLSTM(dim0)(input0)
    h0 = Lambda(lambda x: x[:,:,:dim0])(hc0)
    c_prev0 = Lambda(lambda x: x[:,:,dim0:2*dim0])(hc0)
    c_new0 = Lambda(lambda x: x[:,:,2*dim0:])(hc0)

    maxlen, dim1 = input_shapes[1]
    tot_dim += dim1
    input1 = Input(shape=(maxlen, dim1), dtype='float64')
    hc1 = customLSTM(dim1)(input1)
    h1 = Lambda(lambda x: x[:,:,:dim1])(hc1)
    c_prev1 = Lambda(lambda x: x[:,:,dim1:2*dim1])(hc1)
    c_new1 = Lambda(lambda x: x[:,:,2*dim1:])(hc1)

    c_prev = Concatenate(axis=-1)([c_prev0, c_prev1])
    c_new = Concatenate(axis=-1)([c_new0, c_new1])
    cStar = Concatenate(axis=-1)([c_prev, c_new])

    attention = Sequential([
        TimeDistributed(Dense(256 ,activation='relu')),
        Dropout(rate=0.5),
        TimeDistributed(Dense(2*tot_dim ,activation='softmax'))
    ])(cStar)

    attended = Multiply()([cStar, attention])
    cell = MultiviewGatedMemory(mem_size)
    lstm_cell = RNN(cell)
    mem = lstm_cell(attended) # equation 9-11

    h = Concatenate(axis=-1)([h0, h1])
    last_hs = Lambda(lambda x: x[:,-1,:])(h)

    final = Concatenate(axis=-1)([last_hs, mem]) # Output of MFN

    output = Sequential([
        Dense(128 ,activation='relu'),
        Dropout(rate=0.5),
        Dense(output_classes, activation='softmax')
    ], name='final_output')(final)

    model = Model(inputs=[input0, input1], outputs=output)
    return model

def MFN_trimodal(input_shapes, output_classes, mem_size=256):
    tot_dim = 0

    maxlen, dim0 = input_shapes[0]
    tot_dim += dim0
    input0 = Input(shape=(maxlen, dim0), dtype='float64')
    hc0 = customLSTM(dim0)(input0)
    h0 = Lambda(lambda x: x[:,:,:dim0])(hc0)
    c_prev0 = Lambda(lambda x: x[:,:,dim0:2*dim0])(hc0)
    c_new0 = Lambda(lambda x: x[:,:,2*dim0:])(hc0)

    maxlen, dim1 = input_shapes[1]
    tot_dim += dim1
    input1 = Input(shape=(maxlen, dim1), dtype='float64')
    hc1 = customLSTM(dim1)(input1)
    h1 = Lambda(lambda x: x[:,:,:dim1])(hc1)
    c_prev1 = Lambda(lambda x: x[:,:,dim1:2*dim1])(hc1)
    c_new1 = Lambda(lambda x: x[:,:,2*dim1:])(hc1)

    maxlen, dim2 = input_shapes[2]
    tot_dim += dim2
    input2 = Input(shape=(maxlen, dim2), dtype='float64')
    hc2 = customLSTM(dim2)(input2)
    h2 = Lambda(lambda x: x[:,:,:dim2])(hc2)
    c_prev2 = Lambda(lambda x: x[:,:,dim2:2*dim2])(hc2)
    c_new2 = Lambda(lambda x: x[:,:,2*dim2:])(hc2)

    c_prev = Concatenate(axis=-1)([c_prev0, c_prev1, c_prev2])
    c_new = Concatenate(axis=-1)([c_new0, c_new1, c_new2])
    cStar = Concatenate(axis=-1)([c_prev, c_new])

    attention = Sequential([
        TimeDistributed(Dense(256 ,activation='relu')),
        Dropout(rate=0.5),
        TimeDistributed(Dense(2*tot_dim ,activation='softmax'))
    ])(cStar)

    attended = Multiply()([cStar, attention])
    cell = MultiviewGatedMemory(mem_size)
    lstm_cell = RNN(cell)
    mem = lstm_cell(attended) # equation 9-11

    h = Concatenate(axis=-1)([h0, h1, h2])
    last_hs = Lambda(lambda x: x[:,-1,:])(h)

    final = Concatenate(axis=-1)([last_hs, mem]) # Output of MFN

    output = Sequential([
        Dense(128 ,activation='relu'),
        Dropout(rate=0.5),
        Dense(output_classes, activation='softmax')
    ], name='final_output')(final)

    model = Model(inputs=[input0, input1, input2], outputs=output)
    return model

def MFN(input_shapes, output_classes, mem_size=256):
    k = len(input_shapes)
    assert k in [1, 2, 3]
    if k == 1:
        return MFN_unimodal(input_shapes, output_classes, mem_size=256)
    elif k == 2:
        return MFN_bimodal(input_shapes, output_classes, mem_size=256)
    else:
        return MFN_trimodal(input_shapes, output_classes, mem_size=256)