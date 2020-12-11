import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Dropout, Layer, RNN
from tensorflow.keras.layers import Lambda, Multiply, TimeDistributed, LSTMCell
from tensorflow.keras.models import Model, Sequential

def _generate_dropout_mask(ones, rate, training=None, count=1):
    """Get the dropout mask for RNN cell's input.
    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.
    Args:
        inputs: The input tensor whose shape will be used to generate dropout
        mask.
        training: Boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
        count: Int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
        List of mask tensor, generated or cached mask based on context.
    """
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(dropped_inputs, ones, training=training) for _ in range(count)]
    return K.in_train_phase(dropped_inputs, ones, training=training)

class Mod_LSTMCELL(LSTMCell):
    """Cell class for the LSTM layer.
    Following Arguments are present in the parent class LSTMCell, which 
    have been used in this implementation. Reimplementation is done so 
    that we can extract the complete hidden sequence h,c which cannot 
    be obtained directly using the native LSTM implementation.
    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass `None`, no activation is applied (ie. "linear"
            activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix, used for
            the linear transformation of the inputs. Default: `glorot_uniform`.
            recurrent_initializer: Initializer for the `recurrent_kernel` weights
            matrix, used for the linear transformation of the recurrent state.
            Default: `orthogonal`.
        bias_initializer: Initializer for the bias vector. Default: `zeros`.
        unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
            the forget gate at initialization. Setting it to true will also force
            `bias_initializer="zeros"`. This is recommended in Jozefowicz et al.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector. Default:
            `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the `recurrent_kernel`
            weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector. Default:
            `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the linear
            transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
            the linear transformation of the recurrent state. Default: 0.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of smaller dot
            products and additions, whereas mode 2 (default) will batch them into
            fewer, larger operations. These modes will have different performance
            profiles on different hardware and for different applications. Default: 2.
    """
    def call(self, inputs, states, training=None):
        """The function that contains the logic for one RNN step calculation.
        Args:
            inputs: the input tensor, which is a slide from the overall RNN input by
                the time dimension (usually the second dimension).
            states: the state tensor from previous step, which has the same shape
                as `(batch, state_size)`. In the case of timestep 0, it will be the
                initial state user specified, or zero filled tensor otherwise.
            training: Python boolean indicating whether the layer should behave in
                training mode or in inference mode. Only relevant when dropout or
                recurrent_dropout is used.
        Returns:
            A tuple of two tensors:
                1. output tensor for the current timestep, with size `output_size`.
                2. state tensor for next step, which has the shape of `state_size`.
        """
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
    """RNN applied over an implemented LSTM cell to output a
    custom LSTM which can be used as a Keras Layer.
    Args:
        dim: defines the state size for the LSTM
    Returns:
        LSTM layer formed from Mod_LSTMCELL
    """
    regularizer = tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-3)
    cell = Mod_LSTMCELL(dim, dropout=0.5, kernel_regularizer=regularizer, recurrent_regularizer=regularizer)
    lstm = RNN(cell, return_sequences=True)
    return lstm


class MultiviewGatedMemory(Layer):
    """
    Multi-view Gated Memory is the neural component that
    stores a history of cross-view interactions over time. It
    acts as a unifying memory for the memories in System of
    LSTMs. This class has been implemented as a keras Layer and
    overloads the functions of the Layer class.
    state_size: size(s) of state(s) used by this cell.
    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    def __init__(self, mem_dim, **kwargs):
        self.state_size = mem_dim
        super(MultiviewGatedMemory, self).__init__(**kwargs)

    def build(self, input_shape):
        """Creates the variables of the layer (optional, for subclass implementers).
        This is a method that implementers of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.
        This is typically used to create the weights of `Layer` subclasses.
        Args:
            input_shape: Instance of `TensorShape`, or list of instances of
                `TensorShape` if the layer expects a list of inputs
                (one instance per input).
        """
        # D_u
        self.chat = Sequential([Dense(128, activation='relu'),
                                Dropout(rate=0.5),
                                Dense(self.state_size, activation='tanh'),])
        # D_gamma1
        self.gamma1 = Sequential([Dense(128, activation='relu'),
                                  Dropout(rate=0.5),
                                  Dense(self.state_size, activation='sigmoid'),])
        # D_gamma2
        self.gamma2 = Sequential([Dense(128, activation='relu'),
                                  Dropout(rate=0.5),
                                  Dense(self.state_size, activation='sigmoid'),])

    def call(self, inputs, states):
        """The function that contains the logic for one RNN step calculation.
        Args:
            inputs: the input tensor, which is a slide from the overall RNN input by
                the time dimension (usually the second dimension).
            states: the state tensor from previous step, which has the same shape
                as `(batch, state_size)`. In the case of timestep 0, it will be the
                initial state user specified, or zero filled tensor otherwise.
        Returns:
            A tuple of two tensors:
                1. output tensor for the current timestep, with size `output_size`.
                2. state tensor for next step, which has the shape of `state_size`.
        """
        prev_mem = states[0]
        both = K.concatenate([inputs, prev_mem], axis=-1)
        g1 = self.gamma1(both)
        g2 = self.gamma2(both)
        cHat = self.chat(inputs)
        step_out = g1 * prev_mem + g2 * cHat
        return step_out, [step_out]
    
    def get_config(self):
        """Returns the config of the layer.
        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.
        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).
        Returns:
            Python dictionary.
        """
        config = super().get_config().copy()
        config.update({
            'state_size': self.state_size,
        })
        return config


def MFN_unimodal(input_shapes, output_classes, mem_size=256):
    """Returns the unimodal version of the Memory Fusion model.
    Args:
        input_shapes: list of length 1 containing a tuple of 
            (seq_length, feature_size) for one modality
        output_classes: the number of output classes
        mem_size: the size of states of MultiviewGatedMemory
    Returns:
        an instance of keras Model class with the required 
            architecture defined. 
    """
    tot_dim = 0

    maxlen, dim0 = input_shapes[0]
    tot_dim += dim0
    input0 = Input(shape=(maxlen, dim0), dtype='float64')
    # get the hidden states, memory states and the previous memory
    # states from the LSTM for a single modality
    hc0 = customLSTM(dim0)(input0)
    # Split the above states from the concatenated result from LSTM
    h = Lambda(lambda x: x[:,:,:dim0])(hc0)
    # Combine all the modalities of current and previous time steps
    cStar = Lambda(lambda x: x[:,:,dim0:])(hc0)

    # the self attention network referred to as DMAN
    attention = Sequential([
        TimeDistributed(Dense(256 ,activation='relu')),
        Dropout(rate=0.5),
        TimeDistributed(Dense(2*tot_dim ,activation='softmax'))
    ])(cStar)

    # attention applied to the memories
    attended = Multiply()([cStar, attention])
    cell = MultiviewGatedMemory(mem_size)
    lstm_cell = RNN(cell)
    mem = lstm_cell(attended) # equation 9-11

    last_hs = Lambda(lambda x: x[:,-1,:])(h)

    # this is treated as the final output given by MFN
    # to be used for classification
    final = Concatenate(axis=-1)([last_hs, mem]) # Output of MFN

    # Dense network applied to MFN output 
    # Softmax classifier (with 1 hidden layer)
    output = Sequential([
        Dense(128 ,activation='relu'),
        Dropout(rate=0.5),
        Dense(output_classes, activation='softmax')
    ], name='final_output')(final)

    model = Model(inputs=[input0], outputs=output)
    return model

def MFN_bimodal(input_shapes, output_classes, mem_size=256):
    """Returns the bimodal version of the Memory Fusion model.
    Args:
        input_shapes: list of length 2 containing tuples of 
            (seq_length, feature_size) for 2 modalities
        output_classes: the number of output classes
        mem_size: the size of states of MultiviewGatedMemory
    Returns:
        an instance of keras Model class with the required 
            architecture defined. 
    """
    tot_dim = 0

    maxlen, dim0 = input_shapes[0]
    tot_dim += dim0
    input0 = Input(shape=(maxlen, dim0), dtype='float64')
    # get the hidden states, memory states and the previous memory
    # states from the LSTM for a single modality
    hc0 = customLSTM(dim0)(input0)
    # Split the above states from the concatenated result from LSTM
    h0 = Lambda(lambda x: x[:,:,:dim0])(hc0)
    c_prev0 = Lambda(lambda x: x[:,:,dim0:2*dim0])(hc0)
    c_new0 = Lambda(lambda x: x[:,:,2*dim0:])(hc0)

    maxlen, dim1 = input_shapes[1]
    tot_dim += dim1
    input1 = Input(shape=(maxlen, dim1), dtype='float64')
    # Same procedure is followed for each modality
    hc1 = customLSTM(dim1)(input1)
    h1 = Lambda(lambda x: x[:,:,:dim1])(hc1)
    c_prev1 = Lambda(lambda x: x[:,:,dim1:2*dim1])(hc1)
    c_new1 = Lambda(lambda x: x[:,:,2*dim1:])(hc1)

    # Combine all the modalities of current and previous time steps
    c_prev = Concatenate(axis=-1)([c_prev0, c_prev1])
    c_new = Concatenate(axis=-1)([c_new0, c_new1])
    cStar = Concatenate(axis=-1)([c_prev, c_new])

    # the self attention network referred to as DMAN
    attention = Sequential([
        TimeDistributed(Dense(256 ,activation='relu')),
        Dropout(rate=0.5),
        TimeDistributed(Dense(2*tot_dim ,activation='softmax'))
    ])(cStar)

    # attention applied to the memories
    attended = Multiply()([cStar, attention])
    cell = MultiviewGatedMemory(mem_size)
    lstm_cell = RNN(cell)
    mem = lstm_cell(attended) # equation 9-11

    h = Concatenate(axis=-1)([h0, h1])
    last_hs = Lambda(lambda x: x[:,-1,:])(h)

    # this is treated as the final output given by MFN
    # to be used for classification
    final = Concatenate(axis=-1)([last_hs, mem]) # Output of MFN

    # Dense network applied to MFN output 
    # Softmax classifier (with 1 hidden layer)
    output = Sequential([
        Dense(128 ,activation='relu'),
        Dropout(rate=0.5),
        Dense(output_classes, activation='softmax')
    ], name='final_output')(final)

    model = Model(inputs=[input0, input1], outputs=output)
    return model

def MFN_trimodal(input_shapes, output_classes, mem_size=256):
    """Returns the trimodal version of the Memory Fusion model.
    Args:
        input_shapes: list of length 3 containing tuples of 
            (seq_length, feature_size) for 3 modalities
        output_classes: the number of output classes
        mem_size: the size of states of MultiviewGatedMemory
    Returns:
        an instance of keras Model class with the required 
            architecture defined. 
    """
    tot_dim = 0

    maxlen, dim0 = input_shapes[0]
    tot_dim += dim0
    input0 = Input(shape=(maxlen, dim0), dtype='float64')
    # get the hidden states, memory states and the previous memory
    # states from the LSTM for a single modality
    hc0 = customLSTM(dim0)(input0)
    # Split the above states from the concatenated result from LSTM
    h0 = Lambda(lambda x: x[:,:,:dim0])(hc0)                
    c_prev0 = Lambda(lambda x: x[:,:,dim0:2*dim0])(hc0)
    c_new0 = Lambda(lambda x: x[:,:,2*dim0:])(hc0)

    maxlen, dim1 = input_shapes[1]
    tot_dim += dim1
    input1 = Input(shape=(maxlen, dim1), dtype='float64')
    # Same procedure is followed for each modality
    hc1 = customLSTM(dim1)(input1)
    h1 = Lambda(lambda x: x[:,:,:dim1])(hc1)
    c_prev1 = Lambda(lambda x: x[:,:,dim1:2*dim1])(hc1)
    c_new1 = Lambda(lambda x: x[:,:,2*dim1:])(hc1)

    maxlen, dim2 = input_shapes[2]
    tot_dim += dim2
    input2 = Input(shape=(maxlen, dim2), dtype='float64')
    # Same procedure is followed for each modality
    hc2 = customLSTM(dim2)(input2)
    h2 = Lambda(lambda x: x[:,:,:dim2])(hc2)
    c_prev2 = Lambda(lambda x: x[:,:,dim2:2*dim2])(hc2)
    c_new2 = Lambda(lambda x: x[:,:,2*dim2:])(hc2)

    # Combine all the modalities of current and previous time steps
    c_prev = Concatenate(axis=-1)([c_prev0, c_prev1, c_prev2])
    c_new = Concatenate(axis=-1)([c_new0, c_new1, c_new2])
    cStar = Concatenate(axis=-1)([c_prev, c_new])

    # the self attention network referred to as DMAN
    attention = Sequential([
        TimeDistributed(Dense(256 ,activation='relu')),
        Dropout(rate=0.5),
        TimeDistributed(Dense(2*tot_dim ,activation='softmax'))
    ])(cStar)

    # attention applied to the memories
    attended = Multiply()([cStar, attention])
    cell = MultiviewGatedMemory(mem_size)
    lstm_cell = RNN(cell)
    mem = lstm_cell(attended) # equation 9-11

    h = Concatenate(axis=-1)([h0, h1, h2])
    last_hs = Lambda(lambda x: x[:,-1,:])(h)

    # this is treated as the final output given by MFN
    # to be used for classification
    final = Concatenate(axis=-1)([last_hs, mem])

    # Dense network applied to MFN output 
    # Softmax classifier (with 1 hidden layer)
    output = Sequential([
        Dense(128 ,activation='relu'),
        Dropout(rate=0.5),
        Dense(output_classes, activation='softmax')
    ], name='final_output')(final)

    model = Model(inputs=[input0, input1, input2], outputs=output)
    return model

def MFN(input_shapes, output_classes, mem_size=256):
    """This is just a function to abstract between the number
    of modalities to be provided to MFN.
    MFN can be extended to an arbitrary number of modalities.
    Here implementation of upto 3 modalities has been done.
    Args:
        input_shapes: list of length atmost 3 containing tuples 
            of (seq_length, feature_size) for upto 3 modalities
        output_classes: the number of output classes
        mem_size: the size of states of MultiviewGatedMemory
    Returns:
        an instance of keras Model class with the required 
            architecture defined. 
    """
    k = len(input_shapes)
    assert k in [1, 2, 3]
    if k == 1:
        return MFN_unimodal(input_shapes, output_classes, mem_size=256)
    elif k == 2:
        return MFN_bimodal(input_shapes, output_classes, mem_size=256)
    else:
        return MFN_trimodal(input_shapes, output_classes, mem_size=256)