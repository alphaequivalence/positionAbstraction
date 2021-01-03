import tensorflow as tf
from dataset import DataReader


class Encoder():
    def __init__(self):
        pass


class Decoder():
    def __init__(self):
        pass


class LocalModel():
    def __init__(self,
        position,
        num_filters_0, kernel_size_0,
        num_filters_1, kernel_size_1,
        num_filters_2, kernel_size_2,
        hidden_units, dropout_rate, regularization_rate
    ):
        xs = []
        inputs = []
        feature_maps = []
        for _, channel in DataReader.channels.items():
            ts = tf.keras.Input(shape=(500,), name=position+'_'+channel)  # 3D tensor with shape: (batch_size, steps, input_dim)
            x = tf.keras.layers.Reshape((500, 1))(ts)
            # xs.append(x)

            x = tf.keras.layers.Conv1D(
                filters=num_filters_0,
                kernel_size=kernel_size_0,
                strides=2,
                padding='valid',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(regularization_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularization_rate),
                activity_regularizer=tf.keras.regularizers.l2(regularization_rate),
                input_shape=(None, 500, 1))(x)
            x = tf.keras.layers.MaxPooling1D()(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Conv1D(
                filters=num_filters_1,
                kernel_size=kernel_size_1,
                strides=2,
                padding='valid',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(regularization_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularization_rate),
                activity_regularizer=tf.keras.regularizers.l2(regularization_rate))(x)
            x = tf.keras.layers.MaxPooling1D()(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Conv1D(
                filters=num_filters_2,
                kernel_size=kernel_size_2,
                strides=2,
                padding='valid',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(regularization_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularization_rate),
                activity_regularizer=tf.keras.regularizers.l2(regularization_rate))(x)
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
            x = tf.keras.layers.BatchNormalization()(x)

            inputs.append(ts)
            feature_maps.append(x)

        x = tf.keras.layers.concatenate(feature_maps)  # , axis=-1)

        x = tf.keras.layers.Dense(
            units=hidden_units,
            activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        class_output = tf.keras.layers.Dense(8, activation='softmax', name='class_output')(x)

        model = tf.keras.Model(inputs=inputs, outputs=class_output)

        tf.keras.utils.plot_model(model, position+'_LocalModel.png', show_shapes=True)

        # return model
        self.model = model