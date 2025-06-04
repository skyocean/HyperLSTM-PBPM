import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Concatenate, Flatten, RepeatVector, Dropout, Masking, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers.schedules import ExponentialDecay, InverseTimeDecay, PiecewiseConstantDecay, PolynomialDecay

import keras_tuner as kt
from keras_tuner import RandomSearch, HyperModel, Hyperband, HyperParameters
from tensorflow.keras.metrics import F1Score

class DurationEmbeddingLSTMModel(HyperModel):
    """A hypermodel for tuning LSTM models with keras-tuner."""
    def __init__(self, event_input_shape, duration_embedding_shape, num_sequence_features, num_classes):
        super(DurationEmbeddingLSTMModel, self).__init__()
        self.event_input_shape = event_input_shape
        self.duration_embedding_shape = duration_embedding_shape
        self.num_sequence_features = num_sequence_features
        self.num_classes = num_classes

    """Configures and returns a learning rate schedule based on hyperparameters."""
    def configure_lr_schedule(self, hp):
        lr_schedule_choice = hp.Choice('lr_schedule_type', ['exponential', 'inverse_time', 'piecewise_constant', 'polynomial'])
        initial_learning_rate = hp.Float('init_lr', min_value=1e-4, max_value=1e-2, sampling='LOG')
        if lr_schedule_choice == 'exponential':
            return ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True)
        elif lr_schedule_choice == 'inverse_time':
            return InverseTimeDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.5, staircase=True)
        elif lr_schedule_choice == 'piecewise_constant':
            boundaries = [1000, 5000]
            values = [initial_learning_rate, initial_learning_rate * 0.5, initial_learning_rate * 0.1]
            return PiecewiseConstantDecay(boundaries, values)
        elif lr_schedule_choice == 'polynomial':
            return PolynomialDecay(initial_learning_rate, decay_steps=10000, end_learning_rate=1e-5, power=0.5)
   
    """Configures and returns an optimizer schedule based on hyperparameters."""
    def configure_optimizer(self, hp, lr_schedule):
        optimizer_name = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
        if optimizer_name == 'adam':
            beta_1 = hp.Float('adam_beta_1', min_value=0.85, max_value=0.99, step=0.01, default=0.9)  # Default value in Keras is 0.9
            beta_2 = hp.Float('adam_beta_2', min_value=0.99, max_value=0.999, step=0.001, default=0.999)  # Default value in Keras is 0.999
            return Adam(learning_rate=lr_schedule, beta_1=beta_1, beta_2=beta_2)
        elif optimizer_name == 'sgd':
            return SGD(learning_rate=lr_schedule, momentum=hp.Float('sgd_momentum', 0.0, 0.9))
        elif optimizer_name == 'rmsprop':
            return RMSprop(learning_rate=lr_schedule)

    """Builds the LSTM model based on hyperparameters."""        
    def build(self, hp):
        event_input = Input(shape=self.event_input_shape, name='event_input')
        # Adding Masking layer immediately after the input layer
        x = Masking(mask_value=-1.0)(event_input)  # Use -1 as the mask value if that's your encoding for missing data

        # LSTM layer configuration
        for i in range(hp.Int('event_num_lstm_layers', 1, 3)):
            x = LSTM(
                units=hp.Int('event_lstm_units_l' + str(i), 32, 256, step=32),
                return_sequences=True,
                kernel_regularizer=l2(hp.Float('event_l2_reg_l' + str(i), 1e-5, 1e-2, sampling='LOG'))
            )(x)
            # Optionally add batch normalization
            if hp.Boolean('event_batch_norm_l' + str(i)):
                x = BatchNormalization(
                    momentum=hp.Float('event_batch_norm_momentum_' + str(i), 0.01, 0.999, step=0.1),
                    epsilon=hp.Float('event_batch_norm_epsilon_' + str(i), 1e-5, 1e-2, sampling='LOG')
                )(x)
            x = Dropout(rate=hp.Float('event_dropout_l' + str(i), 0.2, 0.5))(x)

        duration_embedding_input = Input(shape = self.duration_embedding_shape, name='duration_embedding_input')
        p = duration_embedding_input

        # LSTM layer configuration
        for m in range(hp.Int('duration_num_lstm_layers', 1, 3)):
            p = LSTM(
                units=hp.Int('duration_lstm_units_l' + str(m), 32, 256, step=32),
                return_sequences=True,
                kernel_regularizer=l2(hp.Float('duration_l2_reg_l' + str(m), 1e-5, 1e-2, sampling='LOG'))
            )(p)
            # Optionally add batch normalization
            if hp.Boolean('duration_batch_norm_l' + str(m)):
                p = BatchNormalization(
                    momentum=hp.Float('duration_batch_norm_momentum_' + str(m), 0.01, 0.999, step=0.1),
                    epsilon=hp.Float('duration_batch_norm_epsilon_' + str(m), 1e-5, 1e-2, sampling='LOG')
                )(p)
            p = Dropout(rate=hp.Float('duration_dropout_l' + str(m), 0.2, 0.5))(p)
        
        x = Concatenate()([x, p])
        # LSTM layer configuration
        for n in range(hp.Int('concat_num_lstm_layers', 1, 3)):
            x = LSTM(
                units=hp.Int('concat_lstm_units_l' + str(n), 32, 256, step=32),
                return_sequences=True if n < hp.get('concat_num_lstm_layers') - 1 else False,
                kernel_regularizer=l2(hp.Float('concat_l2_reg_l' + str(n), 1e-5, 1e-2, sampling='LOG'))
            )(x)
            # Optionally add batch normalization
            if hp.Boolean('concat_batch_norm_l' + str(n)):
                x = BatchNormalization(
                    momentum=hp.Float('concat_batch_norm_momentum_' + str(n), 0.01, 0.999, step=0.1),
                    epsilon=hp.Float('concat_batch_norm_epsilon_' + str(n), 1e-5, 1e-2, sampling='LOG')
                )(x)
            x = Dropout(rate=hp.Float('concat_dropout_l' + str(n), 0.2, 0.5))(x)
        
        sequence_input = Input(shape=(self.num_sequence_features,), name='sequence_input')
        x = Concatenate()([x, sequence_input])

        # Dense layer configuration
        for j in range(hp.Int('num_dense_layers', 1, 3)):
            x = Dense(
                units=hp.Int('dense_units_' + str(j), 32, 256, step=32),
                kernel_regularizer=l2(hp.Float('l2_dense_' + str(j), 1e-5, 1e-2, sampling='LOG'))
            )(x)
            # Select activation function
            activation_choice = hp.Choice('dense_activation_' + str(j), ['relu', 'tanh', 'softmax', 'leaky_relu'])
            if activation_choice == 'leaky_relu':
                x = LeakyReLU(alpha=hp.Float('leaky_alpha_' + str(j), 0.01, 0.3))(x)
            else:
                x = Activation(activation_choice)(x)
            x = Dropout(rate=hp.Float('dropout_dense_' + str(j), 0.1, 0.7))(x)

        # Output layer
        output = Dense(self.num_classes, activation='softmax')(x)

        # Configure learning rate schedule and optimizer
        lr_schedule = self.configure_lr_schedule(hp)
        optimizer = self.configure_optimizer(hp, lr_schedule)

        model = Model(inputs=[event_input, duration_embedding_input, sequence_input], outputs=output)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[F1Score(average="weighted", name="f1_score")])
        
        return model
        
    
def print_best_hp_duration(best_hps):
    print("Best hyperparameters found were:")
    print(f"Number of Event LSTM layers: {best_hps.get('event_num_lstm_layers')}")
    for i in range(best_hps.get('event_num_lstm_layers')):
        print(f"  LSTM Layer {i}:")
        print(f"    Units: {best_hps.get('event_lstm_units_l' + str(i))}")
        print(f"    Dropout Rate: {best_hps.get('event_dropout_l' + str(i))}")
        print(f"    L2 Regularization: {best_hps.get('event_l2_reg_l' + str(i))}")
        if best_hps.get('event_batch_norm_l' + str(i)):
            print(f"    Batch Norm Momentum: {best_hps.get('event_batch_norm_momentum_' + str(i))}")
            print(f"    Batch Norm Epsilon: {best_hps.get('event_batch_norm_epsilon_' + str(i))}")
    
    print(f"Number of Duration Embedding LSTM layers: {best_hps.get('duration_num_lstm_layers')}")
    for m in range(best_hps.get('duration_num_lstm_layers')):
        print(f"  LSTM Layer {m}:")
        print(f"    Units: {best_hps.get('duration_lstm_units_l' + str(m))}")
        print(f"    Dropout Rate: {best_hps.get('duration_dropout_l' + str(m))}")
        print(f"    L2 Regularization: {best_hps.get('duration_l2_reg_l' + str(m))}")
        if best_hps.get('duration_batch_norm_l' + str(m)):
            print(f"    Batch Norm Momentum: {best_hps.get('duration_batch_norm_momentum_' + str(m))}")
            print(f"    Batch Norm Epsilon: {best_hps.get('duration_batch_norm_epsilon_' + str(m))}")
    
    print(f"Number of Concatenate Event Feature LSTM layers: {best_hps.get('concat_num_lstm_layers')}")
    for n in range(best_hps.get('concat_num_lstm_layers')):
        print(f"  LSTM Layer {n}:")
        print(f"    Units: {best_hps.get('concat_lstm_units_l' + str(n))}")
        print(f"    Dropout Rate: {best_hps.get('concat_dropout_l' + str(n))}")
        print(f"    L2 Regularization: {best_hps.get('concat_l2_reg_l' + str(n))}")
        if best_hps.get('concat_batch_norm_l' + str(n)):
            print(f"    Batch Norm Momentum: {best_hps.get('concat_batch_norm_momentum_' + str(n))}")
            print(f"    Batch Norm Epsilon: {best_hps.get('concat_batch_norm_epsilon_' + str(n))}")
    
    print(f"Number of Dense layers: {best_hps.get('num_dense_layers')}")
    for j in range(best_hps.get('num_dense_layers')):
        print(f"  Dense Layer {j}:")
        print(f"    Units: {best_hps.get('dense_units_' + str(j))}")
        print(f"    Activation: {best_hps.get('dense_activation_' + str(j))}")
        if best_hps.get('dense_activation_' + str(j)) == 'leaky_relu':
            print(f"    Leaky ReLU Alpha: {best_hps.get('leaky_alpha_' + str(j))}")
        print(f"    Dropout Rate: {best_hps.get('dropout_dense_' + str(j))}")
        print(f"    L2 Regularization: {best_hps.get('l2_dense_' + str(j))}")
    
    print(f"Optimizer: {best_hps.get('optimizer')}")
    if best_hps.get('optimizer') == 'adam':
        print(f"  Learning Rate (Adam): {best_hps.get('init_lr')}")
        print(f"  Beta 1 (Adam): {best_hps.get('adam_beta_1')}")
        print(f"  Beta 2 (Adam): {best_hps.get('adam_beta_2')}")
    elif best_hps.get('optimizer') == 'sgd':
        print(f"  Learning Rate (SGD): {best_hps.get('init_lr')}")
        print(f"  Momentum (SGD): {best_hps.get('sgd_momentum')}")
    elif best_hps.get('optimizer') == 'rmsprop':
        print(f"  Learning Rate (RMSprop): {best_hps.get('init_lr')}")

    print(f"Learning Rate Schedule: {best_hps.get('lr_schedule_type')}")
    print('Best batch size:', best_hps.get('batch_size'))
    
    #Alternative, 
    #    for key, value in best_hps.values.items():        print(f"{key}: {value}")
