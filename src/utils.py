import matplotlib.pyplot as plt
import torch
import os
import optuna


# Non smooth_curve
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_accuracy'])
    plt.plot(history['test_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.ylim(0.2, 1.2)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'])
    plt.plot(history['test_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.ylim(0, 3)

    plt.show()

def plot_training_history_im(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_f1'])
    plt.plot(history['test_f1'])
    plt.title('Model F1')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.ylim(0.2, 1.2)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'])
    plt.plot(history['test_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.ylim(0, 3)

    plt.show()

# Non smooth_curve
def plot_training_history_lstm(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.ylim(0.2, 1.2)

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.ylim(0, 3)

    plt.show()


def plot_training_history_LSTMim(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['f1_score'])
    plt.plot(history['val_f1_score'])
    plt.title('Model F1')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.ylim(0.2, 1.2)

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.ylim(0, 3)

    plt.show()

def smooth_curve(points, factor=0.1):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

#  smoothed accuracy - for trending
def plot_training_history_smooth(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(smooth_curve(history['train_accuracy']), label='Smoothed training acc')
    plt.plot(smooth_curve(history['test_accuracy']), label='Smoothed validation acc')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'])
    plt.plot(history['test_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()


def print_best_hp_gcn(model_name, model_save_path, device):
    best_hps = torch.load(model_save_path, map_location = device)
    print(f"Output Size: {best_hps.get('output_dim')}")
    print('Best batch size:', best_hps.get('batch_size'))
    print('Best epoch:', best_hps.get('epoch')+1)
    print(f"Best accuracy: {best_hps.get('best_accuracy'):.4f}")
    print(f"Best loss: {best_hps.get('best_loss'):.4f}")
    print(f"Best loss std: {best_hps.get('best_std_dev'):.4f}")
    print("Best hyperparameters found were:")
    
    if (model_name == "GCNModel"):
        print(f"Input Size: {best_hps.get('num_comb_features')}")
    if model_name == "EventSequenceEmbeddingGCNModel":
        print(f" Embedding Features Input Size: {best_hps.get('num_embedding_features')}")   
        print(f"Embedding Dimensions: {best_hps.get('embedding_dims')}")
        print(f"Number of Embedding Features GCN layers: {len(best_hps.get('gcn_hidden_dims_embedding'))}")
        for i in range(len(best_hps.get('gcn_hidden_dims_embedding'))):
            print(f"  Embedding Features GCN Layer {i+1}:")
            print(f"    Units: {best_hps.get('gcn_hidden_dims_embedding')[i]}") 
            if best_hps.get('gcn_batch_norm_flag_embedding')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum_embedding')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps_embedding')[i]:.4e}")
            print(f"    Activation: {best_hps.get('gcn_activation_embedding')[i]}") 
            if best_hps.get('gcn_dropout_flag_embedding')[i]:
                print(f"    Dropout: {best_hps.get('gcn_dropout_rate_embedding')[i]:.4f}")  
    if (model_name == "EventSequenceGCNModel")  or (model_name == "EventSequenceDurationGCNModel") or (model_name == "EventSequenceEmbeddingGCNModel"):
        print(f"Event Input Size: {best_hps.get('num_event_features')}")
    print(f"Number of GCN layers: {len(best_hps.get('gcn_hidden_dims'))}")
    for i in range(len(best_hps.get('gcn_hidden_dims'))):
        print(f"  GCN Layer {i+1}:")
        print(f"    Units: {best_hps.get('gcn_hidden_dims')[i]}") 
        if (model_name == "GCNModel") or (model_name == "EventSequenceGCNModel"):
            print(f"    Skip Connections: {best_hps.get('gcn_skip_connections')[i]}") 
        if best_hps.get('gcn_batch_norm_flag')[i]:
            print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum')[i]:.4f}")
            print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps')[i]:.4e}")
        print(f"    Activation: {best_hps.get('gcn_activation')[i]}") 
        if best_hps.get('gcn_dropout_flag')[i]:
            print(f"    Dropout: {best_hps.get('gcn_dropout_rate')[i]:.4f}")  
    if model_name == "EventSequenceDurationGCNModel":
        print(f" Duration Embedding Input Size: {best_hps.get('num_duration_features')}")   
        print(f"Number of Duration Embedding GCN layers: {len(best_hps.get('gcn_hidden_dims_duration'))}")
        for i in range(len(best_hps.get('gcn_hidden_dims_duration'))):
            print(f"  Duration Embedding GCN Layer {i+1}:")
            print(f"    Units: {best_hps.get('gcn_hidden_dims_duration')[i]}") 
            if best_hps.get('gcn_batch_norm_flag_duration')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum_duration')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps_duration')[i]:.4e}")
            print(f"    Activation: {best_hps.get('gcn_activation_duration')[i]}") 
            if best_hps.get('gcn_dropout_flag_duration')[i]:
                print(f"    Dropout: {best_hps.get('gcn_dropout_rate_duration')[i]:.4f}")  
        
    if (model_name == "EventSequenceDurationGCNModel") or (model_name == "EventSequenceEmbeddingGCNModel"):
        print(f"Number of Concatenated GCN layers: {len(best_hps.get('gcn_hidden_dims_concat'))}")
        for i in range(len(best_hps.get('gcn_hidden_dims_concat'))):
            print(f"  Concatenated GCN Layer {i+1}:")
            print(f"    Units: {best_hps.get('gcn_hidden_dims_concat')[i]}") 
            print(f"    Skip Connections: {best_hps.get('gcn_skip_connections')[i]}") 
            if best_hps.get('gcn_batch_norm_flag_concat')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum_concat')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps_concat')[i]:.4e}")
            print(f"    Activation: {best_hps.get('gcn_activation_concat')[i]}") 
            if best_hps.get('gcn_dropout_flag_concat')[i]:
                print(f"    Dropout: {best_hps.get('gcn_dropout_rate_concat')[i]:.4f}") 
    
    print(f"Pooling Method: {best_hps.get('pooling_method')}")
    
    if (model_name == "EventSequenceGCNModel") or (model_name == "EventSequenceDurationGCNModel") or (model_name == "EventSequenceEmbeddingGCNModel"):
        print(f"Sequence Input Size: {best_hps.get('num_sequence_features')}")
        print(f"Number of Sequence Dense layers: {len(best_hps.get('fc_hidden_dims'))}")
    elif model_name == "GCNModel":
         print(f"Number of Dense layers: {len(best_hps.get('fc_hidden_dims'))}")
    for i in range(len(best_hps.get('fc_hidden_dims'))):
        if model_name == "GCNModel":
            print(f"  Dense Layer {i+1}:")
        elif (model_name == "EventSequenceGCNModel") or (model_name == "EventSequenceDurationGCNModel") or (model_name == "EventSequenceEmbeddingGCNModel"):
            print(f"  Sequence Dense Layer {i+1}:")
        print(f"    Units: {best_hps.get('fc_hidden_dims')[i]}") 
        if best_hps.get('fc_batch_norm_flag')[i]:
            print(f"    Batch Norm Momentum: {best_hps.get('fc_momentum')[i]:.4f}")
            print(f"    Batch Norm Epsilon: {best_hps.get('fc_eps')[i]:.4e}")
        print(f"    Activation: {best_hps.get('fc_activation')[i]}") 
        if best_hps.get('fc_dropout_flag')[i]:
            print(f"    Dropout: {best_hps.get('fc_dropout_rate')[i]:.4f}")  

    if (model_name == "EventSequenceGCNModel") or (model_name == "EventSequenceDurationGCNModel") or (model_name == "EventSequenceEmbeddingGCNModel"):
        print(f"Number of Dense layers: {len(best_hps.get('fc_hidden_dims_concat'))}")
    
        for i in range(len(best_hps.get('fc_hidden_dims_concat'))):
            print(f"  Dense Layer {i+1}:")
            print(f"    Units: {best_hps.get('fc_hidden_dims_concat')[i]}") 
            if best_hps.get('fc_batch_norm_flag_concat')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('fc_momentum_concat')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('fc_eps_concat')[i]:.4e}")
            print(f"    Activation: {best_hps.get('fc_activation_concat')[i]}") 
            if best_hps.get('fc_dropout_flag_concat')[i]:
                print(f"    Dropout: {best_hps.get('fc_dropout_rate_concat')[i]:.4f}")  

    print(f"Optimizer: {best_hps.get('optimizer_name')}")
    if best_hps.get('optimizer_name') == 'Adam':
        print(f"  Learning Rate (Adam): {best_hps.get('learning_rate'):.4e}")
        print(f"  Beta 1 (Adam): {best_hps.get('beta1'):.4f}")
        print(f"  Beta 2 (Adam): {best_hps.get('beta2'):.4f}")
        print(f"  Weight Decay(Adam): {best_hps.get('weight_decay'):.4e}")
    elif best_hps.get('optimizer_name') == 'SGD':
        print(f"  Learning Rate (SGD): {best_hps.get('learning_rate'):.4e}")
        print(f"  Momentum (SGD): {best_hps.get('momentum'):.4f}")
        print(f"  Weight Decay (SDG): {best_hps.get('weight_decay'):.4e}")
    elif best_hps.get('optimizer_name') == 'RMSprop':
        print(f"  Learning Rate (RMSprop): {best_hps.get('learning_rate'):.4e}")
        print(f"  Weight Decay (RMSprop): {best_hps.get('weight_decay'):.4e}")
        print(f"  Momentum (RMSprop): {best_hps.get('momentum_rms'):.4f}")
        print(f"  Alpha (RMSprop): {best_hps.get('alpha'):.4f}")
        print(f"  Eps (RMSprop): {best_hps.get('eps_rms'):.4e}")
    
    print(f"Learning Rate Schedule: {best_hps.get('lr_scheduler_name')}")
    if best_hps.get('lr_scheduler_name') == 'StepLR':
        print(f"  Step Size: {best_hps.get('step_size'):.4f}")
        print(f"  Gamma: {best_hps.get('stepLRgamma'):.4f}")
    elif best_hps.get('lr_scheduler_name') == 'ExponentialLR':
        print(f"  Gamma: {best_hps.get('exLRgamma'):.4f}")
    elif best_hps.get('lr_scheduler_name') == 'ReduceLROnPlateau':
        print("   Mode: Max")
        print(f"  Factor: {best_hps.get('factor'):.4f}")
        print(f"  Patience: {best_hps.get('lr_patience'):.4f}")
        print(f"  Threshold: {best_hps.get('lr_threshold'):.4e}")
        print(f"  Eps: {best_hps.get('lr_eps'):.4e}")
    elif best_hps.get('lr_scheduler_name') == 'PolynomialLR':
        print(f"  Total Iters: {best_hps.get('total_iters')}")
        print(f"  Power: {best_hps.get('power'):.4f}")
    elif best_hps.get('lr_scheduler_name') ==  'CosineAnnealingLR':
        print(f"  T_max: {best_hps.get('T_max'):.4f}")
        print(f"  Eta_min: {best_hps.get('eta_min'):.4e}")
    elif best_hps.get('lr_scheduler_name') ==  'CyclicLR':
        print(f"  Base_lr: {best_hps.get('base_lr'):.4e}")
        print(f"  Max_lr: {best_hps.get('max_lr_cy'):.4e}")
        print(f"  Step_size_up: {best_hps.get('step_size_up'):.4f}")
    elif best_hps.get('lr_scheduler_name')  == 'OneCycleLR':
        print(f"  Max_lr: {best_hps.get('max_lr_1'):.4e}")
        print(f"  Total_steps: {best_hps.get('total_steps')}")
        print(f"  Pct_start: {best_hps.get('pct_start'):.4f}") 
    print('Loss function:', best_hps.get('loss_function'))
    print('l1 lambda:', f"{best_hps.get('l1_lambda'):.4e}")

def print_best_hp_graphconv(model_name, model_save_path, device):
    best_hps = torch.load(model_save_path, map_location = device)
    print(f"Output Size: {best_hps.get('output_dim')}")
    print('Best batch size:', best_hps.get('batch_size'))
    print('Best epoch:', best_hps.get('epoch')+1)
    print(f"Best accuracy: {best_hps.get('best_accuracy'):.4f}")
    print(f"Best loss: {best_hps.get('best_loss'):.4f}")
    print(f"Best loss std: {best_hps.get('best_std_dev'):.4f}")
    print("Best hyperparameters found were:")
    
    if (model_name == "GraphConvModel"):
        print(f"Input Size: {best_hps.get('num_comb_features')}")
    if model_name == "EventSequenceEmbeddingGraphConvModel":
        print(f" Embedding Features Input Size: {best_hps.get('num_embedding_features')}")   
        print(f"Embedding Dimensions: {best_hps.get('embedding_dims')}")
        print(f"Number of Embedding Features GCN layers: {len(best_hps.get('gcn_hidden_dims_embedding'))}")
        for i in range(len(best_hps.get('gcn_hidden_dims_embedding'))):
            print(f"  Embedding Features GCN Layer {i+1}:")
            print(f"    Units: {best_hps.get('gcn_hidden_dims_embedding')[i]}") 
            print(f"    Aggregation Method: {best_hps.get('gcn_aggrs_embedding')[i]}") 
            if best_hps.get('gcn_batch_norm_flag_embedding')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum_embedding')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps_embedding')[i]:.4e}")
            print(f"    Activation: {best_hps.get('gcn_activation_embedding')[i]}") 
            if best_hps.get('gcn_dropout_flag_embedding')[i]:
                print(f"    Dropout: {best_hps.get('gcn_dropout_rate_embedding')[i]:.4f}")  
    if (model_name == "EventSequenceGraphConvModel")  or (model_name == "EventSequenceDurationGraphConvModel") or (model_name == "EventSequenceEmbeddingGraphConvModel"):
        print(f"Event Input Size: {best_hps.get('num_event_features')}")
    print(f"Number of GCN layers: {len(best_hps.get('gcn_hidden_dims'))}")
    for i in range(len(best_hps.get('gcn_hidden_dims'))):
        print(f"  GCN Layer {i+1}:")
        print(f"    Units: {best_hps.get('gcn_hidden_dims')[i]}")
        print(f"    Aggregation Method: {best_hps.get('gcn_aggrs')[i]}") 
        if (model_name == "GraphConvModel") or (model_name == "EventSequenceGraphConvModel"):
            print(f"    Skip Connections: {best_hps.get('gcn_skip_connections')[i]}") 
        if best_hps.get('gcn_batch_norm_flag')[i]:
            print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum')[i]:.4f}")
            print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps')[i]:.4e}")
        print(f"    Activation: {best_hps.get('gcn_activation')[i]}") 
        if best_hps.get('gcn_dropout_flag')[i]:
            print(f"    Dropout: {best_hps.get('gcn_dropout_rate')[i]:.4f}")  
    if model_name == "EventSequenceDurationGraphConvModel":
        print(f" Duration Embedding Input Size: {best_hps.get('num_duration_features')}")   
        print(f"Number of Duration Embedding GCN layers: {len(best_hps.get('gcn_hidden_dims_duration'))}")
        for i in range(len(best_hps.get('gcn_hidden_dims_duration'))):
            print(f"  Duration Embedding GCN Layer {i+1}:")
            print(f"    Units: {best_hps.get('gcn_hidden_dims_duration')[i]}") 
            print(f"    Aggregation Method: {best_hps.get('gcn_aggrs_duration')[i]}") 
            if best_hps.get('gcn_batch_norm_flag_duration')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum_duration')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps_duration')[i]:.4e}")
            print(f"    Activation: {best_hps.get('gcn_activation_duration')[i]}") 
            if best_hps.get('gcn_dropout_flag_duration')[i]:
                print(f"    Dropout: {best_hps.get('gcn_dropout_rate_duration')[i]:.4f}")  
    if (model_name == "EventSequenceDurationGraphConvModel") or (model_name == "EventSequenceEmbeddingGraphConvModel"):
        print(f"Number of Concatenated GCN layers: {len(best_hps.get('gcn_hidden_dims_concat'))}")
        for i in range(len(best_hps.get('gcn_hidden_dims_concat'))):
            print(f"  Concatenated GCN Layer {i+1}:")
            print(f"    Units: {best_hps.get('gcn_hidden_dims_concat')[i]}") 
            print(f"    Aggregation Method: {best_hps.get('gcn_aggrs_concat')[i]}") 
            print(f"    Skip Connections: {best_hps.get('gcn_skip_connections')[i]}") 
            if best_hps.get('gcn_batch_norm_flag_concat')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum_concat')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps_concat')[i]:.4e}")
            print(f"    Activation: {best_hps.get('gcn_activation_concat')[i]}") 
            if best_hps.get('gcn_dropout_flag_concat')[i]:
                print(f"    Dropout: {best_hps.get('gcn_dropout_rate_concat')[i]:.4f}") 
    
    print(f"Pooling Method: {best_hps.get('pooling_method')}")
    
    if (model_name == "EventSequenceGraphConvModel") or (model_name == "EventSequenceDurationGraphConvModel") or (model_name == "EventSequenceEmbeddingGraphConvModel"):
        print(f"Sequence Input Size: {best_hps.get('num_sequence_features')}")
        print(f"Number of Sequence Dense layers: {len(best_hps.get('fc_hidden_dims'))}")
    elif model_name == "GraphConvModel":
         print(f"Number of Dense layers: {len(best_hps.get('fc_hidden_dims'))}")
    for i in range(len(best_hps.get('fc_hidden_dims'))):
        if model_name == "GraphConvModel":
            print(f"  Dense Layer {i+1}:")
        elif (model_name == "EventSequenceGraphConvModel") or (model_name == "EventSequenceDurationGraphConvModel") or (model_name == "EventSequenceEmbeddingGraphConvModel"):
            print(f"  Sequence Dense Layer {i+1}:")
        print(f"    Units: {best_hps.get('fc_hidden_dims')[i]}") 
        if best_hps.get('fc_batch_norm_flag')[i]:
            print(f"    Batch Norm Momentum: {best_hps.get('fc_momentum')[i]:.4f}")
            print(f"    Batch Norm Epsilon: {best_hps.get('fc_eps')[i]:.4e}")
        print(f"    Activation: {best_hps.get('fc_activation')[i]}") 
        if best_hps.get('fc_dropout_flag')[i]:
            print(f"    Dropout: {best_hps.get('fc_dropout_rate')[i]:.4f}")  

    if (model_name == "EventSequenceGraphConvModel") or (model_name == "EventSequenceDurationGraphConvModel") or (model_name == "EventSequenceEmbeddingGraphConvModel"):
        print(f"Number of Dense layers: {len(best_hps.get('fc_hidden_dims_concat'))}")
    
        for i in range(len(best_hps.get('fc_hidden_dims_concat'))):
            print(f"  Dense Layer {i+1}:")
            print(f"    Units: {best_hps.get('fc_hidden_dims_concat')[i]}") 
            if best_hps.get('fc_batch_norm_flag_concat')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('fc_momentum_concat')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('fc_eps_concat')[i]:.4e}")
            print(f"    Activation: {best_hps.get('fc_activation_concat')[i]}") 
            if best_hps.get('fc_dropout_flag_concat')[i]:
                print(f"    Dropout: {best_hps.get('fc_dropout_rate_concat')[i]:.4f}")  

    print(f"Optimizer: {best_hps.get('optimizer_name')}")
    if best_hps.get('optimizer_name') == 'Adam':
        print(f"  Learning Rate (Adam): {best_hps.get('learning_rate'):.4e}")
        print(f"  Beta 1 (Adam): {best_hps.get('beta1'):.4f}")
        print(f"  Beta 2 (Adam): {best_hps.get('beta2'):.4f}")
        print(f"  Weight Decay(Adam): {best_hps.get('weight_decay'):.4e}")
    elif best_hps.get('optimizer_name') == 'SGD':
        print(f"  Learning Rate (SGD): {best_hps.get('learning_rate'):.4e}")
        print(f"  Momentum (SGD): {best_hps.get('momentum'):.4f}")
        print(f"  Weight Decay(SDG): {best_hps.get('weight_decay'):.4e}")
    elif best_hps.get('optimizer_name') == 'RMSprop':
        print(f"  Learning Rate (RMSprop): {best_hps.get('learning_rate'):.4e}")
        print(f"  Weight Decay (RMSprop): {best_hps.get('weight_decay'):.4f}")
        print(f"  Momentum (RMSprop): {best_hps.get('momentum_rms'):.4f}")
        print(f"  Alpha (RMSprop): {best_hps.get('alpha'):.4f}")
        print(f"  Eps (RMSprop): {best_hps.get('eps_rms'):.4e}")
    
    print(f"Learning Rate Schedule: {best_hps.get('lr_scheduler_name')}")
    if best_hps.get('lr_scheduler_name') == 'StepLR':
        print(f"  Step Size: {best_hps.get('step_size'):.4f}")
        print(f"  Gamma: {best_hps.get('stepLRgamma'):.4f}")
    elif best_hps.get('lr_scheduler_name') == 'ExponentialLR':
        print(f"  Gamma: {best_hps.get('exLRgamma'):.4f}")
    elif best_hps.get('lr_scheduler_name') == 'ReduceLROnPlateau':
        print("   Mode: Max")
        print(f"  Factor: {best_hps.get('factor'):.4f}")
        print(f"  Patience: {best_hps.get('lr_patience'):.4f}")
        print(f"  Threshold: {best_hps.get('lr_threshold'):.4e}")
        print(f"  Eps: {best_hps.get('lr_eps'):.4e}")
    elif best_hps.get('lr_scheduler_name') == 'PolynomialLR':
        print(f"  Total Iters: {best_hps.get('total_iters')}")
        print(f"  Power: {best_hps.get('power'):.4f}")
    elif best_hps.get('lr_scheduler_name') ==  'CosineAnnealingLR':
        print(f"  T_max: {best_hps.get('T_max'):.4f}")
        print(f"  Eta_min: {best_hps.get('eta_min'):.4e}")
    elif best_hps.get('lr_scheduler_name') ==  'CyclicLR':
        print(f"  Base_lr: {best_hps.get('base_lr'):.4e}")
        print(f"  Max_lr: {best_hps.get('max_lr_cy'):.4e}")
        print(f"  Step_size_up: {best_hps.get('step_size_up'):.4f}")
    elif best_hps.get('lr_scheduler_name')  == 'OneCycleLR':
        print(f"  Max_lr: {best_hps.get('max_lr_1'):.4e}")
        print(f"  Total_steps: {best_hps.get('total_steps')}")
        print(f"  Pct_start: {best_hps.get('pct_start'):.4f}") 
    print('Loss function:', best_hps.get('loss_function'))
    print('l1 lambda:', f"{best_hps.get('l1_lambda'):.4e}")

def best_trial_path(study, model_save_folder):
    max_accuracy = max(t.value for t in study.trials)
    best_accuracy_trials = [t for t in study.trials if t.value == max_accuracy]
    min_loss_dev = min(x.user_attrs['loss_dev'] for x in best_accuracy_trials)
    best_loss_trials = [t for t in best_accuracy_trials if t.user_attrs['loss_dev'] == min_loss_dev]

    #best_trial = min(best_accuracy_trials, key=lambda t: t.user_attrs['loss_dev'])
    best_trial = min(best_loss_trials, key=lambda t: t.user_attrs['test_loss'])

    best_trial = best_trial.number
    model_save_path = os.path.join(model_save_folder, f'trial_{best_trial}.pth')
    
    return model_save_path


def print_best_hp_gcn_im(model_name, model_save_path, device):
    best_hps = torch.load(model_save_path, map_location = device)
    print(f"Output Size: {best_hps.get('output_dim')}")
    print('Best batch size:', best_hps.get('batch_size'))
    print('Best epoch:', best_hps.get('epoch')+1)
    print(f"Best accuracy: {best_hps.get('best_accuracy'):.4f}")
    print(f"Best loss: {best_hps.get('best_loss'):.4f}")
    print(f"Best loss std: {best_hps.get('best_std_dev'):.4f}")
    print(f"Best F1: {best_hps.get('best_f1'):.4f}")
    print("Best hyperparameters found were:")
    
    if (model_name == "GCNModel"):
        print(f"Input Size: {best_hps.get('num_comb_features')}")
    if model_name == "EventSequenceEmbeddingGCNModel":
        print(f" Embedding Features Input Size: {best_hps.get('num_embedding_features')}")   
        print(f"Embedding Dimensions: {best_hps.get('embedding_dims')}")
        print(f"Number of Embedding Features GCN layers: {len(best_hps.get('gcn_hidden_dims_embedding'))}")
        for i in range(len(best_hps.get('gcn_hidden_dims_embedding'))):
            print(f"  Embedding Features GCN Layer {i+1}:")
            print(f"    Units: {best_hps.get('gcn_hidden_dims_embedding')[i]}") 
            if best_hps.get('gcn_batch_norm_flag_embedding')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum_embedding')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps_embedding')[i]:.4e}")
            print(f"    Activation: {best_hps.get('gcn_activation_embedding')[i]}") 
            if best_hps.get('gcn_dropout_flag_embedding')[i]:
                print(f"    Dropout: {best_hps.get('gcn_dropout_rate_embedding')[i]:.4f}")  
    if (model_name == "EventSequenceGCNModel")  or (model_name == "EventSequenceDurationGCNModel") or (model_name == "EventSequenceEmbeddingGCNModel"):
        print(f"Event Input Size: {best_hps.get('num_event_features')}")
    print(f"Number of GCN layers: {len(best_hps.get('gcn_hidden_dims'))}")
    for i in range(len(best_hps.get('gcn_hidden_dims'))):
        print(f"  GCN Layer {i+1}:")
        print(f"    Units: {best_hps.get('gcn_hidden_dims')[i]}") 
        if (model_name == "GCNModel") or (model_name == "EventSequenceGCNModel"):
            print(f"    Skip Connections: {best_hps.get('gcn_skip_connections')[i]}") 
        if best_hps.get('gcn_batch_norm_flag')[i]:
            print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum')[i]:.4f}")
            print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps')[i]:.4e}")
        print(f"    Activation: {best_hps.get('gcn_activation')[i]}") 
        if best_hps.get('gcn_dropout_flag')[i]:
            print(f"    Dropout: {best_hps.get('gcn_dropout_rate')[i]:.4f}")  
    if model_name == "EventSequenceDurationGCNModel":
        print(f" Duration Embedding Input Size: {best_hps.get('num_duration_features')}")   
        print(f"Number of Duration Embedding GCN layers: {len(best_hps.get('gcn_hidden_dims_duration'))}")
        for i in range(len(best_hps.get('gcn_hidden_dims_duration'))):
            print(f"  Duration Embedding GCN Layer {i+1}:")
            print(f"    Units: {best_hps.get('gcn_hidden_dims_duration')[i]}") 
            if best_hps.get('gcn_batch_norm_flag_duration')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum_duration')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps_duration')[i]:.4e}")
            print(f"    Activation: {best_hps.get('gcn_activation_duration')[i]}") 
            if best_hps.get('gcn_dropout_flag_duration')[i]:
                print(f"    Dropout: {best_hps.get('gcn_dropout_rate_duration')[i]:.4f}")  
    if (model_name == "EventSequenceDurationGCNModel") or (model_name == "EventSequenceEmbeddingGCNModel"):   
        print(f"Number of Concatenated GCN layers: {len(best_hps.get('gcn_hidden_dims_concat'))}")
        for i in range(len(best_hps.get('gcn_hidden_dims_concat'))):
            print(f"  Concatenated GCN Layer {i+1}:")
            print(f"    Units: {best_hps.get('gcn_hidden_dims_concat')[i]}") 
            print(f"    Skip Connections: {best_hps.get('gcn_skip_connections')[i]}") 
            if best_hps.get('gcn_batch_norm_flag_concat')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum_concat')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps_concat')[i]:.4e}")
            print(f"    Activation: {best_hps.get('gcn_activation_concat')[i]}") 
            if best_hps.get('gcn_dropout_flag_concat')[i]:
                print(f"    Dropout: {best_hps.get('gcn_dropout_rate_concat')[i]:.4f}") 
    
    print(f"Pooling Method: {best_hps.get('pooling_method')}")
    
    if (model_name == "EventSequenceGCNModel") or (model_name == "EventSequenceDurationGCNModel") or (model_name == "EventSequenceEmbeddingGCNModel"):
        print(f"Sequence Input Size: {best_hps.get('num_sequence_features')}")
        print(f"Number of Sequence Dense layers: {len(best_hps.get('fc_hidden_dims'))}")
    elif model_name == "GCNModel":
         print(f"Number of Dense layers: {len(best_hps.get('fc_hidden_dims'))}")
    for i in range(len(best_hps.get('fc_hidden_dims'))):
        if model_name == "GCNModel":
            print(f"  Dense Layer {i+1}:")
        elif (model_name == "EventSequenceGCNModel") or (model_name == "EventSequenceDurationGCNModel") or (model_name == "EventSequenceEmbeddingGCNModel"):
            print(f"  Sequence Dense Layer {i+1}:")
        print(f"    Units: {best_hps.get('fc_hidden_dims')[i]}") 
        if best_hps.get('fc_batch_norm_flag')[i]:
            print(f"    Batch Norm Momentum: {best_hps.get('fc_momentum')[i]:.4f}")
            print(f"    Batch Norm Epsilon: {best_hps.get('fc_eps')[i]:.4e}")
        print(f"    Activation: {best_hps.get('fc_activation')[i]}") 
        if best_hps.get('fc_dropout_flag')[i]:
            print(f"    Dropout: {best_hps.get('fc_dropout_rate')[i]:.4f}")  

    if (model_name == "EventSequenceGCNModel") or (model_name == "EventSequenceDurationGCNModel") or (model_name == "EventSequenceEmbeddingGCNModel"):
        print(f"Number of Dense layers: {len(best_hps.get('fc_hidden_dims_concat'))}")
    
        for i in range(len(best_hps.get('fc_hidden_dims_concat'))):
            print(f"  Dense Layer {i+1}:")
            print(f"    Units: {best_hps.get('fc_hidden_dims_concat')[i]}") 
            if best_hps.get('fc_batch_norm_flag_concat')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('fc_momentum_concat')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('fc_eps_concat')[i]:.4e}")
            print(f"    Activation: {best_hps.get('fc_activation_concat')[i]}") 
            if best_hps.get('fc_dropout_flag_concat')[i]:
                print(f"    Dropout: {best_hps.get('fc_dropout_rate_concat')[i]:.4f}")  

    print(f"Optimizer: {best_hps.get('optimizer_name')}")
    if best_hps.get('optimizer_name') == 'Adam':
        print(f"  Learning Rate (Adam): {best_hps.get('learning_rate'):.4e}")
        print(f"  Beta 1 (Adam): {best_hps.get('beta1'):.4f}")
        print(f"  Beta 2 (Adam): {best_hps.get('beta2'):.4f}")
        print(f"  Weight Decay(Adam): {best_hps.get('weight_decay'):.4e}")
    elif best_hps.get('optimizer_name') == 'SGD':
        print(f"  Learning Rate (SGD): {best_hps.get('learning_rate'):.4e}")
        print(f"  Momentum (SGD): {best_hps.get('momentum'):.4f}")
        print(f"  Weight Decay (SDG): {best_hps.get('weight_decay'):.4e}")
    elif best_hps.get('optimizer_name') == 'RMSprop':
        print(f"  Learning Rate (RMSprop): {best_hps.get('learning_rate'):.4e}")
        print(f"  Weight Decay (RMSprop): {best_hps.get('weight_decay'):.4e}")
        print(f"  Momentum (RMSprop): {best_hps.get('momentum_rms'):.4f}")
        print(f"  Alpha (RMSprop): {best_hps.get('alpha'):.4f}")
        print(f"  Eps (RMSprop): {best_hps.get('eps_rms'):.4e}")
    
    print(f"Learning Rate Schedule: {best_hps.get('lr_scheduler_name')}")
    if best_hps.get('lr_scheduler_name') == 'StepLR':
        print(f"  Step Size: {best_hps.get('step_size'):.4f}")
        print(f"  Gamma: {best_hps.get('stepLRgamma'):.4f}")
    elif best_hps.get('lr_scheduler_name') == 'ExponentialLR':
        print(f"  Gamma: {best_hps.get('exLRgamma'):.4f}")
    elif best_hps.get('lr_scheduler_name') == 'ReduceLROnPlateau':
        print("   Mode: Max")
        print(f"  Factor: {best_hps.get('factor'):.4f}")
        print(f"  Patience: {best_hps.get('lr_patience'):.4f}")
        print(f"  Threshold: {best_hps.get('lr_threshold'):.4e}")
        print(f"  Eps: {best_hps.get('lr_eps'):.4e}")
    elif best_hps.get('lr_scheduler_name') == 'PolynomialLR':
        print(f"  Total Iters: {best_hps.get('total_iters')}")
        print(f"  Power: {best_hps.get('power'):.4f}")
    elif best_hps.get('lr_scheduler_name') ==  'CosineAnnealingLR':
        print(f"  T_max: {best_hps.get('T_max'):.4f}")
        print(f"  Eta_min: {best_hps.get('eta_min'):.4e}")
    elif best_hps.get('lr_scheduler_name') ==  'CyclicLR':
        print(f"  Base_lr: {best_hps.get('base_lr'):.4e}")
        print(f"  Max_lr: {best_hps.get('max_lr_cy'):.4e}")
        print(f"  Step_size_up: {best_hps.get('step_size_up'):.4f}")
    elif best_hps.get('lr_scheduler_name')  == 'OneCycleLR':
        print(f"  Max_lr: {best_hps.get('max_lr_1'):.4e}")
        print(f"  Total_steps: {best_hps.get('total_steps')}")
        print(f"  Pct_start: {best_hps.get('pct_start'):.4f}") 
    print('Loss function:', best_hps.get('loss_function'))
    print('l1 lambda:', f"{best_hps.get('l1_lambda'):.4e}")

def print_best_hp_graphconv_im(model_name, model_save_path, device):
    best_hps = torch.load(model_save_path, map_location = device)
    print(f"Output Size: {best_hps.get('output_dim')}")
    print('Best batch size:', best_hps.get('batch_size'))
    print('Best epoch:', best_hps.get('epoch')+1)
    print(f"Best accuracy: {best_hps.get('best_accuracy'):.4f}")
    print(f"Best loss: {best_hps.get('best_loss'):.4f}")
    print(f"Best loss std: {best_hps.get('best_std_dev'):.4f}")
    print(f"Best F1: {best_hps.get('best_f1'):.4f}")
    print("Best hyperparameters found were:")
    
    if (model_name == "GraphConvModel"):
        print(f"Input Size: {best_hps.get('num_comb_features')}")
    if model_name == "EventSequenceEmbeddingGraphConvModel":
        print(f" Embedding Features Input Size: {best_hps.get('num_embedding_features')}")   
        print(f"Embedding Dimensions: {best_hps.get('embedding_dims')}")
        print(f"Number of Embedding Features GCN layers: {len(best_hps.get('gcn_hidden_dims_embedding'))}")
        for i in range(len(best_hps.get('gcn_hidden_dims_embedding'))):
            print(f"  Embedding Features GCN Layer {i+1}:")
            print(f"    Units: {best_hps.get('gcn_hidden_dims_embedding')[i]}") 
            print(f"    Aggregation Method: {best_hps.get('gcn_aggrs_embedding')[i]}") 
            if best_hps.get('gcn_batch_norm_flag_embedding')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum_embedding')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps_embedding')[i]:.4e}")
            print(f"    Activation: {best_hps.get('gcn_activation_embedding')[i]}") 
            if best_hps.get('gcn_dropout_flag_embedding')[i]:
                print(f"    Dropout: {best_hps.get('gcn_dropout_rate_embedding')[i]:.4f}")  
    if (model_name == "EventSequenceGraphConvModel")  or (model_name == "EventSequenceDurationGraphConvModel") or (model_name == "EventSequenceEmbeddingGraphConvModel"):
        print(f"Event Input Size: {best_hps.get('num_event_features')}")
    print(f"Number of GCN layers: {len(best_hps.get('gcn_hidden_dims'))}")
    for i in range(len(best_hps.get('gcn_hidden_dims'))):
        print(f"  GCN Layer {i+1}:")
        print(f"    Units: {best_hps.get('gcn_hidden_dims')[i]}")
        print(f"    Aggregation Method: {best_hps.get('gcn_aggrs')[i]}") 
        if (model_name == "GraphConvModel") or (model_name == "EventSequenceGraphConvModel"):
            print(f"    Skip Connections: {best_hps.get('gcn_skip_connections')[i]}") 
        if best_hps.get('gcn_batch_norm_flag')[i]:
            print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum')[i]:.4f}")
            print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps')[i]:.4e}")
        print(f"    Activation: {best_hps.get('gcn_activation')[i]}") 
        if best_hps.get('gcn_dropout_flag')[i]:
            print(f"    Dropout: {best_hps.get('gcn_dropout_rate')[i]:.4f}")  
    if model_name == "EventSequenceDurationGraphConvModel":
        print(f" Duration Embedding Input Size: {best_hps.get('num_duration_features')}")   
        print(f"Number of Duration Embedding GCN layers: {len(best_hps.get('gcn_hidden_dims_duration'))}")
        for i in range(len(best_hps.get('gcn_hidden_dims_duration'))):
            print(f"  Duration Embedding GCN Layer {i+1}:")
            print(f"    Units: {best_hps.get('gcn_hidden_dims_duration')[i]}") 
            print(f"    Aggregation Method: {best_hps.get('gcn_aggrs_duration')[i]}") 
            if best_hps.get('gcn_batch_norm_flag_duration')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum_duration')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps_duration')[i]:.4e}")
            print(f"    Activation: {best_hps.get('gcn_activation_duration')[i]}") 
            if best_hps.get('gcn_dropout_flag_duration')[i]:
                print(f"    Dropout: {best_hps.get('gcn_dropout_rate_duration')[i]:.4f}")  
    if (model_name == "EventSequenceDurationGraphConvModel") or (model_name == "EventSequenceEmbeddingGraphConvModel"):   
         print(f"Number of Concatenated GCN layers: {len(best_hps.get('gcn_hidden_dims_concat'))}")
         for i in range(len(best_hps.get('gcn_hidden_dims_concat'))):
             print(f"  Concatenated GCN Layer {i+1}:")
             print(f"    Units: {best_hps.get('gcn_hidden_dims_concat')[i]}") 
             print(f"    Aggregation Method: {best_hps.get('gcn_aggrs_concat')[i]}") 
             print(f"    Skip Connections: {best_hps.get('gcn_skip_connections')[i]}") 
             if best_hps.get('gcn_batch_norm_flag_concat')[i]:
                 print(f"    Batch Norm Momentum: {best_hps.get('gcn_momentum_concat')[i]:.4f}")
                 print(f"    Batch Norm Epsilon: {best_hps.get('gcn_eps_concat')[i]:.4e}")
             print(f"    Activation: {best_hps.get('gcn_activation_concat')[i]}") 
             if best_hps.get('gcn_dropout_flag_concat')[i]:
                print(f"    Dropout: {best_hps.get('gcn_dropout_rate_concat')[i]:.4f}") 
    print(f"Pooling Method: {best_hps.get('pooling_method')}")
    
    if (model_name == "EventSequenceGraphConvModel") or (model_name == "EventSequenceDurationGraphConvModel") or (model_name == "EventSequenceEmbeddingGraphConvModel"):
        print(f"Sequence Input Size: {best_hps.get('num_sequence_features')}")
        print(f"Number of Sequence Dense layers: {len(best_hps.get('fc_hidden_dims'))}")
    elif model_name == "GraphConvModel":
         print(f"Number of Dense layers: {len(best_hps.get('fc_hidden_dims'))}")
    for i in range(len(best_hps.get('fc_hidden_dims'))):
        if model_name == "GraphConvModel":
            print(f"  Dense Layer {i+1}:")
        elif (model_name == "EventSequenceGraphConvModel") or (model_name == "EventSequenceDurationGraphConvModel") or (model_name == "EventSequenceEmbeddingGraphConvModel"):
            print(f"  Sequence Dense Layer {i+1}:")
        print(f"    Units: {best_hps.get('fc_hidden_dims')[i]}") 
        if best_hps.get('fc_batch_norm_flag')[i]:
            print(f"    Batch Norm Momentum: {best_hps.get('fc_momentum')[i]:.4f}")
            print(f"    Batch Norm Epsilon: {best_hps.get('fc_eps')[i]:.4e}")
        print(f"    Activation: {best_hps.get('fc_activation')[i]}") 
        if best_hps.get('fc_dropout_flag')[i]:
            print(f"    Dropout: {best_hps.get('fc_dropout_rate')[i]:.4f}")  

    if (model_name == "EventSequenceGraphConvModel") or (model_name == "EventSequenceDurationGraphConvModel") or (model_name == "EventSequenceEmbeddingGraphConvModel"):
        print(f"Number of Dense layers: {len(best_hps.get('fc_hidden_dims_concat'))}")
    
        for i in range(len(best_hps.get('fc_hidden_dims_concat'))):
            print(f"  Dense Layer {i+1}:")
            print(f"    Units: {best_hps.get('fc_hidden_dims_concat')[i]}") 
            if best_hps.get('fc_batch_norm_flag_concat')[i]:
                print(f"    Batch Norm Momentum: {best_hps.get('fc_momentum_concat')[i]:.4f}")
                print(f"    Batch Norm Epsilon: {best_hps.get('fc_eps_concat')[i]:.4e}")
            print(f"    Activation: {best_hps.get('fc_activation_concat')[i]}") 
            if best_hps.get('fc_dropout_flag_concat')[i]:
                print(f"    Dropout: {best_hps.get('fc_dropout_rate_concat')[i]:.4f}")  

    print(f"Optimizer: {best_hps.get('optimizer_name')}")
    if best_hps.get('optimizer_name') == 'Adam':
        print(f"  Learning Rate (Adam): {best_hps.get('learning_rate'):.4e}")
        print(f"  Beta 1 (Adam): {best_hps.get('beta1'):.4f}")
        print(f"  Beta 2 (Adam): {best_hps.get('beta2'):.4f}")
        print(f"  Weight Decay(Adam): {best_hps.get('weight_decay'):.4e}")
    elif best_hps.get('optimizer_name') == 'SGD':
        print(f"  Learning Rate (SGD): {best_hps.get('learning_rate'):.4e}")
        print(f"  Momentum (SGD): {best_hps.get('momentum'):.4f}")
        print(f"  Weight Decay(SDG): {best_hps.get('weight_decay'):.4e}")
    elif best_hps.get('optimizer_name') == 'RMSprop':
        print(f"  Learning Rate (RMSprop): {best_hps.get('learning_rate'):.4e}")
        print(f"  Weight Decay (RMSprop): {best_hps.get('weight_decay'):.4e}")
        print(f"  Momentum (RMSprop): {best_hps.get('momentum_rms'):.4f}")
        print(f"  Alpha (RMSprop): {best_hps.get('alpha'):.4f}")
        print(f"  Eps (RMSprop): {best_hps.get('eps_rms'):.4e}")
    
    print(f"Learning Rate Schedule: {best_hps.get('lr_scheduler_name')}")
    if best_hps.get('lr_scheduler_name') == 'StepLR':
        print(f"  Step Size: {best_hps.get('step_size'):.4f}")
        print(f"  Gamma: {best_hps.get('stepLRgamma'):.4f}")
    elif best_hps.get('lr_scheduler_name') == 'ExponentialLR':
        print(f"  Gamma: {best_hps.get('exLRgamma'):.4f}")
    elif best_hps.get('lr_scheduler_name') == 'ReduceLROnPlateau':
        print("   Mode: Max")
        print(f"  Factor: {best_hps.get('factor'):.4f}")
        print(f"  Patience: {best_hps.get('lr_patience'):.4f}")
        print(f"  Threshold: {best_hps.get('lr_threshold'):.4e}")
        print(f"  Eps: {best_hps.get('lr_eps'):.4e}")
    elif best_hps.get('lr_scheduler_name') == 'PolynomialLR':
        print(f"  Total Iters: {best_hps.get('total_iters')}")
        print(f"  Power: {best_hps.get('power'):.4f}")
    elif best_hps.get('lr_scheduler_name') ==  'CosineAnnealingLR':
        print(f"  T_max: {best_hps.get('T_max'):.4f}")
        print(f"  Eta_min: {best_hps.get('eta_min'):.4e}")
    elif best_hps.get('lr_scheduler_name') ==  'CyclicLR':
        print(f"  Base_lr: {best_hps.get('base_lr'):.4e}")
        print(f"  Max_lr: {best_hps.get('max_lr_cy'):.4e}")
        print(f"  Step_size_up: {best_hps.get('step_size_up'):.4f}")
    elif best_hps.get('lr_scheduler_name')  == 'OneCycleLR':
        print(f"  Max_lr: {best_hps.get('max_lr_1'):.4e}")
        print(f"  Total_steps: {best_hps.get('total_steps')}")
        print(f"  Pct_start: {best_hps.get('pct_start'):.4f}") 
    print('Loss function:', best_hps.get('loss_function'))
    print('l1 lambda:', f"{best_hps.get('l1_lambda'):.4e}")

def best_trial_path_im(study, model_save_folder):
    
    max_f1 = max(t.value for t in study.trials)
    best_f1_trials = [t for t in study.trials if t.value == max_f1]
    min_loss = min(x.user_attrs['test_loss'] for x in best_f1_trials)
    best_loss_trials = [t for t in best_f1_trials if t.user_attrs['test_loss'] == min_loss]

    #best_trial = min(best_accuracy_trials, key=lambda t: t.user_attrs['loss_dev'])
    best_trial = min(best_loss_trials, key=lambda t: t.user_attrs['loss_dev'])

    best_trial = best_trial.number
    model_save_path = os.path.join(model_save_folder, f'trial_{best_trial}.pth')
    
    return model_save_path
