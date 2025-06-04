from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, StandardScaler
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

import numpy as np
import pandas as pd

def custom_onehot_encode(data, categorical_columns, missing_value):
    """
    Custom one-hot encoding to handle '<no_desc>' as a missing category, encoding it differently.

    Args:
    data (DataFrame): The DataFrame containing the data.
    categorical_columns (list): The names of the categorical columns to encode.
    missing_value (str): The placeholder in the data that indicates missing values.

    Returns:
    Tuple[numpy.ndarray, OneHotEncoder]: The custom one-hot encoded matrix and the fitted encoder.
    """
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit and transform the data
    data_encoded = encoder.fit_transform(data[categorical_columns])

    # Create a mask where data is missing_value
    mask = data[categorical_columns] == missing_value

    # Convert mask to match the shape of data_encoded
    expanded_mask = np.column_stack([mask[col] for col in categorical_columns for _ in range(len(encoder.categories_[categorical_columns.index(col)]))])

    # Apply the mask, setting encoded rows to -1 where the original data was missing
    data_encoded[expanded_mask] = -1

    return data_encoded, encoder

def onehot_encode(data, categorical_columns):
    """
    Perform one-hot encoding on specified categorical columns of a DataFrame.

    Args:
    data (DataFrame): The DataFrame containing categorical data.
    categorical_columns (list): List of column names to be one-hot encoded.

    Returns:
    numpy.ndarray: An array containing the one-hot encoded data.
    """
    encoder = OneHotEncoder(sparse_output=False)
    data_encoded = encoder.fit_transform(data[categorical_columns])
    return data_encoded, encoder

def custom_scale_encode(data, numerical_columns):
    """
    Scales numerical columns of a DataFrame where -1 indicates missing values.

    Args:
    data (DataFrame): DataFrame containing the data data.
    numerical_columns (list): List of column names to be scaled.

    Returns:
    DataFrame: A DataFrame with scaled numerical data, where -1 values are kept intact.
    """

    data = data.copy()
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(index=data.index, columns=numerical_columns)
    
    for col in numerical_columns:
        # Replace -1 with NaN for scaling purposes
        valid_data = data[col].replace(-1, np.nan).dropna()
        if not valid_data.empty:
            # Fit scaler only on valid, non-missing data
            scaler.fit(valid_data.values.reshape(-1, 1))
            # Apply scaling to non-missing values
            data_scaled.loc[data[col] != -1, col] = scaler.transform(data[col][data[col] != -1].values.reshape(-1, 1)).flatten()
    
    # Replace NaNs with -1 in the scaled data
    data_scaled.fillna(-1, inplace=True)

    return data_scaled.values, scaler

def median_scale_encode(data, numerical_columns):
    """
    Scale numerical columns of a DataFrame, handling missing values with median imputation.

    Args:
    data (DataFrame): The DataFrame containing numerical data.
    numerical_columns (list): List of column names to be scaled.

    Returns:
    numpy.ndarray: An array containing the scaled numerical data.
    """
    data = data.copy()
    # Handle -1 as NaN for numerical features and replace with the median
    data[numerical_columns] = data[numerical_columns].replace(-1, np.nan)
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[numerical_columns])
    return data_scaled, scaler

def encode_pad_event_concurr(event, cat_col_event, num_col_event, case_index, start_time_col, cat_mask = False, num_mask = False, eos = True):
    """
    Encode sequence level features and handle co-occurrence by appending up to certain number of co-occurring events
    into a single array within each sequence step.
    
    Parameters:
    - event: DataFrame including event level features.
    - cat_col_event: List of column names for categorical features.
    - num_col_event: List of column names for numerical features; -1 representing NaN.
    - case_index: Column name for sequence index.
    - start_time_col: Column name for the event start time, used to determine co-occurrences.
    
    Returns:
    - padded_sequences: Encoded and padded sequence features ready for model input.
    """   
    # Initialize combined_features_bulk
    combined_features_bulk = np.array([])
    
    # Custom encoding for categorical columns
    if cat_col_event:
        if cat_mask:        
            event_encoded, encoder = custom_onehot_encode(event, cat_col_event, "<NO_DESC>")
        else:        
            event_encoded, encoder = onehot_encode(event, cat_col_event)            
        combined_features_bulk = event_encoded
        
    # Apply median scaling
    if num_col_event:
        if num_mask:
            event_scaled, scaler = custom_scale_encode(event, num_col_event)
        else:            
            event_scaled, scaler = median_scale_encode(event, num_col_event)  
            
        if combined_features_bulk.size == 0:
            combined_features_bulk = event_scaled
        else:
            # Combine encoded categorical data and normalized numerical data
            combined_features_bulk = np.hstack((combined_features_bulk, event_scaled))
    
    # Group by sequence ID and StartTime, then count the events
    group_counts = event.groupby([case_index, start_time_col]).size()
    # Find the maximum number of co-occurring events
    max_co_occur = group_counts.max()
    
    # Prepare sequences with co-occurrence
    encoded_sequences = []    
   
    feature_length = combined_features_bulk.shape[1] * max_co_occur  # Maximum features in a single stack
    if eos:
        eos_token = np.zeros((1, feature_length))

    for _, group in event.groupby(case_index):
        group_features = combined_features_bulk[group.index]
        
        # Stack co-occurring events into the same array
        stacked_features = []
        i = 0
        while i < len(group_features):
            # Start a new stack with the current event
            stack = [group_features[i]]
            count = 1
            # Look ahead to stack up to `max_co_occur` events occurring at the same time
            while i + count < len(group_features) and group.iloc[i + count][start_time_col] == group.iloc[i][start_time_col] and count < max_co_occur:
                stack.append(group_features[i + count])
                count += 1
            # Advance the index by the number of stacked events
            i += count
            # Concatenate stacked events into a single array
            stack_array = np.concatenate(stack, axis=0)
            # Ensure all stacks have the same number of features by padding
            padded_stack = np.pad(stack_array, (0, feature_length - stack_array.size), mode='constant', constant_values = -1.0)
            stacked_features.append(padded_stack)
    
        if eos:
            #Add the <EOS> token to the end of each sequence
            sequence_with_eos = np.vstack([stacked_features, eos_token])
            #Append the processed features of the group to the encoded sequences list
            encoded_sequences.append(sequence_with_eos)
        else:
            encoded_sequences.append(np.array(stacked_features))
    
    # Pad sequences
    padded_sequences = pad_sequences(encoded_sequences, padding='post', dtype='float32', value = -1)
    
    return padded_sequences

def encode_pad_event(event, cat_col_event, num_col_event, case_index, cat_mask=False, num_mask=False, eos = True):
    """
    encode sequence level features
    
    Parameters:
    - event Dateframe including event level features.
    - cat_col_event: List of column names of categorical features 
    - num_col_event: List of column names of numrical features; -1 respenting NAN
    - case_index: column name of sequence index
    
    Returns:
    - Encoding event level features and padding with the same length ready for input to the model.
    """
    

   # Custom encoding for categorical columns
    if cat_col_event:
        if cat_mask:        
            event_encoded, encoder = custom_onehot_encode(event, cat_col_event, "<NO_DESC>")
        else:        
            event_encoded, encoder = onehot_encode(event, cat_col_event)            
        combined_features_bulk = event_encoded
        
    # Apply median scaling
    if num_col_event:
        if num_mask:
            event_scaled, scaler = custom_scale_encode(event, num_col_event)
        else:            
            event_scaled, scaler = median_scale_encode(event, num_col_event)  
            
        if combined_features_bulk.size == 0:
            combined_features_bulk = event_scaled
        else:
            # Combine encoded categorical data and normalized numerical data
            combined_features_bulk = np.hstack((combined_features_bulk, event_scaled))
    
    # Prepare sequences
    encoded_sequences = []

    feature_length = combined_features_bulk.shape[1]  # Maximum features in a single stack
    if eos:
        eos_token = np.zeros((1, feature_length))
        
    for _, group in event.groupby(case_index):
        group_indices = group.index
        group_combined_features = combined_features_bulk[group_indices]

        if eos:
            # Append EOS token
            group_combined_features_with_eos = np.vstack([group_combined_features, eos_token])
            encoded_sequences.append(group_combined_features_with_eos)
        else:
            encoded_sequences.append(group_combined_features)
    
    # Pad sequences
    padded_sequences = pad_sequences(encoded_sequences, padding='post', dtype='float32', value = -1)
    
    return padded_sequences

def encode_pad_sequence(sequence, cat_col_seq, num_col_seq, cat_mask = False, num_mask = False):
    """
    Encode sequence level features.
    
    Parameters:
    - sequence: DataFrame including sequence level features.
    - cat_col_seq: List of column names of categorical features.
    - num_col_seq: List of column names of numerical features; -1 representing NAN.
    
    Returns:
    numpy.ndarray: Array of combined features ready for model input.
    """
    # Initialize combined_features_bulk
    combined_features_bulk = np.array([])

    if cat_col_seq:
        if cat_mask:
             sequence_encoded, encoder = custom_onehot_encode(sequence, cat_col_seq)
        else:
            # Apply one-hot encoding
            sequence_encoded, encoder = onehot_encode(sequence, cat_col_seq)
        combined_features_bulk = sequence_encoded
    
    if num_col_seq:
        if num_mask:
            # Apply median scaling
            sequence_scaled, scaler = custom_scale_encode(sequence, num_col_seq)
        else:            
            sequence_scaled, scaler = median_scale_encode(sequence, num_col_seq)
        if combined_features_bulk.size == 0:
            combined_features_bulk = sequence_scaled
        else:# Combine encoded categorical data and scaled numerical data
            combined_features_bulk = np.hstack((combined_features_bulk, sequence_scaled))
    
    return combined_features_bulk


def encode_y(y_col):        
    # Create a label encoder
    encoder = LabelEncoder()
    
    # Encode the labels (converts strings to integers)
    y_int = encoder.fit_transform(y_col)
    
    # Convert integer encoded labels to one-hot encoding
    y_encode = to_categorical(y_int)

    return y_encode

def encode_textual_event(event, text_col, case_index, eos = True, pad_value = 0):

    tokenizer = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(event[text_col])
    text_bulk = tokenizer.texts_to_sequences(event[text_col])
    vocab_size = len(tokenizer.word_index) + 1  # For embedding layer
    
    # Prepare sequences
    encoded_sequences = []
    text_feature_bulk = np.array(text_bulk)
    feature_length = text_feature_bulk.shape[1]  # Maximum features in a single stack
    
    if eos:
        eos_token = np.zeros((1, feature_length))
        
    for _, group in event.groupby(case_index):
        group_indices = group.index
        group_combined_features = text_feature_bulk[group_indices]

        if eos:
            # Append EOS token
            group_combined_features_with_eos = np.vstack([group_combined_features, eos_token])
            encoded_sequences.append(group_combined_features_with_eos)
        else:
            encoded_sequences.append(group_combined_features)
    
    # Pad sequences
    # for LSTM pad_value should be default and set as 0
    padded_sequences = pad_sequences(encoded_sequences, padding='post', dtype='float32', value = pad_value)
    
    return padded_sequences, vocab_size


