import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

def custom_concat(event, columns, separator="_", na_val='<NO_DESC>'):
    """
    Concatenate specified columns in a DataFrame based on given conditions.

    This function concatenates values from multiple columns in each row of the DataFrame,
    joining them with a specified separator, but only includes values that are not equal
    to a specified 'na_val' and are not NaN.

    Parameters:
    - event (pd.DataFrame): The DataFrame containing the data to concatenate.
    - columns (list of str): List of column names to include in the concatenation.
    - separator (str): The string to use between values in the concatenated output.
      Default is "_".
    - na_val (str): The value to treat as 'not available' which should be excluded
      from the concatenation. Default is '<NO_DESC>'.

    Returns:
    - pd.Series: A pandas Series object representing the concatenated string from
      the specified columns for each row.
    """
    # Define the concatenation directly in the apply function
    concat_features = event.apply(
        lambda row: separator.join([str(row[col]) for col in columns if row[col] != na_val and pd.notna(row[col])]),
        axis=1)
    return concat_features



def tfidf_embedding_features(event, case_index, act_col, features, separator, na_val):
    """
    Generates a TF-IDF embedding for event data by first concatenating specified features with a custom separator,
    then using this concatenated string to create a TF-IDF vectorized representation grouped by case index.

    Args:
        event (pd.DataFrame): The input DataFrame containing the event data.
        case_index (str): The column name in `event` that serves as the unique case identifier.
        act_col (str): The column name in `event` representing the activity codes.
        features (list of str): List of column names to be concatenated for TF-IDF vectorization.
        separator (str): Separator string used to concatenate the features.
        na_val (str): A placeholder value in `features` that should not be included in the concatenation.

    Returns:
        pd.DataFrame: The original DataFrame augmented with TF-IDF scores for each unique concatenated feature.

    Example:
        Assuming `event` is a DataFrame with columns 'activity', 'duration_bin', 'case_id':
        >>> tfidf_embedding_features(event, 'case_id', 'activity', ['resource', 'dummyf'], '_', '<NO_DESC>')
    """
    # Copy the DataFrame to avoid modifying the original data
    event = event.copy()
    event['concate_features'] = custom_concat(event, features, separator, na_val)

    # Create documents for TF-IDF by concatenating activity codes and  features with a delimiter
    
    documents_event = event.groupby(case_index).apply(
        lambda x: ' '.join(x[act_col] + "|" + x['concate_features'].astype(str))
    ).reset_index(name='document')
    corpus = documents_event['document'].tolist()

    # Setup the TF-IDF vectorizer with a custom tokenization pattern
    vectorizer = TfidfVectorizer(lowercase = False, token_pattern=r'(?u)\b\w+\|\w+\b')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores_event = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=documents_event[case_index])

    # Handle NaN values by filling with zero
    #tfidf_scores_event.fillna(0, inplace=True)

    # Apply Min-Max scaling
    scaler = MinMaxScaler()
    tfidf_scores_scaled = scaler.fit_transform(tfidf_scores_event)
    tfidf_scores_scaled_df = pd.DataFrame(tfidf_scores_scaled, columns=tfidf_scores_event.columns, index=tfidf_scores_event.index)

    # Initialize TF-IDF score columns
    for feature_code in np.sort(event['concate_features'].unique()):
        # Initialize with 0 for missing TF-IDF data for later masking
        event[str(feature_code)] = 0.0

     # Populate TF-IDF scores based on matching concept_duration
    for index, row in event.iterrows():
        # Combine concept and duration to match the TF-IDF feature name
        concept_feature = f"{row[act_col]}|{row['concate_features']}"
        # Check if the concept_duration is a feature in our TF-IDF matrix
        if concept_feature in feature_names:
            # Retrieve the TF-IDF score for this concept_duration
            tfidf_score = tfidf_scores_scaled_df.loc[row[case_index], concept_feature]
            # Update the corresponding duration column for this row in the original DataFrame
            event.at[index, str(row['concate_features'])] = tfidf_score
    return event



def feature_embedding_layer(event, embedding_cols_names, case_index, eos = True):
    
    # Prepare sequences
    encoded_sequences = []
    combined_features_bulk = event[embedding_cols_names].to_numpy()
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