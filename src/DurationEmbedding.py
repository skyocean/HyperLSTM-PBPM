import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

def bin_cutoff(event, duration_col, fix_cut, cut_num):
    """
    Bins the duration data into specified number of quantile bins, treating values under a fixed cut-off separately.

    Parameters:
        event (DataFrame): The pandas DataFrame containing the data.
        duration_col (str): The name of the column containing duration values.
        fix_cut (int): Threshold below which each unique value gets its own bin.
        cut_num (int): Number of bins for values above or equal to the threshold.

    Returns:
        DataFrame: The original DataFrame with a new 'bin' column added.
    """
    if event.empty:
        print("Alert: The input DataFrame is empty. No processing will be done.")
        return event

    if not pd.api.types.is_numeric_dtype(event[duration_col]):
        print(f"Alert: The column {duration_col} contains non-numeric values. Binning cannot be performed.")
        return event

    if event[duration_col].isnull().any():
        print(f"Alert: The column {duration_col} contains NaN values. Consider handling them before binning.")
        return event

    if event[duration_col].nunique() == 1:
        print("Alert: All values in the duration column are the same. Binning cannot be performed.")
        return event

    if cut_num < 2:
        print("Alert: The number of bins (cut_num) must be at least 2 for meaningful quantile binning.")
        return event

    unique_below_cut = event[event[duration_col] < fix_cut][duration_col].nunique()
    if unique_below_cut > 50:  # Adjust as needed
        print(f"Alert: There are {unique_below_cut} unique values below the fixed cut-off, resulting in many bins.")
    
    event['bin'] = pd.Series(index=event.index, dtype='object')
    event['bin_range'] = pd.Series(index=event.index, dtype='object')

    # Segment the data into two parts
    mask = event[duration_col] < fix_cut

    # For values under fix_cut, each unique value becomes a bin
    event.loc[mask, 'bin_range'] = event.loc[mask, duration_col].apply(lambda x: f"[{x}, {x+1})")
    event.loc[mask, 'bin'] = event.loc[mask, duration_col].apply(lambda x: f"{str(int(x))}")

    # Calculate quantiles for the rest
    quantiles = np.quantile(event.loc[~mask, duration_col], np.linspace(0, 1, num=cut_num))

    if len(np.unique(quantiles)) != len(quantiles):
        print("Alert: Duplicate quantile values detected, which may cause issues with binning.")

    # Remove duplicates and ensure full coverage of the data range
    unique_quantiles = np.unique(quantiles.astype(int))
    max_value = event.loc[~mask, duration_col].max()
    if unique_quantiles[-1] != max_value:
        unique_quantiles = np.append(unique_quantiles, max_value)

    # Adjust quantiles to ensure full range coverage
    adjusted_quantiles = np.concatenate([[unique_quantiles[0]], np.unique(unique_quantiles[1:] + 1)])

    # Apply these quantiles as bins
    range_labels = [f"[{adjusted_quantiles[i]}, {adjusted_quantiles[i+1]})" for i in range(len(adjusted_quantiles) - 1)]
    bin_labels = [f"{adjusted_quantiles[i]}" for i in range(len(adjusted_quantiles) - 1)]
    event.loc[~mask, 'bin_range'] = pd.cut(event.loc[~mask, duration_col], bins=adjusted_quantiles, labels=range_labels, right=False)
    event.loc[~mask, 'bin'] = pd.cut(event.loc[~mask, duration_col], bins=adjusted_quantiles, labels=bin_labels, right=False)

    # Print the frequency of each bin
    bin_counts = event['bin'].value_counts().sort_index()
    print(bin_counts)

    # Display the cut-off numbers explicitly for bins
    print("Cut-off values for bins:")
    for i in range(len(adjusted_quantiles) - 1):
        print(f"[{adjusted_quantiles[i]}, {adjusted_quantiles[i+1]})")

    return event



def tfidf_embedding(event, case_index, act_col, duration_bin):
    """
    Apply TF-IDF embedding to event data based on a concatenation of activity codes and duration bins,
    using a custom tokenization pattern to handle underscores and custom delimiters.

    Args:
    event (pd.DataFrame): The input DataFrame containing the event data.
    case_index (str): The column name in `event` that serves as the unique case identifier.
    act_col (str): The column name in `event` representing the activity codes.
    duration_bin (str): The column name in `event` representing the duration bins.

    Returns:
    pd.DataFrame: The modified DataFrame with new TF-IDF score columns for each unique duration.
    """
    # Copy the DataFrame to avoid modifying the original data
    event = event.copy()

    # Create documents for TF-IDF by concatenating activity codes and duration bins with a delimiter
    delimiter = '|'
    documents_event = event.groupby(case_index).apply(
        lambda x: ' '.join(x[act_col] + delimiter + x[duration_bin].astype(str))
    ).reset_index(name='document')
    corpus = documents_event['document'].tolist()

    # Setup the TF-IDF vectorizer with a custom tokenization pattern
    vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\|\w+\b')
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
    for duration in np.sort(event[duration_bin].unique()):
        # Initialize with -1 for missing TF-IDF data for later masking
        event[str(duration)] = 0.0

     # Populate TF-IDF scores based on matching concept_duration
    for index, row in event.iterrows():
        # Combine concept and duration to match the TF-IDF feature name
        concept_duration = f"{row[act_col]}|{row[duration_bin]}"
        # Check if the concept_duration is a feature in our TF-IDF matrix
        if concept_duration in feature_names:
            # Retrieve the TF-IDF score for this concept_duration
            tfidf_score = tfidf_scores_scaled_df.loc[row[case_index], concept_duration]
            # Update the corresponding duration column for this row in the original DataFrame
            event.at[index, str(row[duration_bin])] = tfidf_score
    return event

def duration_embedding_layer(event, embedding_cols_names, case_index, eos = True):
    
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
