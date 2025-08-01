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
    event.loc[mask, 'bin_range'] = event.loc[mask, duration_col].apply(lambda x: f"[{x}, {x + 1})")
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
    range_labels = [f"[{adjusted_quantiles[i]}, {adjusted_quantiles[i + 1]})" for i in
                    range(len(adjusted_quantiles) - 1)]
    bin_labels = [f"{adjusted_quantiles[i]}" for i in range(len(adjusted_quantiles) - 1)]
    event.loc[~mask, 'bin_range'] = pd.cut(event.loc[~mask, duration_col], bins=adjusted_quantiles, labels=range_labels,
                                           right=False)
    event.loc[~mask, 'bin'] = pd.cut(event.loc[~mask, duration_col], bins=adjusted_quantiles, labels=bin_labels,
                                     right=False)

    # Print the frequency of each bin
    bin_counts = event['bin'].value_counts().sort_index()
    print(bin_counts)

    # Display the cut-off numbers explicitly for bins
    print("Cut-off values for bins:")
    for i in range(len(adjusted_quantiles) - 1):
        print(f"[{adjusted_quantiles[i]}, {adjusted_quantiles[i + 1]})")

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
    event[duration_bin] = event[duration_bin].apply(lambda x: str(int(x)) if x == int(x) else str(x))

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
    # tfidf_scores_event.fillna(0, inplace=True)

    # Apply Min-Max scaling
    scaler = MinMaxScaler()
    tfidf_scores_scaled = scaler.fit_transform(tfidf_scores_event)
    tfidf_scores_scaled_df = pd.DataFrame(tfidf_scores_scaled, columns=tfidf_scores_event.columns,
                                          index=tfidf_scores_event.index)

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


def duration_embedding_layer(event, embedding_cols_names, case_index, eos=True):
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
    padded_sequences = pad_sequences(encoded_sequences, padding='post', dtype='float32', value=-1)

    return padded_sequences


import numpy as np
import pandas as pd
from collections import Counter


def apply_two_phase_binning(event, duration_col, t_cut, q_bins):
    """
    Apply two-phase binning:
    - T_di < T_cut: individual bins (fine-grained for short durations)
    - T_di >= T_cut: q-quantile binning (equal-sized bins for long durations)

    Args:
        event: DataFrame with duration column
        duration_col: name of duration column
        t_cut: cutoff threshold separating short/long durations
        q_bins: number of quantile bins for long durations (exact number)

    Returns:
        DataFrame with 'bin' column, bin_mapping dict, quantile_ranges list
    """
    # Ensure parameters are integers
    t_cut = int(t_cut)
    q_bins = int(q_bins)

    durations = event[duration_col].values

    # Phase 1: Individual bins for short durations (< T_cut)
    short_mask = durations < t_cut
    short_durations = durations[short_mask]

    # Phase 2: Quantile binning for long durations (>= T_cut)
    long_mask = durations >= t_cut
    long_durations = durations[long_mask]

    bin_mapping = {}
    quantile_ranges = []

    # Phase 1: Map each unique short duration to itself
    unique_short = np.unique(short_durations)
    for duration in unique_short:
        bin_mapping[duration] = duration

    print(f"Phase 1 - Individual bins: {len(unique_short)} unique short durations (< {t_cut})")

    # Phase 2: Create exactly q equal-sized quantile bins for long durations
    if len(long_durations) > 0:
        # Sort long durations for quantile calculation
        sorted_long = np.sort(long_durations)
        n_long = len(sorted_long)

        # Calculate exact quantile boundaries to create q bins
        quantile_boundaries = []
        for i in range(q_bins + 1):
            if i == 0:
                quantile_boundaries.append(sorted_long[0])
            elif i == q_bins:
                quantile_boundaries.append(sorted_long[-1])
            else:
                # Calculate position for quantile
                pos = int((i * n_long) // q_bins)  # Ensure integer division result
                quantile_boundaries.append(sorted_long[pos])

        # Remove duplicates while preserving order
        unique_boundaries = []
        for boundary in quantile_boundaries:
            if not unique_boundaries or boundary != unique_boundaries[-1]:
                unique_boundaries.append(boundary)

        actual_q_bins = len(unique_boundaries) - 1
        print(f"Phase 2 - Quantile bins: {actual_q_bins} bins for long durations (>= {t_cut})")

        # Create bin mapping for quantile bins
        for i in range(actual_q_bins):
            lower_bound = unique_boundaries[i]
            upper_bound = unique_boundaries[i + 1]

            # Find all durations in this quantile range
            if i == actual_q_bins - 1:  # Last bin includes upper boundary
                mask = (long_durations >= lower_bound) & (long_durations <= upper_bound)
            else:
                mask = (long_durations >= lower_bound) & (long_durations < upper_bound)

            durations_in_bin = long_durations[mask]

            # Use the minimum value in the bin as the bin identifier
            if len(durations_in_bin) > 0:
                bin_identifier = int(np.min(durations_in_bin))  # Use actual minimum value in bin
                for duration in np.unique(durations_in_bin):
                    bin_mapping[duration] = bin_identifier

                quantile_ranges.append((int(lower_bound), int(upper_bound)))

    # Apply mapping to create binned column
    event_binned = event.copy()
    event_binned['bin'] = event_binned[duration_col].map(bin_mapping)

    return event_binned, bin_mapping, quantile_ranges


def evaluate_binning_balance(event_binned, t_cut):
    """
    Evaluate how balanced the quantile bins are
    """
    bin_counts = event_binned['bin'].value_counts()

    # Separate individual bins (< t_cut) from quantile bins (>= t_cut)
    individual_bins = bin_counts[bin_counts.index < t_cut].sort_index()
    quantile_bins = bin_counts[bin_counts.index >= t_cut].sort_index()

    balance_score = 0
    if len(quantile_bins) > 1:
        balance_score = quantile_bins.max() - quantile_bins.min()

    return balance_score, individual_bins, quantile_bins


def analyze_data_distribution(event, duration_col):
    """
    Analyze data distribution to suggest reasonable parameter ranges
    """
    durations = event[duration_col]
    duration_counts = durations.value_counts().sort_index()

    print("=== Data Distribution Analysis ===")
    print(f"Total data points: {len(durations):,}")
    print(f"Unique duration values: {durations.nunique():,}")
    print(f"Duration range: {durations.min()} - {durations.max()}")

    # Analyze small values (candidates for individual bins)
    small_value_analysis = {}
    cumulative_count = 0
    cumulative_pct = 0

    print(f"\nSmall value analysis:")
    print(f"{'Value':<8} {'Count':<10} {'Cumulative':<12} {'Cum %':<8}")
    print("-" * 40)

    for val in sorted(duration_counts.index[:20]):  # Check first 20 unique values
        count = duration_counts[val]
        cumulative_count += count
        cumulative_pct = (cumulative_count / len(durations)) * 100

        small_value_analysis[val] = {
            'count': count,
            'cumulative_count': cumulative_count,
            'cumulative_pct': cumulative_pct
        }

        print(f"{val:<8} {count:<10,} {cumulative_count:<12,} {cumulative_pct:<8.1f}%")

        if cumulative_pct > 80:  # Stop if we've covered 80% of data
            break

    return small_value_analysis


def suggest_parameter_ranges(event, duration_col, target_total_bins=None):
    """
    Suggest reasonable parameter ranges based on data distribution
    """
    durations = event[duration_col]
    duration_counts = durations.value_counts().sort_index()
    n_total = len(durations)

    # Analyze distribution
    distribution_analysis = analyze_data_distribution(event, duration_col)

    # Suggest T_cut based on data characteristics
    suggested_t_cuts = []

    # Method 1: Find where cumulative percentage reaches certain thresholds
    thresholds = [5, 10, 15, 20, 25]  # percentages
    for threshold in thresholds:
        cumulative = 0
        for val in sorted(duration_counts.index):
            cumulative += duration_counts[val]
            if (cumulative / n_total) * 100 >= threshold:
                if val <= 20:  # Only consider reasonable small values
                    suggested_t_cuts.append(int(val + 1))  # Ensure integer
                break

    # Method 2: Find values with high individual counts
    high_freq_threshold = n_total * 0.01  # Values with >1% of total data
    high_freq_values = duration_counts[duration_counts >= high_freq_threshold].index
    if len(high_freq_values) > 0:
        suggested_t_cuts.append(int(max(high_freq_values) + 1))  # Ensure integer

    # Method 3: Natural breaks in frequency
    sorted_values = sorted(duration_counts.index[:50])  # Check first 50 values
    for i in range(1, min(len(sorted_values), 20)):
        val = sorted_values[i]
        count = duration_counts[val]
        prev_count = duration_counts[sorted_values[i - 1]]

        # If there's a significant drop in frequency, consider this as cutoff
        if prev_count > count * 3 and count < n_total * 0.005:  # Big drop and small count
            suggested_t_cuts.append(int(val))  # Ensure integer

    # Clean and deduplicate t_cut suggestions
    suggested_t_cuts = sorted(list(set([t for t in suggested_t_cuts if 2 <= t <= 25])))

    # Suggest Q_bins based on remaining data and target total bins
    if target_total_bins is None:
        # Estimate reasonable total bins based on data size
        if n_total < 1000:
            target_total_bins = min(10, durations.nunique())
        elif n_total < 10000:
            target_total_bins = min(20, durations.nunique())
        elif n_total < 100000:
            target_total_bins = min(30, durations.nunique())
        else:
            target_total_bins = min(50, durations.nunique())

    target_total_bins = int(target_total_bins)  # Ensure integer

    suggested_q_bins = []
    for t_cut in suggested_t_cuts:
        remaining_bins = max(5, target_total_bins - t_cut)
        suggested_q_bins.extend([remaining_bins - 2, remaining_bins, remaining_bins + 2])

    # Remove duplicates and filter reasonable ranges, ensure integers
    suggested_q_bins = sorted(list(set([int(q) for q in suggested_q_bins if 5 <= q <= 50])))

    print(f"\n=== Parameter Suggestions ===")
    print(f"Suggested T_cut values: {suggested_t_cuts}")
    print(f"Suggested Q_bins values: {suggested_q_bins}")
    print(f"Target total bins: {target_total_bins}")

    # Provide defaults if no good suggestions found
    if not suggested_t_cuts:
        suggested_t_cuts = [3, 5, 7, 10]
        print("No clear T_cut found, using defaults:", suggested_t_cuts)

    if not suggested_q_bins:
        suggested_q_bins = [10, 15, 20, 25]
        print("No clear Q_bins found, using defaults:", suggested_q_bins)

    return suggested_t_cuts, suggested_q_bins, target_total_bins


def find_optimal_parameters(event, duration_col,
                            t_cut_range=None,  # Will be auto-determined
                            q_bins_range=None,  # Will be auto-determined
                            max_evaluations=50,
                            target_total_bins=None):
    """
    Find optimal T_cut and q parameters for balanced quantile bins
    Now with automatic parameter range detection!
    """
    from itertools import product
    import random

    # Auto-determine parameter ranges if not provided
    if t_cut_range is None or q_bins_range is None:
        print("Auto-detecting optimal parameter ranges...")
        suggested_t_cuts, suggested_q_bins, estimated_total = suggest_parameter_ranges(
            event, duration_col, target_total_bins
        )

        if t_cut_range is None:
            t_cut_range = suggested_t_cuts
        if q_bins_range is None:
            q_bins_range = suggested_q_bins

    # Generate all combinations
    param_combinations = list(product(t_cut_range, q_bins_range))

    # If too many combinations, sample intelligently
    if len(param_combinations) > max_evaluations:
        # Prioritize combinations closer to estimated optimal
        if target_total_bins:
            # Sort by how close total bins are to target
            param_combinations.sort(key=lambda x: abs((x[0] + x[1]) - target_total_bins))
            param_combinations = param_combinations[:max_evaluations]
        else:
            param_combinations = random.sample(param_combinations, max_evaluations)

    best_params = None
    best_balance = float('inf')
    best_result = None

    print(f"Evaluating {len(param_combinations)} parameter combinations...")
    print(f"T_cut range: {min(t_cut_range)} - {max(t_cut_range)}")
    print(f"Q_bins range: {min(q_bins_range)} - {max(q_bins_range)}")

    for i, (t_cut, q_bins) in enumerate(param_combinations):
        if i % 10 == 0:
            print(
                f"  Progress: {i}/{len(param_combinations)} - Current best: T_cut={best_params[0] if best_params else 'None'}, Q_bins={best_params[1] if best_params else 'None'}, Balance={best_balance:.1f}")

        try:
            # Apply binning
            binned_event, mapping, ranges = apply_two_phase_binning(
                event, duration_col, t_cut, q_bins
            )

            # Evaluate balance
            balance_score, individual_bins, quantile_bins = evaluate_binning_balance(
                binned_event, t_cut
            )

            # Multi-criteria scoring: balance + reasonable bin count
            total_quantile_bins = len(quantile_bins)
            if total_quantile_bins < 3:  # Too few quantile bins
                balance_score += 1000
            elif total_quantile_bins > 50:  # Too many quantile bins
                balance_score += 100

            # Check if quantile bins are reasonably balanced
            if balance_score < best_balance:
                best_balance = balance_score
                best_params = (t_cut, q_bins)
                best_result = (binned_event, mapping, ranges)

                # Early stopping if very well balanced
                if balance_score <= 10:
                    print(f"  Early stop: Found excellent solution at T_cut={t_cut}, Q_bins={q_bins}")
                    break

        except Exception as e:
            continue

    if best_result is None:
        print("Warning: No good parameters found, using data-driven defaults")
        # Fallback to middle values
        fallback_t_cut = t_cut_range[len(t_cut_range) // 2] if t_cut_range else 5
        fallback_q_bins = q_bins_range[len(q_bins_range) // 2] if q_bins_range else 15
        return apply_two_phase_binning(event, duration_col, fallback_t_cut, fallback_q_bins)

    print(f"\nOptimal parameters found:")
    print(f"  T_cut: {best_params[0]} (individual bins for values < {best_params[0]})")
    print(f"  Q_bins: {best_params[1]} (quantile bins for values >= {best_params[0]})")
    print(f"  Total bins: ~{best_params[0] + best_params[1]}")
    print(f"  Balance score: {best_balance:.1f}")

    return best_result


def analyze_two_phase_result(binned_event, bin_mapping, quantile_ranges, t_cut):
    """
    Analyze and display two-phase binning results
    """
    bin_counts = binned_event['bin'].value_counts().sort_index()

    # Separate results
    individual_bins = bin_counts[bin_counts.index < t_cut]
    quantile_bins = bin_counts[bin_counts.index >= t_cut]

    print(f"\n=== Two-Phase Binning Results ===")
    print(f"T_cut threshold: {t_cut}")
    print(f"Individual bins (< {t_cut}): {len(individual_bins)} bins")
    print(f"Quantile bins (>= {t_cut}): {len(quantile_bins)} bins")

    print(f"\nBin distribution:")
    print("bin")
    for bin_val, count in bin_counts.items():
        if bin_val < t_cut:
            print(f"{bin_val:>8}     {count:>5}")
        else:
            print(f"{bin_val:>8}     {count:>5}")

    if quantile_ranges:
        print(f"\nQuantile ranges for long durations:")
        for i, (start, end) in enumerate(quantile_ranges, 1):
            print(f"b*{i}: [{start}, {end})")

    # Balance analysis for quantile bins
    if len(quantile_bins) > 1:
        print(f"\nQuantile bin balance analysis:")
        print(f"  Mean size: {quantile_bins.mean():.1f}")
        print(f"  Std deviation: {quantile_bins.std():.1f}")
        print(f"  Min size: {quantile_bins.min()}")
        print(f"  Max size: {quantile_bins.max()}")
        print(f"  Balance score (max-min): {quantile_bins.max() - quantile_bins.min()}")

    print(f"\nIndividual bin summary (short durations < {t_cut}):")
    print(f"  Total count: {individual_bins.sum()}")
    print(f"  Unique values: {len(individual_bins)}")
    if len(individual_bins) > 0:
        print(f"  Range: {individual_bins.index.min()} - {individual_bins.index.max()}")

    return bin_counts


def pseudo_embedding_pipeline_two_phase(event, duration_col, t_cut=5, q_bins=19, optimize=True):
    """
    Main pipeline for two-phase binning

    Args:
        event: DataFrame with duration data
        duration_col: name of duration column
        t_cut: cutoff threshold (default 5 to match your manual cut: 0,1,2,3,4 individual)
        q_bins: number of quantile bins (default 19 to get 24 total bins: 5 individual + 19 quantile)
        optimize: whether to search for optimal parameters
    """
    print("Starting two-phase binning pipeline...")

    if optimize and (t_cut is None or q_bins is None):
        print("Optimizing T_cut and q_bins parameters...")
        binned_event, bin_mapping, quantile_ranges = find_optimal_parameters(event, duration_col)

        # Extract the parameters used
        bin_counts = binned_event['bin'].value_counts()
        individual_bins = bin_counts[bin_counts.index < max(bin_counts.index)]
        if len(individual_bins) > 0:
            t_cut = individual_bins.index.max() + 1
        q_bins = len(quantile_ranges)

    else:
        # Use provided parameters or defaults
        print(f"Using T_cut={t_cut}, q_bins={q_bins} (Total bins: ~{t_cut + q_bins})")
        binned_event, bin_mapping, quantile_ranges = apply_two_phase_binning(
            event, duration_col, t_cut, q_bins
        )

    # Analyze results
    analyze_two_phase_result(binned_event, bin_mapping, quantile_ranges, t_cut)

    return binned_event


def auto_binning_pipeline(event, duration_col, target_total_bins=None, verbose=True):
    """
    Fully automatic binning pipeline for unknown datasets

    Args:
        event: DataFrame with duration data
        duration_col: name of duration column
        target_total_bins: desired total number of bins (will be estimated if None)
        verbose: whether to show detailed analysis

    This function:
    1. Analyzes data distribution
    2. Suggests reasonable parameter ranges
    3. Finds optimal T_cut and Q_bins
    4. Returns binned results
    """
    if verbose:
        print("=== Automatic Binning Pipeline for Unknown Dataset ===")

    # Let the system automatically determine everything
    binned_event, bin_mapping, quantile_ranges = find_optimal_parameters(
        event, duration_col, target_total_bins=target_total_bins
    )

    # Extract final parameters
    bin_counts = binned_event['bin'].value_counts()
    all_bins = sorted(bin_counts.index)

    # Find the cutoff (where individual bins end and quantile bins begin)
    individual_values = []
    quantile_values = []

    for bin_val in all_bins:
        # Check if this bin represents a single duration value
        original_durations = event[event[duration_col].map(bin_mapping) == bin_val][duration_col].unique()
        if len(original_durations) == 1 and original_durations[0] == bin_val:
            individual_values.append(bin_val)
        else:
            quantile_values.append(bin_val)

    detected_t_cut = max(individual_values) + 1 if individual_values else 5
    detected_q_bins = len(quantile_values)

    if verbose:
        print(f"\n=== Final Configuration ===")
        print(f"Detected T_cut: {detected_t_cut}")
        print(f"Detected Q_bins: {detected_q_bins}")
        print(f"Total bins: {len(all_bins)}")
        print(f"Individual bins (< {detected_t_cut}): {len(individual_values)}")
        print(f"Quantile bins (>= {detected_t_cut}): {len(quantile_values)}")

        # Analyze results
        analyze_two_phase_result(binned_event, bin_mapping, quantile_ranges, detected_t_cut)

    return binned_event


def find_natural_small_value_cutoff(durations, total_count, min_percentage_threshold=1.0):
    """
    Dynamically find where "small values with substantial representation" end

    Uses multiple heuristics:
    1. Frequency drop detection (big drops in count)
    2. Percentage threshold (values representing <1% of data)
    3. Cumulative analysis (diminishing returns)
    4. Natural gaps in the sequence
    """
    # Get frequency distribution for small values
    value_counts = {}
    max_check = min(50, int(durations.max()) + 1)  # Don't check beyond 50

    for val in range(1, max_check):
        count = (durations == val).sum()
        if count > 0:
            value_counts[val] = {
                'count': count,
                'percentage': count / total_count * 100
            }

    if not value_counts:
        return 1, {}  # No non-zero values found

    # Sort by value
    sorted_values = sorted(value_counts.keys())

    # Heuristic 1: Find significant frequency drops
    frequency_drops = []
    for i in range(1, len(sorted_values)):
        current_val = sorted_values[i]
        prev_val = sorted_values[i - 1]

        current_count = value_counts[current_val]['count']
        prev_count = value_counts[prev_val]['count']

        # Check for significant drops (current is <50% of previous)
        if current_count < prev_count * 0.5 and prev_count > total_count * 0.01:
            frequency_drops.append(current_val)

    # Heuristic 2: Find where percentage drops below threshold
    percentage_cutoffs = []
    for val in sorted_values:
        if value_counts[val]['percentage'] < min_percentage_threshold:
            percentage_cutoffs.append(val)
            break

    # Heuristic 3: Find cumulative representation cutoff
    cumulative_count = 0
    cumulative_cutoffs = []
    for val in sorted_values:
        cumulative_count += value_counts[val]['count']
        cumulative_percentage = cumulative_count / total_count * 100

        # If we've captured a substantial portion with few values, consider stopping
        if cumulative_percentage > 20 and val <= 10:  # 20% with ≤10 values
            cumulative_cutoffs.append(val + 1)
            break

    # Heuristic 4: Find natural gaps (missing consecutive values)
    gap_cutoffs = []
    for i in range(len(sorted_values) - 1):
        current_val = sorted_values[i]
        next_val = sorted_values[i + 1]

        # If there's a gap of >2 values, consider it a natural break
        if next_val - current_val > 2 and current_val <= 15:
            gap_cutoffs.append(current_val + 1)

    # Combine all heuristics and choose the best cutoff
    all_cutoffs = frequency_drops + percentage_cutoffs + cumulative_cutoffs + gap_cutoffs

    if all_cutoffs:
        # Choose the median cutoff (balanced approach)
        suggested_cutoff = int(np.median(all_cutoffs))
        # But cap it at reasonable range
        suggested_cutoff = max(2, min(suggested_cutoff, 20))
    else:
        # Fallback: find where we have at least 1% representation
        suggested_cutoff = 2
        for val in sorted_values:
            if value_counts[val]['percentage'] >= 1.0:
                suggested_cutoff = val + 1
            else:
                break

    return suggested_cutoff, value_counts


def detect_dominant_zero_pattern(event, duration_col, zero_threshold=0.5):
    """
    Detect if zero values dominate the dataset and suggest binary vs two-phase binning
    Now with dynamic detection of small value cutoffs!

    Args:
        event: DataFrame with duration data
        duration_col: name of duration column
        zero_threshold: threshold for considering zero as dominant (default 50%)

    Returns:
        dict with analysis results and recommendations
    """
    durations = event[duration_col]
    total_count = len(durations)
    zero_count = (durations == 0).sum()
    zero_percentage = zero_count / total_count

    # Analyze non-zero distribution
    non_zero_durations = durations[durations > 0]
    non_zero_unique = non_zero_durations.nunique()

    # DYNAMICALLY find where small values end
    small_value_cutoff, small_value_analysis = find_natural_small_value_cutoff(
        durations, total_count
    )

    # Calculate total representation of detected small values
    small_value_total = sum(v['count'] for k, v in small_value_analysis.items()
                            if k < small_value_cutoff)
    small_value_percentage = small_value_total / total_count

    # Decision logic
    recommendation = {
        'zero_percentage': zero_percentage * 100,
        'zero_count': zero_count,
        'non_zero_unique': non_zero_unique,
        'detected_small_cutoff': small_value_cutoff,
        'small_value_analysis': small_value_analysis,
        'small_value_percentage': small_value_percentage * 100,
        'strategy': None,
        'reasoning': [],
        'suggested_params': {}
    }

    # Rule 1: If zero is overwhelming (>85%), consider binary
    if zero_percentage > 0.85:
        recommendation['strategy'] = 'binary_zero_nonzero'
        recommendation['reasoning'].append(f"Zero overwhelmingly dominates {zero_percentage * 100:.1f}% of data")
        recommendation['suggested_params'] = {'binary_split': True}

    # Rule 2: If zero is dominant but small values have good representation
    elif zero_percentage > zero_threshold:
        # Use the dynamically detected cutoff
        if small_value_percentage > 0.10:  # If small values represent >10% of data
            recommendation['strategy'] = 'two_phase_preserve_small'
            recommendation['reasoning'].extend([
                f"Zero is dominant ({zero_percentage * 100:.1f}%) but values 1-{small_value_cutoff - 1} are substantial ({small_value_percentage * 100:.1f}%)",
                f"Detected natural cutoff at {small_value_cutoff} based on frequency patterns",
                "Preserving individual small values maintains information"
            ])

            recommendation['suggested_params'] = {
                't_cut': small_value_cutoff,
                'q_bins': max(8, min(25, non_zero_unique // 3))
            }
        else:
            recommendation['strategy'] = 'binary_zero_nonzero'
            recommendation['reasoning'].extend([
                f"Zero dominates ({zero_percentage * 100:.1f}%)",
                f"Small non-zero values have low representation ({small_value_percentage * 100:.1f}%)",
                f"Detected small values 1-{small_value_cutoff - 1} are not substantial enough"
            ])
            recommendation['suggested_params'] = {'binary_split': True}

    # Rule 3: Balanced distribution - use standard two-phase
    else:
        recommendation['strategy'] = 'two_phase_standard'
        recommendation['reasoning'].append(f"Balanced distribution (zero: {zero_percentage * 100:.1f}%)")
        # Use existing logic for parameter detection
        recommendation['suggested_params'] = {
            't_cut': None,  # Will be auto-detected
            'q_bins': None  # Will be auto-detected
        }

    return recommendation


def apply_binary_zero_nonzero_binning(event, duration_col):
    """
    Apply simple binary binning: zero vs non-zero
    """
    event_binned = event.copy()
    event_binned['bin'] = (event_binned[duration_col] > 0).astype(int)

    bin_mapping = {0: 0}  # Zero maps to bin 0
    # All non-zero values map to bin 1
    non_zero_values = event[event[duration_col] > 0][duration_col].unique()
    for val in non_zero_values:
        bin_mapping[val] = 1

    return event_binned, bin_mapping, []


def smart_auto_binning_pipeline(event, duration_col,
                                zero_dominance_threshold=0.6,
                                verbose=True):
    """
    Smart automatic binning that handles different data patterns:
    1. Zero-dominated datasets
    2. Balanced datasets
    3. Datasets with significant small values

    Args:
        event: DataFrame with duration data
        duration_col: name of duration column
        zero_dominance_threshold: threshold for considering zero dominant
        verbose: show detailed analysis
    """
    if verbose:
        print("=== Smart Auto-Binning Pipeline ===")
        print("Analyzing data pattern...")

    # Step 1: Detect data pattern
    pattern_analysis = detect_dominant_zero_pattern(
        event, duration_col, zero_dominance_threshold
    )

    if verbose:
        print(f"\n=== Data Pattern Analysis ===")
        print(f"Zero percentage: {pattern_analysis['zero_percentage']:.1f}%")
        print(f"Non-zero unique values: {pattern_analysis['non_zero_unique']}")
        print(f"Detected small values cutoff: {pattern_analysis.get('detected_small_cutoff', 'N/A')}")
        print(
            f"Small values (1-{pattern_analysis.get('detected_small_cutoff', 1) - 1}) represent: {pattern_analysis.get('small_value_percentage', 0):.1f}% of data")
        print(f"Recommended strategy: {pattern_analysis['strategy']}")
        print("Reasoning:")
        for reason in pattern_analysis['reasoning']:
            print(f"  - {reason}")

        # Show the detected small value analysis
        if pattern_analysis.get('small_value_analysis'):
            print(f"\nDetected small value frequencies:")
            for val, info in sorted(pattern_analysis['small_value_analysis'].items())[:10]:
                print(f"  Value {val}: {info['count']:,} samples ({info['percentage']:.1f}%)")

    # Step 2: Apply recommended strategy
    if pattern_analysis['strategy'] == 'binary_zero_nonzero':
        if verbose:
            print(f"\n=== Applying Binary Zero/Non-Zero Binning ===")

        binned_event, bin_mapping, quantile_ranges = apply_binary_zero_nonzero_binning(
            event, duration_col
        )

        if verbose:
            bin_counts = binned_event['bin'].value_counts().sort_index()
            print(f"Bin 0 (zero): {bin_counts[0]:,} samples")
            print(f"Bin 1 (non-zero): {bin_counts[1]:,} samples")
            print(f"Balance ratio: {max(bin_counts) / min(bin_counts):.1f}:1")

    elif pattern_analysis['strategy'] == 'two_phase_preserve_small':
        if verbose:
            print(f"\n=== Applying Two-Phase Binning (Preserve Small Values) ===")

        params = pattern_analysis['suggested_params']

        binned_event, bin_mapping, quantile_ranges = apply_two_phase_binning(
            event, duration_col, params['t_cut'], params['q_bins']
        )

        if verbose:
            print(f"T_cut: {params['t_cut']} (individual bins for 0 to {params['t_cut'] - 1})")
            print(f"Q_bins: {params['q_bins']} (quantile bins for {params['t_cut']}+)")

    else:  # two_phase_standard
        if verbose:
            print(f"\n=== Applying Standard Two-Phase Binning ===")

        binned_event, bin_mapping, quantile_ranges = find_optimal_parameters(
            event, duration_col
        )

    # Step 3: Final analysis
    if verbose:
        print(f"\n=== Final Binning Results ===")
        bin_counts = binned_event['bin'].value_counts().sort_index()
        total_bins = len(bin_counts)
        print(f"Total bins created: {total_bins}")
        print(f"Bin distribution:")
        for bin_val, count in bin_counts.items():
            percentage = count / len(binned_event) * 100
            print(f"  Bin {bin_val}: {count:,} samples ({percentage:.1f}%)")

        # Balance analysis
        if total_bins > 2:
            balance_score = bin_counts.max() - bin_counts.min()
            print(f"Balance score (max-min): {balance_score}")

        print(f"\n✅ Strategy used: {pattern_analysis['strategy']}")

    return binned_event, pattern_analysis


# Integration with existing code
def enhanced_auto_binning_pipeline(event, duration_col,
                                   zero_dominance_threshold=0.6,
                                   target_total_bins=None,
                                   verbose=True):
    """
    Enhanced version of your auto_binning_pipeline that handles zero-dominated data

    This function first analyzes the data pattern, then chooses the best binning strategy:
    - Binary (zero/non-zero) for extremely zero-dominated data
    - Two-phase with small value preservation for moderately zero-dominated data
    - Standard two-phase optimization for balanced data
    """
    if verbose:
        print("=== Enhanced Auto-Binning Pipeline ===")

    # Use the smart pipeline
    binned_event, pattern_analysis = smart_auto_binning_pipeline(
        event, duration_col, zero_dominance_threshold, verbose
    )

    return binned_event


