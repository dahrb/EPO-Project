"""
Experiment Processing Pipeline for EPO Patent Decisions

Builds the final train/test splits used by the ML experiment runner from the
preprocessed Patent Refusal and Opposition Division datasets.

Pipeline:
        1. Resolve data/output paths
        2. Build Patent Refusal splits (Experiment 1 + Experiment 2)
        3. Build Opposition Division splits (Experiment 1 + Experiment 2)
        4. Build Combined (PR + OD) splits (Experiment 1 + Experiment 2)
        5. Clean text (remove HTML) and encode labels (Affirmed=1, Reversed=0)
        6. Save all outputs to Data/Final_Processed

Outputs (saved to Data/Final_Processed):
        - X_Train_{exp}_{name}.pkl
        - y_Train_{exp}_{name}.pkl
        - X_test_{exp}_{name}.pkl
        - y_test_{exp}_{name}.pkl
where:
        - exp ∈ {1, 2}
        - name ∈ {pf, op, both}

Last Updated: 03.04.26

Status: Done
"""

import os
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Paths
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_data_dir() -> Path:
    """Resolve `Data/` directory from current working directory or parent."""
    cwd = Path(os.getcwd()).resolve()
    candidates = [cwd / 'Data', cwd.parent / 'Data']
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return cwd / 'Data'


DATA_DIR = resolve_data_dir()
OUTPUT_DIR = DATA_DIR / 'Final_Processed'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Data dir   : {DATA_DIR}')
print(f'Output dir : {OUTPUT_DIR}')


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Shared Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def RemoveHTMLTags(string: str):
    """Remove basic HTML tags from a text string."""
    return re.compile(r'<[^>]+>').sub(' ', str(string))

def _downsample_majority(data: pd.DataFrame, label_col: str = 'Outcome'):
    """Downsample majority class to match minority class."""
    
    #calc difference
    n_aff = (data[label_col] == 'Affirmed').sum()
    n_rev = (data[label_col] == 'Reversed').sum()
    diff = n_aff - n_rev

    if diff == 0:
        return data.copy(), data.iloc[0:0].copy()

    majority = 'Affirmed' if diff > 0 else 'Reversed'
    n_drop = abs(diff)
    drop_df = data.loc[data[label_col] == majority].sample(n=n_drop, random_state=42)
    balanced = data.drop(drop_df.index)
    
    return balanced, drop_df

def pre_processing_experiments(df: pd.DataFrame,
                               opposition: bool = False,
                               combined: bool = False,
                               experiment: str = '1',
                               year: int = 2020):
    """
    Build train/test splits for Patent Refusal, Opposition Division, or Combined.

    Parameters
    ----------
    df         : DataFrame with columns 'New Summary Facts', 'Outcome', 'Year',
                 and optionally 'Matches' (opposition=True) or
                 'Category' (combined=True).
    opposition : If True, include one-hot-encoded 'Matches' columns in X.
    combined   : If True, include one-hot-encoded `Category` plus opposition
                 sub-match (`OD_Match`, based on `Matches` values 1/2/3).
    experiment : '1' → random stratified split; '2' → temporal split.
    year       : Cut-off year for temporal split (test = year and above).
    """
    
    if opposition and combined:
        raise ValueError("`opposition` and `combined` cannot both be True.")

    feature_cols = ['Category', 'OD_Match'] if combined else (['Matches'] if opposition else None)

    data = df.copy()

    if combined:
        data['OD_Match'] = '0'
        has_matches = 'Matches' in data.columns
        if has_matches:
            op_mask = data['Category'] == 'Opposition Division'
            data.loc[op_mask, 'OD_Match'] = data.loc[op_mask, 'Matches'].astype(str)
    
    #temporal holdout first (Exp2)
    if experiment == '2':
        test = data[data['Year'] >= year].copy()
        train = data[data['Year'] < year].copy()
    else:
        test = None
        train = data.copy()
    
    #balance train pool for BOTH Exp1 and Exp2
    new_data, test_sample = _downsample_majority(train, label_col='Outcome')
    if experiment == '2':
        new_data = new_data.sort_values('Year')

    print(f'Length of dataset after balancing: {len(new_data)}')

    #build feature matrix
    if feature_cols is None:
        X = new_data['New Summary Facts']
    else:
        X = new_data[['New Summary Facts'] + feature_cols].copy()
        dummies = pd.get_dummies(X[feature_cols], prefix=feature_cols).astype(int)
        X = pd.concat([X.drop(columns=feature_cols), dummies], axis=1)

    y = new_data['Outcome']

    #train/test split
    if experiment == '2':
        X_train, y_train = X, y

        if feature_cols is None:
            X_test = test['New Summary Facts']
        else:
            X_test = test[['New Summary Facts'] + feature_cols].copy()
            dummies = pd.get_dummies(X_test[feature_cols], prefix=feature_cols).astype(int)
            X_test = pd.concat([X_test.drop(columns=feature_cols), dummies], axis=1)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        y_test = test['Outcome']
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42)

        if feature_cols is None:
            x_test_2 = test_sample['New Summary Facts']
        else:
            x_test_2 = test_sample[['New Summary Facts'] + feature_cols].copy()
            x_d = pd.get_dummies(x_test_2[feature_cols], prefix=feature_cols).astype(int)
            x_test_2 = pd.concat([x_test_2.drop(columns=feature_cols), x_d], axis=1)
            x_test_2 = x_test_2.reindex(columns=X_train.columns, fill_value=0)

        y_test_2 = test_sample['Outcome']
        X_test = pd.concat([X_test, x_test_2], axis=0)
        y_test = pd.concat([y_test, y_test_2], axis=0)

    #clean HTML, encode labels, save
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    X_train['New Summary Facts'] = X_train['New Summary Facts'].apply(RemoveHTMLTags)
    X_test['New Summary Facts'] = X_test['New Summary Facts'].apply(RemoveHTMLTags)

    y_train_processed = pd.DataFrame([1 if i == 'Affirmed' else 0 for i in y_train])
    y_test_processed = pd.DataFrame([1 if i == 'Affirmed' else 0 for i in y_test])

    if combined:
        name = 'both'
    else:
        name = 'op' if opposition else 'pf'

    X_train.to_pickle(OUTPUT_DIR / f'X_Train_{experiment}_{name}.pkl')
    y_train_processed.to_pickle(OUTPUT_DIR / f'y_Train_{experiment}_{name}.pkl')
    X_test.to_pickle(OUTPUT_DIR / f'X_test_{experiment}_{name}.pkl')
    y_test_processed.to_pickle(OUTPUT_DIR / f'y_test_{experiment}_{name}.pkl')

    print(f'  Saved splits → {OUTPUT_DIR} (suffix _{experiment}_{name})')
    return X_train, y_train, X_test, y_test

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Run All Split Generation
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    
    #Patent Refusal
    pf_path = DATA_DIR / 'Train&TestData_1.0_PatentRefusal.pkl'
    binary_refusal_df = pd.read_pickle(pf_path)
    
    print(f'\nLoaded Patent Refusal data (raw): {len(binary_refusal_df):,} rows')
    
    #Keep only known outcomes
    binary_refusal_df = binary_refusal_df[
        binary_refusal_df['Outcome'].isin(['Affirmed', 'Reversed'])
    ].copy()
    
    print(f'Patent Refusal (known outcomes): {len(binary_refusal_df):,} rows')

    print('\n--- PF Experiment 1 ---')
    pre_processing_experiments(binary_refusal_df, opposition=False, experiment='1')

    print('\n--- PF Experiment 2 (temporal, year >= 2019) ---')
    pre_processing_experiments(binary_refusal_df, opposition=False, experiment='2', year=2020)

    #Opposition Division
    op_path = DATA_DIR / 'Train&TestData_1.0_OppositionDivision.pkl'
    opposition_df = pd.read_pickle(op_path)
    
    print(f'\nLoaded Opposition Division data (raw): {len(opposition_df):,} rows')
    
    #Keep only known outcomes and valid sub-classifications (1, 2, 3 — no 'Other')
    opposition_df = opposition_df[
        opposition_df['Outcome'].isin(['Affirmed', 'Reversed']) &
        opposition_df['Matches'].isin(['1', '2', '3'])
    ].copy()
    
    print(f'Opposition Division (known outcomes, Matches 1/2/3): {len(opposition_df):,} rows')

    print('\n--- OP Experiment 1 ---')
    pre_processing_experiments(opposition_df, opposition=True, experiment='1')

    print('\n--- OP Experiment 2 (temporal, year >= 2019) ---')
    pre_processing_experiments(opposition_df, opposition=True, experiment='2', year=2020)

    #Combined (PR + OD)
    pf_known = binary_refusal_df.copy()  
    op_known = opposition_df.copy()   
    pf_known['Category'] = 'Patent Refusal'
    op_known['Category'] = 'Opposition Division'
    combined_df = pd.concat([pf_known, op_known], ignore_index=True)
    print(f'\nCombined known-outcome rows: {len(combined_df):,}')

    print('\n--- Combined Experiment 1 ---')
    pre_processing_experiments(combined_df, combined=True, experiment='1')

    print('\n--- Combined Experiment 2 (temporal, year >= 2019) ---')
    pre_processing_experiments(combined_df, combined=True, experiment='2', year=2020)

    print('\nAll splits saved to', OUTPUT_DIR)
