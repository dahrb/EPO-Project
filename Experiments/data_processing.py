"""
Data Processing Pipeline for EPO Patent Decisions

Processes the raw EPO XML file into cleaned, classified datasets for
Patent Refusal and Opposition Division experiments.

Pipeline:
    1. Parse XML → raw DataFrame (via TableCreator / PatentExtract)
    2. Filter & clean (T-court, English, 2000-2024, dedup by ECLI)
    3. Match Legal Provisions → article names
    4. Classify case types (Patent Refusal / Opposition Division /
       Admissibility / Other) via spaCy Matcher
    5. Pre-process text (chop first 35 chars of Summary Facts)
    6. Sub-classify Opposition Division cases by appellant type
       (Patentee / Opponent / Both / Other)
    7. Extract Outcomes per subset (Patent Refusal & Opposition Division)
    8. Save outputs

Outputs (saved to ../Data/):
    - EPO_Data.pkl                          : full cleaned dataset (all case types)
    - Train&TestData_1.0_PatentRefusal.pkl  : Patent Refusal cases with Outcome
    - Train&TestData_2.0_OppositionDivision : Opposition Division cases with Outcome

Last Updated: 02.04.26

Status: Done
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher
from xml.dom import pulldom

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "Data")

#insert your EPO XML here - note this has only been tested with EPDecisions_March2025.xml
XML_PATH = os.path.join(DATA_DIR, "EPDecisions_March2025.xml")

#add project root so we can import Utilities
sys.path.insert(0, PROJECT_DIR)
from Utilities.utils import PatentExtract, TableCreator

# ═══════════════════════════════════════════════════════════════════════════════
# 1. XML Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_xml(xml_path: str) -> pd.DataFrame:
    """
    Parse the EPO decisions XML file into a raw DataFrame.
    Uses xml.dom.pulldom for memory-efficient streaming and
    PatentExtract from Utilities.utils for generic DOM helpers.
    """
    print(f"[1/8] Parsing XML: {xml_path}")
    
    #pre-processes XML data
    table = TableCreator()
    
    doc = pulldom.parse(xml_path)

    for event, node in doc:
        if event == pulldom.START_ELEMENT and node.tagName == "ep-appeal-decision":
            try:
                doc.expandNode(node)
            except Exception:
                pass

            extract = PatentExtract(node)

            #procedure lang
            lang = extract.getAttribute(attr='lang')
            table.append(table.procedure_lang, lang)

            #get court type e.g. T and appeal number
            appeal_num, court_type = extract.multiTagTextExtract(
                tag='ep-case-num', attr='code', inner_tag='ep-appeal-num'
            )
            table.append(table.appeal_num, appeal_num)
            table.append(table.court_type, court_type)

            #ECLI
            ecli = extract.singleTagTextExtract(tag='ep-ecli')
            table.append(table.ecli, ecli)

            #get summary of facts
            sum_facts, fact_lang = extract.multiTagTextExtract(
                tag='ep-summary-of-facts', attr='lang', inner_tag='p'
            )
            table.append(table.summary_facts, sum_facts)
            table.append(table.fact_lang, fact_lang)

            #get reason for decision
            reasons = extract.multiTagTextExtract(
                tag='ep-reasons-for-decision', inner_tag='p'
            )
            table.append(table.decision_reason, reasons)

            #get legal provisions
            legal_prov = extract.multiTagTextExtract(
                tag='ep-legal-citation', inner_tag='ep-legal-ref-presentation'
            )
            table.append(table.legal_provisions, legal_prov)

            #get order
            order = extract.multiTagTextExtract(
                tag='ep-appeal-order', inner_tag='p'
            )
            table.append(table.order, order)

            #date
            date = extract.multiTagTextExtract(
                tag='ep-date-of-decision', inner_tag='date'
            )
            table.append(table.date, date)

            #invention title
            title = extract.singleTagTextExtract(tag='invention-title')
            table.append(table.title, title)

            #board of appeal code i.e. 3.2.03
            code = extract.singleTagTextExtract(tag='ep-board-of-appeal-code')
            table.append(table.board_code, code)

            #patent keywords
            keywords = extract.multiTagTextExtract(
                tag='ep-keywords', inner_tag='keyword'
            )
            table.append(table.keywords, keywords)

            #references cited in text
            _, reference = extract.multiTagTextExtract(
                tag='ep-appeal-bib-data', attr='reference', inner_tag=None
            )
            table.append(table.reference, reference)

            #classification 
            classification = extract.classificationExtract()
            table.append(table.classification, classification)

            #app number
            app_no = extract.singleTagTextExtract(tag='doc-number')
            table.append(table.appno, app_no)

            #cited decisions
            cited = extract.cited_decisions()
            table.append(table.cited, cited)

    table.create_table()
    print(f"       Parsed {len(table.df):,} records")
    return table.df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Cleaning & Filtering
# ═══════════════════════════════════════════════════════════════════════════════

def _flatten_list_col(series: pd.Series) -> pd.Series:
    """Flatten single-element lists to scalars, multi-element to joined str."""
    def _flat(val):
        if isinstance(val, list):
            if len(val) == 0:
                return np.nan
            if len(val) == 1:
                return val[0]
            return val  # keep as list (e.g. Legal Provisions)
        return val
    return series.apply(_flat)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame:
      1. Flatten list columns → join to strings
      2. Fix dates from ECLI when too short
      3. Convert dates to datetime, extract Year
      4. Replace empty strings with NaN
      5. Filter: Court Type T, English (Procedure Language), Year 2000-2024,
         Order not null
      6. De-duplicate by ECLI
    """
    print("[2/8] Cleaning DataFrame …")

    #1. Flatten all columns that may contain lists → join to strings
    cols_to_join = [
        "Reference", "Procedure Language", "Court Type", "Appeal Number",
        "Document Language", "Summary Facts", "Decision Reasons", "Order",
        "Date", "ECLI", "Invention Title", "Board Code", "Classification",
        "Keywords", "App No",
    ]
    for col in cols_to_join:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: " ".join(x) if isinstance(x, list) else x
            )

    #2. Fix dates by taking from the end of the ECLI number when too short
    df["Date"] = df.apply(
        lambda row: row["ECLI"].split(".")[1]
        if isinstance(row["ECLI"], str)
        and isinstance(row["Date"], str)
        and len(row["Date"]) < 8
        else row["Date"],
        axis=1,
    )

    #3. Convert dates to datetime and extract Year
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")

    #4. Replace empty / whitespace-only strings with NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    df["Year"] = df["Date"].dt.year

    #5. Filter: Technical Board (T), English, Year 2000-2024, Order present
    df = df[
        (df["Court Type"] == "T")
        & (df["Procedure Language"] == "en")
        & (df["Year"] >= 2000)
        & (df["Year"] < 2025)
        & (df["Order"].notna())
    ].copy()
    print(f"{len(df):,} T-court English records (2000-2024) with Order")

    #6. De-duplicate by ECLI (notebook approach)
    dups = df[df.duplicated(subset=["ECLI"], keep=False)]
    dups_no_title = dups[dups["Invention Title"].isna()]
    df = df.drop(dups_no_title.index)
    df = df[df["ECLI"].str.len() >= 20]
    df = df.drop_duplicates(subset=["ECLI"], keep="last")

    print(f"{len(df):,} after ECLI de-duplication")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Legal-Provision → Article matching
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_provision_list(val) -> list:
    """Safely convert Legal Provisions to a list of strings."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [val]
    if pd.isna(val):
        return []
    return []

def match_legal_provisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process Legal Provisions and create 'Matched Articles' column.
      1. Filter provisions containing 'Art' and 'EPC'
      2. Split on 'Art' to get code portion
      3. Handle '_(2007)' suffix
      4. Strip leading underscore, remove empty / too-long / 'icle' items
      5. Map to article names
    """
    print("[3/8] Matching Legal Provisions → Articles …")

    def _split_item(item):
        try:
            return item.split('_(2007)')[0]
        except IndexError:
            return item

    def _process_provisions(provisions):
        provisions = _safe_provision_list(provisions)
        #Step 1: filter for 'Art' and 'EPC'
        provisions = [item for item in provisions if 'Art' in item and 'EPC' in item]
        #Step 2: split on 'Art' to get code portion
        provisions = [item.split('Art')[1] for item in provisions]
        #Step 3: handle '_(2007)' suffix
        provisions = [_split_item(i) for i in provisions]
        #Step 4: remove empty strings
        provisions = [item for item in provisions if item != '']
        #Step 5: remove too-long strings
        provisions = [item for item in provisions if len(item) < 15]
        #Step 6: remove 'icle' remnants
        provisions = [item for item in provisions if 'icle' not in item]
        #Step 7: strip leading underscore → get code like '054'
        provisions = [item.split('_')[1] if '_' in item else item for item in provisions]
        return provisions

    df["Legal Provisions"] = df["Legal Provisions"].apply(_process_provisions)

    #now map to article names (notebook approach)
    articles = {
        "052(2)": "Subject Matter", "052(3)": "Subject Matter",
        "052(4)": "Subject Matter", "053": "Subject Matter",
        "054": "Novelty", "055": "Novelty",
        "056": "Inventive Step",
        "057": "Industrial Applicability",
    }

    def _match_articles(provisions):
        matched = [articles[prov] for prov in provisions if prov in articles]
        seen = set()
        unique = []
        for m in matched:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        return unique if unique else ["Other"]

    df["Matched Articles"] = df["Legal Provisions"].apply(_match_articles)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Case-type classification via spaCy Matcher
# ═══════════════════════════════════════════════════════════════════════════════

def _build_case_matcher(nlp):
    """
    Build a single spaCy Matcher for Patent Refusal, Opposition Division,
    and Admissibility.
    """
    matcher = Matcher(nlp.vocab)

    # Opposition Division
    matcher.add("Opposition Division", [
        [{"LOWER": "opposition"}, {"LOWER": "division"}],
    ])

    # Admissibility  (restricted … admissibility)
    matcher.add("Admissibility", [
        [{"LOWER": "restricted"}, {"OP": "*"}, {"OP": "*"},
         {"LOWER": {"FUZZY": "admissibility"}}],
    ])

    # Patent Refusal — uses LEMMA so "refused"/"refusal"/"refusing" all map to "refuse"
    matcher.add("Patent Refusal", [
        [{"LEMMA": "refuse"}, {"OP": "*"},
         {"LOWER": "european", "OP": "*"},
         {"LOWER": "patent", "OP": "*"},
         {"LOWER": "application"}],
    ])

    return matcher


def _get_case_match(doc, matcher, nlp):
    """Classify a single spaCy Doc using the matcher — mirrors notebook logic."""
    match = matcher(doc)
    if match:
        string_ids = []
        for match_id, start, end in match:
            string_ids.append(nlp.vocab.strings[match_id])
        if "Admissibility" in string_ids:
            return "Admissibility"
        elif "Opposition Division" in string_ids:
            return "Opposition Division"
        elif "Patent Refusal" in string_ids:
            return "Patent Refusal"
    return "Other"


def _add_custom_lemma_pipe(nlp):
    """Add a custom component that normalises refuse/refusal/refused/refusing lemmas."""
    from spacy.language import Language

    custom_lookup = {
        "refusal": "refuse", "refusing": "refuse",
        "refuse": "refuse", "refused": "refuse",
    }

    @Language.component("change_lemma")
    def change_lemma_property(doc):
        for token in doc:
            if token.text in custom_lookup:
                token.lemma_ = custom_lookup[token.text]
        return doc

    if "change_lemma" not in nlp.pipe_names:
        nlp.add_pipe("change_lemma", first=True)


def classify_cases(df: pd.DataFrame, nlp) -> pd.DataFrame:
    """
    Classify each case as Patent Refusal, Opposition Division,
    Admissibility, or Other using spaCy Matcher on Summary Facts.
    
    custom lemma component, then matches.
    """
    print("[4/8] Classifying case types via spaCy Matcher …")

    #add custom lemma normalisation for refuse/refusal etc.
    _add_custom_lemma_pipe(nlp)

    matcher = _build_case_matcher(nlp)

    #strip HTML before tokenising
    texts = df["Summary Facts"].apply(
        lambda x: _HTML_RE.sub(" ", str(x))
    ).tolist()

    matches_list = []
    nlp_docs = []

    for doc in nlp.pipe(texts, batch_size=500):
        nlp_docs.append(doc)
        matches_list.append(_get_case_match(doc, matcher, nlp))

    df["nlp"] = nlp_docs
    df["Matches"] = matches_list

    print(f"       Case-type distribution:")
    for k, v in df["Matches"].value_counts().items():
        print(f"         {k:25s} {v:>6,}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Text pre-processing (HTML removal, cleaning)
# ═══════════════════════════════════════════════════════════════════════════════

_HTML_RE = re.compile(r"<[^>]+>")


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'New Summary Facts' by chopping the first 35 characters
    (standard preamble) from Summary Facts — matches the notebook.
    """
    print("[5/8] Pre-processing text …")

    df["New Summary Facts"] = df["Summary Facts"].apply(lambda x: str(x)[35:])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Opposition Division sub-classification (appellant type)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_opposition_subclass_matcher(nlp):
    """
    Build a spaCy Matcher to sub-classify Opposition Division cases by
    who filed the appeal: Patentee ("1"), Opponent ("2"), Both ("3").
    """
    matcher = Matcher(nlp.vocab)

    #Both ("3")
    Both = [{"LOWER": {"FUZZY": "patent"}}, {"LOWER": "proprietor"}, {"LOWER": "and"}, {"OP": "{,5}"}, {"LOWER": {"FUZZY": "opponent"}}]
    Both2 = [{"LOWER": {"FUZZY": "proprietor"}}, {"OP": "{,3}"}, {"LOWER": "("}, {"OP": "{,5}"}, {"LOWER": {"FUZZY": "appellant"}}, {"OP": "{,5}"}, {"LOWER": ")"}, {"OP": "{,10}"},
             {"LOWER": {"FUZZY": "opponent"}}, {"OP": "{,3}"}, {"LOWER": "("}, {"OP": "{,5}"}, {"LOWER": {"FUZZY": "appellant"}}, {"OP": "{,5}"}, {"LOWER": ")"}]
    Both3 = [{"LOWER": {"FUZZY": "opponent"}}, {"OP": "{,3}"}, {"LOWER": {"FUZZY": "proprietor"}}, {"LOWER": {"FUZZY": "appeal"}}]
    matcher.add("3", [Both, Both2, Both3])

    #Opponent ("2")
    Opponent  = [{"LOWER": {"FUZZY": "appellant"}}, {"OP": "{,3}"}, {"LOWER": "("}, {"OP": "{,5}"}, {"LOWER": {"FUZZY": "opponent"}}, {"OP": "{,5}"}, {"LOWER": ")"}]
    Opponent2 = [{"LOWER": {"FUZZY": "opponent"}}, {"OP": "{,3}"}, {"LOWER": "("}, {"OP": "{,5}"}, {"LOWER": {"FUZZY": "appellant"}}, {"OP": "{,3}"}, {"LOWER": ")"}]
    Opponent3 = [{"LOWER": {"FUZZY": "opponent"}}, {"OP": "{,3}"}, {"LOWER": {"FUZZY": "appeal"}}]
    Opponent4 = [{"LOWER": {"FUZZY": "appeal"}}, {"LOWER": "of"}, {"LOWER": {"FUZZY": "opponent"}}]
    matcher.add("2", [Opponent, Opponent2, Opponent3, Opponent4])

    #Patentee ("1")
    Patentee  = [{"LOWER": {"FUZZY": "appellant"}}, {"OP": "{,3}"}, {"LOWER": "("}, {"OP": "{,5}"}, {"LOWER": {"FUZZY": "patent"}}, {"OP": "{,3}"}, {"LOWER": ")"}]
    Patentee2 = [{"LOWER": {"FUZZY": "proprietor"}}, {"OP": "{,3}"}, {"LOWER": "("}, {"OP": "{,5}"}, {"LOWER": {"FUZZY": "appellant"}}, {"OP": "{,3}"}, {"LOWER": ")"}]
    Patentee3 = [{"LOWER": {"FUZZY": "appellant"}}, {"OP": "{,3}"}, {"LOWER": "("}, {"OP": "{,5}"}, {"LOWER": {"FUZZY": "proprietor"}}, {"OP": "{,3}"}, {"LOWER": ")"}]
    Patentee4 = [{"LOWER": {"FUZZY": "patent"}}, {"OP": "{,3}"}, {"LOWER": "("}, {"OP": "{,5}"}, {"LOWER": {"FUZZY": "appellant"}}, {"OP": "{,3}"}, {"LOWER": ")"}]
    Patentee5 = [{"LOWER": {"FUZZY": "proprietor"}}, {"OP": "{,3}"}, {"LOWER": {"FUZZY": "appeal"}}]
    Patentee6 = [{"LOWER": {"FUZZY": "appeal"}}, {"LOWER": "by"}, {"LOWER": "the"}, {"LOWER": {"FUZZY": "proprietor"}}]
    matcher.add("1", [Patentee, Patentee2, Patentee3, Patentee4, Patentee5, Patentee6])

    return matcher


def _get_opposition_subclass(doc, matcher, nlp):
    """Return '1' (Patentee), '2' (Opponent), '3' (Both), or 'Other'."""
    match = matcher(doc)
    if match:
        string_ids = [nlp.vocab.strings[mid] for mid, s, e in match]
        if "3" in string_ids:
            return "3"
        elif "2" in string_ids and "1" in string_ids:
            return "3"
        elif "2" in string_ids:
            return "2"
        elif "1" in string_ids:
            return "1"
    return "Other"


def classify_opposition_subcases(df: pd.DataFrame, nlp) -> pd.DataFrame:
    """
    Sub-classify Opposition Division cases by appellant type.
    Overwrites the 'Matches' column for OD rows only:
      '1' = Patentee, '2' = Opponent, '3' = Both, 'Other' = unclassified.
    Uses the existing 'nlp' column (spaCy docs from step 4).
    """
    print("[6/8] Sub-classifying Opposition Division cases …")

    od_mask = df["Matches"] == "Opposition Division"
    od_idx = df.index[od_mask]

    matcher = _build_opposition_subclass_matcher(nlp)

    subcats = []
    for idx in od_idx:
        doc = df.at[idx, "nlp"]
        subcats.append(_get_opposition_subclass(doc, matcher, nlp))

    df.loc[od_idx, "Matches"] = subcats

    print("       Opposition Division sub-classification:")
    for k, v in df.loc[od_idx, "Matches"].value_counts().items():
        print(f"         {k:25s} {v:>6,}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Outcome extraction via spaCy Matcher (per-subset)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_outcome_matcher(nlp):
    """
    Build outcome matcher — same patterns for both PR and OD.
    Uses FUZZY matching on 'appeal'.
    """
    matcher = Matcher(nlp.vocab)

    # Affirmed: appeal dismissed / rejected
    matcher.add("Affirmed", [
        [{"LOWER": {"FUZZY": "appeal"}}, {"OP": "*"}, {"LOWER": "dismissed"}],
        [{"LOWER": {"FUZZY": "appeal"}}, {"OP": "*"}, {"LOWER": "rejected"}],
    ])

    # Reversed: set aside
    matcher.add("Reversed", [
        [{"LOWER": {"FUZZY": "appeal"}}, {"OP": "*"},
         {"LOWER": "set"}, {"LOWER": "aside"}],
    ])

    return matcher


def _get_outcome_match(doc, matcher, nlp):
    """Return 'Affirmed', 'Reversed', or 'Unknown' for a single doc."""
    match = matcher(doc)
    if match:
        string_ids = [nlp.vocab.strings[mid] for mid, s, e in match]
        if "Affirmed" in string_ids:
            return "Affirmed"
        elif "Reversed" in string_ids:
            return "Reversed"
    return "Unknown"


def _extract_outcomes_for_subset(df_subset: pd.DataFrame, nlp, matcher, label: str) -> pd.DataFrame:
    """
    Run PreProcessingOutcome on a case-type subset:
      1. Strip HTML from Order
      2. Create spaCy docs via nlp.pipe
      3. Apply outcome matcher
    Returns the subset with an 'Outcome' column added.
    """
    subset = df_subset.copy()
    # Strip HTML from Order and create spaCy docs
    order_texts = subset["Order"].astype(str).apply(
        lambda x: _HTML_RE.sub(" ", x)
    ).tolist()
    order_docs = list(nlp.pipe(order_texts, batch_size=500))

    subset["Outcome"] = [_get_outcome_match(d, matcher, nlp) for d in order_docs]

    known = (subset["Outcome"] != "Unknown").sum()
    total = len(subset)
    print(f"       {label}: {known:,} / {total:,} outcomes extracted")
    print(f"         Affirmed: {(subset['Outcome'] == 'Affirmed').sum():,}")
    print(f"         Reversed: {(subset['Outcome'] == 'Reversed').sum():,}")
    print(f"         Unknown:  {(subset['Outcome'] == 'Unknown').sum():,}")
    return subset


def extract_outcomes(df: pd.DataFrame, nlp, od_indices) -> tuple:
    """
    Extract outcomes separately for Patent Refusal and Opposition Division.
    
    Parameters
    ----------
    df : full DataFrame (Matches already sub-classified for OD)
    nlp : spaCy Language model
    od_indices : pandas Index of rows that were originally Opposition Division
                 (captured before sub-classification overwrote Matches)

    Returns (pr_subset, od_subset).
    """
    print("[7/8] Extracting Outcomes from Order text …")

    matcher = _build_outcome_matcher(nlp)

    # Patent Refusal subset
    pr_mask = df["Matches"] == "Patent Refusal"
    pr_df = _extract_outcomes_for_subset(df[pr_mask], nlp, matcher, "Patent Refusal")

    # Opposition Division subset
    od_df = _extract_outcomes_for_subset(df.loc[od_indices], nlp, matcher, "Opposition Division")

    return pr_df, od_df


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Save outputs
# ═══════════════════════════════════════════════════════════════════════════════

def save_outputs(df: pd.DataFrame, pr_df: pd.DataFrame, od_df: pd.DataFrame) -> None:
    """
    Save:
      1. EPO_Data.pkl — full cleaned dataset (nlp column dropped to save memory)
      2. Train&TestData_1.0_PatentRefusal.pkl — Patent Refusal with known Outcome
      3. Train&TestData_1.0_OppositionDivision — Opposition Division with Outcome
    """
    print("[8/8] Saving outputs …")

    # ── Full dataset ──
    # Drop the heavy 'nlp' column (spaCy Doc objects) to avoid OOM on pickle
    epo_save = df.drop(columns=["nlp"], errors="ignore")
    epo_path = os.path.join(DATA_DIR, "EPO_Data.pkl")
    epo_save.to_pickle(epo_path)
    print(f"       Saved EPO_Data.pkl  ({len(epo_save):,} rows)")

    # ── Patent Refusal subset with known Outcome ──
    pr_binary = pr_df[pr_df["Outcome"] != "Unknown"].copy()
    pr_binary = pr_binary.drop(columns=["nlp"], errors="ignore")
    pr_path = os.path.join(DATA_DIR, "Train&TestData_1.0_PatentRefusal.pkl")
    pr_binary.to_pickle(pr_path)
    print(f"       Saved Train&TestData_1.0_PatentRefusal.pkl  ({len(pr_binary):,} rows)")
    print(f"         Affirmed: {(pr_binary['Outcome'] == 'Affirmed').sum():,}")
    print(f"         Reversed: {(pr_binary['Outcome'] == 'Reversed').sum():,}")

    # ── Opposition Division with Outcome ──
    od_save = od_df.drop(columns=["nlp"], errors="ignore")
    od_path = os.path.join(DATA_DIR, "Train&TestData_1.0_OppositionDivision")
    od_save.to_pickle(od_path)
    print(f"       Saved Train&TestData_1.0_OppositionDivision  ({len(od_save):,} rows)")
    od_binary = od_save[od_save["Outcome"] != "Unknown"]
    print(f"         With known outcome: {len(od_binary):,}")
    print(f"         Affirmed: {(od_binary['Outcome'] == 'Affirmed').sum():,}")
    print(f"         Reversed: {(od_binary['Outcome'] == 'Reversed').sum():,}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print(" EPO Data Processing Pipeline")
    print("=" * 70)

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.max_length = 5_000_000

    #1. Parse XML
    df = parse_xml(XML_PATH)

    #2. Clean
    df = clean_dataframe(df)

    #3. Legal Provisions → Articles
    df = match_legal_provisions(df)

    #4. Classify case types
    df = classify_cases(df, nlp)

    #5. Pre-process text
    df = preprocess_text(df)

    #6. Sub-classify Opposition Division by appellant type
    #keep track of opposition division cases 
    od_indices = df.index[df["Matches"] == "Opposition Division"]
    df = classify_opposition_subcases(df, nlp)

    #7. Extract outcomes (per subset: PR and OD separately)
    pr_df, od_df = extract_outcomes(df, nlp, od_indices)

    #8. Save
    save_outputs(df, pr_df, od_df)

    print()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
