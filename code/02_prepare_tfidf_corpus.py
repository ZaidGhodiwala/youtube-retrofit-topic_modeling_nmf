from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)


NMF_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    NMF_ROOT
    / "data"
    / "input"
    / "core_comments_substantive_ge3_words.csv"
)

PROCESSED_DIR = NMF_ROOT / "data" / "processed"
AUDIT_DIR = NMF_ROOT / "outputs" / "audit"
TABLE_DIR = NMF_ROOT / "outputs" / "tables"
MODEL_DIR = NMF_ROOT / "outputs" / "models"
CONFIG_DIR = NMF_ROOT / "config"

EXPECTED_ROWS = 42_487
EXPECTED_VIDEOS = 1_159

EXPECTED_INPUT_SHA256 = (
    "1fd8fdbf84bfb22aa7b22bb1e8633821938ee78cfb04917de7dd199f11b68350"
)

SOURCE_TEXT_COLUMN = "comment_text_no_urls"
MODEL_TEXT_COLUMN = "nmf_text_conservative"

# These words are excluded from the standard English stop-word list
# because they may carry practical, comparative or help-seeking meaning.
PRESERVED_WORDS = {
    "no",
    "not",
    "nor",
    "never",
    "without",
    "against",
    "before",
    "after",
    "above",
    "below",
    "under",
    "over",
    "between",
    "through",
    "during",
    "more",
    "less",
    "most",
    "least",
    "how",
    "why",
    "what",
    "where",
    "when",
    "should",
    "could",
    "would",
    "must",
}

CUSTOM_STOP_WORDS = sorted(
    set(ENGLISH_STOP_WORDS) - PRESERVED_WORDS
)

# Retains terms such as heat-pump, u-value, r-value and co2 while
# excluding isolated single-character tokens.
TOKEN_PATTERN = (
    r"(?u)\b(?:"
    r"[a-zA-Z][a-zA-Z0-9]*(?:[-_][a-zA-Z0-9]+)+"
    r"|"
    r"[a-zA-Z][a-zA-Z0-9]+"
    r")\b"
)


def calculate_sha256(file_path: Path) -> str:
    """Return the SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()


def normalise_model_text(value: object) -> str:
    """Apply conservative text normalisation for TF-IDF modelling."""

    if pd.isna(value):
        text = ""
    else:
        text = str(value)

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
    )

    # Defensive URL removal, even though the selected source column
    # has already had URLs removed.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = text.lower()

    contraction_replacements = (
        (r"\bwon't\b", "will not"),
        (r"\bcan't\b", "can not"),
        (r"\bcannot\b", "can not"),
        (r"n't\b", " not"),
        (r"'re\b", " are"),
        (r"'ve\b", " have"),
        (r"'ll\b", " will"),
        (r"'m\b", " am"),
        (r"'d\b", " would"),
        (r"'s\b", ""),
    )

    for pattern, replacement in contraction_replacements:
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def main() -> None:
    for directory in (
        PROCESSED_DIR,
        AUDIT_DIR,
        TABLE_DIR,
        MODEL_DIR,
        CONFIG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Input corpus not found:\n"
            f"{INPUT_FILE}"
        )

    input_hash = calculate_sha256(INPUT_FILE)

    if input_hash != EXPECTED_INPUT_SHA256:
        raise ValueError(
            "The input file does not match the audited corpus.\n"
            f"Expected SHA-256: {EXPECTED_INPUT_SHA256}\n"
            f"Observed SHA-256: {input_hash}"
        )

    print(f"Reading: {INPUT_FILE}")

    dataframe = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8",
        low_memory=False,
    )

    required_columns = {
        "video_id",
        "comment_id",
        SOURCE_TEXT_COLUMN,
        "retrofit_topic",
        "creator_type",
        "video_type",
        "primary_theme",
    }

    missing_columns = sorted(
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    if len(dataframe) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows but found "
            f"{len(dataframe):,}."
        )

    unique_videos = dataframe["video_id"].nunique()

    if unique_videos != EXPECTED_VIDEOS:
        raise ValueError(
            f"Expected {EXPECTED_VIDEOS:,} videos but found "
            f"{unique_videos:,}."
        )

    model_text = dataframe[SOURCE_TEXT_COLUMN].map(
        normalise_model_text
    )

    blank_text_count = int(model_text.eq("").sum())

    if blank_text_count:
        raise ValueError(
            f"{blank_text_count:,} comments became blank before "
            "vectorisation."
        )

    vectorizer = TfidfVectorizer(
        lowercase=False,
        strip_accents="unicode",
        stop_words=CUSTOM_STOP_WORDS,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.85,
        max_features=None,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm="l2",
        dtype=np.float64,
    )

    print("Building TF-IDF matrix...")

    full_matrix = vectorizer.fit_transform(model_text)

    row_nonzero_counts = np.asarray(
        full_matrix.getnnz(axis=1)
    ).ravel()

    included_mask = row_nonzero_counts > 0
    excluded_mask = ~included_mask

    model_matrix = full_matrix[included_mask].tocsr()

    model_corpus = dataframe.loc[included_mask].copy()
    model_corpus.insert(
        0,
        "tfidf_row_index",
        range(len(model_corpus)),
    )
    model_corpus[MODEL_TEXT_COLUMN] = (
        model_text.loc[included_mask].to_numpy()
    )

    excluded_rows = dataframe.loc[
        excluded_mask,
        [
            "video_id",
            "comment_id",
            "retrofit_topic",
            "creator_type",
            "video_type",
            "primary_theme",
            SOURCE_TEXT_COLUMN,
        ],
    ].copy()

    excluded_rows[MODEL_TEXT_COLUMN] = (
        model_text.loc[excluded_mask].to_numpy()
    )
    excluded_rows["exclusion_reason"] = (
        "No retained TF-IDF features after stop-word and "
        "document-frequency filtering"
    )

    corpus_path = (
        PROCESSED_DIR
        / "02_nmf_model_corpus_conservative.csv"
    )

    matrix_path = (
        PROCESSED_DIR
        / "02_tfidf_matrix_conservative.npz"
    )

    vectorizer_path = (
        MODEL_DIR
        / "02_tfidf_vectorizer_conservative.joblib"
    )

    excluded_path = (
        AUDIT_DIR
        / "02_zero_vector_comments.csv"
    )

    model_corpus.to_csv(
        corpus_path,
        index=False,
        encoding="utf-8-sig",
    )

    excluded_rows.to_csv(
        excluded_path,
        index=False,
        encoding="utf-8-sig",
    )

    sparse.save_npz(
        matrix_path,
        model_matrix,
        compressed=True,
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
        compress=3,
    )

    feature_names = np.asarray(
        vectorizer.get_feature_names_out()
    )

    # Each CSR matrix row contains at most one stored value for each
    # feature, so counting column indices gives document frequency.
    document_frequency = np.bincount(
        full_matrix.indices,
        minlength=full_matrix.shape[1],
    )

    summed_tfidf = np.asarray(
        full_matrix.sum(axis=0)
    ).ravel()

    feature_table = pd.DataFrame(
        {
            "feature": feature_names,
            "ngram_size": [
                feature.count(" ") + 1
                for feature in feature_names
            ],
            "document_frequency": document_frequency,
            "document_frequency_percent": (
                100
                * document_frequency
                / full_matrix.shape[0]
            ),
            "idf": vectorizer.idf_,
            "mean_tfidf_all_comments": (
                summed_tfidf
                / full_matrix.shape[0]
            ),
            "mean_tfidf_when_present": np.divide(
                summed_tfidf,
                document_frequency,
                out=np.zeros_like(summed_tfidf),
                where=document_frequency > 0,
            ),
        }
    )

    feature_table = feature_table.sort_values(
        by=[
            "document_frequency",
            "mean_tfidf_all_comments",
            "feature",
        ],
        ascending=[False, False, True],
        kind="stable",
    )

    feature_path = (
        TABLE_DIR
        / "02_tfidf_feature_inventory.csv"
    )

    top_feature_path = (
        TABLE_DIR
        / "02_top_250_tfidf_features.csv"
    )

    feature_table.to_csv(
        feature_path,
        index=False,
        encoding="utf-8-sig",
    )

    feature_table.head(250).to_csv(
        top_feature_path,
        index=False,
        encoding="utf-8-sig",
    )

    configuration = {
        "analysis_stage": (
            "Conservative TF-IDF corpus preparation"
        ),
        "input_file": str(INPUT_FILE),
        "input_sha256": input_hash,
        "source_text_column": SOURCE_TEXT_COLUMN,
        "model_text_column": MODEL_TEXT_COLUMN,
        "expected_input_rows": EXPECTED_ROWS,
        "expected_unique_videos": EXPECTED_VIDEOS,
        "normalisation": [
            "HTML entity decoding",
            "Unicode NFKC normalisation",
            "apostrophe standardisation",
            "defensive URL removal",
            "lowercase conversion",
            "common contraction expansion",
            "whitespace normalisation",
        ],
        "stemming": False,
        "lemmatisation": False,
        "preserved_words_removed_from_stop_list": sorted(
            PRESERVED_WORDS
        ),
        "vectorizer": {
            "representation": "TF-IDF",
            "ngram_range": [1, 2],
            "min_df": 10,
            "max_df": 0.85,
            "max_features": None,
            "sublinear_tf": True,
            "use_idf": True,
            "smooth_idf": True,
            "norm": "l2",
            "strip_accents": "unicode",
            "token_pattern": TOKEN_PATTERN,
            "dtype": "float64",
        },
    }

    config_path = (
        CONFIG_DIR
        / "02_conservative_tfidf_config.json"
    )

    config_path.write_text(
        json.dumps(
            configuration,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    matrix_density = (
        model_matrix.nnz
        / (
            model_matrix.shape[0]
            * model_matrix.shape[1]
        )
    )

    checks = {
        "Input hash matches audited corpus": (
            input_hash == EXPECTED_INPUT_SHA256
        ),
        "Expected input row count": (
            len(dataframe) == EXPECTED_ROWS
        ),
        "Expected unique-video count": (
            unique_videos == EXPECTED_VIDEOS
        ),
        "No blank normalised text": (
            blank_text_count == 0
        ),
        "Model matrix rows match model corpus": (
            model_matrix.shape[0] == len(model_corpus)
        ),
        "No zero rows in saved model matrix": (
            int(
                (
                    np.asarray(
                        model_matrix.getnnz(axis=1)
                    ).ravel()
                    == 0
                ).sum()
            )
            == 0
        ),
        "TF-IDF vocabulary is non-empty": (
            model_matrix.shape[1] > 0
        ),
    }

    overall_status = (
        "PASS"
        if all(checks.values())
        else "REVIEW REQUIRED"
    )

    report_lines = [
        "YOUTUBE RETROFIT CONSERVATIVE TF-IDF AUDIT",
        "=" * 47,
        "",
        f"Overall status: {overall_status}",
        "",
        "Source corpus",
        "-------------",
        f"Input rows: {len(dataframe):,}",
        f"Unique videos: {unique_videos:,}",
        f"Input SHA-256: {input_hash}",
        f"Source text column: {SOURCE_TEXT_COLUMN}",
        "",
        "Normalised corpus",
        "-----------------",
        f"Blank normalised texts: {blank_text_count:,}",
        (
            "Comments retained in TF-IDF model corpus: "
            f"{len(model_corpus):,}"
        ),
        (
            "Comments excluded as zero vectors: "
            f"{int(excluded_mask.sum()):,}"
        ),
        "",
        "TF-IDF matrix",
        "-------------",
        (
            "Matrix dimensions: "
            f"{model_matrix.shape[0]:,} rows x "
            f"{model_matrix.shape[1]:,} features"
        ),
        f"Non-zero values: {model_matrix.nnz:,}",
        f"Matrix density: {matrix_density:.6f}",
        (
            "Unigram features: "
            f"{int((feature_table['ngram_size'] == 1).sum()):,}"
        ),
        (
            "Bigram features: "
            f"{int((feature_table['ngram_size'] == 2).sum()):,}"
        ),
        "",
        "Vectorisation settings",
        "----------------------",
        "N-grams: unigrams and bigrams",
        "Minimum document frequency: 10 comments",
        "Maximum document frequency: 85% of comments",
        "Sublinear term frequency: enabled",
        "L2 normalisation: enabled",
        "Stemming: not used",
        "Lemmatisation: not used",
        "",
        "Validation checks",
        "-----------------",
    ]

    for check_name, passed in checks.items():
        report_lines.append(
            f"{'PASS' if passed else 'FAIL'}: {check_name}"
        )

    report_lines.extend(
        [
            "",
            "Output hashes",
            "-------------",
            (
                "Model corpus SHA-256: "
                f"{calculate_sha256(corpus_path)}"
            ),
            (
                "TF-IDF matrix SHA-256: "
                f"{calculate_sha256(matrix_path)}"
            ),
            (
                "Vectorizer SHA-256: "
                f"{calculate_sha256(vectorizer_path)}"
            ),
            "",
            "Interpretive note",
            "-----------------",
            (
                "Comments producing zero vectors are retained in a "
                "separate audit table. They are not silently deleted. "
                "They contain no vocabulary surviving the documented "
                "stop-word and document-frequency criteria and cannot "
                "receive a meaningful NMF topic allocation."
            ),
        ]
    )

    report_path = (
        AUDIT_DIR
        / "02_conservative_tfidf_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("\n".join(report_lines))
    print()
    print("Created:")
    print(f"  {corpus_path}")
    print(f"  {matrix_path}")
    print(f"  {vectorizer_path}")
    print(f"  {excluded_path}")
    print(f"  {feature_path}")
    print(f"  {top_feature_path}")
    print(f"  {config_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()