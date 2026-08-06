from __future__ import annotations

import hashlib
import json
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

# ---------------------------------------------------------------------
# Existing inclusive representation
# ---------------------------------------------------------------------

INCLUSIVE_CORPUS_FILE = (
    NMF_ROOT
    / "data"
    / "processed"
    / "02_nmf_model_corpus_conservative.csv"
)

INCLUSIVE_MATRIX_FILE = (
    NMF_ROOT
    / "data"
    / "processed"
    / "02_tfidf_matrix_conservative.npz"
)

INCLUSIVE_VECTORIZER_FILE = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "02_tfidf_vectorizer_conservative.joblib"
)

INCLUSIVE_FEATURE_FILE = (
    NMF_ROOT
    / "outputs"
    / "tables"
    / "02_tfidf_feature_inventory.csv"
)

TECHNICAL_ANCHOR_FILE = (
    NMF_ROOT
    / "outputs"
    / "tables"
    / "03_technical_anchor_review.csv"
)

INITIAL_ZERO_VECTOR_FILE = (
    NMF_ROOT
    / "outputs"
    / "audit"
    / "02_zero_vector_comments.csv"
)
# ---------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------

PROCESSED_DIR = NMF_ROOT / "data" / "processed"
AUDIT_DIR = NMF_ROOT / "outputs" / "audit"
TABLE_DIR = NMF_ROOT / "outputs" / "tables"
MODEL_DIR = NMF_ROOT / "outputs" / "models"
CONFIG_DIR = NMF_ROOT / "config"

EXPECTED_CORPUS_SHA256 = (
    "9baccc2467c1dc5a68d2c01ae23ae708"
    "bb94bc92d20a24e33a75e0dd00dde86d"
)

EXPECTED_MATRIX_SHA256 = (
    "897f5989f9861f16b851e9e04ce98ba6"
    "ee196a84bff2c4e1a5fe2499348469a7"
)

EXPECTED_VECTORIZER_SHA256 = (
    "ca257f25ff4680dd54589019f7eebc537"
    "5aee8aaf90ab3257f137c38a183ebec"
)

EXPECTED_ROWS = 42_443

EXPECTED_AUDITED_SOURCE_VIDEOS = 1_159
EXPECTED_INCLUSIVE_MODEL_VIDEOS = 1_158

EXPECTED_ZERO_VECTOR_ONLY_VIDEO_IDS = {
    "g-xfY5FjsXA",
}

SOURCE_TEXT_COLUMN = "nmf_text_conservative"

# ---------------------------------------------------------------------
# Stop-word decisions
# ---------------------------------------------------------------------

# These normally excluded English terms are retained because they may
# indicate negation, comparison, help-seeking or building relationships.
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

# This deliberately narrow list is used only for the content-focused
# sensitivity representation. The inclusive primary model retains these
# terms so social endorsement and platform interaction remain discoverable.
ADDITIONAL_SENSITIVITY_STOP_WORDS = {
    # Platform references
    "video",
    "videos",
    "youtube",
    "channel",
    "subscribe",
    "subscribed",
    "subscriber",
    "subscribers",
    "watch",
    "watching",
    "watched",
    "content",
    "comment",
    "comments",

    # Generic gratitude and strong praise
    "thank",
    "thanks",
    "great",
    "nice",
    "excellent",
    "amazing",
    "awesome",

    # Clear web-address residue
    "com",
}

# Ambiguous terms deliberately retained:
#
# good   — may describe technical suitability or performance
# love   — may indicate endorsement
# liked  — may refer to approval or platform behaviour
# like   — may indicate comparison, appearance or preference
# just   — may affect how users qualify or simplify advice
#
# Practical outcome terms such as worked, works, working, helped,
# helpful, fixed, solved, effective, successful, failed and recommend
# are also retained.

SENSITIVITY_STOP_WORDS = sorted(
    (
        set(ENGLISH_STOP_WORDS)
        - PRESERVED_WORDS
    )
    | ADDITIONAL_SENSITIVITY_STOP_WORDS
)

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


def verify_file(
    file_path: Path,
    expected_hash: str,
    description: str,
) -> str:
    """Verify that an existing Stage 3 artefact is unchanged."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{file_path}"
        )

    observed_hash = calculate_sha256(file_path)

    if observed_hash != expected_hash:
        raise ValueError(
            f"{description} does not match the audited Stage 3 artefact.\n"
            f"Expected SHA-256: {expected_hash}\n"
            f"Observed SHA-256: {observed_hash}"
        )

    return observed_hash


def create_feature_inventory(
    matrix,
    vectorizer: TfidfVectorizer,
) -> pd.DataFrame:
    """Create a documented inventory of TF-IDF features."""

    feature_names = np.asarray(
        vectorizer.get_feature_names_out()
    )

    document_frequency = np.bincount(
        matrix.indices,
        minlength=matrix.shape[1],
    )

    summed_tfidf = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    inventory = pd.DataFrame(
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
                / matrix.shape[0]
            ),
            "idf": vectorizer.idf_,
            "mean_tfidf_all_comments": (
                summed_tfidf
                / matrix.shape[0]
            ),
            "mean_tfidf_when_present": np.divide(
                summed_tfidf,
                document_frequency,
                out=np.zeros_like(summed_tfidf),
                where=document_frequency > 0,
            ),
        }
    )

    return inventory.sort_values(
        by=[
            "document_frequency",
            "mean_tfidf_all_comments",
            "feature",
        ],
        ascending=[False, False, True],
        kind="stable",
    )


def main() -> None:
    for directory in (
        PROCESSED_DIR,
        AUDIT_DIR,
        TABLE_DIR,
        MODEL_DIR,
        CONFIG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Verify the already frozen inclusive primary representation
    # -----------------------------------------------------------------

    corpus_hash = verify_file(
        INCLUSIVE_CORPUS_FILE,
        EXPECTED_CORPUS_SHA256,
        "Inclusive modelling corpus",
    )

    matrix_hash = verify_file(
        INCLUSIVE_MATRIX_FILE,
        EXPECTED_MATRIX_SHA256,
        "Inclusive TF-IDF matrix",
    )

    vectorizer_hash = verify_file(
        INCLUSIVE_VECTORIZER_FILE,
        EXPECTED_VECTORIZER_SHA256,
        "Inclusive TF-IDF vectorizer",
    )

    inclusive_corpus = pd.read_csv(
        INCLUSIVE_CORPUS_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    inclusive_matrix = sparse.load_npz(
        INCLUSIVE_MATRIX_FILE
    ).tocsr()

    inclusive_vectorizer = joblib.load(
        INCLUSIVE_VECTORIZER_FILE
    )

    if len(inclusive_corpus) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} inclusive corpus rows but "
            f"found {len(inclusive_corpus):,}."
        )

    inclusive_video_count = int(
    inclusive_corpus["video_id"].nunique()
)

    if (
        inclusive_video_count
        != EXPECTED_INCLUSIVE_MODEL_VIDEOS
    ):
        raise ValueError(
           "Unexpected unique-video count in the inclusive "
           "modelling corpus.\n"
           f"Expected: {EXPECTED_INCLUSIVE_MODEL_VIDEOS:,}\n"
           f"Observed: {inclusive_video_count:,}"
        )
        
    if not INITIAL_ZERO_VECTOR_FILE.exists():
        raise FileNotFoundError(
            "The Stage 3 zero-vector audit file was not found:\n"
            f"{INITIAL_ZERO_VECTOR_FILE}"
        )

    initial_zero_vectors = pd.read_csv(
        INITIAL_ZERO_VECTOR_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    zero_vector_video_ids = set(
        initial_zero_vectors["video_id"]
        .dropna()
        .astype(str)
    )

    inclusive_video_ids = set(
        inclusive_corpus["video_id"]
        .dropna()
        .astype(str)
    )

    zero_vector_only_video_ids = (
        zero_vector_video_ids
        - inclusive_video_ids
    )

    if (
        zero_vector_only_video_ids
        != EXPECTED_ZERO_VECTOR_ONLY_VIDEO_IDS
    ):
        raise ValueError(
            "The set of videos excluded entirely by zero-vector "
            "filtering was unexpected.\n"
            f"Expected: "
            f"{sorted(EXPECTED_ZERO_VECTOR_ONLY_VIDEO_IDS)}\n"
            f"Observed: {sorted(zero_vector_only_video_ids)}"
        )    

    if inclusive_matrix.shape[0] != len(inclusive_corpus):
        raise ValueError(
            "Inclusive matrix rows do not match inclusive corpus rows."
        )

    if (
        inclusive_matrix.shape[1]
        != len(
            inclusive_vectorizer.get_feature_names_out()
        )
    ):
        raise ValueError(
            "Inclusive matrix columns do not match vectorizer features."
        )

    if SOURCE_TEXT_COLUMN not in inclusive_corpus.columns:
        raise ValueError(
            f"Required text column is missing: {SOURCE_TEXT_COLUMN}"
        )

    model_text = (
        inclusive_corpus[SOURCE_TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if model_text.eq("").any():
        raise ValueError(
            "The inclusive modelling corpus contains blank text."
        )

    # -----------------------------------------------------------------
    # Build the content-focused sensitivity representation
    # -----------------------------------------------------------------

    sensitivity_vectorizer = TfidfVectorizer(
        lowercase=False,
        strip_accents="unicode",
        stop_words=SENSITIVITY_STOP_WORDS,
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

    print(
        "Building content-focused sensitivity TF-IDF matrix..."
    )

    full_sensitivity_matrix = (
        sensitivity_vectorizer.fit_transform(model_text)
    )

    nonzero_counts = np.asarray(
        full_sensitivity_matrix.getnnz(axis=1)
    ).ravel()

    included_mask = nonzero_counts > 0
    excluded_mask = ~included_mask

    sensitivity_matrix = (
        full_sensitivity_matrix[included_mask]
        .tocsr()
    )

    sensitivity_corpus = (
        inclusive_corpus.loc[included_mask]
        .copy()
    )

    if "tfidf_row_index" in sensitivity_corpus.columns:
        sensitivity_corpus = sensitivity_corpus.rename(
            columns={
                "tfidf_row_index":
                "inclusive_tfidf_row_index"
            }
        )

    sensitivity_corpus.insert(
        0,
        "content_focused_tfidf_row_index",
        range(len(sensitivity_corpus)),
    )

    excluded_rows = inclusive_corpus.loc[
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

    excluded_rows["exclusion_reason"] = (
        "No retained TF-IDF features after applying the "
        "content-focused sensitivity stop-word list"
    )

    sensitivity_corpus_path = (
        PROCESSED_DIR
        / "04_nmf_model_corpus_content_focused.csv"
    )

    sensitivity_matrix_path = (
        PROCESSED_DIR
        / "04_tfidf_matrix_content_focused.npz"
    )

    sensitivity_vectorizer_path = (
        MODEL_DIR
        / "04_tfidf_vectorizer_content_focused.joblib"
    )

    sensitivity_excluded_path = (
        AUDIT_DIR
        / "04_content_focused_zero_vector_comments.csv"
    )

    sensitivity_corpus.to_csv(
        sensitivity_corpus_path,
        index=False,
        encoding="utf-8-sig",
    )

    sparse.save_npz(
        sensitivity_matrix_path,
        sensitivity_matrix,
        compressed=True,
    )

    joblib.dump(
        sensitivity_vectorizer,
        sensitivity_vectorizer_path,
        compress=3,
    )

    excluded_rows.to_csv(
        sensitivity_excluded_path,
        index=False,
        encoding="utf-8-sig",
    )

    sensitivity_features = create_feature_inventory(
        full_sensitivity_matrix,
        sensitivity_vectorizer,
    )

    sensitivity_feature_path = (
        TABLE_DIR
        / "04_content_focused_feature_inventory.csv"
    )

    sensitivity_top_feature_path = (
        TABLE_DIR
        / "04_content_focused_top_250_features.csv"
    )

    sensitivity_features.to_csv(
        sensitivity_feature_path,
        index=False,
        encoding="utf-8-sig",
    )

    sensitivity_features.head(250).to_csv(
        sensitivity_top_feature_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Compare the two representations
    # -----------------------------------------------------------------

    inclusive_features = pd.read_csv(
        INCLUSIVE_FEATURE_FILE,
        encoding="utf-8-sig",
    )

    inclusive_feature_set = set(
        inclusive_vectorizer.get_feature_names_out()
    )

    sensitivity_feature_set = set(
        sensitivity_vectorizer.get_feature_names_out()
    )

    removed_features = inclusive_features.loc[
        ~inclusive_features["feature"].isin(
            sensitivity_feature_set
        )
    ].copy()

    def classify_removal(feature: object) -> str:
        feature_text = str(feature).lower()
        tokens = set(feature_text.split())

        if (
            feature_text
            in ADDITIONAL_SENSITIVITY_STOP_WORDS
        ):
            return "Direct sensitivity stop word"

        if (
            tokens
            & ADDITIONAL_SENSITIVITY_STOP_WORDS
        ):
            return (
                "N-gram containing a sensitivity stop word"
            )

        return (
            "No longer met documented document-frequency "
            "criteria after sensitivity preprocessing"
        )

    removed_features["removal_basis"] = (
        removed_features["feature"].map(
            classify_removal
        )
    )

    removed_features = removed_features.sort_values(
        by=[
            "document_frequency",
            "feature",
        ],
        ascending=[False, True],
        kind="stable",
    )

    removed_feature_path = (
        TABLE_DIR
        / "04_features_removed_in_content_focused_sensitivity.csv"
    )

    removed_features.to_csv(
        removed_feature_path,
        index=False,
        encoding="utf-8-sig",
    )

    representation_summary = pd.DataFrame(
        [
            {
                "representation_id": "inclusive_primary",
                "analytical_role": "Primary",
                "description": (
                    "Retains technical, evaluative, social, "
                    "gratitude and platform-mediated language."
                ),
                "corpus_file": str(
                    INCLUSIVE_CORPUS_FILE
                ),
                "matrix_file": str(
                    INCLUSIVE_MATRIX_FILE
                ),
                "vectorizer_file": str(
                    INCLUSIVE_VECTORIZER_FILE
                ),
                "comment_rows": (
                    inclusive_matrix.shape[0]
                ),
                "feature_count": (
                    inclusive_matrix.shape[1]
                ),
                "nonzero_values": (
                    inclusive_matrix.nnz
                ),
                "zero_vector_comments_at_this_stage": 0,
            },
            {
                "representation_id": (
                    "content_focused_sensitivity"
                ),
                "analytical_role": "Sensitivity",
                "description": (
                    "Removes a narrow set of generic platform, "
                    "gratitude and praise terms while preserving "
                    "practical outcome and ambiguous evaluative terms."
                ),
                "corpus_file": str(
                    sensitivity_corpus_path
                ),
                "matrix_file": str(
                    sensitivity_matrix_path
                ),
                "vectorizer_file": str(
                    sensitivity_vectorizer_path
                ),
                "comment_rows": (
                    sensitivity_matrix.shape[0]
                ),
                "feature_count": (
                    sensitivity_matrix.shape[1]
                ),
                "nonzero_values": (
                    sensitivity_matrix.nnz
                ),
                "zero_vector_comments_at_this_stage": (
                    int(excluded_mask.sum())
                ),
            },
        ]
    )

    summary_path = (
        TABLE_DIR
        / "04_parallel_tfidf_representation_summary.csv"
    )

    representation_summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Technical-anchor comparison
    # -----------------------------------------------------------------

    technical_present_inclusive = None
    technical_present_sensitivity = None
    technical_total = None

    technical_comparison_path = (
        TABLE_DIR
        / "04_parallel_technical_anchor_comparison.csv"
    )

    if TECHNICAL_ANCHOR_FILE.exists():
        technical = pd.read_csv(
            TECHNICAL_ANCHOR_FILE,
            encoding="utf-8-sig",
        )

        technical["present_in_inclusive_primary"] = (
            technical["technical_anchor"]
            .astype(str)
            .str.lower()
            .isin(inclusive_feature_set)
        )

        technical[
            "present_in_content_focused_sensitivity"
        ] = (
            technical["technical_anchor"]
            .astype(str)
            .str.lower()
            .isin(sensitivity_feature_set)
        )

        technical_total = len(technical)

        technical_present_inclusive = int(
            technical[
                "present_in_inclusive_primary"
            ].sum()
        )

        technical_present_sensitivity = int(
            technical[
                "present_in_content_focused_sensitivity"
            ].sum()
        )

        technical.to_csv(
            technical_comparison_path,
            index=False,
            encoding="utf-8-sig",
        )

    # -----------------------------------------------------------------
    # Configuration and audit report
    # -----------------------------------------------------------------

    configuration = {
        "analysis_stage": (
            "Parallel TF-IDF representation freeze"
        ),
        "primary_representation": {
            "id": "inclusive_primary",
            "source": (
                "Existing Stage 3 conservative TF-IDF "
                "representation"
            ),
            "purpose": (
                "Capture technical content alongside trust, "
                "endorsement, gratitude, social validation "
                "and platform-mediated interaction."
            ),
            "corpus_file": str(
                INCLUSIVE_CORPUS_FILE
            ),
            "matrix_file": str(
                INCLUSIVE_MATRIX_FILE
            ),
            "vectorizer_file": str(
                INCLUSIVE_VECTORIZER_FILE
            ),
        },
        "sensitivity_representation": {
            "id": "content_focused_sensitivity",
            "purpose": (
                "Assess whether generic platform and praise "
                "language obscures more specific technical "
                "retrofit themes."
            ),
            "additional_stop_words": sorted(
                ADDITIONAL_SENSITIVITY_STOP_WORDS
            ),
            "deliberately_retained_ambiguous_terms": [
                "good",
                "love",
                "liked",
                "like",
                "just",
            ],
            "deliberately_retained_outcome_terms": [
                "worked",
                "works",
                "working",
                "helped",
                "helpful",
                "fixed",
                "solved",
                "successful",
                "effective",
                "failed",
                "recommend",
                "recommended",
            ],
            "corpus_file": str(
                sensitivity_corpus_path
            ),
            "matrix_file": str(
                sensitivity_matrix_path
            ),
            "vectorizer_file": str(
                sensitivity_vectorizer_path
            ),
        },
        "shared_vectorizer_settings": {
            "representation": "TF-IDF",
            "ngram_range": [1, 2],
            "min_df": 10,
            "max_df": 0.85,
            "sublinear_tf": True,
            "use_idf": True,
            "smooth_idf": True,
            "norm": "l2",
            "strip_accents": "unicode",
            "token_pattern": TOKEN_PATTERN,
            "dtype": "float64",
            "stemming": False,
            "lemmatisation": False,
        },
    }

    config_path = (
        CONFIG_DIR
        / "04_parallel_tfidf_representations.json"
    )

    config_path.write_text(
        json.dumps(
            configuration,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    checks = {
        "Inclusive corpus hash verified": (
            corpus_hash
            == EXPECTED_CORPUS_SHA256
        ),
        "Inclusive matrix hash verified": (
            matrix_hash
            == EXPECTED_MATRIX_SHA256
        ),
        "Inclusive vectorizer hash verified": (
            vectorizer_hash
            == EXPECTED_VECTORIZER_SHA256
        ),
        "Inclusive matrix rows match corpus": (
            inclusive_matrix.shape[0]
            == len(inclusive_corpus)
        ),
        "Sensitivity matrix rows match corpus": (
            sensitivity_matrix.shape[0]
            == len(sensitivity_corpus)
        ),
        "No zero rows in saved sensitivity matrix": (
            int(
                (
                    np.asarray(
                        sensitivity_matrix.getnnz(axis=1)
                    ).ravel()
                    == 0
                ).sum()
            )
            == 0
        ),
        "Inclusive representation retains video": (
            "video" in inclusive_feature_set
        ),
        "Sensitivity representation removes video": (
            "video" not in sensitivity_feature_set
        ),
        "Inclusive representation retains thanks": (
            "thanks" in inclusive_feature_set
        ),
        "Sensitivity representation removes thanks": (
            "thanks" not in sensitivity_feature_set
        ),
        "Practical outcome term retained": (
            "worked" in sensitivity_feature_set
        ),
        "Negation retained": (
            "not" in sensitivity_feature_set
            and "no" in sensitivity_feature_set
        ),
        "Core technical phrase retained": (
            "heat pump" in sensitivity_feature_set
        ),
    }

    overall_status = (
        "PASS"
        if all(checks.values())
        else "REVIEW REQUIRED"
    )

    report_lines = [
        "YOUTUBE RETROFIT PARALLEL TF-IDF FREEZE",
        "=" * 43,
        "",
        f"Overall status: {overall_status}",
        "",
        "Design decision",
        "---------------",
        (
            "The inclusive representation is the primary "
            "representation. It retains technical discussion "
            "alongside praise, gratitude, endorsement and "
            "platform-mediated interaction."
        ),
        (
            "The content-focused representation is a "
            "preprocessing sensitivity condition. It removes "
            "only a narrow set of generic platform, gratitude "
            "and strong-praise terms."
        ),
        "",
        "Inclusive primary representation",
        "--------------------------------",
        (
            "Comments: "
            f"{inclusive_matrix.shape[0]:,}"
        ),
        (
            "Unique videos represented: "
            f"{inclusive_video_count:,}"
        ),
        (
            "Videos absent after zero-vector filtering: "
            f"{len(zero_vector_only_video_ids):,}"
        ),
        (
            "Features: "
            f"{inclusive_matrix.shape[1]:,}"
        ),
        (
            "Non-zero values: "
            f"{inclusive_matrix.nnz:,}"
        ),
        (
            "Matrix dimensions: "
            f"{inclusive_matrix.shape[0]:,} x "
            f"{inclusive_matrix.shape[1]:,}"
        ),
        "",
        "Content-focused sensitivity representation",
        "------------------------------------------",
        (
            "Comments retained: "
            f"{sensitivity_matrix.shape[0]:,}"
        ),
        (
            "Additional zero-vector comments: "
            f"{int(excluded_mask.sum()):,}"
        ),
        (
            "Features: "
            f"{sensitivity_matrix.shape[1]:,}"
        ),
        (
            "Non-zero values: "
            f"{sensitivity_matrix.nnz:,}"
        ),
        (
            "Matrix dimensions: "
            f"{sensitivity_matrix.shape[0]:,} x "
            f"{sensitivity_matrix.shape[1]:,}"
        ),
        (
            "Features removed relative to inclusive "
            f"representation: {len(removed_features):,}"
        ),
        "",
        "Terms deliberately retained in sensitivity model",
        "------------------------------------------------",
        (
            "Ambiguous evaluative terms: "
            "good, love, liked, like, just"
        ),
        (
            "Practical outcome terms: worked, works, "
            "working, helped, helpful, fixed, solved, "
            "successful, effective, failed, recommend "
            "and recommended"
        ),
        (
            "Negation and help-seeking terms remain "
            "available."
        ),
    ]

    if technical_total is not None:
        report_lines.extend(
            [
                "",
                "Technical-anchor comparison",
                "---------------------------",
                (
                    "Inclusive primary anchors present: "
                    f"{technical_present_inclusive:,} "
                    f"of {technical_total:,}"
                ),
                (
                    "Content-focused sensitivity anchors "
                    f"present: {technical_present_sensitivity:,} "
                    f"of {technical_total:,}"
                ),
            ]
        )

    report_lines.extend(
        [
            "",
            "Validation checks",
            "-----------------",
        ]
    )

    for check_name, passed in checks.items():
        report_lines.append(
            f"{'PASS' if passed else 'FAIL'}: "
            f"{check_name}"
        )

    report_lines.extend(
        [
            "",
            "Output hashes",
            "-------------",
            (
                "Sensitivity corpus SHA-256: "
                f"{calculate_sha256(sensitivity_corpus_path)}"
            ),
            (
                "Sensitivity matrix SHA-256: "
                f"{calculate_sha256(sensitivity_matrix_path)}"
            ),
            (
                "Sensitivity vectorizer SHA-256: "
                f"{calculate_sha256(sensitivity_vectorizer_path)}"
            ),
            "",
            "Interpretive note",
            "-----------------",
            (
                "Neither representation is treated as the "
                "objectively correct preprocessing condition. "
                "Candidate NMF solutions will be fitted using "
                "the inclusive primary representation, with "
                "the content-focused version used to evaluate "
                "whether major findings depend on generic "
                "social and platform vocabulary."
            ),
        ]
    )

    report_path = (
        AUDIT_DIR
        / "04_parallel_tfidf_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("\n".join(report_lines))
    print()
    print("Created:")
    print(f"  {sensitivity_corpus_path}")
    print(f"  {sensitivity_matrix_path}")
    print(f"  {sensitivity_vectorizer_path}")
    print(f"  {sensitivity_excluded_path}")
    print(f"  {sensitivity_feature_path}")
    print(f"  {sensitivity_top_feature_path}")
    print(f"  {removed_feature_path}")
    print(f"  {summary_path}")
    print(f"  {technical_comparison_path}")
    print(f"  {config_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()