from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

NMF_ROOT = Path(__file__).resolve().parents[1]

# Inclusive primary representation
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

INCLUSIVE_MODEL_FILE = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "05_inclusive_candidates"
    / "inclusive_nmf_k16.joblib"
)

INCLUSIVE_DOCUMENT_TOPIC_FILE = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "05_inclusive_candidates"
    / "inclusive_document_topic_k16.npz"
)

# Content-focused sensitivity representation
SENSITIVITY_CORPUS_FILE = (
    NMF_ROOT
    / "data"
    / "processed"
    / "04_nmf_model_corpus_content_focused.csv"
)

SENSITIVITY_MATRIX_FILE = (
    NMF_ROOT
    / "data"
    / "processed"
    / "04_tfidf_matrix_content_focused.npz"
)

SENSITIVITY_VECTORIZER_FILE = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "04_tfidf_vectorizer_content_focused.joblib"
)

# Outputs
MODEL_DIR = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "07_preprocessing_sensitivity"
)

TABLE_DIR = NMF_ROOT / "outputs" / "tables"
AUDIT_DIR = NMF_ROOT / "outputs" / "audit"
CONFIG_DIR = NMF_ROOT / "config"
REVIEW_DIR = NMF_ROOT / "outputs" / "review"


# ---------------------------------------------------------------------
# Frozen sensitivity-input hashes
# ---------------------------------------------------------------------

EXPECTED_SENSITIVITY_CORPUS_SHA256 = (
    "bfb9ca0cb355f1476be9f51e9923e718"
    "108b3938c609f6071748c6542cb3bcf7"
)

EXPECTED_SENSITIVITY_MATRIX_SHA256 = (
    "742b3dcf55ef8848ec148151c16fde282"
    "590d901747c3ef5769708bf1c2c9186"
)

EXPECTED_SENSITIVITY_VECTORIZER_SHA256 = (
    "197c43f3b2d45f487504ef50acff83c322"
    "aa76fb0c629a3e2799e27e358d9041"
)


# ---------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------

TOPIC_COUNT = 16

RANDOM_STATE = 20_260_806
MAX_ITER = 500
TOLERANCE = 1e-4

TOP_TERMS_REPORTED = 20
TOP_TERMS_FOR_JACCARD = 20

REPRESENTATIVE_COMMENTS_PER_TOPIC = 8
MAX_REPRESENTATIVES_PER_VIDEO = 2


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def calculate_sha256(file_path: Path) -> str:
    """Return SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def verify_file(
    file_path: Path,
    expected_hash: str,
    description: str,
) -> None:
    """Verify a frozen file against its expected hash."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{file_path}"
        )

    observed_hash = calculate_sha256(file_path)

    if observed_hash != expected_hash:
        raise ValueError(
            f"{description} does not match the frozen artefact.\n"
            f"Expected: {expected_hash}\n"
            f"Observed: {observed_hash}"
        )


def top_indices(
    components: np.ndarray,
    number_of_terms: int,
) -> np.ndarray:
    """Return descending top-feature indices."""

    return np.argsort(
        components,
        axis=1,
    )[:, -number_of_terms:][:, ::-1]


def jaccard(
    first_terms: set[str],
    second_terms: set[str],
) -> float:
    """Return Jaccard similarity between two term sets."""

    union = first_terms | second_terms

    if not union:
        return 1.0

    return len(
        first_terms & second_terms
    ) / len(union)


def get_topic_shares(
    document_topic: np.ndarray,
) -> np.ndarray:
    """Return dominant-topic proportions."""

    dominant = np.argmax(
        document_topic,
        axis=1,
    )

    counts = np.bincount(
        dominant,
        minlength=document_topic.shape[1],
    )

    return counts / len(dominant)


def select_representatives(
    corpus: pd.DataFrame,
    document_topic: np.ndarray,
    topic_number: int,
) -> list[dict[str, object]]:
    """Select high-loading representative comments."""

    topic_index = topic_number - 1

    row_sums = document_topic.sum(
        axis=1
    )

    relative_loading = np.divide(
        document_topic[:, topic_index],
        row_sums,
        out=np.zeros(
            document_topic.shape[0],
            dtype=np.float64,
        ),
        where=row_sums > 0,
    )

    ordered_rows = np.argsort(
        document_topic[:, topic_index]
    )[::-1]

    selected = []
    per_video_count: dict[str, int] = {}

    text_column = (
        "comment_text_for_coding"
        if "comment_text_for_coding"
        in corpus.columns
        else "nmf_text_conservative"
    )

    for row_index in ordered_rows:
        if (
            len(selected)
            >= REPRESENTATIVE_COMMENTS_PER_TOPIC
        ):
            break

        video_id = str(
            corpus.iloc[row_index]["video_id"]
        )

        existing_count = per_video_count.get(
            video_id,
            0,
        )

        if (
            existing_count
            >= MAX_REPRESENTATIVES_PER_VIDEO
        ):
            continue

        record = {
            "topic_number": topic_number,
            "representative_rank": (
                len(selected) + 1
            ),
            "topic_weight": float(
                document_topic[
                    row_index,
                    topic_index,
                ]
            ),
            "relative_topic_loading": float(
                relative_loading[row_index]
            ),
            "video_id": video_id,
            "comment_id": corpus.iloc[
                row_index
            ]["comment_id"],
            "comment_text": corpus.iloc[
                row_index
            ][text_column],
        }

        for column in [
            "retrofit_topic",
            "creator_type",
            "video_type",
            "primary_theme",
        ]:
            if column in corpus.columns:
                record[column] = corpus.iloc[
                    row_index
                ][column]

        selected.append(record)

        per_video_count[video_id] = (
            existing_count + 1
        )

    return selected


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    for directory in (
        MODEL_DIR,
        TABLE_DIR,
        AUDIT_DIR,
        CONFIG_DIR,
        REVIEW_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -----------------------------------------------------------------
    # Verify sensitivity representation
    # -----------------------------------------------------------------

    verify_file(
        SENSITIVITY_CORPUS_FILE,
        EXPECTED_SENSITIVITY_CORPUS_SHA256,
        "Sensitivity corpus",
    )

    verify_file(
        SENSITIVITY_MATRIX_FILE,
        EXPECTED_SENSITIVITY_MATRIX_SHA256,
        "Sensitivity matrix",
    )

    verify_file(
        SENSITIVITY_VECTORIZER_FILE,
        EXPECTED_SENSITIVITY_VECTORIZER_SHA256,
        "Sensitivity vectorizer",
    )

    required_existing_files = [
        INCLUSIVE_CORPUS_FILE,
        INCLUSIVE_MATRIX_FILE,
        INCLUSIVE_VECTORIZER_FILE,
        INCLUSIVE_MODEL_FILE,
        INCLUSIVE_DOCUMENT_TOPIC_FILE,
    ]

    for file_path in required_existing_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required inclusive artefact not found:\n"
                f"{file_path}"
            )

    # -----------------------------------------------------------------
    # Load both representations
    # -----------------------------------------------------------------

    inclusive_corpus = pd.read_csv(
        INCLUSIVE_CORPUS_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    sensitivity_corpus = pd.read_csv(
        SENSITIVITY_CORPUS_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    inclusive_matrix = sparse.load_npz(
        INCLUSIVE_MATRIX_FILE
    ).tocsr()

    sensitivity_matrix = sparse.load_npz(
        SENSITIVITY_MATRIX_FILE
    ).tocsr()

    inclusive_vectorizer = joblib.load(
        INCLUSIVE_VECTORIZER_FILE
    )

    sensitivity_vectorizer = joblib.load(
        SENSITIVITY_VECTORIZER_FILE
    )

    inclusive_model = joblib.load(
        INCLUSIVE_MODEL_FILE
    )

    inclusive_document_topic = (
        sparse.load_npz(
            INCLUSIVE_DOCUMENT_TOPIC_FILE
        )
        .toarray()
    )

    inclusive_features = np.asarray(
        inclusive_vectorizer
        .get_feature_names_out()
    )

    sensitivity_features = np.asarray(
        sensitivity_vectorizer
        .get_feature_names_out()
    )

    # -----------------------------------------------------------------
    # Fit k=16 sensitivity model
    # -----------------------------------------------------------------

    print(
        "Fitting k=16 NMF to content-focused "
        "sensitivity representation..."
    )

    sensitivity_model = NMF(
        n_components=TOPIC_COUNT,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        tol=TOLERANCE,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
        shuffle=False,
    )

    start_time = time.perf_counter()

    with warnings.catch_warnings(
        record=True
    ) as recorded_warnings:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        sensitivity_document_topic = (
            sensitivity_model.fit_transform(
                sensitivity_matrix
            )
        )

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    convergence_warning = any(
        issubclass(
            warning.category,
            ConvergenceWarning,
        )
        for warning in recorded_warnings
    )

    print(
        f"Completed in {runtime_seconds:.1f}s "
        f"after {sensitivity_model.n_iter_} iterations."
    )

    # -----------------------------------------------------------------
    # Align vocabularies
    # -----------------------------------------------------------------

    inclusive_lookup = {
        feature: index
        for index, feature
        in enumerate(inclusive_features)
    }

    sensitivity_lookup = {
        feature: index
        for index, feature
        in enumerate(sensitivity_features)
    }

    shared_features = sorted(
        set(inclusive_features)
        & set(sensitivity_features)
    )

    inclusive_shared_indices = np.asarray(
        [
            inclusive_lookup[feature]
            for feature in shared_features
        ]
    )

    sensitivity_shared_indices = np.asarray(
        [
            sensitivity_lookup[feature]
            for feature in shared_features
        ]
    )

    inclusive_shared_components = (
        inclusive_model.components_[
            :,
            inclusive_shared_indices,
        ]
    )

    sensitivity_shared_components = (
        sensitivity_model.components_[
            :,
            sensitivity_shared_indices,
        ]
    )

    similarity_matrix = cosine_similarity(
        inclusive_shared_components,
        sensitivity_shared_components,
    )

    inclusive_topic_indices, sensitivity_topic_indices = (
        linear_sum_assignment(
            -similarity_matrix
        )
    )

    matched_cosines = similarity_matrix[
        inclusive_topic_indices,
        sensitivity_topic_indices,
    ]

    # -----------------------------------------------------------------
    # Top-term comparisons
    # -----------------------------------------------------------------

    inclusive_top_indices = top_indices(
        inclusive_model.components_,
        TOP_TERMS_REPORTED,
    )

    sensitivity_top_indices = top_indices(
        sensitivity_model.components_,
        TOP_TERMS_REPORTED,
    )

    inclusive_shares = get_topic_shares(
        inclusive_document_topic
    )

    sensitivity_shares = get_topic_shares(
        sensitivity_document_topic
    )

    comparison_rows = []

    for (
        inclusive_index,
        sensitivity_index,
        cosine_value,
    ) in zip(
        inclusive_topic_indices,
        sensitivity_topic_indices,
        matched_cosines,
        strict=True,
    ):
        inclusive_terms = [
            str(term)
            for term in inclusive_features[
                inclusive_top_indices[
                    inclusive_index
                ]
            ]
        ]

        sensitivity_terms = [
            str(term)
            for term in sensitivity_features[
                sensitivity_top_indices[
                    sensitivity_index
                ]
            ]
        ]

        inclusive_term_set = set(
            inclusive_terms[
                :TOP_TERMS_FOR_JACCARD
            ]
        )

        sensitivity_term_set = set(
            sensitivity_terms[
                :TOP_TERMS_FOR_JACCARD
            ]
        )

        shared_top_terms = (
            inclusive_term_set
            & sensitivity_term_set
        )

        comparison_rows.append(
            {
                "inclusive_topic_number": (
                    int(inclusive_index) + 1
                ),
                "matched_sensitivity_topic_number": (
                    int(sensitivity_index) + 1
                ),
                "shared_vocabulary_cosine_similarity": (
                    float(cosine_value)
                ),
                "top_20_term_jaccard": float(
                    jaccard(
                        inclusive_term_set,
                        sensitivity_term_set,
                    )
                ),
                "shared_top_20_term_count": (
                    len(shared_top_terms)
                ),
                "inclusive_dominant_share": float(
                    inclusive_shares[
                        inclusive_index
                    ]
                ),
                "sensitivity_dominant_share": float(
                    sensitivity_shares[
                        sensitivity_index
                    ]
                ),
                "absolute_prevalence_difference": float(
                    abs(
                        sensitivity_shares[
                            sensitivity_index
                        ]
                        - inclusive_shares[
                            inclusive_index
                        ]
                    )
                ),
                "inclusive_top_20_terms": (
                    " | ".join(
                        inclusive_terms
                    )
                ),
                "sensitivity_top_20_terms": (
                    " | ".join(
                        sensitivity_terms
                    )
                ),
                "shared_top_terms": (
                    " | ".join(
                        sorted(
                            shared_top_terms
                        )
                    )
                ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    ).sort_values(
        "inclusive_topic_number"
    )

    # -----------------------------------------------------------------
    # Compare dominant assignments on comments present in both corpora
    # -----------------------------------------------------------------

    inclusive_id_to_row = {
        str(comment_id): row_index
        for row_index, comment_id
        in enumerate(
            inclusive_corpus[
                "comment_id"
            ].astype(str)
        )
    }

    sensitivity_id_to_row = {
        str(comment_id): row_index
        for row_index, comment_id
        in enumerate(
            sensitivity_corpus[
                "comment_id"
            ].astype(str)
        )
    }

    shared_comment_ids = sorted(
        set(inclusive_id_to_row)
        & set(sensitivity_id_to_row)
    )

    inclusive_dominant = np.argmax(
        inclusive_document_topic,
        axis=1,
    )

    sensitivity_dominant = np.argmax(
        sensitivity_document_topic,
        axis=1,
    )

    sensitivity_to_inclusive_topic = {
        int(sensitivity_index):
        int(inclusive_index)
        for inclusive_index, sensitivity_index
        in zip(
            inclusive_topic_indices,
            sensitivity_topic_indices,
            strict=True,
        )
    }

    inclusive_labels = []
    mapped_sensitivity_labels = []
    assignment_rows = []

    for comment_id in shared_comment_ids:
        inclusive_row = (
            inclusive_id_to_row[
                comment_id
            ]
        )

        sensitivity_row = (
            sensitivity_id_to_row[
                comment_id
            ]
        )

        inclusive_topic = int(
            inclusive_dominant[
                inclusive_row
            ]
        )

        raw_sensitivity_topic = int(
            sensitivity_dominant[
                sensitivity_row
            ]
        )

        mapped_sensitivity_topic = (
            sensitivity_to_inclusive_topic[
                raw_sensitivity_topic
            ]
        )

        inclusive_labels.append(
            inclusive_topic
        )

        mapped_sensitivity_labels.append(
            mapped_sensitivity_topic
        )

        assignment_rows.append(
            {
                "comment_id": comment_id,
                "video_id": inclusive_corpus.iloc[
                    inclusive_row
                ]["video_id"],
                "inclusive_topic_number": (
                    inclusive_topic + 1
                ),
                "raw_sensitivity_topic_number": (
                    raw_sensitivity_topic + 1
                ),
                "mapped_sensitivity_topic_number": (
                    mapped_sensitivity_topic + 1
                ),
                "same_dominant_topic_after_mapping": (
                    inclusive_topic
                    == mapped_sensitivity_topic
                ),
            }
        )

    assignment_comparison = pd.DataFrame(
        assignment_rows
    )

    dominant_assignment_agreement = float(
        assignment_comparison[
            "same_dominant_topic_after_mapping"
        ].mean()
    )

    adjusted_rand = float(
        adjusted_rand_score(
            inclusive_labels,
            mapped_sensitivity_labels,
        )
    )

    # -----------------------------------------------------------------
    # Sensitivity representative comments
    # -----------------------------------------------------------------

    representative_rows = []

    for topic_number in range(
        1,
        TOPIC_COUNT + 1,
    ):
        representative_rows.extend(
            select_representatives(
                corpus=sensitivity_corpus,
                document_topic=(
                    sensitivity_document_topic
                ),
                topic_number=topic_number,
            )
        )

    representatives = pd.DataFrame(
        representative_rows
    )

    # -----------------------------------------------------------------
    # Save model and matrices
    # -----------------------------------------------------------------

    sensitivity_model_path = (
        MODEL_DIR
        / "07_content_focused_nmf_k16.joblib"
    )

    sensitivity_document_topic_path = (
        MODEL_DIR
        / "07_content_focused_document_topic_k16.npz"
    )

    joblib.dump(
        sensitivity_model,
        sensitivity_model_path,
        compress=3,
    )

    sparse.save_npz(
        sensitivity_document_topic_path,
        sparse.csr_matrix(
            sensitivity_document_topic
        ),
        compressed=True,
    )

    # -----------------------------------------------------------------
    # Save tables
    # -----------------------------------------------------------------

    comparison_path = (
        TABLE_DIR
        / "07_preprocessing_topic_matching.csv"
    )

    assignment_path = (
        TABLE_DIR
        / "07_preprocessing_assignment_comparison.csv"
    )

    representative_path = (
        REVIEW_DIR
        / "07_content_focused_representative_comments.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    assignment_comparison.to_csv(
        assignment_path,
        index=False,
        encoding="utf-8-sig",
    )

    representatives.to_csv(
        representative_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Global diagnostics
    # -----------------------------------------------------------------

    mean_cosine = float(
        comparison[
            "shared_vocabulary_cosine_similarity"
        ].mean()
    )

    median_cosine = float(
        comparison[
            "shared_vocabulary_cosine_similarity"
        ].median()
    )

    minimum_cosine = float(
        comparison[
            "shared_vocabulary_cosine_similarity"
        ].min()
    )

    mean_jaccard = float(
        comparison[
            "top_20_term_jaccard"
        ].mean()
    )

    median_jaccard = float(
        comparison[
            "top_20_term_jaccard"
        ].median()
    )

    mean_prevalence_difference = float(
        comparison[
            "absolute_prevalence_difference"
        ].mean()
    )

    matches_090 = float(
        (
            comparison[
                "shared_vocabulary_cosine_similarity"
            ]
            >= 0.90
        ).mean()
    )

    matches_080 = float(
        (
            comparison[
                "shared_vocabulary_cosine_similarity"
            ]
            >= 0.80
        ).mean()
    )

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    configuration = {
        "analysis_stage": (
            "k=16 preprocessing sensitivity comparison"
        ),
        "inclusive_model": str(
            INCLUSIVE_MODEL_FILE
        ),
        "content_focused_matrix": str(
            SENSITIVITY_MATRIX_FILE
        ),
        "topic_count": TOPIC_COUNT,
        "topic_matching": (
            "Hungarian one-to-one assignment "
            "using cosine similarity after projection "
            "into shared vocabulary"
        ),
        "shared_vocabulary_features": (
            len(shared_features)
        ),
        "nmf_parameters": {
            "init": "nndsvda",
            "solver": "cd",
            "beta_loss": "frobenius",
            "tol": TOLERANCE,
            "max_iter": MAX_ITER,
            "random_state": RANDOM_STATE,
            "alpha_W": 0.0,
            "alpha_H": 0.0,
            "l1_ratio": 0.0,
            "shuffle": False,
        },
    }

    config_path = (
        CONFIG_DIR
        / "07_preprocessing_sensitivity_config.json"
    )

    config_path.write_text(
        json.dumps(
            configuration,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    report_lines = [
        "YOUTUBE RETROFIT NMF PREPROCESSING SENSITIVITY",
        "=" * 48,
        "",
        "Overall status: PASS",
        "",
        "Design",
        "------",
        (
            "Selected topic count: k=16"
        ),
        (
            "Primary representation: inclusive TF-IDF"
        ),
        (
            "Sensitivity representation: "
            "content-focused TF-IDF"
        ),
        (
            "Shared vocabulary features used for topic "
            f"comparison: {len(shared_features):,}"
        ),
        "",
        "Sensitivity model fit",
        "---------------------",
        (
            "Comments: "
            f"{sensitivity_matrix.shape[0]:,}"
        ),
        (
            "Features: "
            f"{sensitivity_matrix.shape[1]:,}"
        ),
        (
            "Iterations: "
            f"{sensitivity_model.n_iter_:,}"
        ),
        (
            "Convergence warning: "
            f"{convergence_warning}"
        ),
        (
            "Runtime seconds: "
            f"{runtime_seconds:.1f}"
        ),
        "",
        "Topic-structure comparison",
        "--------------------------",
        (
            "Mean matched cosine similarity: "
            f"{mean_cosine:.4f}"
        ),
        (
            "Median matched cosine similarity: "
            f"{median_cosine:.4f}"
        ),
        (
            "Minimum matched cosine similarity: "
            f"{minimum_cosine:.4f}"
        ),
        (
            "Matched topics >= 0.90 cosine: "
            f"{100 * matches_090:.1f}%"
        ),
        (
            "Matched topics >= 0.80 cosine: "
            f"{100 * matches_080:.1f}%"
        ),
        (
            "Mean top-20 term Jaccard: "
            f"{mean_jaccard:.4f}"
        ),
        (
            "Median top-20 term Jaccard: "
            f"{median_jaccard:.4f}"
        ),
        (
            "Mean absolute topic-prevalence difference: "
            f"{100 * mean_prevalence_difference:.2f} "
            "percentage points"
        ),
        "",
        "Comment-level comparison",
        "------------------------",
        (
            "Shared comments: "
            f"{len(shared_comment_ids):,}"
        ),
        (
            "Dominant-topic agreement after topic matching: "
            f"{100 * dominant_assignment_agreement:.1f}%"
        ),
        (
            "Adjusted Rand Index: "
            f"{adjusted_rand:.4f}"
        ),
        "",
        "Matched topic details",
        "---------------------",
    ]

    for _, row in comparison.iterrows():
        report_lines.extend(
            [
                (
                    f"Inclusive topic "
                    f"{int(row['inclusive_topic_number']):02d} "
                    f"-> sensitivity topic "
                    f"{int(row['matched_sensitivity_topic_number']):02d}"
                ),
                (
                    "  cosine="
                    f"{row['shared_vocabulary_cosine_similarity']:.4f}; "
                    "top-20 Jaccard="
                    f"{row['top_20_term_jaccard']:.4f}; "
                    "prevalence difference="
                    f"{100 * row['absolute_prevalence_difference']:.2f} pp"
                ),
                (
                    "  inclusive: "
                    f"{row['inclusive_top_20_terms']}"
                ),
                (
                    "  sensitivity: "
                    f"{row['sensitivity_top_20_terms']}"
                ),
                "",
            ]
        )

    report_lines.extend(
        [
            "Interpretive rule",
            "-----------------",
            (
                "The sensitivity representation is not expected "
                "to reproduce the inclusive model perfectly because "
                "generic platform, gratitude and strong-praise terms "
                "were deliberately removed. Particular attention "
                "should therefore be paid to whether substantive "
                "technical topics remain recognisable and stable."
            ),
            (
                "A reorganisation of the inclusive social-endorsement "
                "topic is expected and should not by itself be treated "
                "as evidence of poor preprocessing robustness."
            ),
            "",
            "Outputs",
            "-------",
            str(comparison_path),
            str(assignment_path),
            str(representative_path),
            str(sensitivity_model_path),
            str(sensitivity_document_topic_path),
            str(config_path),
        ]
    )

    report_path = (
        AUDIT_DIR
        / "07_preprocessing_sensitivity_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()