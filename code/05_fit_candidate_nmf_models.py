from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import cosine_similarity


NMF_ROOT = Path(__file__).resolve().parents[1]

CORPUS_FILE = (
    NMF_ROOT
    / "data"
    / "processed"
    / "02_nmf_model_corpus_conservative.csv"
)

MATRIX_FILE = (
    NMF_ROOT
    / "data"
    / "processed"
    / "02_tfidf_matrix_conservative.npz"
)

VECTORIZER_FILE = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "02_tfidf_vectorizer_conservative.joblib"
)

MODEL_DIR = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "05_inclusive_candidates"
)

TABLE_DIR = NMF_ROOT / "outputs" / "tables"
AUDIT_DIR = NMF_ROOT / "outputs" / "audit"
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
EXPECTED_FEATURES = 14_427

TOPIC_COUNTS = [8, 10, 12, 14, 16, 20]

RANDOM_STATE = 20_260_806
MAX_ITER = 500
TOLERANCE = 1e-4

TOP_TERMS_REPORTED = 20
TOP_TERMS_FOR_COHERENCE = 10
REPRESENTATIVE_COMMENTS_PER_TOPIC = 10
MAX_REPRESENTATIVES_PER_VIDEO = 2


def calculate_sha256(file_path: Path) -> str:
    """Return the SHA-256 hash of a file."""

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
    """Verify a frozen input artefact."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{file_path}"
        )

    observed_hash = calculate_sha256(file_path)

    if observed_hash != expected_hash:
        raise ValueError(
            f"{description} does not match the frozen artefact.\n"
            f"Expected SHA-256: {expected_hash}\n"
            f"Observed SHA-256: {observed_hash}"
        )


def topic_top_indices(
    components: np.ndarray,
    number_of_terms: int,
) -> np.ndarray:
    """Return descending top-feature indices for every topic."""

    return np.argsort(
        components,
        axis=1,
    )[:, -number_of_terms:][:, ::-1]


def calculate_topic_diversity(
    top_indices: np.ndarray,
) -> float:
    """Calculate the proportion of unique top terms."""

    unique_terms = np.unique(top_indices).size
    total_terms = top_indices.size

    return float(unique_terms / total_terms)


def calculate_topic_similarity(
    components: np.ndarray,
) -> tuple[float, float]:
    """Calculate mean and maximum inter-topic cosine similarity."""

    similarity = cosine_similarity(components)

    upper_triangle = similarity[
        np.triu_indices_from(
            similarity,
            k=1,
        )
    ]

    if upper_triangle.size == 0:
        return 0.0, 0.0

    return (
        float(upper_triangle.mean()),
        float(upper_triangle.max()),
    )


def calculate_topic_npmi(
    matrix: sparse.csr_matrix,
    top_indices: np.ndarray,
) -> np.ndarray:
    """Calculate mean NPMI for each topic's leading features."""

    document_count = matrix.shape[0]
    topic_scores: list[float] = []

    for feature_indices in top_indices:
        binary = matrix[:, feature_indices].copy()
        binary.data = np.ones_like(
            binary.data,
            dtype=np.float64,
        )

        document_frequency = np.asarray(
            binary.sum(axis=0)
        ).ravel()

        cooccurrence = (
            binary.T @ binary
        ).toarray()

        pair_scores: list[float] = []

        for first in range(len(feature_indices)):
            for second in range(
                first + 1,
                len(feature_indices),
            ):
                joint_count = cooccurrence[first, second]

                if joint_count <= 0:
                    pair_scores.append(-1.0)
                    continue

                probability_first = (
                    document_frequency[first]
                    / document_count
                )

                probability_second = (
                    document_frequency[second]
                    / document_count
                )

                joint_probability = (
                    joint_count
                    / document_count
                )

                pointwise_mutual_information = math.log(
                    joint_probability
                    / (
                        probability_first
                        * probability_second
                    )
                )

                denominator = -math.log(
                    joint_probability
                )

                if denominator == 0:
                    pair_scores.append(0.0)
                else:
                    pair_scores.append(
                        pointwise_mutual_information
                        / denominator
                    )

        topic_scores.append(
            float(np.mean(pair_scores))
            if pair_scores
            else float("nan")
        )

    return np.asarray(topic_scores)


def select_representative_comments(
    corpus: pd.DataFrame,
    topic_weights: np.ndarray,
    topic_number: int,
    topic_count: int,
) -> list[dict[str, object]]:
    """Select high-loading comments while limiting video repetition."""

    row_sums = topic_weights.sum(axis=1)

    relative_loading = np.divide(
        topic_weights[:, topic_number],
        row_sums,
        out=np.zeros(
            topic_weights.shape[0],
            dtype=np.float64,
        ),
        where=row_sums > 0,
    )

    ordered_indices = np.argsort(
        topic_weights[:, topic_number]
    )[::-1]

    selected: list[dict[str, object]] = []
    per_video_counts: dict[str, int] = {}

    display_column = (
        "comment_text_for_coding"
        if "comment_text_for_coding" in corpus.columns
        else "nmf_text_conservative"
    )

    optional_columns = [
        "retrofit_topic",
        "creator_type",
        "video_type",
        "primary_theme",
    ]

    for row_index in ordered_indices:
        if len(selected) >= REPRESENTATIVE_COMMENTS_PER_TOPIC:
            break

        video_id = str(
            corpus.iloc[row_index]["video_id"]
        )

        current_video_count = per_video_counts.get(
            video_id,
            0,
        )

        if (
            current_video_count
            >= MAX_REPRESENTATIVES_PER_VIDEO
        ):
            continue

        record = {
            "candidate_topic_count": topic_count,
            "topic_number": topic_number + 1,
            "representative_rank": len(selected) + 1,
            "tfidf_row_index": int(row_index),
            "topic_weight": float(
                topic_weights[
                    row_index,
                    topic_number,
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
            ][display_column],
        }

        for column in optional_columns:
            if column in corpus.columns:
                record[column] = corpus.iloc[
                    row_index
                ][column]

        selected.append(record)

        per_video_counts[video_id] = (
            current_video_count + 1
        )

    return selected


def main() -> None:
    for directory in (
        MODEL_DIR,
        TABLE_DIR,
        AUDIT_DIR,
        CONFIG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    verify_file(
        CORPUS_FILE,
        EXPECTED_CORPUS_SHA256,
        "Inclusive corpus",
    )

    verify_file(
        MATRIX_FILE,
        EXPECTED_MATRIX_SHA256,
        "Inclusive TF-IDF matrix",
    )

    verify_file(
        VECTORIZER_FILE,
        EXPECTED_VECTORIZER_SHA256,
        "Inclusive TF-IDF vectorizer",
    )

    corpus = pd.read_csv(
        CORPUS_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    matrix = sparse.load_npz(
        MATRIX_FILE
    ).tocsr()

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    feature_names = np.asarray(
        vectorizer.get_feature_names_out()
    )

    if len(corpus) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} corpus rows but "
            f"found {len(corpus):,}."
        )

    if matrix.shape != (
        EXPECTED_ROWS,
        EXPECTED_FEATURES,
    ):
        raise ValueError(
            "Unexpected inclusive matrix dimensions.\n"
            f"Expected: "
            f"{EXPECTED_ROWS:,} x {EXPECTED_FEATURES:,}\n"
            f"Observed: "
            f"{matrix.shape[0]:,} x "
            f"{matrix.shape[1]:,}"
        )

    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            "Matrix columns do not match vectorizer features."
        )

    matrix_frobenius_norm = float(
        np.sqrt(matrix.multiply(matrix).sum())
    )

    diagnostic_rows: list[dict[str, object]] = []
    term_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []
    topic_summary_rows: list[dict[str, object]] = []

    for topic_count in TOPIC_COUNTS:
        print()
        print(
            f"Fitting inclusive NMF with "
            f"{topic_count} topics..."
        )

        start_time = time.perf_counter()

        model = NMF(
            n_components=topic_count,
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

        with warnings.catch_warnings(
            record=True
        ) as recorded_warnings:
            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )

            document_topic = model.fit_transform(
                matrix
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

        components = model.components_

        top_20_indices = topic_top_indices(
            components,
            TOP_TERMS_REPORTED,
        )

        top_10_indices = top_20_indices[
            :,
            :TOP_TERMS_FOR_COHERENCE,
        ]

        topic_npmi = calculate_topic_npmi(
            matrix,
            top_10_indices,
        )

        topic_diversity = (
            calculate_topic_diversity(
                top_20_indices
            )
        )

        (
            mean_topic_similarity,
            maximum_topic_similarity,
        ) = calculate_topic_similarity(
            components
        )

        dominant_topic = np.argmax(
            document_topic,
            axis=1,
        )

        row_sums = document_topic.sum(
            axis=1
        )

        dominant_weight = np.max(
            document_topic,
            axis=1,
        )

        relative_dominant_loading = np.divide(
            dominant_weight,
            row_sums,
            out=np.zeros_like(
                dominant_weight
            ),
            where=row_sums > 0,
        )

        topic_counts = np.bincount(
            dominant_topic,
            minlength=topic_count,
        )

        topic_shares = (
            topic_counts
            / matrix.shape[0]
        )

        small_topic_count = int(
            (topic_shares < 0.01).sum()
        )

        relative_reconstruction_error = (
            model.reconstruction_err_
            / matrix_frobenius_norm
        )

        diagnostic_rows.append(
            {
                "representation": (
                    "inclusive_primary"
                ),
                "topic_count": topic_count,
                "runtime_seconds": runtime_seconds,
                "iterations": int(
                    model.n_iter_
                ),
                "maximum_iterations": MAX_ITER,
                "convergence_warning": (
                    convergence_warning
                ),
                "reconstruction_error": float(
                    model.reconstruction_err_
                ),
                "relative_reconstruction_error": float(
                    relative_reconstruction_error
                ),
                "mean_topic_npmi": float(
                    np.nanmean(topic_npmi)
                ),
                "minimum_topic_npmi": float(
                    np.nanmin(topic_npmi)
                ),
                "maximum_topic_npmi": float(
                    np.nanmax(topic_npmi)
                ),
                "topic_diversity_top_20": float(
                    topic_diversity
                ),
                "mean_intertopic_cosine_similarity": (
                    mean_topic_similarity
                ),
                "maximum_intertopic_cosine_similarity": (
                    maximum_topic_similarity
                ),
                "minimum_dominant_topic_comments": int(
                    topic_counts.min()
                ),
                "maximum_dominant_topic_comments": int(
                    topic_counts.max()
                ),
                "minimum_dominant_topic_share": float(
                    topic_shares.min()
                ),
                "maximum_dominant_topic_share": float(
                    topic_shares.max()
                ),
                "topics_below_one_percent": (
                    small_topic_count
                ),
                "median_relative_dominant_loading": (
                    float(
                        np.median(
                            relative_dominant_loading
                        )
                    )
                ),
                "mean_relative_dominant_loading": (
                    float(
                        np.mean(
                            relative_dominant_loading
                        )
                    )
                ),
            }
        )

        for topic_index in range(topic_count):
            feature_indices = top_20_indices[
                topic_index
            ]

            top_terms = feature_names[
                feature_indices
            ]

            top_weights = components[
                topic_index,
                feature_indices,
            ]

            for term_rank, (
                term,
                weight,
            ) in enumerate(
                zip(
                    top_terms,
                    top_weights,
                    strict=True,
                ),
                start=1,
            ):
                term_rows.append(
                    {
                        "representation": (
                            "inclusive_primary"
                        ),
                        "candidate_topic_count": (
                            topic_count
                        ),
                        "topic_number": (
                            topic_index + 1
                        ),
                        "term_rank": term_rank,
                        "term": term,
                        "term_weight": float(
                            weight
                        ),
                    }
                )

            topic_summary_rows.append(
                {
                    "representation": (
                        "inclusive_primary"
                    ),
                    "candidate_topic_count": (
                        topic_count
                    ),
                    "topic_number": (
                        topic_index + 1
                    ),
                    "topic_npmi_top_10": float(
                        topic_npmi[
                            topic_index
                        ]
                    ),
                    "dominant_comment_count": int(
                        topic_counts[
                            topic_index
                        ]
                    ),
                    "dominant_comment_share": float(
                        topic_shares[
                            topic_index
                        ]
                    ),
                    "top_10_terms": " | ".join(
                        top_terms[:10]
                    ),
                    "top_20_terms": " | ".join(
                        top_terms
                    ),
                }
            )

            representative_rows.extend(
                select_representative_comments(
                    corpus=corpus,
                    topic_weights=document_topic,
                    topic_number=topic_index,
                    topic_count=topic_count,
                )
            )

        model_path = (
            MODEL_DIR
            / (
                f"inclusive_nmf_"
                f"k{topic_count:02d}.joblib"
            )
        )

        document_topic_path = (
            MODEL_DIR
            / (
                f"inclusive_document_topic_"
                f"k{topic_count:02d}.npz"
            )
        )

        joblib.dump(
            model,
            model_path,
            compress=3,
        )

        sparse.save_npz(
            document_topic_path,
            sparse.csr_matrix(
                document_topic
            ),
            compressed=True,
        )

        print(
            f"Completed k={topic_count}: "
            f"error={model.reconstruction_err_:.4f}, "
            f"mean NPMI={np.nanmean(topic_npmi):.4f}, "
            f"diversity={topic_diversity:.4f}, "
            f"iterations={model.n_iter_}, "
            f"time={runtime_seconds:.1f}s"
        )

    diagnostics = pd.DataFrame(
        diagnostic_rows
    ).sort_values(
        "topic_count"
    )

    topic_terms = pd.DataFrame(
        term_rows
    )

    topic_summaries = pd.DataFrame(
        topic_summary_rows
    )

    representatives = pd.DataFrame(
        representative_rows
    )

    diagnostics_path = (
        TABLE_DIR
        / "05_candidate_model_diagnostics.csv"
    )

    topic_terms_path = (
        TABLE_DIR
        / "05_candidate_topic_terms.csv"
    )

    topic_summary_path = (
        TABLE_DIR
        / "05_candidate_topic_summaries.csv"
    )

    representatives_path = (
        TABLE_DIR
        / "05_candidate_representative_comments.csv"
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
        encoding="utf-8-sig",
    )

    topic_terms.to_csv(
        topic_terms_path,
        index=False,
        encoding="utf-8-sig",
    )

    topic_summaries.to_csv(
        topic_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    representatives.to_csv(
        representatives_path,
        index=False,
        encoding="utf-8-sig",
    )

    configuration = {
        "analysis_stage": (
            "Inclusive primary candidate NMF screening"
        ),
        "topic_counts": TOPIC_COUNTS,
        "input_corpus": str(CORPUS_FILE),
        "input_matrix": str(MATRIX_FILE),
        "input_vectorizer": str(
            VECTORIZER_FILE
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
        "diagnostics": {
            "coherence": (
                "Mean normalized pointwise mutual "
                "information across top 10 features"
            ),
            "topic_diversity": (
                "Unique features divided by all top-20 "
                "topic-feature positions"
            ),
            "topic_similarity": (
                "Cosine similarity between complete "
                "topic-term vectors"
            ),
            "representatives_per_topic": (
                REPRESENTATIVE_COMMENTS_PER_TOPIC
            ),
            "maximum_representatives_per_video": (
                MAX_REPRESENTATIVES_PER_VIDEO
            ),
        },
    }

    config_path = (
        CONFIG_DIR
        / "05_candidate_nmf_config.json"
    )

    config_path.write_text(
        json.dumps(
            configuration,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "YOUTUBE RETROFIT INCLUSIVE NMF CANDIDATE SCREENING",
        "=" * 52,
        "",
        "Overall status: PASS",
        "",
        "Input representation",
        "--------------------",
        (
            "Representation: inclusive primary "
            "TF-IDF matrix"
        ),
        f"Comments: {matrix.shape[0]:,}",
        f"Features: {matrix.shape[1]:,}",
        (
            "Unique videos represented: "
            f"{corpus['video_id'].nunique():,}"
        ),
        "",
        "Candidate models",
        "----------------",
        (
            "Topic counts: "
            + ", ".join(
                str(value)
                for value in TOPIC_COUNTS
            )
        ),
        "Initialisation: NNDSVDa",
        "Solver: coordinate descent",
        "Loss: Frobenius",
        f"Maximum iterations: {MAX_ITER:,}",
        "",
        "Diagnostic summary",
        "------------------",
    ]

    for _, row in diagnostics.iterrows():
        report_lines.append(
            (
                f"k={int(row['topic_count']):>2}: "
                f"relative error="
                f"{row['relative_reconstruction_error']:.5f}; "
                f"mean NPMI="
                f"{row['mean_topic_npmi']:.4f}; "
                f"diversity="
                f"{row['topic_diversity_top_20']:.3f}; "
                f"mean similarity="
                f"{row['mean_intertopic_cosine_similarity']:.3f}; "
                f"max similarity="
                f"{row['maximum_intertopic_cosine_similarity']:.3f}; "
                f"small topics="
                f"{int(row['topics_below_one_percent'])}; "
                f"iterations="
                f"{int(row['iterations'])}; "
                f"warning="
                f"{bool(row['convergence_warning'])}"
            )
        )

    report_lines.extend(
        [
            "",
            "Selection rule",
            "--------------",
            (
                "No topic count is selected automatically. "
                "The next stage will jointly review "
                "reconstruction improvement, topic coherence, "
                "topic diversity, redundancy, topic size and "
                "representative-comment interpretability."
            ),
            (
                "Stability testing will be restricted to the "
                "strongest two or three candidate topic counts."
            ),
            "",
            "Created outputs",
            "---------------",
            str(diagnostics_path),
            str(topic_terms_path),
            str(topic_summary_path),
            str(representatives_path),
            str(config_path),
            str(MODEL_DIR),
        ]
    )

    report_path = (
        AUDIT_DIR
        / "05_candidate_nmf_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()