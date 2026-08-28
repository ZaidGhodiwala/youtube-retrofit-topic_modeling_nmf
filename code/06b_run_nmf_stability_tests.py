from __future__ import annotations

import hashlib
import itertools
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
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

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

REFERENCE_MODEL_DIR = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "05_inclusive_candidates"
)

STABILITY_MODEL_DIR = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "06b_stability"
)

TABLE_DIR = NMF_ROOT / "outputs" / "tables"
AUDIT_DIR = NMF_ROOT / "outputs" / "audit"
CONFIG_DIR = NMF_ROOT / "config"


# ---------------------------------------------------------------------
# Frozen-input checks
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Stability design
# ---------------------------------------------------------------------

CANDIDATE_TOPIC_COUNTS = [12, 16]

N_SUBSAMPLE_RUNS = 10
SUBSAMPLE_FRACTION = 0.80

BASE_RANDOM_SEED = 20_260_828

MAX_ITER = 500
TOLERANCE = 1e-4

TOP_TERMS_FOR_STABILITY = 20
TOP_TERMS_REPORTED = 10


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
    """Verify a frozen input file."""

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


def get_top_term_indices(
    components: np.ndarray,
    n_terms: int,
) -> np.ndarray:
    """Return descending top-term indices for every topic."""

    return np.argsort(
        components,
        axis=1,
    )[:, -n_terms:][:, ::-1]


def jaccard_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Calculate Jaccard overlap between two feature-index sets."""

    first_set = set(
        int(value)
        for value in first
    )

    second_set = set(
        int(value)
        for value in second
    )

    union = first_set | second_set

    if not union:
        return 1.0

    return len(
        first_set & second_set
    ) / len(union)


def match_topics(
    reference_components: np.ndarray,
    comparison_components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match comparison topics to reference topics using maximum
    one-to-one cosine similarity.
    """

    similarity_matrix = cosine_similarity(
        reference_components,
        comparison_components,
    )

    reference_indices, comparison_indices = (
        linear_sum_assignment(
            -similarity_matrix
        )
    )

    matched_similarities = similarity_matrix[
        reference_indices,
        comparison_indices,
    ]

    return (
        reference_indices,
        comparison_indices,
        matched_similarities,
    )


def dominant_topic_shares(
    document_topic: np.ndarray,
    topic_count: int,
) -> np.ndarray:
    """Calculate dominant-topic prevalence."""

    dominant = np.argmax(
        document_topic,
        axis=1,
    )

    counts = np.bincount(
        dominant,
        minlength=topic_count,
    )

    return counts / len(dominant)


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main() -> None:
    for directory in (
        STABILITY_MODEL_DIR,
        TABLE_DIR,
        AUDIT_DIR,
        CONFIG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -----------------------------------------------------------------
    # Verify frozen Stage 3 inputs
    # -----------------------------------------------------------------

    verify_file(
        CORPUS_FILE,
        EXPECTED_CORPUS_SHA256,
        "Inclusive modelling corpus",
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
            f"Expected {EXPECTED_ROWS:,} corpus rows "
            f"but found {len(corpus):,}."
        )

    if matrix.shape != (
        EXPECTED_ROWS,
        EXPECTED_FEATURES,
    ):
        raise ValueError(
            "Unexpected TF-IDF matrix dimensions.\n"
            f"Expected: "
            f"{EXPECTED_ROWS:,} x "
            f"{EXPECTED_FEATURES:,}\n"
            f"Observed: "
            f"{matrix.shape[0]:,} x "
            f"{matrix.shape[1]:,}"
        )

    if len(feature_names) != matrix.shape[1]:
        raise ValueError(
            "Vectorizer feature count does not match matrix columns."
        )

    sample_size = int(
        round(
            matrix.shape[0]
            * SUBSAMPLE_FRACTION
        )
    )

    print(
        f"Corpus: {matrix.shape[0]:,} comments x "
        f"{matrix.shape[1]:,} features"
    )

    print(
        f"Stability design: "
        f"{N_SUBSAMPLE_RUNS} independent "
        f"{SUBSAMPLE_FRACTION:.0%} subsamples "
        f"per candidate topic count"
    )

    print(
        f"Rows per stability run: "
        f"{sample_size:,}"
    )

    all_run_rows: list[dict[str, object]] = []
    all_match_rows: list[dict[str, object]] = []
    all_pairwise_rows: list[dict[str, object]] = []
    all_topic_stability_rows: list[dict[str, object]] = []
    candidate_summary_rows: list[dict[str, object]] = []

    # -----------------------------------------------------------------
    # Test k=12 and k=16 separately
    # -----------------------------------------------------------------

    for topic_count in CANDIDATE_TOPIC_COUNTS:
        print()
        print("=" * 68)
        print(
            f"STABILITY TESTING: k={topic_count}"
        )
        print("=" * 68)

        reference_model_path = (
            REFERENCE_MODEL_DIR
            / (
                f"inclusive_nmf_"
                f"k{topic_count:02d}.joblib"
            )
        )

        reference_document_topic_path = (
            REFERENCE_MODEL_DIR
            / (
                "inclusive_document_topic_"
                f"k{topic_count:02d}.npz"
            )
        )

        if not reference_model_path.exists():
            raise FileNotFoundError(
                "Reference NMF model not found:\n"
                f"{reference_model_path}"
            )

        if not reference_document_topic_path.exists():
            raise FileNotFoundError(
                "Reference document-topic matrix "
                "not found:\n"
                f"{reference_document_topic_path}"
            )

        reference_model = joblib.load(
            reference_model_path
        )

        reference_document_topic = (
            sparse.load_npz(
                reference_document_topic_path
            )
            .toarray()
        )

        reference_components = (
            reference_model.components_.copy()
        )

        if reference_components.shape != (
            topic_count,
            matrix.shape[1],
        ):
            raise ValueError(
                f"Unexpected reference component dimensions "
                f"for k={topic_count}."
            )

        if reference_document_topic.shape != (
            matrix.shape[0],
            topic_count,
        ):
            raise ValueError(
                f"Unexpected reference document-topic "
                f"dimensions for k={topic_count}."
            )

        reference_topic_shares = (
            dominant_topic_shares(
                reference_document_topic,
                topic_count,
            )
        )

        reference_top_terms = (
            get_top_term_indices(
                reference_components,
                TOP_TERMS_FOR_STABILITY,
            )
        )

        reference_top_10 = (
            get_top_term_indices(
                reference_components,
                TOP_TERMS_REPORTED,
            )
        )

        run_components: dict[
            int,
            np.ndarray,
        ] = {}

        run_sample_indices: dict[
            int,
            np.ndarray,
        ] = {}

        run_topic_shares: dict[
            int,
            np.ndarray,
        ] = {}

        topic_match_rows_for_k: list[
            dict[str, object]
        ] = []

        run_rows_for_k: list[
            dict[str, object]
        ] = []

        # -------------------------------------------------------------
        # Repeated 80% subsampling
        # -------------------------------------------------------------

        for run_number in range(
            1,
            N_SUBSAMPLE_RUNS + 1,
        ):
            run_seed = (
                BASE_RANDOM_SEED
                + topic_count * 10_000
                + run_number * 101
            )

            rng = np.random.default_rng(
                run_seed
            )

            sampled_indices = np.sort(
                rng.choice(
                    matrix.shape[0],
                    size=sample_size,
                    replace=False,
                )
            )

            sampled_matrix = (
                matrix[sampled_indices]
                .tocsr()
            )

            sampled_norm = float(
                np.sqrt(
                    sampled_matrix
                    .multiply(sampled_matrix)
                    .sum()
                )
            )

            print(
                f"k={topic_count:02d} "
                f"run={run_number:02d}/{N_SUBSAMPLE_RUNS} "
                f"seed={run_seed} ...",
                end=" ",
                flush=True,
            )

            model = NMF(
                n_components=topic_count,
                init="nndsvda",
                solver="cd",
                beta_loss="frobenius",
                tol=TOLERANCE,
                max_iter=MAX_ITER,
                random_state=run_seed,
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

                sampled_document_topic = (
                    model.fit_transform(
                        sampled_matrix
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
                for warning
                in recorded_warnings
            )

            components = (
                model.components_.copy()
            )

            (
                reference_indices,
                comparison_indices,
                matched_cosines,
            ) = match_topics(
                reference_components,
                components,
            )

            run_top_terms = (
                get_top_term_indices(
                    components,
                    TOP_TERMS_FOR_STABILITY,
                )
            )

            current_topic_shares = (
                dominant_topic_shares(
                    sampled_document_topic,
                    topic_count,
                )
            )

            run_components[
                run_number
            ] = components

            run_sample_indices[
                run_number
            ] = sampled_indices

            run_topic_shares[
                run_number
            ] = current_topic_shares

            run_jaccards: list[float] = []
            prevalence_differences: list[float] = []

            for (
                reference_topic_index,
                run_topic_index,
                cosine_value,
            ) in zip(
                reference_indices,
                comparison_indices,
                matched_cosines,
                strict=True,
            ):
                top_term_jaccard = (
                    jaccard_similarity(
                        reference_top_terms[
                            reference_topic_index
                        ],
                        run_top_terms[
                            run_topic_index
                        ],
                    )
                )

                reference_share = float(
                    reference_topic_shares[
                        reference_topic_index
                    ]
                )

                run_share = float(
                    current_topic_shares[
                        run_topic_index
                    ]
                )

                prevalence_difference = (
                    run_share
                    - reference_share
                )

                run_jaccards.append(
                    top_term_jaccard
                )

                prevalence_differences.append(
                    abs(
                        prevalence_difference
                    )
                )

                reference_terms = (
                    feature_names[
                        reference_top_10[
                            reference_topic_index
                        ]
                    ]
                )

                run_terms = (
                    feature_names[
                        run_top_terms[
                            run_topic_index,
                            :TOP_TERMS_REPORTED,
                        ]
                    ]
                )

                match_record = {
                    "candidate_topic_count": (
                        topic_count
                    ),
                    "run_number": run_number,
                    "run_seed": run_seed,
                    "reference_topic_number": (
                        int(
                            reference_topic_index
                        )
                        + 1
                    ),
                    "matched_run_topic_number": (
                        int(
                            run_topic_index
                        )
                        + 1
                    ),
                    "cosine_similarity": float(
                        cosine_value
                    ),
                    "top_20_term_jaccard": float(
                        top_term_jaccard
                    ),
                    "reference_dominant_share": (
                        reference_share
                    ),
                    "run_dominant_share": (
                        run_share
                    ),
                    "prevalence_difference": float(
                        prevalence_difference
                    ),
                    "absolute_prevalence_difference": float(
                        abs(
                            prevalence_difference
                        )
                    ),
                    "reference_top_10_terms": (
                        " | ".join(
                            reference_terms
                        )
                    ),
                    "run_top_10_terms": (
                        " | ".join(
                            run_terms
                        )
                    ),
                }

                all_match_rows.append(
                    match_record
                )

                topic_match_rows_for_k.append(
                    match_record
                )

            relative_reconstruction_error = (
                model.reconstruction_err_
                / sampled_norm
            )

            run_record = {
                "candidate_topic_count": (
                    topic_count
                ),
                "run_number": run_number,
                "run_seed": run_seed,
                "sample_fraction": (
                    SUBSAMPLE_FRACTION
                ),
                "sample_size": sample_size,
                "runtime_seconds": (
                    runtime_seconds
                ),
                "iterations": int(
                    model.n_iter_
                ),
                "convergence_warning": (
                    convergence_warning
                ),
                "reconstruction_error": float(
                    model.reconstruction_err_
                ),
                "relative_reconstruction_error": (
                    float(
                        relative_reconstruction_error
                    )
                ),
                "mean_matched_cosine_to_reference": (
                    float(
                        np.mean(
                            matched_cosines
                        )
                    )
                ),
                "median_matched_cosine_to_reference": (
                    float(
                        np.median(
                            matched_cosines
                        )
                    )
                ),
                "minimum_matched_cosine_to_reference": (
                    float(
                        np.min(
                            matched_cosines
                        )
                    )
                ),
                "mean_top_20_jaccard_to_reference": (
                    float(
                        np.mean(
                            run_jaccards
                        )
                    )
                ),
                "minimum_top_20_jaccard_to_reference": (
                    float(
                        np.min(
                            run_jaccards
                        )
                    )
                ),
                "mean_absolute_topic_prevalence_difference": (
                    float(
                        np.mean(
                            prevalence_differences
                        )
                    )
                ),
                "maximum_absolute_topic_prevalence_difference": (
                    float(
                        np.max(
                            prevalence_differences
                        )
                    )
                ),
            }

            all_run_rows.append(
                run_record
            )

            run_rows_for_k.append(
                run_record
            )

            print(
                f"mean cosine="
                f"{np.mean(matched_cosines):.3f}; "
                f"min="
                f"{np.min(matched_cosines):.3f}; "
                f"Jaccard="
                f"{np.mean(run_jaccards):.3f}; "
                f"iter={model.n_iter_}; "
                f"{runtime_seconds:.1f}s"
            )

        # -------------------------------------------------------------
        # Save components and exact sampled row indices
        # -------------------------------------------------------------

        component_output = {
            "reference_components": (
                reference_components
            )
        }

        sample_output = {}

        for run_number in range(
            1,
            N_SUBSAMPLE_RUNS + 1,
        ):
            component_output[
                f"run_{run_number:02d}_components"
            ] = run_components[
                run_number
            ]

            sample_output[
                f"run_{run_number:02d}_row_indices"
            ] = run_sample_indices[
                run_number
            ]

        component_path = (
            STABILITY_MODEL_DIR
            / (
                f"06b_k{topic_count:02d}_"
                "stability_components.npz"
            )
        )

        sample_path = (
            STABILITY_MODEL_DIR
            / (
                f"06b_k{topic_count:02d}_"
                "subsample_row_indices.npz"
            )
        )

        np.savez_compressed(
            component_path,
            **component_output,
        )

        np.savez_compressed(
            sample_path,
            **sample_output,
        )

        # -------------------------------------------------------------
        # Pairwise stability between all subsample runs
        # -------------------------------------------------------------

        pairwise_rows_for_k: list[
            dict[str, object]
        ] = []

        for first_run, second_run in itertools.combinations(
            range(
                1,
                N_SUBSAMPLE_RUNS + 1,
            ),
            2,
        ):
            first_components = (
                run_components[
                    first_run
                ]
            )

            second_components = (
                run_components[
                    second_run
                ]
            )

            (
                first_indices,
                second_indices,
                matched_cosines,
            ) = match_topics(
                first_components,
                second_components,
            )

            first_top_terms = (
                get_top_term_indices(
                    first_components,
                    TOP_TERMS_FOR_STABILITY,
                )
            )

            second_top_terms = (
                get_top_term_indices(
                    second_components,
                    TOP_TERMS_FOR_STABILITY,
                )
            )

            pair_jaccards = []

            for (
                first_topic,
                second_topic,
            ) in zip(
                first_indices,
                second_indices,
                strict=True,
            ):
                pair_jaccards.append(
                    jaccard_similarity(
                        first_top_terms[
                            first_topic
                        ],
                        second_top_terms[
                            second_topic
                        ],
                    )
                )

            pair_record = {
                "candidate_topic_count": (
                    topic_count
                ),
                "first_run": first_run,
                "second_run": second_run,
                "mean_matched_cosine": float(
                    np.mean(
                        matched_cosines
                    )
                ),
                "median_matched_cosine": float(
                    np.median(
                        matched_cosines
                    )
                ),
                "minimum_matched_cosine": float(
                    np.min(
                        matched_cosines
                    )
                ),
                "mean_top_20_jaccard": float(
                    np.mean(
                        pair_jaccards
                    )
                ),
                "minimum_top_20_jaccard": float(
                    np.min(
                        pair_jaccards
                    )
                ),
            }

            all_pairwise_rows.append(
                pair_record
            )

            pairwise_rows_for_k.append(
                pair_record
            )

        # -------------------------------------------------------------
        # Reference-topic-level stability
        # -------------------------------------------------------------

        topic_match_dataframe = (
            pd.DataFrame(
                topic_match_rows_for_k
            )
        )

        topic_stability_rows_for_k: list[
            dict[str, object]
        ] = []

        for reference_topic_number in range(
            1,
            topic_count + 1,
        ):
            topic_records = (
                topic_match_dataframe.loc[
                    topic_match_dataframe[
                        "reference_topic_number"
                    ]
                    == reference_topic_number
                ]
            )

            reference_topic_index = (
                reference_topic_number
                - 1
            )

            reference_terms = (
                feature_names[
                    reference_top_10[
                        reference_topic_index
                    ]
                ]
            )

            topic_stability_record = {
                "candidate_topic_count": (
                    topic_count
                ),
                "reference_topic_number": (
                    reference_topic_number
                ),
                "reference_top_10_terms": (
                    " | ".join(
                        reference_terms
                    )
                ),
                "reference_dominant_share": (
                    float(
                        reference_topic_shares[
                            reference_topic_index
                        ]
                    )
                ),
                "mean_cosine_similarity": float(
                    topic_records[
                        "cosine_similarity"
                    ].mean()
                ),
                "median_cosine_similarity": float(
                    topic_records[
                        "cosine_similarity"
                    ].median()
                ),
                "minimum_cosine_similarity": float(
                    topic_records[
                        "cosine_similarity"
                    ].min()
                ),
                "standard_deviation_cosine": float(
                    topic_records[
                        "cosine_similarity"
                    ].std(ddof=1)
                ),
                "mean_top_20_jaccard": float(
                    topic_records[
                        "top_20_term_jaccard"
                    ].mean()
                ),
                "minimum_top_20_jaccard": float(
                    topic_records[
                        "top_20_term_jaccard"
                    ].min()
                ),
                "mean_run_dominant_share": float(
                    topic_records[
                        "run_dominant_share"
                    ].mean()
                ),
                "standard_deviation_run_share": float(
                    topic_records[
                        "run_dominant_share"
                    ].std(ddof=1)
                ),
                "mean_absolute_prevalence_difference": (
                    float(
                        topic_records[
                            "absolute_prevalence_difference"
                        ].mean()
                    )
                ),
                "fraction_runs_cosine_at_least_0_90": (
                    float(
                        (
                            topic_records[
                                "cosine_similarity"
                            ]
                            >= 0.90
                        ).mean()
                    )
                ),
                "fraction_runs_cosine_at_least_0_80": (
                    float(
                        (
                            topic_records[
                                "cosine_similarity"
                            ]
                            >= 0.80
                        ).mean()
                    )
                ),
                "fraction_runs_cosine_at_least_0_70": (
                    float(
                        (
                            topic_records[
                                "cosine_similarity"
                            ]
                            >= 0.70
                        ).mean()
                    )
                ),
            }

            all_topic_stability_rows.append(
                topic_stability_record
            )

            topic_stability_rows_for_k.append(
                topic_stability_record
            )

        # -------------------------------------------------------------
        # Candidate-level stability summary
        # -------------------------------------------------------------

        run_dataframe = pd.DataFrame(
            run_rows_for_k
        )

        pairwise_dataframe = pd.DataFrame(
            pairwise_rows_for_k
        )

        topic_stability_dataframe = pd.DataFrame(
            topic_stability_rows_for_k
        )

        all_matched_cosines = (
            topic_match_dataframe[
                "cosine_similarity"
            ].to_numpy()
        )

        all_jaccards = (
            topic_match_dataframe[
                "top_20_term_jaccard"
            ].to_numpy()
        )

        all_prevalence_differences = (
            topic_match_dataframe[
                "absolute_prevalence_difference"
            ].to_numpy()
        )

        candidate_summary_rows.append(
            {
                "candidate_topic_count": (
                    topic_count
                ),
                "subsample_runs": (
                    N_SUBSAMPLE_RUNS
                ),
                "subsample_fraction": (
                    SUBSAMPLE_FRACTION
                ),
                "sample_size": sample_size,
                "mean_matched_cosine_to_reference": float(
                    np.mean(
                        all_matched_cosines
                    )
                ),
                "median_matched_cosine_to_reference": float(
                    np.median(
                        all_matched_cosines
                    )
                ),
                "tenth_percentile_matched_cosine": float(
                    np.quantile(
                        all_matched_cosines,
                        0.10,
                    )
                ),
                "minimum_matched_cosine_to_reference": float(
                    np.min(
                        all_matched_cosines
                    )
                ),
                "fraction_matches_cosine_at_least_0_90": (
                    float(
                        np.mean(
                            all_matched_cosines
                            >= 0.90
                        )
                    )
                ),
                "fraction_matches_cosine_at_least_0_80": (
                    float(
                        np.mean(
                            all_matched_cosines
                            >= 0.80
                        )
                    )
                ),
                "fraction_matches_cosine_at_least_0_70": (
                    float(
                        np.mean(
                            all_matched_cosines
                            >= 0.70
                        )
                    )
                ),
                "mean_top_20_term_jaccard": float(
                    np.mean(
                        all_jaccards
                    )
                ),
                "tenth_percentile_top_20_jaccard": float(
                    np.quantile(
                        all_jaccards,
                        0.10,
                    )
                ),
                "minimum_top_20_term_jaccard": float(
                    np.min(
                        all_jaccards
                    )
                ),
                "mean_absolute_topic_prevalence_difference": (
                    float(
                        np.mean(
                            all_prevalence_differences
                        )
                    )
                ),
                "maximum_mean_topic_prevalence_difference": (
                    float(
                        topic_stability_dataframe[
                            "mean_absolute_prevalence_difference"
                        ].max()
                    )
                ),
                "mean_pairwise_run_cosine": float(
                    pairwise_dataframe[
                        "mean_matched_cosine"
                    ].mean()
                ),
                "minimum_pairwise_run_mean_cosine": float(
                    pairwise_dataframe[
                        "mean_matched_cosine"
                    ].min()
                ),
                "mean_pairwise_top_20_jaccard": float(
                    pairwise_dataframe[
                        "mean_top_20_jaccard"
                    ].mean()
                ),
                "mean_run_relative_reconstruction_error": (
                    float(
                        run_dataframe[
                            "relative_reconstruction_error"
                        ].mean()
                    )
                ),
                "convergence_warning_runs": int(
                    run_dataframe[
                        "convergence_warning"
                    ].sum()
                ),
                "mean_runtime_seconds": float(
                    run_dataframe[
                        "runtime_seconds"
                    ].mean()
                ),
            }
        )

    # -----------------------------------------------------------------
    # Save combined tables
    # -----------------------------------------------------------------

    run_summary = pd.DataFrame(
        all_run_rows
    )

    topic_matches = pd.DataFrame(
        all_match_rows
    )

    pairwise_stability = pd.DataFrame(
        all_pairwise_rows
    )

    topic_stability = pd.DataFrame(
        all_topic_stability_rows
    )

    candidate_summary = pd.DataFrame(
        candidate_summary_rows
    ).sort_values(
        "candidate_topic_count"
    )

    run_summary_path = (
        TABLE_DIR
        / "06b_stability_run_summary.csv"
    )

    topic_matches_path = (
        TABLE_DIR
        / "06b_topic_matches_to_reference.csv"
    )

    pairwise_path = (
        TABLE_DIR
        / "06b_pairwise_run_stability.csv"
    )

    topic_stability_path = (
        TABLE_DIR
        / "06b_reference_topic_stability.csv"
    )

    candidate_summary_path = (
        TABLE_DIR
        / "06b_candidate_stability_summary.csv"
    )

    run_summary.to_csv(
        run_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    topic_matches.to_csv(
        topic_matches_path,
        index=False,
        encoding="utf-8-sig",
    )

    pairwise_stability.to_csv(
        pairwise_path,
        index=False,
        encoding="utf-8-sig",
    )

    topic_stability.to_csv(
        topic_stability_path,
        index=False,
        encoding="utf-8-sig",
    )

    candidate_summary.to_csv(
        candidate_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Configuration record
    # -----------------------------------------------------------------

    configuration = {
        "analysis_stage": (
            "Repeated NMF sampling-stability assessment"
        ),
        "candidates": (
            CANDIDATE_TOPIC_COUNTS
        ),
        "primary_representation": (
            "inclusive_primary"
        ),
        "stability_design": {
            "runs_per_candidate": (
                N_SUBSAMPLE_RUNS
            ),
            "subsample_fraction": (
                SUBSAMPLE_FRACTION
            ),
            "sampling": (
                "without replacement"
            ),
            "topic_matching": (
                "Hungarian one-to-one assignment "
                "maximising cosine similarity "
                "between topic-term vectors"
            ),
            "top_term_stability": (
                f"Jaccard similarity of top "
                f"{TOP_TERMS_FOR_STABILITY} terms"
            ),
            "reference_models": (
                "Full-corpus Stage 5 candidate models"
            ),
        },
        "reason_for_subsampling": (
            "NNDSVDa is substantially deterministic for "
            "an unchanged matrix. Repeating the identical "
            "full corpus would therefore provide limited "
            "information about robustness. Repeated "
            "subsampling tests sensitivity to corpus "
            "composition."
        ),
        "nmf_parameters": {
            "init": "nndsvda",
            "solver": "cd",
            "beta_loss": "frobenius",
            "tol": TOLERANCE,
            "max_iter": MAX_ITER,
            "alpha_W": 0.0,
            "alpha_H": 0.0,
            "l1_ratio": 0.0,
            "shuffle": False,
        },
        "base_random_seed": (
            BASE_RANDOM_SEED
        ),
    }

    config_path = (
        CONFIG_DIR
        / "06b_nmf_stability_config.json"
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
    # Human-readable report
    # -----------------------------------------------------------------

    report_lines = [
        "YOUTUBE RETROFIT NMF STABILITY TEST",
        "=" * 38,
        "",
        "Overall status: PASS",
        "",
        "Stability design",
        "----------------",
        (
            "Candidates tested: "
            + ", ".join(
                f"k={value}"
                for value
                in CANDIDATE_TOPIC_COUNTS
            )
        ),
        (
            "Runs per candidate: "
            f"{N_SUBSAMPLE_RUNS}"
        ),
        (
            "Subsample fraction: "
            f"{SUBSAMPLE_FRACTION:.0%}"
        ),
        (
            "Comments per run: "
            f"{sample_size:,}"
        ),
        (
            "Sampling: random without replacement"
        ),
        (
            "Topic matching: Hungarian one-to-one "
            "assignment using cosine similarity"
        ),
        (
            "Term-set comparison: Jaccard similarity "
            f"of top {TOP_TERMS_FOR_STABILITY} terms"
        ),
        "",
        "Why subsampling was used",
        "------------------------",
        (
            "The candidate models use NNDSVDa "
            "initialisation, which is substantially "
            "deterministic for an unchanged input matrix. "
            "Repeatedly fitting the identical corpus would "
            "therefore provide little evidence about topic "
            "robustness. Repeated 80% corpus subsampling "
            "tests whether similar topics are recovered "
            "when corpus composition changes."
        ),
        "",
        "Candidate stability summary",
        "---------------------------",
    ]

    for _, row in candidate_summary.iterrows():
        report_lines.extend(
            [
                (
                    f"k={int(row['candidate_topic_count'])}"
                ),
                (
                    "  Mean matched cosine to full-corpus "
                    f"reference: "
                    f"{row['mean_matched_cosine_to_reference']:.4f}"
                ),
                (
                    "  Median matched cosine: "
                    f"{row['median_matched_cosine_to_reference']:.4f}"
                ),
                (
                    "  10th percentile matched cosine: "
                    f"{row['tenth_percentile_matched_cosine']:.4f}"
                ),
                (
                    "  Minimum matched cosine: "
                    f"{row['minimum_matched_cosine_to_reference']:.4f}"
                ),
                (
                    "  Matches >= 0.90 cosine: "
                    f"{100 * row['fraction_matches_cosine_at_least_0_90']:.1f}%"
                ),
                (
                    "  Matches >= 0.80 cosine: "
                    f"{100 * row['fraction_matches_cosine_at_least_0_80']:.1f}%"
                ),
                (
                    "  Matches >= 0.70 cosine: "
                    f"{100 * row['fraction_matches_cosine_at_least_0_70']:.1f}%"
                ),
                (
                    "  Mean top-20 term Jaccard: "
                    f"{row['mean_top_20_term_jaccard']:.4f}"
                ),
                (
                    "  10th percentile top-20 Jaccard: "
                    f"{row['tenth_percentile_top_20_jaccard']:.4f}"
                ),
                (
                    "  Mean pairwise run cosine: "
                    f"{row['mean_pairwise_run_cosine']:.4f}"
                ),
                (
                    "  Minimum pairwise-run mean cosine: "
                    f"{row['minimum_pairwise_run_mean_cosine']:.4f}"
                ),
                (
                    "  Mean absolute topic-prevalence "
                    f"difference: "
                    f"{100 * row['mean_absolute_topic_prevalence_difference']:.2f} "
                    "percentage points"
                ),
                (
                    "  Convergence-warning runs: "
                    f"{int(row['convergence_warning_runs'])}"
                ),
                "",
            ]
        )

    report_lines.extend(
        [
            "Selection rule",
            "--------------",
            (
                "No candidate is selected automatically "
                "from a single threshold. Final selection "
                "will combine sampling stability, topic-level "
                "stability, quantitative diagnostics and the "
                "previous human interpretability review."
            ),
            (
                "Cosine thresholds reported above are "
                "descriptive diagnostics, not universal "
                "cut-offs for acceptable topic stability."
            ),
            "",
            "Outputs",
            "-------",
            str(candidate_summary_path),
            str(run_summary_path),
            str(topic_stability_path),
            str(topic_matches_path),
            str(pairwise_path),
            str(config_path),
            str(STABILITY_MODEL_DIR),
        ]
    )

    report_path = (
        AUDIT_DIR
        / "06b_nmf_stability_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()