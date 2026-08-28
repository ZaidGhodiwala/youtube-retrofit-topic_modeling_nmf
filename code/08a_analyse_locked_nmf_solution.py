from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


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

DOCUMENT_TOPIC_FILE = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "05_inclusive_candidates"
    / "inclusive_document_topic_k16.npz"
)

LOCKED_INTERPRETATION_FILE = (
    NMF_ROOT
    / "outputs"
    / "review"
    / "07c_final_k16_topic_interpretation_locked.csv"
)

TABLE_DIR = (
    NMF_ROOT
    / "outputs"
    / "tables"
)

AUDIT_DIR = (
    NMF_ROOT
    / "outputs"
    / "audit"
)

CONFIG_DIR = (
    NMF_ROOT
    / "config"
)


# ---------------------------------------------------------------------
# Frozen interpretation check
# ---------------------------------------------------------------------

EXPECTED_LOCKED_INTERPRETATION_SHA256 = (
    "de464bf50139656e1210e93a7b4d5dcc"
    "4c51cb510d57c05298abe37ca9a687c3"
)

TOPIC_COUNT = 16

EXPECTED_COMMENTS = 42_443


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sha256(path: Path) -> str:
    """Return SHA-256 hash for a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def verify_locked_interpretation() -> None:
    """Confirm that the frozen interpretation has not changed."""

    if not LOCKED_INTERPRETATION_FILE.exists():
        raise FileNotFoundError(
            "Locked interpretation file not found:\n"
            f"{LOCKED_INTERPRETATION_FILE}"
        )

    observed_hash = sha256(
        LOCKED_INTERPRETATION_FILE
    )

    if (
        observed_hash
        != EXPECTED_LOCKED_INTERPRETATION_SHA256
    ):
        raise ValueError(
            "Locked interpretation file has changed.\n"
            f"Expected SHA-256: "
            f"{EXPECTED_LOCKED_INTERPRETATION_SHA256}\n"
            f"Observed SHA-256: "
            f"{observed_hash}"
        )


def percentile(
    values: np.ndarray,
    q: float,
) -> float:
    """Calculate a percentile safely."""

    if len(values) == 0:
        return float("nan")

    return float(
        np.quantile(
            values,
            q,
        )
    )


def entropy_from_relative_weights(
    weights: np.ndarray,
) -> np.ndarray:
    """
    Shannon entropy for row-normalised NMF weights.

    These are relative NMF weights, not probabilities in a
    generative-topic-model sense. Entropy is used only as a
    descriptive measure of how concentrated or diffuse each
    comment's topic membership is.
    """

    safe_weights = np.where(
        weights > 0,
        weights,
        1.0,
    )

    entropy = -np.sum(
        np.where(
            weights > 0,
            weights * np.log(
                safe_weights
            ),
            0.0,
        ),
        axis=1,
    )

    return entropy


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    for directory in (
        TABLE_DIR,
        AUDIT_DIR,
        CONFIG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -----------------------------------------------------------------
    # Verify files
    # -----------------------------------------------------------------

    verify_locked_interpretation()

    for file_path in (
        CORPUS_FILE,
        DOCUMENT_TOPIC_FILE,
    ):
        if not file_path.exists():
            raise FileNotFoundError(
                "Required file not found:\n"
                f"{file_path}"
            )

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------

    corpus = pd.read_csv(
        CORPUS_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    document_topic = (
        sparse.load_npz(
            DOCUMENT_TOPIC_FILE
        )
        .toarray()
        .astype(
            np.float64,
            copy=False,
        )
    )

    locked = pd.read_csv(
        LOCKED_INTERPRETATION_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if len(corpus) != EXPECTED_COMMENTS:
        raise ValueError(
            f"Expected {EXPECTED_COMMENTS:,} "
            f"comments but found {len(corpus):,}."
        )

    if document_topic.shape != (
        EXPECTED_COMMENTS,
        TOPIC_COUNT,
    ):
        raise ValueError(
            "Unexpected document-topic matrix shape.\n"
            f"Expected: "
            f"{EXPECTED_COMMENTS:,} x {TOPIC_COUNT}\n"
            f"Observed: "
            f"{document_topic.shape[0]:,} x "
            f"{document_topic.shape[1]:,}"
        )

    if len(locked) != TOPIC_COUNT:
        raise ValueError(
            "Locked interpretation does not contain "
            "exactly 16 topics."
        )

    topic_numbers = (
        locked[
            "topic_number"
        ]
        .astype(int)
        .tolist()
    )

    if topic_numbers != list(
        range(
            1,
            TOPIC_COUNT + 1,
        )
    ):
        raise ValueError(
            "Locked topics are not ordered 1-16."
        )

    topic_labels = {
        int(row["topic_number"]):
        str(row["researcher_final_label"])
        for _, row in locked.iterrows()
    }

    topic_types = {
        int(row["topic_number"]):
        str(row["researcher_topic_type"])
        for _, row in locked.iterrows()
    }

    topic_confidence = {
        int(row["topic_number"]):
        int(row["researcher_confidence_1_to_5"])
        for _, row in locked.iterrows()
    }

    # -----------------------------------------------------------------
    # Convert NMF W matrix into relative within-comment weights
    #
    # Raw NMF weights are not probabilities.
    # Row-normalisation is used only to make relative contribution
    # within a comment interpretable and comparable.
    # -----------------------------------------------------------------

    row_sums = document_topic.sum(
        axis=1
    )

    zero_rows = (
        row_sums <= 0
    )

    if zero_rows.any():
        raise ValueError(
            f"{int(zero_rows.sum())} document-topic rows "
            "have zero total NMF weight."
        )

    relative_weights = (
        document_topic
        / row_sums[:, None]
    )

    row_sum_check = (
        relative_weights.sum(
            axis=1
        )
    )

    if not np.allclose(
        row_sum_check,
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Relative topic weights do not sum to 1."
        )

    # -----------------------------------------------------------------
    # Dominant and secondary topics
    # -----------------------------------------------------------------

    ordered_topics = np.argsort(
        relative_weights,
        axis=1,
    )

    dominant_indices = (
        ordered_topics[
            :,
            -1,
        ]
    )

    secondary_indices = (
        ordered_topics[
            :,
            -2,
        ]
    )

    dominant_relative_weights = (
        relative_weights[
            np.arange(
                len(corpus)
            ),
            dominant_indices,
        ]
    )

    secondary_relative_weights = (
        relative_weights[
            np.arange(
                len(corpus)
            ),
            secondary_indices,
        ]
    )

    dominance_margin = (
        dominant_relative_weights
        - secondary_relative_weights
    )

    entropy = (
        entropy_from_relative_weights(
            relative_weights
        )
    )

    effective_topic_count = np.exp(
        entropy
    )

    # -----------------------------------------------------------------
    # Topic prevalence
    #
    # Two complementary measures:
    #
    # 1. Dominant prevalence:
    #    fraction of comments for which the topic has the highest
    #    relative NMF weight.
    #
    # 2. Soft prevalence:
    #    mean relative NMF weight across all comments.
    #
    # The second measure preserves NMF's multi-topic structure.
    # -----------------------------------------------------------------

    prevalence_rows = []

    for topic_index in range(
        TOPIC_COUNT
    ):
        topic_number = (
            topic_index + 1
        )

        dominant_mask = (
            dominant_indices
            == topic_index
        )

        topic_dominant_weights = (
            dominant_relative_weights[
                dominant_mask
            ]
        )

        topic_secondary_weights = (
            secondary_relative_weights[
                dominant_mask
            ]
        )

        topic_margins = (
            dominance_margin[
                dominant_mask
            ]
        )

        topic_effective_counts = (
            effective_topic_count[
                dominant_mask
            ]
        )

        dominant_count = int(
            dominant_mask.sum()
        )

        dominant_share = (
            dominant_count
            / len(corpus)
        )

        soft_prevalence = float(
            relative_weights[
                :,
                topic_index,
            ].mean()
        )

        prevalence_rows.append(
            {
                "topic_number": (
                    topic_number
                ),
                "topic_label": (
                    topic_labels[
                        topic_number
                    ]
                ),
                "topic_type": (
                    topic_types[
                        topic_number
                    ]
                ),
                "interpretation_confidence": (
                    topic_confidence[
                        topic_number
                    ]
                ),
                "dominant_comment_count": (
                    dominant_count
                ),
                "dominant_comment_share": (
                    dominant_share
                ),
                "soft_prevalence_mean_relative_weight": (
                    soft_prevalence
                ),
                "dominant_share_minus_soft_prevalence": (
                    dominant_share
                    - soft_prevalence
                ),
                "mean_dominant_relative_weight": (
                    float(
                        topic_dominant_weights.mean()
                    )
                ),
                "median_dominant_relative_weight": (
                    float(
                        np.median(
                            topic_dominant_weights
                        )
                    )
                ),
                "dominant_relative_weight_q25": (
                    percentile(
                        topic_dominant_weights,
                        0.25,
                    )
                ),
                "dominant_relative_weight_q75": (
                    percentile(
                        topic_dominant_weights,
                        0.75,
                    )
                ),
                "mean_secondary_relative_weight_within_dominant_comments": (
                    float(
                        topic_secondary_weights.mean()
                    )
                ),
                "median_dominance_margin": (
                    float(
                        np.median(
                            topic_margins
                        )
                    )
                ),
                "mean_dominance_margin": (
                    float(
                        topic_margins.mean()
                    )
                ),
                "median_effective_topic_count": (
                    float(
                        np.median(
                            topic_effective_counts
                        )
                    )
                ),
                "unique_videos_with_dominant_comments": int(
                    corpus.loc[
                        dominant_mask,
                        "video_id",
                    ].nunique()
                ),
            }
        )

    prevalence = pd.DataFrame(
        prevalence_rows
    )

    if not np.isclose(
        prevalence[
            "dominant_comment_share"
        ].sum(),
        1.0,
    ):
        raise ValueError(
            "Dominant-topic shares do not sum to 1."
        )

    if not np.isclose(
        prevalence[
            "soft_prevalence_mean_relative_weight"
        ].sum(),
        1.0,
    ):
        raise ValueError(
            "Soft topic prevalence does not sum to 1."
        )

    # -----------------------------------------------------------------
    # Comment-level membership summary
    # -----------------------------------------------------------------

    comment_rows = pd.DataFrame(
        {
            "comment_id": (
                corpus[
                    "comment_id"
                ].astype(str)
            ),
            "video_id": (
                corpus[
                    "video_id"
                ].astype(str)
            ),
            "dominant_topic_number": (
                dominant_indices + 1
            ),
            "secondary_topic_number": (
                secondary_indices + 1
            ),
            "dominant_relative_weight": (
                dominant_relative_weights
            ),
            "secondary_relative_weight": (
                secondary_relative_weights
            ),
            "dominance_margin": (
                dominance_margin
            ),
            "topic_weight_entropy": (
                entropy
            ),
            "effective_topic_count": (
                effective_topic_count
            ),
        }
    )

    comment_rows[
        "dominant_topic_label"
    ] = comment_rows[
        "dominant_topic_number"
    ].map(
        topic_labels
    )

    comment_rows[
        "secondary_topic_label"
    ] = comment_rows[
        "secondary_topic_number"
    ].map(
        topic_labels
    )

    # -----------------------------------------------------------------
    # Dominant-secondary topic combinations
    # -----------------------------------------------------------------

    pair_rows = []

    pair_grouped = (
        comment_rows.groupby(
            [
                "dominant_topic_number",
                "secondary_topic_number",
            ],
            observed=True,
        )
    )

    for (
        dominant_topic_number,
        secondary_topic_number,
    ), group in pair_grouped:

        pair_rows.append(
            {
                "dominant_topic_number": (
                    int(
                        dominant_topic_number
                    )
                ),
                "dominant_topic_label": (
                    topic_labels[
                        int(
                            dominant_topic_number
                        )
                    ]
                ),
                "secondary_topic_number": (
                    int(
                        secondary_topic_number
                    )
                ),
                "secondary_topic_label": (
                    topic_labels[
                        int(
                            secondary_topic_number
                        )
                    ]
                ),
                "comment_count": int(
                    len(group)
                ),
                "share_of_all_comments": (
                    float(
                        len(group)
                        / len(corpus)
                    )
                ),
                "share_within_dominant_topic": (
                    float(
                        len(group)
                        / (
                            comment_rows[
                                "dominant_topic_number"
                            ]
                            == dominant_topic_number
                        ).sum()
                    )
                ),
                "mean_secondary_relative_weight": (
                    float(
                        group[
                            "secondary_relative_weight"
                        ].mean()
                    )
                ),
                "median_secondary_relative_weight": (
                    float(
                        group[
                            "secondary_relative_weight"
                        ].median()
                    )
                ),
                "mean_dominance_margin": (
                    float(
                        group[
                            "dominance_margin"
                        ].mean()
                    )
                ),
            }
        )

    dominant_secondary_pairs = (
        pd.DataFrame(
            pair_rows
        )
        .sort_values(
            [
                "comment_count",
                "mean_secondary_relative_weight",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # -----------------------------------------------------------------
    # Soft profiles conditional on dominant topic
    #
    # Example:
    # Among comments dominated by Topic 5, what is the average
    # relative weight assigned to every other topic?
    #
    # This avoids forcing comments into strictly exclusive categories.
    # -----------------------------------------------------------------

    soft_profile_rows = []

    for dominant_topic_index in range(
        TOPIC_COUNT
    ):

        dominant_topic_number = (
            dominant_topic_index + 1
        )

        dominant_mask = (
            dominant_indices
            == dominant_topic_index
        )

        subset_weights = (
            relative_weights[
                dominant_mask,
                :,
            ]
        )

        for associated_topic_index in range(
            TOPIC_COUNT
        ):

            associated_topic_number = (
                associated_topic_index + 1
            )

            weights = (
                subset_weights[
                    :,
                    associated_topic_index,
                ]
            )

            soft_profile_rows.append(
                {
                    "dominant_topic_number": (
                        dominant_topic_number
                    ),
                    "dominant_topic_label": (
                        topic_labels[
                            dominant_topic_number
                        ]
                    ),
                    "associated_topic_number": (
                        associated_topic_number
                    ),
                    "associated_topic_label": (
                        topic_labels[
                            associated_topic_number
                        ]
                    ),
                    "same_topic": (
                        dominant_topic_number
                        == associated_topic_number
                    ),
                    "dominant_comment_count": int(
                        dominant_mask.sum()
                    ),
                    "mean_relative_weight": (
                        float(
                            weights.mean()
                        )
                    ),
                    "median_relative_weight": (
                        float(
                            np.median(
                                weights
                            )
                        )
                    ),
                    "q75_relative_weight": (
                        percentile(
                            weights,
                            0.75,
                        )
                    ),
                    "q90_relative_weight": (
                        percentile(
                            weights,
                            0.90,
                        )
                    ),
                }
            )

    soft_profiles = pd.DataFrame(
        soft_profile_rows
    )

    # -----------------------------------------------------------------
    # Global membership diagnostics
    # -----------------------------------------------------------------

    global_summary = pd.DataFrame(
        [
            {
                "comments": int(
                    len(corpus)
                ),
                "videos": int(
                    corpus[
                        "video_id"
                    ].nunique()
                ),
                "topics": (
                    TOPIC_COUNT
                ),
                "mean_dominant_relative_weight": float(
                    dominant_relative_weights.mean()
                ),
                "median_dominant_relative_weight": float(
                    np.median(
                        dominant_relative_weights
                    )
                ),
                "dominant_relative_weight_q25": (
                    percentile(
                        dominant_relative_weights,
                        0.25,
                    )
                ),
                "dominant_relative_weight_q75": (
                    percentile(
                        dominant_relative_weights,
                        0.75,
                    )
                ),
                "mean_secondary_relative_weight": float(
                    secondary_relative_weights.mean()
                ),
                "median_secondary_relative_weight": float(
                    np.median(
                        secondary_relative_weights
                    )
                ),
                "mean_dominance_margin": float(
                    dominance_margin.mean()
                ),
                "median_dominance_margin": float(
                    np.median(
                        dominance_margin
                    )
                ),
                "mean_effective_topic_count": float(
                    effective_topic_count.mean()
                ),
                "median_effective_topic_count": float(
                    np.median(
                        effective_topic_count
                    )
                ),
                "effective_topic_count_q25": (
                    percentile(
                        effective_topic_count,
                        0.25,
                    )
                ),
                "effective_topic_count_q75": (
                    percentile(
                        effective_topic_count,
                        0.75,
                    )
                ),
            }
        ]
    )

    # -----------------------------------------------------------------
    # Highest-frequency secondary topic for each dominant topic
    # -----------------------------------------------------------------

    leading_secondary_rows = []

    for topic_number in range(
        1,
        TOPIC_COUNT + 1,
    ):

        topic_pairs = (
            dominant_secondary_pairs.loc[
                dominant_secondary_pairs[
                    "dominant_topic_number"
                ]
                == topic_number
            ]
            .sort_values(
                [
                    "comment_count",
                    "mean_secondary_relative_weight",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

        if topic_pairs.empty:
            continue

        first = (
            topic_pairs.iloc[0]
        )

        leading_secondary_rows.append(
            {
                "dominant_topic_number": (
                    topic_number
                ),
                "dominant_topic_label": (
                    topic_labels[
                        topic_number
                    ]
                ),
                "most_common_secondary_topic_number": int(
                    first[
                        "secondary_topic_number"
                    ]
                ),
                "most_common_secondary_topic_label": (
                    first[
                        "secondary_topic_label"
                    ]
                ),
                "pair_comment_count": int(
                    first[
                        "comment_count"
                    ]
                ),
                "share_within_dominant_topic": float(
                    first[
                        "share_within_dominant_topic"
                    ]
                ),
                "mean_secondary_relative_weight": float(
                    first[
                        "mean_secondary_relative_weight"
                    ]
                ),
            }
        )

    leading_secondary = pd.DataFrame(
        leading_secondary_rows
    )

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------

    prevalence_path = (
        TABLE_DIR
        / "08a_topic_prevalence_and_strength.csv"
    )

    comment_path = (
        TABLE_DIR
        / "08a_comment_membership_summary.csv"
    )

    pair_path = (
        TABLE_DIR
        / "08a_dominant_secondary_topic_pairs.csv"
    )

    profile_path = (
        TABLE_DIR
        / "08a_dominant_topic_soft_profiles.csv"
    )

    global_path = (
        TABLE_DIR
        / "08a_global_membership_summary.csv"
    )

    leading_secondary_path = (
        TABLE_DIR
        / "08a_leading_secondary_topic_by_dominant_topic.csv"
    )

    prevalence.to_csv(
        prevalence_path,
        index=False,
        encoding="utf-8-sig",
    )

    comment_rows.to_csv(
        comment_path,
        index=False,
        encoding="utf-8-sig",
    )

    dominant_secondary_pairs.to_csv(
        pair_path,
        index=False,
        encoding="utf-8-sig",
    )

    soft_profiles.to_csv(
        profile_path,
        index=False,
        encoding="utf-8-sig",
    )

    global_summary.to_csv(
        global_path,
        index=False,
        encoding="utf-8-sig",
    )

    leading_secondary.to_csv(
        leading_secondary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    configuration = {
        "analysis_stage": (
            "Standalone descriptive analysis "
            "of locked k=16 NMF solution"
        ),
        "model_status": (
            "frozen; no model fitting performed"
        ),
        "selected_model": (
            "inclusive primary TF-IDF k=16 NMF"
        ),
        "comments": int(
            len(corpus)
        ),
        "videos": int(
            corpus[
                "video_id"
            ].nunique()
        ),
        "topics": int(
            TOPIC_COUNT
        ),
        "locked_interpretation_sha256": (
            sha256(
                LOCKED_INTERPRETATION_FILE
            )
        ),
        "relative_weight_method": (
            "Each comment's 16 NMF document-topic weights "
            "were divided by their row sum. These values are "
            "descriptive relative NMF weights and are not "
            "interpreted as generative probabilities."
        ),
        "prevalence_measures": {
            "dominant_prevalence": (
                "Share of comments for which a topic has "
                "the largest relative NMF weight."
            ),
            "soft_prevalence": (
                "Mean relative NMF weight for a topic "
                "across all comments."
            ),
        },
        "membership_dispersion": (
            "Shannon entropy and exp(entropy) effective-topic "
            "count calculated from relative NMF weights for "
            "descriptive purposes."
        ),
        "important_exclusion": (
            "No predefined rule-based comment theme or "
            "knowledge-sharing classification is used in "
            "Stage 08a."
        ),
    }

    config_path = (
        CONFIG_DIR
        / "08a_standalone_nmf_analysis_config.json"
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

    global_row = (
        global_summary.iloc[0]
    )

    report_lines = [
        "YOUTUBE RETROFIT STANDALONE NMF ANALYSIS — STAGE 08a",
        "=" * 58,
        "",
        "Overall status: PASS",
        "",
        "Scope",
        "-----",
        (
            "This stage analyses the locked k=16 NMF "
            "solution on its own."
        ),
        (
            "No predefined rule-based comment themes or "
            "knowledge-sharing modes are used."
        ),
        "",
        "Corpus",
        "------",
        (
            f"Comments: "
            f"{int(global_row['comments']):,}"
        ),
        (
            f"Videos: "
            f"{int(global_row['videos']):,}"
        ),
        (
            f"Topics: "
            f"{TOPIC_COUNT}"
        ),
        "",
        "Overall topic-membership structure",
        "----------------------------------",
        (
            "Median dominant relative NMF weight: "
            f"{global_row['median_dominant_relative_weight']:.4f}"
        ),
        (
            "IQR dominant relative NMF weight: "
            f"{global_row['dominant_relative_weight_q25']:.4f}"
            " – "
            f"{global_row['dominant_relative_weight_q75']:.4f}"
        ),
        (
            "Median secondary relative NMF weight: "
            f"{global_row['median_secondary_relative_weight']:.4f}"
        ),
        (
            "Median dominant-secondary margin: "
            f"{global_row['median_dominance_margin']:.4f}"
        ),
        (
            "Median effective topic count per comment: "
            f"{global_row['median_effective_topic_count']:.2f}"
        ),
        (
            "IQR effective topic count: "
            f"{global_row['effective_topic_count_q25']:.2f}"
            " – "
            f"{global_row['effective_topic_count_q75']:.2f}"
        ),
        "",
        "Topic prevalence",
        "----------------",
    ]

    prevalence_sorted = (
        prevalence.sort_values(
            "dominant_comment_share",
            ascending=False,
        )
    )

    for _, row in prevalence_sorted.iterrows():

        report_lines.extend(
            [
                (
                    f"{int(row['topic_number']):02d}. "
                    f"{row['topic_label']}"
                ),
                (
                    "    Dominant comments: "
                    f"{int(row['dominant_comment_count']):,} "
                    f"("
                    f"{100 * row['dominant_comment_share']:.2f}%"
                    f")"
                ),
                (
                    "    Soft prevalence: "
                    f"{100 * row['soft_prevalence_mean_relative_weight']:.2f}%"
                ),
                (
                    "    Median dominant relative weight: "
                    f"{row['median_dominant_relative_weight']:.3f}"
                ),
                (
                    "    Median dominance margin: "
                    f"{row['median_dominance_margin']:.3f}"
                ),
            ]
        )

    report_lines.extend(
        [
            "",
            "Most common secondary topic",
            "---------------------------",
        ]
    )

    for _, row in (
        leading_secondary.sort_values(
            "dominant_topic_number"
        )
        .iterrows()
    ):
        report_lines.append(
            (
                f"{int(row['dominant_topic_number']):02d} "
                f"{row['dominant_topic_label']} "
                "-> "
                f"{int(row['most_common_secondary_topic_number']):02d} "
                f"{row['most_common_secondary_topic_label']} "
                f"("
                f"{100 * row['share_within_dominant_topic']:.1f}% "
                "of dominant-topic comments; "
                "mean secondary weight "
                f"{row['mean_secondary_relative_weight']:.3f}"
                ")"
            )
        )

    report_lines.extend(
        [
            "",
            "Interpretive cautions",
            "--------------------",
            (
                "NMF provides continuous topic weights. "
                "Dominant-topic prevalence is therefore a "
                "descriptive simplification rather than a claim "
                "that each comment belongs exclusively to one topic."
            ),
            (
                "Soft prevalence preserves information from all "
                "16 topic weights and should be reported alongside "
                "dominant-topic prevalence."
            ),
            (
                "Row-normalised NMF weights are relative weights, "
                "not probabilistic posterior topic assignments."
            ),
            (
                "The effective-topic-count measure describes "
                "membership concentration only and is not an "
                "additional fitted model."
            ),
            "",
            "Outputs",
            "-------",
            str(prevalence_path),
            str(global_path),
            str(leading_secondary_path),
            str(pair_path),
            str(profile_path),
            str(comment_path),
            str(config_path),
        ]
    )

    report_path = (
        AUDIT_DIR
        / "08a_standalone_nmf_analysis_report.txt"
    )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print(
        "\n".join(
            report_lines
        )
    )


if __name__ == "__main__":
    main()