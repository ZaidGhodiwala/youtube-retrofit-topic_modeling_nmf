from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

NMF_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = NMF_ROOT.parent

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

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "metadata_engagement_analysis"
    / "integrated_data"
    / "integrated_video_analysis_master.csv"
)

STAGE_08C_CORRELATION_FILE = (
    NMF_ROOT
    / "outputs"
    / "tables"
    / "08c_topic_weight_engagement_spearman.csv"
)

TABLE_DIR = NMF_ROOT / "outputs" / "tables"
AUDIT_DIR = NMF_ROOT / "outputs" / "audit"
CONFIG_DIR = NMF_ROOT / "config"


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

TOPIC_COUNT = 16

EXPECTED_COMMENTS = 42_443
EXPECTED_VIDEOS = 1_158

SAMPLE_SIZES = [10, 20]

REPETITIONS = 100

BASE_SEED = 20_260_828

METRICS = [
    "duration_minutes",
    "view_count",
    "like_count",
    "api_comment_count",
    "likes_per_1000_views",
    "api_comments_per_1000_views",
    "views_per_day",
    "api_comments_per_day",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def numeric(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def percentile(
    values: np.ndarray,
    q: float,
) -> float:

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return float("nan")

    return float(
        np.quantile(
            values,
            q,
        )
    )


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

    required_files = [
        CORPUS_FILE,
        DOCUMENT_TOPIC_FILE,
        LOCKED_INTERPRETATION_FILE,
        METADATA_FILE,
        STAGE_08C_CORRELATION_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

    # -----------------------------------------------------------------
    # Load
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

    metadata = pd.read_csv(
        METADATA_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    stage_08c = pd.read_csv(
        STAGE_08C_CORRELATION_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if len(corpus) != EXPECTED_COMMENTS:
        raise ValueError(
            f"Expected {EXPECTED_COMMENTS:,} comments "
            f"but found {len(corpus):,}."
        )

    if document_topic.shape != (
        EXPECTED_COMMENTS,
        TOPIC_COUNT,
    ):
        raise ValueError(
            "Unexpected document-topic matrix shape."
        )

    if len(locked) != TOPIC_COUNT:
        raise ValueError(
            "Expected exactly 16 locked topics."
        )

    # -----------------------------------------------------------------
    # Relative NMF weights
    # -----------------------------------------------------------------

    row_sums = document_topic.sum(
        axis=1
    )

    if np.any(
        row_sums <= 0
    ):
        raise ValueError(
            "Zero-total NMF document-topic row found."
        )

    relative_weights = (
        document_topic
        / row_sums[:, None]
    )

    # -----------------------------------------------------------------
    # IDs and metadata
    # -----------------------------------------------------------------

    corpus["video_id"] = (
        corpus[
            "video_id"
        ]
        .astype("string")
        .str.strip()
    )

    metadata["video_id"] = (
        metadata[
            "video_id"
        ]
        .astype("string")
        .str.strip()
    )

    for metric in METRICS:
        if metric not in metadata.columns:
            raise KeyError(
                f"Missing metadata metric: {metric}"
            )

        metadata[
            metric
        ] = numeric(
            metadata[
                metric
            ]
        )

    metadata_lookup = (
        metadata[
            [
                "video_id",
                *METRICS,
            ]
        ]
        .drop_duplicates(
            subset=[
                "video_id"
            ]
        )
        .set_index(
            "video_id"
        )
    )

    topic_labels = {
        int(row["topic_number"]):
        str(row["researcher_final_label"])
        for _, row in locked.iterrows()
    }

    # -----------------------------------------------------------------
    # Comment row indices by video
    # -----------------------------------------------------------------

    video_indices: dict[
        str,
        np.ndarray,
    ] = {}

    for video_id, index_values in (
        corpus.groupby(
            "video_id",
            sort=False,
        ).indices.items()
    ):

        video_indices[
            str(video_id)
        ] = np.asarray(
            index_values,
            dtype=int,
        )

    if len(video_indices) != EXPECTED_VIDEOS:
        raise ValueError(
            f"Expected {EXPECTED_VIDEOS:,} videos "
            f"but found {len(video_indices):,}."
        )

    # -----------------------------------------------------------------
    # Repeated equal-comment sampling
    # -----------------------------------------------------------------

    replicate_rows = []
    sample_summary_rows = []

    for sample_size in SAMPLE_SIZES:

        eligible_video_ids = sorted(
            [
                video_id
                for video_id, indices
                in video_indices.items()
                if len(indices)
                >= sample_size
                and video_id
                in metadata_lookup.index
            ]
        )

        print()
        print(
            f"Equal-comment analysis: "
            f"{sample_size} comments/video"
        )

        print(
            f"Eligible videos: "
            f"{len(eligible_video_ids):,}"
        )

        if len(
            eligible_video_ids
        ) < 50:
            raise ValueError(
                "Too few eligible videos."
            )

        sample_summary_rows.append(
            {
                "comments_sampled_per_video": (
                    sample_size
                ),
                "eligible_videos": int(
                    len(
                        eligible_video_ids
                    )
                ),
                "repetitions": (
                    REPETITIONS
                ),
            }
        )

        engagement = (
            metadata_lookup.loc[
                eligible_video_ids,
                METRICS,
            ]
            .reset_index(
                drop=False
            )
        )

        for repetition in range(
            1,
            REPETITIONS + 1,
        ):

            seed = (
                BASE_SEED
                + sample_size * 100_000
                + repetition
            )

            rng = np.random.default_rng(
                seed
            )

            balanced_weights = np.zeros(
                (
                    len(
                        eligible_video_ids
                    ),
                    TOPIC_COUNT,
                ),
                dtype=np.float64,
            )

            for video_position, video_id in enumerate(
                eligible_video_ids
            ):

                available_indices = (
                    video_indices[
                        video_id
                    ]
                )

                selected_indices = (
                    rng.choice(
                        available_indices,
                        size=sample_size,
                        replace=False,
                    )
                )

                balanced_weights[
                    video_position,
                    :,
                ] = (
                    relative_weights[
                        selected_indices,
                        :
                    ].mean(
                        axis=0
                    )
                )

            for topic_index in range(
                TOPIC_COUNT
            ):

                topic_number = (
                    topic_index + 1
                )

                topic_values = (
                    balanced_weights[
                        :,
                        topic_index,
                    ]
                )

                for metric in METRICS:

                    metric_values = (
                        engagement[
                            metric
                        ].to_numpy(
                            dtype=float
                        )
                    )

                    valid = (
                        np.isfinite(
                            topic_values
                        )
                        & np.isfinite(
                            metric_values
                        )
                    )

                    if valid.sum() < 10:
                        rho = np.nan
                    else:
                        result = spearmanr(
                            topic_values[
                                valid
                            ],
                            metric_values[
                                valid
                            ],
                        )

                        rho = float(
                            result.statistic
                        )

                    replicate_rows.append(
                        {
                            "comments_sampled_per_video": (
                                sample_size
                            ),
                            "repetition": (
                                repetition
                            ),
                            "seed": (
                                seed
                            ),
                            "eligible_videos": int(
                                len(
                                    eligible_video_ids
                                )
                            ),
                            "topic_number": (
                                topic_number
                            ),
                            "topic_label": (
                                topic_labels[
                                    topic_number
                                ]
                            ),
                            "metric": (
                                metric
                            ),
                            "spearman_rho": (
                                rho
                            ),
                        }
                    )

            if (
                repetition == 1
                or repetition % 10 == 0
                or repetition == REPETITIONS
            ):
                print(
                    f"  repetition "
                    f"{repetition:03d}/"
                    f"{REPETITIONS}"
                )

    replicate_results = pd.DataFrame(
        replicate_rows
    )

    # -----------------------------------------------------------------
    # Aggregate resampling results
    # -----------------------------------------------------------------

    aggregate_rows = []

    for (
        sample_size,
        topic_number,
        metric,
    ), group in replicate_results.groupby(
        [
            "comments_sampled_per_video",
            "topic_number",
            "metric",
        ],
        observed=True,
    ):

        values = (
            group[
                "spearman_rho"
            ]
            .to_numpy(
                dtype=float
            )
        )

        finite_values = values[
            np.isfinite(
                values
            )
        ]

        if len(
            finite_values
        ) == 0:
            continue

        positive_fraction = float(
            (
                finite_values
                > 0
            ).mean()
        )

        negative_fraction = float(
            (
                finite_values
                < 0
            ).mean()
        )

        median_rho = float(
            np.median(
                finite_values
            )
        )

        # Corresponding Stage 08c analysis:
        # all comments from videos meeting the same threshold.
        original_row = (
            stage_08c.loc[
                (
                    stage_08c[
                        "minimum_modelled_comments"
                    ]
                    == sample_size
                )
                & (
                    stage_08c[
                        "topic_number"
                    ]
                    == topic_number
                )
                & (
                    stage_08c[
                        "metric"
                    ]
                    == metric
                )
            ]
        )

        if len(
            original_row
        ) == 1:
            original_rho = float(
                original_row.iloc[
                    0
                ][
                    "spearman_rho"
                ]
            )
        else:
            original_rho = np.nan

        aggregate_rows.append(
            {
                "comments_sampled_per_video": (
                    int(
                        sample_size
                    )
                ),
                "eligible_videos": int(
                    group[
                        "eligible_videos"
                    ].iloc[0]
                ),
                "topic_number": int(
                    topic_number
                ),
                "topic_label": (
                    topic_labels[
                        int(
                            topic_number
                        )
                    ]
                ),
                "metric": (
                    metric
                ),
                "repetitions": int(
                    len(
                        finite_values
                    )
                ),
                "median_rho": (
                    median_rho
                ),
                "mean_rho": float(
                    np.mean(
                        finite_values
                    )
                ),
                "rho_q025": (
                    percentile(
                        finite_values,
                        0.025,
                    )
                ),
                "rho_q25": (
                    percentile(
                        finite_values,
                        0.25,
                    )
                ),
                "rho_q75": (
                    percentile(
                        finite_values,
                        0.75,
                    )
                ),
                "rho_q975": (
                    percentile(
                        finite_values,
                        0.975,
                    )
                ),
                "minimum_rho": float(
                    np.min(
                        finite_values
                    )
                ),
                "maximum_rho": float(
                    np.max(
                        finite_values
                    )
                ),
                "positive_fraction": (
                    positive_fraction
                ),
                "negative_fraction": (
                    negative_fraction
                ),
                "stage_08c_threshold_rho": (
                    original_rho
                ),
                "median_minus_stage_08c_rho": (
                    median_rho
                    - original_rho
                    if np.isfinite(
                        original_rho
                    )
                    else np.nan
                ),
            }
        )

    aggregate = pd.DataFrame(
        aggregate_rows
    )

    # -----------------------------------------------------------------
    # Cross-sample-size comparison
    # -----------------------------------------------------------------

    ten = (
        aggregate.loc[
            aggregate[
                "comments_sampled_per_video"
            ]
            == 10
        ][
            [
                "topic_number",
                "topic_label",
                "metric",
                "median_rho",
                "rho_q025",
                "rho_q975",
                "positive_fraction",
                "negative_fraction",
            ]
        ]
        .rename(
            columns={
                "median_rho": (
                    "median_rho_10"
                ),
                "rho_q025": (
                    "q025_10"
                ),
                "rho_q975": (
                    "q975_10"
                ),
                "positive_fraction": (
                    "positive_fraction_10"
                ),
                "negative_fraction": (
                    "negative_fraction_10"
                ),
            }
        )
    )

    twenty = (
        aggregate.loc[
            aggregate[
                "comments_sampled_per_video"
            ]
            == 20
        ][
            [
                "topic_number",
                "metric",
                "median_rho",
                "rho_q025",
                "rho_q975",
                "positive_fraction",
                "negative_fraction",
            ]
        ]
        .rename(
            columns={
                "median_rho": (
                    "median_rho_20"
                ),
                "rho_q025": (
                    "q025_20"
                ),
                "rho_q975": (
                    "q975_20"
                ),
                "positive_fraction": (
                    "positive_fraction_20"
                ),
                "negative_fraction": (
                    "negative_fraction_20"
                ),
            }
        )
    )

    cross_threshold = ten.merge(
        twenty,
        on=[
            "topic_number",
            "metric",
        ],
        how="inner",
        validate="one_to_one",
    )

    cross_threshold[
        "same_median_sign"
    ] = (
        np.sign(
            cross_threshold[
                "median_rho_10"
            ]
        )
        == np.sign(
            cross_threshold[
                "median_rho_20"
            ]
        )
    )

    cross_threshold[
        "minimum_absolute_median_rho"
    ] = np.minimum(
        cross_threshold[
            "median_rho_10"
        ].abs(),
        cross_threshold[
            "median_rho_20"
        ].abs(),
    )

    cross_threshold[
        "resampling_interval_excludes_zero_10"
    ] = (
        (
            cross_threshold[
                "q025_10"
            ]
            > 0
        )
        | (
            cross_threshold[
                "q975_10"
            ]
            < 0
        )
    )

    cross_threshold[
        "resampling_interval_excludes_zero_20"
    ] = (
        (
            cross_threshold[
                "q025_20"
            ]
            > 0
        )
        | (
            cross_threshold[
                "q975_20"
            ]
            < 0
        )
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    replicate_path = (
        TABLE_DIR
        / "08d_equal_comment_sampling_replicates.csv"
    )

    aggregate_path = (
        TABLE_DIR
        / "08d_equal_comment_sampling_summary.csv"
    )

    comparison_path = (
        TABLE_DIR
        / "08d_equal_comment_cross_threshold_summary.csv"
    )

    sample_summary_path = (
        TABLE_DIR
        / "08d_equal_comment_sampling_design.csv"
    )

    replicate_results.to_csv(
        replicate_path,
        index=False,
        encoding="utf-8-sig",
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
        encoding="utf-8-sig",
    )

    cross_threshold.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        sample_summary_rows
    ).to_csv(
        sample_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Report strongest cross-threshold results
    # -----------------------------------------------------------------

    report_candidates = (
        cross_threshold.loc[
            cross_threshold[
                "same_median_sign"
            ]
        ]
        .copy()
    )

    report_candidates = (
        report_candidates.sort_values(
            "minimum_absolute_median_rho",
            ascending=False,
        )
    )

    report_lines = [
        "YOUTUBE RETROFIT NMF EQUAL-COMMENT ROBUSTNESS — STAGE 08d",
        "=" * 62,
        "",
        "Overall status: PASS",
        "",
        "Purpose",
        "-------",
        (
            "This robustness analysis standardises the "
            "number of comments contributing to each "
            "video-level NMF topic profile."
        ),
        (
            "It addresses the strong relationship between "
            "modelled-comment count and apparent topic-mixture "
            "breadth identified in Stage 08c."
        ),
        "",
        "Design",
        "------",
        (
            f"Repeated samples: "
            f"{REPETITIONS} per condition"
        ),
    ]

    for row in sample_summary_rows:
        report_lines.append(
            (
                f"{row['comments_sampled_per_video']} "
                "comments/video: "
                f"{row['eligible_videos']:,} videos"
            )
        )

    report_lines.extend(
        [
            "",
            "Strongest associations with the same median sign",
            "------------------------------------------------",
        ]
    )

    for _, row in (
        report_candidates.head(
            25
        ).iterrows()
    ):

        report_lines.append(
            (
                f"T{int(row['topic_number']):02d} "
                f"{row['topic_label']} × "
                f"{row['metric']}: "
                f"median rho 10="
                f"{row['median_rho_10']:.3f} "
                f"[{row['q025_10']:.3f}, "
                f"{row['q975_10']:.3f}]; "
                f"20="
                f"{row['median_rho_20']:.3f} "
                f"[{row['q025_20']:.3f}, "
                f"{row['q975_20']:.3f}]"
            )
        )

    robust_both = (
        cross_threshold.loc[
            (
                cross_threshold[
                    "same_median_sign"
                ]
            )
            & (
                cross_threshold[
                    "resampling_interval_excludes_zero_10"
                ]
            )
            & (
                cross_threshold[
                    "resampling_interval_excludes_zero_20"
                ]
            )
        ]
    )

    report_lines.extend(
        [
            "",
            "Cross-threshold robustness",
            "--------------------------",
            (
                "Associations whose 2.5th–97.5th percentile "
                "resampling interval excludes zero in both "
                f"conditions: {len(robust_both)}"
            ),
            "",
            "Interpretive caution",
            "--------------------",
            (
                "The percentile ranges are empirical "
                "resampling intervals describing sensitivity "
                "to which comments are selected. They are not "
                "population confidence intervals."
            ),
            (
                "The analysis remains observational and "
                "compositional; it does not establish causal "
                "effects of discussion topics on engagement."
            ),
            (
                "Results that disappear after equal-comment "
                "sampling should be treated as sensitive to "
                "unequal discussion-sample size rather than "
                "as robust topic–engagement relationships."
            ),
            "",
            "Outputs",
            "-------",
            str(sample_summary_path),
            str(aggregate_path),
            str(comparison_path),
            str(replicate_path),
        ]
    )

    report_path = (
        AUDIT_DIR
        / "08d_equal_comment_sampling_robustness_report.txt"
    )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    config = {
        "analysis_stage": (
            "Equal-comment video-profile robustness"
        ),
        "comments_per_video_conditions": (
            SAMPLE_SIZES
        ),
        "repetitions": (
            REPETITIONS
        ),
        "base_seed": (
            BASE_SEED
        ),
        "sampling": (
            "without replacement within each video"
        ),
        "topic_profile": (
            "mean row-normalised NMF topic weights "
            "for exactly N sampled comments"
        ),
        "association": (
            "Spearman rank correlation"
        ),
        "purpose": (
            "Assess sensitivity of Stage 08c associations "
            "to unequal numbers of modelled comments "
            "per video."
        ),
    }

    config_path = (
        CONFIG_DIR
        / "08d_equal_comment_sampling_config.json"
    )

    config_path.write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
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