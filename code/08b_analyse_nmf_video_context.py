from __future__ import annotations

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

TABLE_DIR = NMF_ROOT / "outputs" / "tables"
AUDIT_DIR = NMF_ROOT / "outputs" / "audit"
CONFIG_DIR = NMF_ROOT / "config"


# ---------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------

TOPIC_COUNT = 16
EXPECTED_COMMENTS = 42_443

# Used only for robustness summaries.
VIDEO_COMMENT_THRESHOLDS = [1, 5, 10, 20]

# How many highest-weight videos to report per topic.
TOP_VIDEOS_PER_TOPIC = 20

# Candidate video-publication-date column names.
DATE_COLUMN_CANDIDATES = [
    "final_published_at",
    "video_published_at",
    "published_at",
    "video_publish_date",
    "video_published_date",
    "published_date",
    "video_published",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def percentile(
    values: np.ndarray,
    q: float,
) -> float:

    if len(values) == 0:
        return float("nan")

    return float(
        np.quantile(
            values,
            q,
        )
    )


def entropy_from_weights(
    weights: np.ndarray,
) -> np.ndarray:

    safe = np.where(
        weights > 0,
        weights,
        1.0,
    )

    return -np.sum(
        np.where(
            weights > 0,
            weights * np.log(safe),
            0.0,
        ),
        axis=1,
    )


def find_date_column(
    dataframe: pd.DataFrame,
) -> str | None:

    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in dataframe.columns:
            return candidate

    return None


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

    for file_path in (
        CORPUS_FILE,
        DOCUMENT_TOPIC_FILE,
        LOCKED_INTERPRETATION_FILE,
    ):
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

    if len(corpus) != EXPECTED_COMMENTS:
        raise ValueError(
            f"Expected {EXPECTED_COMMENTS:,} comments; "
            f"found {len(corpus):,}."
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
            "Locked interpretation does not contain "
            "exactly 16 topics."
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

    # -----------------------------------------------------------------
    # Relative comment-level NMF weights
    # -----------------------------------------------------------------

    row_sums = document_topic.sum(
        axis=1
    )

    if np.any(
        row_sums <= 0
    ):
        raise ValueError(
            "At least one row has zero total NMF weight."
        )

    relative_weights = (
        document_topic
        / row_sums[:, None]
    )

    dominant_comment_topic = (
        np.argmax(
            relative_weights,
            axis=1,
        )
        + 1
    )

    # -----------------------------------------------------------------
    # Construct analysis frame
    # -----------------------------------------------------------------

    analysis = pd.DataFrame(
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
                dominant_comment_topic
            ),
        }
    )

    for topic_index in range(
        TOPIC_COUNT
    ):
        analysis[
            f"topic_{topic_index + 1:02d}_relative_weight"
        ] = relative_weights[
            :,
            topic_index,
        ]

    # Carry neutral contextual fields only.
    #
    # We deliberately do NOT use:
    #   retrofit_topic
    #   primary_theme
    #   knowledge-sharing rule classifications
    #
    # in Stage 08b.

    contextual_columns = []

    for column in [
        "creator_type",
        "video_type",
    ]:
        if column in corpus.columns:
            analysis[column] = (
                corpus[column]
            )

            contextual_columns.append(
                column
            )

    date_column = find_date_column(
        corpus
    )

    if date_column is not None:
        analysis[
            "_video_publication_date"
        ] = pd.to_datetime(
            corpus[
                date_column
            ],
            errors="coerce",
            utc=True,
        )

        contextual_columns.append(
            date_column
        )

    # -----------------------------------------------------------------
    # Video-level aggregation
    #
    # Each video receives the mean relative NMF topic weight across
    # its modelled substantive comments.
    # -----------------------------------------------------------------

    topic_weight_columns = [
        f"topic_{topic_number:02d}_relative_weight"
        for topic_number in range(
            1,
            TOPIC_COUNT + 1,
        )
    ]

    video_topic = (
        analysis.groupby(
            "video_id",
            observed=True,
        )[topic_weight_columns]
        .mean()
        .reset_index()
    )

    video_comment_counts = (
        analysis.groupby(
            "video_id",
            observed=True,
        )
        .size()
        .rename(
            "modelled_comment_count"
        )
        .reset_index()
    )

    video_topic = (
        video_topic.merge(
            video_comment_counts,
            on="video_id",
            how="left",
            validate="one_to_one",
        )
    )

    video_weights = (
        video_topic[
            topic_weight_columns
        ].to_numpy()
    )

    video_dominant_indices = np.argmax(
        video_weights,
        axis=1,
    )

    video_sorted = np.argsort(
        video_weights,
        axis=1,
    )

    video_secondary_indices = (
        video_sorted[
            :,
            -2,
        ]
    )

    video_dominant_weights = (
        video_weights[
            np.arange(
                len(video_topic)
            ),
            video_dominant_indices,
        ]
    )

    video_secondary_weights = (
        video_weights[
            np.arange(
                len(video_topic)
            ),
            video_secondary_indices,
        ]
    )

    video_topic[
        "dominant_topic_number"
    ] = (
        video_dominant_indices + 1
    )

    video_topic[
        "secondary_topic_number"
    ] = (
        video_secondary_indices + 1
    )

    video_topic[
        "dominant_topic_label"
    ] = (
        video_topic[
            "dominant_topic_number"
        ].map(
            topic_labels
        )
    )

    video_topic[
        "dominant_relative_weight"
    ] = video_dominant_weights

    video_topic[
        "secondary_relative_weight"
    ] = video_secondary_weights

    video_topic[
        "dominance_margin"
    ] = (
        video_dominant_weights
        - video_secondary_weights
    )

    video_entropy = (
        entropy_from_weights(
            video_weights
        )
    )

    video_topic[
        "effective_topic_count"
    ] = np.exp(
        video_entropy
    )

    # -----------------------------------------------------------------
    # One contextual value per video
    # -----------------------------------------------------------------

    video_context = (
        analysis[
            ["video_id"]
            + [
                column
                for column in [
                    "creator_type",
                    "video_type",
                    "_video_publication_date",
                ]
                if column
                in analysis.columns
            ]
        ]
        .drop_duplicates(
            subset=[
                "video_id"
            ],
            keep="first",
        )
    )

    video_topic = (
        video_topic.merge(
            video_context,
            on="video_id",
            how="left",
            validate="one_to_one",
        )
    )

    # -----------------------------------------------------------------
    # Video-level prevalence
    # -----------------------------------------------------------------

    video_prevalence_rows = []

    for topic_number in range(
        1,
        TOPIC_COUNT + 1,
    ):

        column = (
            f"topic_{topic_number:02d}_relative_weight"
        )

        dominant_mask = (
            video_topic[
                "dominant_topic_number"
            ]
            == topic_number
        )

        video_prevalence_rows.append(
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
                "dominant_video_count": int(
                    dominant_mask.sum()
                ),
                "dominant_video_share": float(
                    dominant_mask.mean()
                ),
                "mean_video_relative_weight": float(
                    video_topic[
                        column
                    ].mean()
                ),
                "median_video_relative_weight": float(
                    video_topic[
                        column
                    ].median()
                ),
                "q25_video_relative_weight": float(
                    video_topic[
                        column
                    ].quantile(
                        0.25
                    )
                ),
                "q75_video_relative_weight": float(
                    video_topic[
                        column
                    ].quantile(
                        0.75
                    )
                ),
            }
        )

    video_prevalence = pd.DataFrame(
        video_prevalence_rows
    )

    # -----------------------------------------------------------------
    # Robustness to low-comment videos
    # -----------------------------------------------------------------

    threshold_rows = []

    for threshold in (
        VIDEO_COMMENT_THRESHOLDS
    ):

        eligible = (
            video_topic.loc[
                video_topic[
                    "modelled_comment_count"
                ]
                >= threshold
            ]
        )

        for topic_number in range(
            1,
            TOPIC_COUNT + 1,
        ):

            column = (
                f"topic_{topic_number:02d}_relative_weight"
            )

            threshold_rows.append(
                {
                    "minimum_modelled_comments": (
                        threshold
                    ),
                    "eligible_videos": int(
                        len(
                            eligible
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
                    "dominant_video_count": int(
                        (
                            eligible[
                                "dominant_topic_number"
                            ]
                            == topic_number
                        ).sum()
                    ),
                    "dominant_video_share": float(
                        (
                            eligible[
                                "dominant_topic_number"
                            ]
                            == topic_number
                        ).mean()
                    )
                    if len(
                        eligible
                    )
                    else np.nan,
                    "mean_video_relative_weight": (
                        float(
                            eligible[
                                column
                            ].mean()
                        )
                        if len(
                            eligible
                        )
                        else np.nan
                    ),
                }
            )

    threshold_sensitivity = (
        pd.DataFrame(
            threshold_rows
        )
    )

    # -----------------------------------------------------------------
    # Topic concentration across videos
    #
    # Sum relative comment weights by video and calculate how much
    # of each topic's total comment-level weight comes from the
    # highest-contributing videos.
    # -----------------------------------------------------------------

    concentration_rows = []
    top_video_rows = []

    for topic_number in range(
        1,
        TOPIC_COUNT + 1,
    ):

        column = (
            f"topic_{topic_number:02d}_relative_weight"
        )

        video_totals = (
            analysis.groupby(
                "video_id",
                observed=True,
            )[column]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        total_weight = float(
            video_totals.sum()
        )

        if total_weight <= 0:
            continue

        shares = (
            video_totals
            / total_weight
        )

        concentration_rows.append(
            {
                "topic_number": (
                    topic_number
                ),
                "topic_label": (
                    topic_labels[
                        topic_number
                    ]
                ),
                "videos_with_positive_weight": int(
                    (
                        video_totals
                        > 0
                    ).sum()
                ),
                "largest_video_share_of_topic_weight": float(
                    shares.iloc[0]
                ),
                "top_5_video_share_of_topic_weight": float(
                    shares.head(
                        5
                    ).sum()
                ),
                "top_10_video_share_of_topic_weight": float(
                    shares.head(
                        10
                    ).sum()
                ),
                "top_20_video_share_of_topic_weight": float(
                    shares.head(
                        20
                    ).sum()
                ),
            }
        )

        for rank, (
            video_id,
            topic_weight,
        ) in enumerate(
            video_totals.head(
                TOP_VIDEOS_PER_TOPIC
            ).items(),
            start=1,
        ):

            row = video_topic.loc[
                video_topic[
                    "video_id"
                ]
                == str(
                    video_id
                )
            ]

            record = {
                "topic_number": (
                    topic_number
                ),
                "topic_label": (
                    topic_labels[
                        topic_number
                    ]
                ),
                "rank": (
                    rank
                ),
                "video_id": str(
                    video_id
                ),
                "summed_comment_relative_weight": float(
                    topic_weight
                ),
                "share_of_topic_total_weight": float(
                    topic_weight
                    / total_weight
                ),
            }

            if not row.empty:
                record[
                    "modelled_comment_count"
                ] = int(
                    row.iloc[0][
                        "modelled_comment_count"
                    ]
                )

                record[
                    "mean_video_relative_weight"
                ] = float(
                    row.iloc[0][
                        column
                    ]
                )

            top_video_rows.append(
                record
            )

    concentration = pd.DataFrame(
        concentration_rows
    )

    top_videos = pd.DataFrame(
        top_video_rows
    )

    # -----------------------------------------------------------------
    # Creator-type and video-type context
    #
    # These are descriptive contextual comparisons.
    # No inferential significance tests are performed here.
    # -----------------------------------------------------------------

    group_rows = []

    for grouping_column in (
        "creator_type",
        "video_type",
    ):

        if grouping_column not in (
            video_topic.columns
        ):
            continue

        grouping_values = (
            video_topic[
                grouping_column
            ]
            .fillna("Missing")
            .astype(str)
        )

        temporary = (
            video_topic.copy()
        )

        temporary[
            "_group"
        ] = grouping_values

        for group_name, group in (
            temporary.groupby(
                "_group",
                observed=True,
            )
        ):

            for topic_number in range(
                1,
                TOPIC_COUNT + 1,
            ):

                column = (
                    f"topic_{topic_number:02d}_relative_weight"
                )

                group_rows.append(
                    {
                        "grouping_variable": (
                            grouping_column
                        ),
                        "group": (
                            group_name
                        ),
                        "videos": int(
                            len(group)
                        ),
                        "topic_number": (
                            topic_number
                        ),
                        "topic_label": (
                            topic_labels[
                                topic_number
                            ]
                        ),
                        "mean_video_relative_weight": float(
                            group[
                                column
                            ].mean()
                        ),
                        "median_video_relative_weight": float(
                            group[
                                column
                            ].median()
                        ),
                        "dominant_video_count": int(
                            (
                                group[
                                    "dominant_topic_number"
                                ]
                                == topic_number
                            ).sum()
                        ),
                        "dominant_video_share": float(
                            (
                                group[
                                    "dominant_topic_number"
                                ]
                                == topic_number
                            ).mean()
                        ),
                    }
                )

    contextual_distributions = (
        pd.DataFrame(
            group_rows
        )
    )

    # -----------------------------------------------------------------
    # Publication-year distributions, only if a date exists
    # -----------------------------------------------------------------

    yearly_rows = []

    if (
        "_video_publication_date"
        in video_topic.columns
    ):

        video_topic[
            "publication_year"
        ] = (
            video_topic[
                "_video_publication_date"
            ]
            .dt.year
        )

        dated = (
            video_topic.dropna(
                subset=[
                    "publication_year"
                ]
            )
            .copy()
        )

        if not dated.empty:

            dated[
                "publication_year"
            ] = (
                dated[
                    "publication_year"
                ]
                .astype(int)
            )

            for year, group in (
                dated.groupby(
                    "publication_year",
                    observed=True,
                )
            ):

                for topic_number in range(
                    1,
                    TOPIC_COUNT + 1,
                ):

                    column = (
                        f"topic_{topic_number:02d}_relative_weight"
                    )

                    yearly_rows.append(
                        {
                            "publication_year": int(
                                year
                            ),
                            "videos": int(
                                len(group)
                            ),
                            "topic_number": (
                                topic_number
                            ),
                            "topic_label": (
                                topic_labels[
                                    topic_number
                                ]
                            ),
                            "mean_video_relative_weight": float(
                                group[
                                    column
                                ].mean()
                            ),
                            "dominant_video_share": float(
                                (
                                    group[
                                        "dominant_topic_number"
                                    ]
                                    == topic_number
                                ).mean()
                            ),
                        }
                    )

    yearly_distribution = (
        pd.DataFrame(
            yearly_rows
        )
    )

    # -----------------------------------------------------------------
    # Global video-membership summary
    # -----------------------------------------------------------------

    global_video_summary = (
        pd.DataFrame(
            [
                {
                    "videos": int(
                        len(
                            video_topic
                        )
                    ),
                    "median_modelled_comments_per_video": float(
                        video_topic[
                            "modelled_comment_count"
                        ].median()
                    ),
                    "q25_modelled_comments_per_video": float(
                        video_topic[
                            "modelled_comment_count"
                        ].quantile(
                            0.25
                        )
                    ),
                    "q75_modelled_comments_per_video": float(
                        video_topic[
                            "modelled_comment_count"
                        ].quantile(
                            0.75
                        )
                    ),
                    "median_video_dominant_relative_weight": float(
                        video_topic[
                            "dominant_relative_weight"
                        ].median()
                    ),
                    "median_video_secondary_relative_weight": float(
                        video_topic[
                            "secondary_relative_weight"
                        ].median()
                    ),
                    "median_video_dominance_margin": float(
                        video_topic[
                            "dominance_margin"
                        ].median()
                    ),
                    "median_video_effective_topic_count": float(
                        video_topic[
                            "effective_topic_count"
                        ].median()
                    ),
                    "q25_video_effective_topic_count": float(
                        video_topic[
                            "effective_topic_count"
                        ].quantile(
                            0.25
                        )
                    ),
                    "q75_video_effective_topic_count": float(
                        video_topic[
                            "effective_topic_count"
                        ].quantile(
                            0.75
                        )
                    ),
                }
            ]
        )
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    video_topic_path = (
        TABLE_DIR
        / "08b_video_level_nmf_profiles.csv"
    )

    prevalence_path = (
        TABLE_DIR
        / "08b_video_topic_prevalence.csv"
    )

    threshold_path = (
        TABLE_DIR
        / "08b_video_comment_count_sensitivity.csv"
    )

    concentration_path = (
        TABLE_DIR
        / "08b_topic_video_concentration.csv"
    )

    top_videos_path = (
        TABLE_DIR
        / "08b_top_videos_by_topic_weight.csv"
    )

    context_path = (
        TABLE_DIR
        / "08b_creator_and_video_type_distributions.csv"
    )

    yearly_path = (
        TABLE_DIR
        / "08b_publication_year_topic_distribution.csv"
    )

    global_path = (
        TABLE_DIR
        / "08b_global_video_membership_summary.csv"
    )

    video_topic.to_csv(
        video_topic_path,
        index=False,
        encoding="utf-8-sig",
    )

    video_prevalence.to_csv(
        prevalence_path,
        index=False,
        encoding="utf-8-sig",
    )

    threshold_sensitivity.to_csv(
        threshold_path,
        index=False,
        encoding="utf-8-sig",
    )

    concentration.to_csv(
        concentration_path,
        index=False,
        encoding="utf-8-sig",
    )

    top_videos.to_csv(
        top_videos_path,
        index=False,
        encoding="utf-8-sig",
    )

    contextual_distributions.to_csv(
        context_path,
        index=False,
        encoding="utf-8-sig",
    )

    if not yearly_distribution.empty:
        yearly_distribution.to_csv(
            yearly_path,
            index=False,
            encoding="utf-8-sig",
        )

    global_video_summary.to_csv(
        global_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------

    config = {
        "analysis_stage": (
            "Standalone contextual analysis "
            "of locked k=16 NMF"
        ),
        "model_status": (
            "frozen; no NMF fitting performed"
        ),
        "videos": int(
            len(
                video_topic
            )
        ),
        "video_topic_aggregation": (
            "Mean row-normalised NMF topic weight "
            "across modelled comments within each video."
        ),
        "contextual_variables_used": [
            value
            for value in [
                "creator_type"
                if "creator_type"
                in video_topic.columns
                else None,
                "video_type"
                if "video_type"
                in video_topic.columns
                else None,
                date_column,
            ]
            if value is not None
        ],
        "explicitly_excluded_from_stage_08b": [
            "retrofit_topic",
            "primary_theme",
            "rule-based knowledge-sharing modes",
        ],
        "low_comment_video_sensitivity_thresholds": (
            VIDEO_COMMENT_THRESHOLDS
        ),
        "inferential_tests": (
            "None. Stage 08b is descriptive."
        ),
    }

    config_path = (
        CONFIG_DIR
        / "08b_nmf_video_context_config.json"
    )

    config_path.write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    summary = (
        global_video_summary.iloc[
            0
        ]
    )

    report_lines = [
        "YOUTUBE RETROFIT STANDALONE NMF ANALYSIS — STAGE 08b",
        "=" * 58,
        "",
        "Overall status: PASS",
        "",
        "Scope",
        "-----",
        (
            "This stage aggregates the frozen k=16 NMF "
            "solution to video level and examines neutral "
            "contextual distributions."
        ),
        (
            "No rule-based comment theme, knowledge-sharing "
            "mode or predefined retrofit-domain classification "
            "is used."
        ),
        "",
        "Video-level corpus",
        "------------------",
        (
            f"Videos represented: "
            f"{int(summary['videos']):,}"
        ),
        (
            "Median modelled comments per video: "
            f"{summary['median_modelled_comments_per_video']:.1f}"
        ),
        (
            "IQR modelled comments per video: "
            f"{summary['q25_modelled_comments_per_video']:.1f}"
            " – "
            f"{summary['q75_modelled_comments_per_video']:.1f}"
        ),
        "",
        "Video-level topic mixture",
        "-------------------------",
        (
            "Median dominant topic weight: "
            f"{summary['median_video_dominant_relative_weight']:.4f}"
        ),
        (
            "Median secondary topic weight: "
            f"{summary['median_video_secondary_relative_weight']:.4f}"
        ),
        (
            "Median dominance margin: "
            f"{summary['median_video_dominance_margin']:.4f}"
        ),
        (
            "Median effective topic count: "
            f"{summary['median_video_effective_topic_count']:.2f}"
        ),
        "",
        "Dominant topic by video",
        "-----------------------",
    ]

    for _, row in (
        video_prevalence.sort_values(
            "dominant_video_share",
            ascending=False,
        )
        .iterrows()
    ):

        report_lines.append(
            (
                f"{int(row['topic_number']):02d}. "
                f"{row['topic_label']}: "
                f"{int(row['dominant_video_count']):,} videos "
                f"("
                f"{100 * row['dominant_video_share']:.2f}%"
                "); mean video weight "
                f"{100 * row['mean_video_relative_weight']:.2f}%"
            )
        )

    report_lines.extend(
        [
            "",
            "Topic concentration across videos",
            "---------------------------------",
        ]
    )

    for _, row in (
        concentration.sort_values(
            "top_10_video_share_of_topic_weight",
            ascending=False,
        )
        .iterrows()
    ):

        report_lines.append(
            (
                f"{int(row['topic_number']):02d}. "
                f"{row['topic_label']}: "
                "largest video "
                f"{100 * row['largest_video_share_of_topic_weight']:.2f}%; "
                "top 10 videos "
                f"{100 * row['top_10_video_share_of_topic_weight']:.2f}%"
            )
        )

    report_lines.extend(
        [
            "",
            "Low-comment-video sensitivity",
            "-----------------------------",
        ]
    )

    for threshold in (
        VIDEO_COMMENT_THRESHOLDS
    ):

        eligible_videos = int(
            threshold_sensitivity.loc[
                threshold_sensitivity[
                    "minimum_modelled_comments"
                ]
                == threshold,
                "eligible_videos",
            ].iloc[0]
        )

        report_lines.append(
            (
                f"At least {threshold} modelled comments: "
                f"{eligible_videos:,} videos"
            )
        )

    report_lines.extend(
        [
            "",
            "Context availability",
            "--------------------",
            (
                "Creator type available: "
                f"{'yes' if 'creator_type' in video_topic.columns else 'no'}"
            ),
            (
                "Video type available: "
                f"{'yes' if 'video_type' in video_topic.columns else 'no'}"
            ),
            (
                "Publication date available: "
                f"{'yes' if date_column is not None else 'no'}"
            ),
        ]
    )

    if date_column is not None:
        report_lines.append(
            (
                "Publication-date source column: "
                f"{date_column}"
            )
        )

    report_lines.extend(
        [
            "",
            "Interpretive cautions",
            "--------------------",
            (
                "Video-level topic weights are averages of "
                "comment-level relative NMF weights. They "
                "describe the composition of the observed "
                "comment discussion, not the technical content "
                "of the video itself."
            ),
            (
                "Videos with few modelled comments provide "
                "less reliable estimates of discussion-topic "
                "composition; threshold summaries are therefore "
                "reported."
            ),
            (
                "Topic concentration based on summed comment "
                "weights is influenced by discussion volume and "
                "is used specifically to assess whether topic "
                "discussion is concentrated in a small set of "
                "videos."
            ),
            (
                "Creator- and video-type comparisons in this "
                "stage are descriptive contextual analyses, "
                "not causal effects."
            ),
            "",
            "Outputs",
            "-------",
            str(video_topic_path),
            str(prevalence_path),
            str(threshold_path),
            str(concentration_path),
            str(top_videos_path),
            str(context_path),
            (
                str(yearly_path)
                if not yearly_distribution.empty
                else (
                    "Publication-year table not created "
                    "because no usable publication-date "
                    "column was available."
                )
            ),
            str(global_path),
            str(config_path),
        ]
    )

    report_path = (
        AUDIT_DIR
        / "08b_standalone_nmf_video_context_report.txt"
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