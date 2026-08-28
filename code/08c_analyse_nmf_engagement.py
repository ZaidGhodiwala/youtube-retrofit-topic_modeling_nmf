from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

NMF_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = NMF_ROOT.parent

VIDEO_PROFILE_FILE = (
    NMF_ROOT
    / "outputs"
    / "tables"
    / "08b_video_level_nmf_profiles.csv"
)

LOCKED_INTERPRETATION_FILE = (
    NMF_ROOT
    / "outputs"
    / "review"
    / "07c_final_k16_topic_interpretation_locked.csv"
)

METADATA_MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "metadata_engagement_analysis"
    / "integrated_data"
    / "integrated_video_analysis_master.csv"
)

TABLE_DIR = NMF_ROOT / "outputs" / "tables"
AUDIT_DIR = NMF_ROOT / "outputs" / "audit"
CONFIG_DIR = NMF_ROOT / "config"


# ---------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------

TOPIC_COUNT = 16
EXPECTED_NMF_VIDEOS = 1_158

COMMENT_THRESHOLDS = [1, 5, 10, 20]

ENGAGEMENT_METRICS = {
    "duration_minutes": "video_context",
    "view_count": "reach",
    "like_count": "reach",
    "api_comment_count": "discussion_volume",
    "likes_per_1000_views": "engagement_rate",
    "api_comments_per_1000_views": "discussion_rate",
    "views_per_day": "age_normalised_provisional",
    "api_comments_per_day": "age_normalised_provisional",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean_id(series: pd.Series) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def bh_adjust(
    p_values: pd.Series,
) -> pd.Series:
    """
    Benjamini-Hochberg FDR adjustment.

    Adjustment is applied within each metric/subsample family.
    """

    values = pd.to_numeric(
        p_values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    adjusted = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(
        values
    )

    if not valid.any():
        return pd.Series(
            adjusted,
            index=p_values.index,
        )

    valid_positions = np.flatnonzero(
        valid
    )

    valid_values = values[
        valid
    ]

    order = np.argsort(
        valid_values
    )

    ordered_p = valid_values[
        order
    ]

    m = len(
        ordered_p
    )

    raw_adjusted = (
        ordered_p
        * m
        / np.arange(
            1,
            m + 1,
        )
    )

    monotonic = np.minimum.accumulate(
        raw_adjusted[::-1]
    )[::-1]

    monotonic = np.minimum(
        monotonic,
        1.0,
    )

    restored = np.empty(
        m,
        dtype=float,
    )

    restored[
        order
    ] = monotonic

    adjusted[
        valid_positions
    ] = restored

    return pd.Series(
        adjusted,
        index=p_values.index,
    )


def epsilon_squared_kw(
    h_value: float,
    group_count: int,
    sample_size: int,
) -> float:
    """
    Approximate epsilon-squared effect size for Kruskal-Wallis.
    """

    if (
        sample_size <= group_count
        or not np.isfinite(
            h_value
        )
    ):
        return float("nan")

    value = (
        h_value
        - group_count
        + 1
    ) / (
        sample_size
        - group_count
    )

    return float(
        max(
            0.0,
            value,
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

    for file_path in (
        VIDEO_PROFILE_FILE,
        LOCKED_INTERPRETATION_FILE,
        METADATA_MASTER_FILE,
    ):
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    profiles = pd.read_csv(
        VIDEO_PROFILE_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    metadata = pd.read_csv(
        METADATA_MASTER_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    locked = pd.read_csv(
        LOCKED_INTERPRETATION_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if len(profiles) != EXPECTED_NMF_VIDEOS:
        raise ValueError(
            f"Expected {EXPECTED_NMF_VIDEOS:,} "
            f"NMF video profiles but found "
            f"{len(profiles):,}."
        )

    if profiles["video_id"].nunique() != len(
        profiles
    ):
        raise ValueError(
            "NMF video profiles do not contain "
            "unique video IDs."
        )

    if metadata["video_id"].nunique() != len(
        metadata
    ):
        raise ValueError(
            "Integrated metadata master does not "
            "contain unique video IDs."
        )

    profiles["video_id"] = clean_id(
        profiles[
            "video_id"
        ]
    )

    metadata["video_id"] = clean_id(
        metadata[
            "video_id"
        ]
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
    # Validate required metadata
    # -----------------------------------------------------------------

    required_metadata = [
        "video_id",
        *ENGAGEMENT_METRICS.keys(),
    ]

    missing_metadata = [
        column
        for column in required_metadata
        if column not in metadata.columns
    ]

    if missing_metadata:
        raise KeyError(
            "Integrated master is missing required columns:\n"
            f"{missing_metadata}"
        )

    topic_weight_columns = [
        f"topic_{topic_number:02d}_relative_weight"
        for topic_number in range(
            1,
            TOPIC_COUNT + 1,
        )
    ]

    missing_topic_columns = [
        column
        for column in topic_weight_columns
        if column not in profiles.columns
    ]

    if missing_topic_columns:
        raise KeyError(
            "Video profile file is missing NMF weight columns:\n"
            f"{missing_topic_columns}"
        )

    # -----------------------------------------------------------------
    # Merge only neutral metadata fields
    #
    # Rule-based comment themes, predefined retrofit domains and
    # knowledge-sharing modes are deliberately not imported.
    # -----------------------------------------------------------------

    metadata_subset = metadata[
        required_metadata
    ].copy()

    merged = profiles.merge(
        metadata_subset,
        on="video_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = (
        merged[
            "_merge"
        ]
        .ne("both")
        .sum()
    )

    if unmatched:
        raise ValueError(
            f"{int(unmatched)} NMF videos did not "
            "match the integrated metadata master."
        )

    merged = merged.drop(
        columns=[
            "_merge"
        ]
    )

    for metric in ENGAGEMENT_METRICS:
        merged[metric] = numeric(
            merged[
                metric
            ]
        )

    # -----------------------------------------------------------------
    # Audit valid denominators
    # -----------------------------------------------------------------

    denominator_rows = []

    for metric, family in (
        ENGAGEMENT_METRICS.items()
    ):

        valid = merged[
            metric
        ].notna()

        denominator_rows.append(
            {
                "metric": metric,
                "metric_family": family,
                "nmf_videos": int(
                    len(
                        merged
                    )
                ),
                "valid_n": int(
                    valid.sum()
                ),
                "missing_n": int(
                    (~valid).sum()
                ),
                "zero_n": int(
                    merged.loc[
                        valid,
                        metric,
                    ]
                    .eq(0)
                    .sum()
                ),
                "median": float(
                    merged.loc[
                        valid,
                        metric,
                    ].median()
                )
                if valid.any()
                else np.nan,
                "q25": float(
                    merged.loc[
                        valid,
                        metric,
                    ].quantile(
                        0.25
                    )
                )
                if valid.any()
                else np.nan,
                "q75": float(
                    merged.loc[
                        valid,
                        metric,
                    ].quantile(
                        0.75
                    )
                )
                if valid.any()
                else np.nan,
            }
        )

    denominators = pd.DataFrame(
        denominator_rows
    )

    # -----------------------------------------------------------------
    # Spearman associations:
    # continuous NMF topic weight versus engagement
    #
    # Repeated at comment-count thresholds because topic profiles based
    # on very few comments are less precisely estimated.
    # -----------------------------------------------------------------

    correlation_rows = []

    for threshold in COMMENT_THRESHOLDS:

        subset = merged.loc[
            merged[
                "modelled_comment_count"
            ]
            >= threshold
        ].copy()

        for topic_number in range(
            1,
            TOPIC_COUNT + 1,
        ):

            topic_column = (
                f"topic_{topic_number:02d}_relative_weight"
            )

            for metric, family in (
                ENGAGEMENT_METRICS.items()
            ):

                valid = (
                    subset[
                        topic_column
                    ].notna()
                    & subset[
                        metric
                    ].notna()
                )

                n = int(
                    valid.sum()
                )

                if n < 10:
                    rho = np.nan
                    p_value = np.nan
                else:
                    result = spearmanr(
                        subset.loc[
                            valid,
                            topic_column,
                        ],
                        subset.loc[
                            valid,
                            metric,
                        ],
                        nan_policy="omit",
                    )

                    rho = float(
                        result.statistic
                    )

                    p_value = float(
                        result.pvalue
                    )

                correlation_rows.append(
                    {
                        "minimum_modelled_comments": (
                            threshold
                        ),
                        "eligible_videos": int(
                            len(
                                subset
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
                        "topic_type": (
                            topic_types[
                                topic_number
                            ]
                        ),
                        "metric": (
                            metric
                        ),
                        "metric_family": (
                            family
                        ),
                        "valid_n": (
                            n
                        ),
                        "spearman_rho": (
                            rho
                        ),
                        "p_value_raw": (
                            p_value
                        ),
                    }
                )

    correlations = pd.DataFrame(
        correlation_rows
    )

    # FDR within each threshold × metric family of 16 topic tests.
    correlations[
        "p_value_bh_fdr"
    ] = np.nan

    for (
        threshold,
        metric,
    ), indices in correlations.groupby(
        [
            "minimum_modelled_comments",
            "metric",
        ]
    ).groups.items():

        correlations.loc[
            indices,
            "p_value_bh_fdr",
        ] = bh_adjust(
            correlations.loc[
                indices,
                "p_value_raw",
            ]
        ).to_numpy()

    # -----------------------------------------------------------------
    # Threshold sensitivity
    #
    # Compare full-video result against >=10-comments result.
    # -----------------------------------------------------------------

    full = (
        correlations.loc[
            correlations[
                "minimum_modelled_comments"
            ]
            == 1
        ][
            [
                "topic_number",
                "topic_label",
                "metric",
                "spearman_rho",
                "valid_n",
            ]
        ]
        .rename(
            columns={
                "spearman_rho": (
                    "rho_all_nmf_videos"
                ),
                "valid_n": (
                    "valid_n_all"
                ),
            }
        )
    )

    ten_plus = (
        correlations.loc[
            correlations[
                "minimum_modelled_comments"
            ]
            == 10
        ][
            [
                "topic_number",
                "metric",
                "spearman_rho",
                "valid_n",
            ]
        ]
        .rename(
            columns={
                "spearman_rho": (
                    "rho_min_10_comments"
                ),
                "valid_n": (
                    "valid_n_min_10"
                ),
            }
        )
    )

    correlation_sensitivity = (
        full.merge(
            ten_plus,
            on=[
                "topic_number",
                "metric",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    correlation_sensitivity[
        "rho_change_min10_minus_all"
    ] = (
        correlation_sensitivity[
            "rho_min_10_comments"
        ]
        - correlation_sensitivity[
            "rho_all_nmf_videos"
        ]
    )

    correlation_sensitivity[
        "absolute_rho_change"
    ] = (
        correlation_sensitivity[
            "rho_change_min10_minus_all"
        ].abs()
    )

    # -----------------------------------------------------------------
    # Dominant-topic engagement summaries
    #
    # This is complementary to the soft-weight analysis.
    # -----------------------------------------------------------------

    dominant_summary_rows = []

    for topic_number in range(
        1,
        TOPIC_COUNT + 1,
    ):

        group = merged.loc[
            merged[
                "dominant_topic_number"
            ]
            == topic_number
        ]

        for metric, family in (
            ENGAGEMENT_METRICS.items()
        ):

            values = (
                group[
                    metric
                ]
                .dropna()
            )

            dominant_summary_rows.append(
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
                        len(
                            group
                        )
                    ),
                    "metric": (
                        metric
                    ),
                    "metric_family": (
                        family
                    ),
                    "valid_n": int(
                        len(
                            values
                        )
                    ),
                    "median": float(
                        values.median()
                    )
                    if len(
                        values
                    )
                    else np.nan,
                    "q25": float(
                        values.quantile(
                            0.25
                        )
                    )
                    if len(
                        values
                    )
                    else np.nan,
                    "q75": float(
                        values.quantile(
                            0.75
                        )
                    )
                    if len(
                        values
                    )
                    else np.nan,
                }
            )

    dominant_summary = pd.DataFrame(
        dominant_summary_rows
    )

    # -----------------------------------------------------------------
    # Kruskal-Wallis across dominant NMF topics
    #
    # This asks whether metric distributions differ somewhere among the
    # 16 dominant-topic groups. It does not establish which topic caused
    # an engagement difference.
    # -----------------------------------------------------------------

    kw_rows = []

    for metric, family in (
        ENGAGEMENT_METRICS.items()
    ):

        groups = []

        group_sizes = []

        for topic_number in range(
            1,
            TOPIC_COUNT + 1,
        ):

            values = (
                merged.loc[
                    merged[
                        "dominant_topic_number"
                    ]
                    == topic_number,
                    metric,
                ]
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )

            if len(
                values
            ) >= 5:
                groups.append(
                    values
                )

                group_sizes.append(
                    len(
                        values
                    )
                )

        group_count = len(
            groups
        )

        total_n = int(
            sum(
                group_sizes
            )
        )

        if group_count < 2:
            h_value = np.nan
            p_value = np.nan
            epsilon_squared = np.nan
        else:
            result = kruskal(
                *groups,
                nan_policy="omit",
            )

            h_value = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

            epsilon_squared = (
                epsilon_squared_kw(
                    h_value,
                    group_count,
                    total_n,
                )
            )

        kw_rows.append(
            {
                "metric": (
                    metric
                ),
                "metric_family": (
                    family
                ),
                "group_count": (
                    group_count
                ),
                "valid_video_count": (
                    total_n
                ),
                "kruskal_h": (
                    h_value
                ),
                "p_value": (
                    p_value
                ),
                "epsilon_squared": (
                    epsilon_squared
                ),
            }
        )

    kw_results = pd.DataFrame(
        kw_rows
    )

    # -----------------------------------------------------------------
    # Topic-mixture diagnostic
    #
    # Video-level mixture breadth can depend on how many comments were
    # available. Check this before interpreting effective topic count.
    # -----------------------------------------------------------------

    mixture_diagnostics = []

    for variable in [
        "dominant_relative_weight",
        "dominance_margin",
        "effective_topic_count",
    ]:

        valid = (
            merged[
                variable
            ].notna()
            & merged[
                "modelled_comment_count"
            ].notna()
        )

        result = spearmanr(
            merged.loc[
                valid,
                variable,
            ],
            merged.loc[
                valid,
                "modelled_comment_count",
            ],
            nan_policy="omit",
        )

        mixture_diagnostics.append(
            {
                "mixture_measure": (
                    variable
                ),
                "comparison_variable": (
                    "modelled_comment_count"
                ),
                "valid_n": int(
                    valid.sum()
                ),
                "spearman_rho": float(
                    result.statistic
                ),
                "p_value": float(
                    result.pvalue
                ),
            }
        )

    mixture_diagnostics = (
        pd.DataFrame(
            mixture_diagnostics
        )
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    merged_path = (
        TABLE_DIR
        / "08c_nmf_video_profiles_with_engagement.csv"
    )

    denominator_path = (
        TABLE_DIR
        / "08c_engagement_metric_denominators.csv"
    )

    correlation_path = (
        TABLE_DIR
        / "08c_topic_weight_engagement_spearman.csv"
    )

    sensitivity_path = (
        TABLE_DIR
        / "08c_topic_engagement_comment_count_sensitivity.csv"
    )

    dominant_path = (
        TABLE_DIR
        / "08c_dominant_topic_engagement_summary.csv"
    )

    kw_path = (
        TABLE_DIR
        / "08c_dominant_topic_engagement_kruskal.csv"
    )

    mixture_path = (
        TABLE_DIR
        / "08c_video_topic_mixture_diagnostics.csv"
    )

    merged.to_csv(
        merged_path,
        index=False,
        encoding="utf-8-sig",
    )

    denominators.to_csv(
        denominator_path,
        index=False,
        encoding="utf-8-sig",
    )

    correlations.to_csv(
        correlation_path,
        index=False,
        encoding="utf-8-sig",
    )

    correlation_sensitivity.to_csv(
        sensitivity_path,
        index=False,
        encoding="utf-8-sig",
    )

    dominant_summary.to_csv(
        dominant_path,
        index=False,
        encoding="utf-8-sig",
    )

    kw_results.to_csv(
        kw_path,
        index=False,
        encoding="utf-8-sig",
    )

    mixture_diagnostics.to_csv(
        mixture_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    full_correlations = (
        correlations.loc[
            correlations[
                "minimum_modelled_comments"
            ]
            == 1
        ]
        .copy()
    )

    full_correlations[
        "absolute_rho"
    ] = (
        full_correlations[
            "spearman_rho"
        ].abs()
    )

    strongest = (
        full_correlations
        .dropna(
            subset=[
                "spearman_rho"
            ]
        )
        .sort_values(
            "absolute_rho",
            ascending=False,
        )
        .head(
            20
        )
    )

    max_sensitivity_change = float(
        correlation_sensitivity[
            "absolute_rho_change"
        ].max()
    )

    report_lines = [
        "YOUTUBE RETROFIT STANDALONE NMF ANALYSIS — STAGE 08c",
        "=" * 58,
        "",
        "Overall status: PASS",
        "",
        "Scope",
        "-----",
        (
            "This stage relates the frozen k=16 NMF "
            "video-level discussion profiles to neutral "
            "video metadata and engagement measures."
        ),
        (
            "No predefined retrofit domain, rule-based "
            "comment theme or rule-based knowledge-sharing "
            "mode is used."
        ),
        "",
        "Denominator",
        "-----------",
        (
            f"NMF videos matched to metadata: "
            f"{len(merged):,}"
        ),
        "",
        "Metric denominators",
        "-------------------",
    ]

    for _, row in denominators.iterrows():

        report_lines.append(
            (
                f"{row['metric']}: "
                f"valid n={int(row['valid_n']):,}; "
                f"missing={int(row['missing_n']):,}; "
                f"zero={int(row['zero_n']):,}; "
                f"median={row['median']:.4g}"
            )
        )

    report_lines.extend(
        [
            "",
            "Strongest full-sample topic–metadata associations",
            "-------------------------------------------------",
        ]
    )

    for _, row in strongest.iterrows():

        report_lines.append(
            (
                f"T{int(row['topic_number']):02d} "
                f"{row['topic_label']} × "
                f"{row['metric']}: "
                f"rho={row['spearman_rho']:.3f}, "
                f"n={int(row['valid_n']):,}, "
                f"BH-FDR p={row['p_value_bh_fdr']:.3g}"
            )
        )

    report_lines.extend(
        [
            "",
            "Dominant-topic Kruskal-Wallis tests",
            "-----------------------------------",
        ]
    )

    for _, row in kw_results.iterrows():

        report_lines.append(
            (
                f"{row['metric']}: "
                f"H={row['kruskal_h']:.3f}, "
                f"n={int(row['valid_video_count']):,}, "
                f"p={row['p_value']:.3g}, "
                f"epsilon-squared="
                f"{row['epsilon_squared']:.4f}"
            )
        )

    report_lines.extend(
        [
            "",
            "Low-comment-video sensitivity",
            "-----------------------------",
            (
                "Maximum absolute change in a Spearman "
                "correlation when restricting from all NMF "
                "videos to videos with at least 10 modelled "
                f"comments: {max_sensitivity_change:.3f}"
            ),
            "",
            "Topic-mixture diagnostics",
            "-------------------------",
        ]
    )

    for _, row in mixture_diagnostics.iterrows():

        report_lines.append(
            (
                f"{row['mixture_measure']} × "
                "modelled_comment_count: "
                f"rho={row['spearman_rho']:.3f}, "
                f"n={int(row['valid_n']):,}"
            )
        )

    report_lines.extend(
        [
            "",
            "Interpretive cautions",
            "--------------------",
            (
                "Associations are observational and do not "
                "show that a discussion topic causes views, "
                "likes or commenting behaviour."
            ),
            (
                "NMF topic weights are compositional: the "
                "16 relative weights within each video sum "
                "to one. Positive or negative correlations "
                "must therefore be interpreted as relative "
                "discussion composition, not independent "
                "topic quantities."
            ),
            (
                "View count, like count and API comment count "
                "are strongly right-skewed; Spearman rank "
                "correlations and median/IQR summaries are "
                "therefore used."
            ),
            (
                "API comment count represents total platform "
                "discussion, not the number of comments "
                "extracted for NMF."
            ),
            (
                "Likes and comments per 1,000 views are "
                "engagement-rate measures, not indicators "
                "of technical quality."
            ),
            (
                "Views per day and API comments per day remain "
                "provisional because the original metadata "
                "snapshot date was not retained."
            ),
            (
                "Kruskal-Wallis tests compare rank "
                "distributions across dominant-topic groups; "
                "they are not strictly tests of medians."
            ),
            (
                "BH-FDR adjusted p-values are exploratory and "
                "should be considered alongside effect sizes, "
                "not used as automatic importance thresholds."
            ),
            "",
            "Outputs",
            "-------",
            str(denominator_path),
            str(correlation_path),
            str(sensitivity_path),
            str(dominant_path),
            str(kw_path),
            str(mixture_path),
            str(merged_path),
        ]
    )

    report_path = (
        AUDIT_DIR
        / "08c_standalone_nmf_engagement_report.txt"
    )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    config = {
        "analysis_stage": (
            "Standalone NMF engagement/context analysis"
        ),
        "model_status": (
            "frozen; no NMF model fitting"
        ),
        "nmf_video_denominator": int(
            len(
                merged
            )
        ),
        "metadata_source": str(
            METADATA_MASTER_FILE
        ),
        "metrics": (
            ENGAGEMENT_METRICS
        ),
        "comment_count_thresholds": (
            COMMENT_THRESHOLDS
        ),
        "correlation_method": (
            "Spearman rank correlation"
        ),
        "multiple_testing_adjustment": (
            "Benjamini-Hochberg within each "
            "threshold × metric family of 16 topics"
        ),
        "dominant_topic_test": (
            "Kruskal-Wallis with approximate "
            "epsilon-squared effect size"
        ),
        "explicit_exclusions": [
            "retrofit_topic",
            "rule-based primary_theme",
            "rule-based knowledge-sharing modes",
            "substantive_comment_count as an outcome",
        ],
    }

    config_path = (
        CONFIG_DIR
        / "08c_nmf_engagement_config.json"
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