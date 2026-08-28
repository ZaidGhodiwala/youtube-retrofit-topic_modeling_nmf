from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


NMF_ROOT = Path(__file__).resolve().parents[1]

CORPUS_FILE = (
    NMF_ROOT
    / "data"
    / "processed"
    / "02_nmf_model_corpus_conservative.csv"
)

DIAGNOSTICS_FILE = (
    NMF_ROOT
    / "outputs"
    / "tables"
    / "05_candidate_model_diagnostics.csv"
)

TOPIC_SUMMARY_FILE = (
    NMF_ROOT
    / "outputs"
    / "tables"
    / "05_candidate_topic_summaries.csv"
)

REPRESENTATIVE_FILE = (
    NMF_ROOT
    / "outputs"
    / "tables"
    / "05_candidate_representative_comments.csv"
)

MODEL_DIR = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "05_inclusive_candidates"
)

REVIEW_DIR = NMF_ROOT / "outputs" / "review"
AUDIT_DIR = NMF_ROOT / "outputs" / "audit"

CANDIDATE_TOPIC_COUNTS = [12, 16, 20]

REPRESENTATIVE_COMMENTS_IN_REPORT = 5

GENERIC_SOCIAL_TERMS = {
    "video",
    "videos",
    "great",
    "good",
    "thanks",
    "thank",
    "nice",
    "love",
    "excellent",
    "amazing",
    "awesome",
    "channel",
    "watching",
    "youtube",
    "content",
}

CHANNEL_COLUMN_CANDIDATES = [
    "channel_title",
    "channel_name",
    "channel_id",
]


def clean_display_text(
    value: object,
    maximum_length: int = 550,
) -> str:
    """Prepare comment text for a compact review report."""

    text = re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()

    if len(text) > maximum_length:
        return text[: maximum_length - 3] + "..."

    return text


def top_distribution(
    values: pd.Series,
    maximum_items: int = 3,
) -> str:
    """Return the leading categories and their within-topic shares."""

    values = (
        values.fillna("Missing")
        .astype(str)
        .str.strip()
        .replace("", "Missing")
    )

    counts = values.value_counts(
        dropna=False
    )

    total = int(counts.sum())

    if total == 0:
        return ""

    entries: list[str] = []

    for label, count in counts.head(
        maximum_items
    ).items():
        entries.append(
            f"{label} "
            f"({int(count):,}; "
            f"{100 * count / total:.1f}%)"
        )

    return " | ".join(entries)


def detect_channel_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """Find an available channel identifier."""

    for column in CHANNEL_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column

    return None


def main() -> None:
    REVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_files = [
        CORPUS_FILE,
        DIAGNOSTICS_FILE,
        TOPIC_SUMMARY_FILE,
        REPRESENTATIVE_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

    corpus = pd.read_csv(
        CORPUS_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    diagnostics = pd.read_csv(
        DIAGNOSTICS_FILE,
        encoding="utf-8-sig",
    )

    topic_summaries = pd.read_csv(
        TOPIC_SUMMARY_FILE,
        encoding="utf-8-sig",
    )

    representatives = pd.read_csv(
        REPRESENTATIVE_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    diagnostics = diagnostics.loc[
        diagnostics["topic_count"].isin(
            CANDIDATE_TOPIC_COUNTS
        )
    ].copy()

    topic_summaries = topic_summaries.loc[
        topic_summaries[
            "candidate_topic_count"
        ].isin(CANDIDATE_TOPIC_COUNTS)
    ].copy()

    representatives = representatives.loc[
        representatives[
            "candidate_topic_count"
        ].isin(CANDIDATE_TOPIC_COUNTS)
    ].copy()

    channel_column = detect_channel_column(
        corpus
    )

    profile_rows: list[dict[str, object]] = []
    reviewer_rows: list[dict[str, object]] = []
    report_lines: list[str] = [
        "YOUTUBE RETROFIT NMF CANDIDATE INTERPRETABILITY REVIEW",
        "=" * 57,
        "",
        "Purpose",
        "-------",
        (
            "Compare the 12-, 16- and 20-topic inclusive NMF "
            "solutions before stability testing and final "
            "topic-count selection."
        ),
        (
            "Topic labels must be based on the combination of "
            "leading terms, representative comments and topic "
            "distribution rather than leading terms alone."
        ),
        "",
    ]

    for topic_count in CANDIDATE_TOPIC_COUNTS:
        diagnostic_row = diagnostics.loc[
            diagnostics["topic_count"]
            == topic_count
        ]

        if len(diagnostic_row) != 1:
            raise ValueError(
                f"Expected one diagnostic row for k={topic_count}."
            )

        diagnostic = diagnostic_row.iloc[0]

        document_topic_path = (
            MODEL_DIR
            / (
                "inclusive_document_topic_"
                f"k{topic_count:02d}.npz"
            )
        )

        if not document_topic_path.exists():
            raise FileNotFoundError(
                "Document-topic matrix not found:\n"
                f"{document_topic_path}"
            )

        document_topic = sparse.load_npz(
            document_topic_path
        ).toarray()

        if document_topic.shape != (
            len(corpus),
            topic_count,
        ):
            raise ValueError(
                f"Unexpected document-topic dimensions for "
                f"k={topic_count}: {document_topic.shape}"
            )

        dominant_topic = np.argmax(
            document_topic,
            axis=1,
        )

        row_sums = document_topic.sum(
            axis=1
        )

        dominant_weights = np.max(
            document_topic,
            axis=1,
        )

        relative_dominant_loading = np.divide(
            dominant_weights,
            row_sums,
            out=np.zeros_like(
                dominant_weights
            ),
            where=row_sums > 0,
        )

        report_lines.extend(
            [
                f"CANDIDATE: {topic_count} TOPICS",
                "-" * 29,
                (
                    "Mean NPMI: "
                    f"{diagnostic['mean_topic_npmi']:.4f}"
                ),
                (
                    "Topic diversity: "
                    f"{diagnostic['topic_diversity_top_20']:.3f}"
                ),
                (
                    "Mean inter-topic similarity: "
                    f"{diagnostic['mean_intertopic_cosine_similarity']:.3f}"
                ),
                (
                    "Maximum inter-topic similarity: "
                    f"{diagnostic['maximum_intertopic_cosine_similarity']:.3f}"
                ),
                (
                    "Relative reconstruction error: "
                    f"{diagnostic['relative_reconstruction_error']:.5f}"
                ),
                "",
            ]
        )

        candidate_summaries = (
            topic_summaries.loc[
                topic_summaries[
                    "candidate_topic_count"
                ]
                == topic_count
            ]
            .sort_values(
                "topic_number"
            )
        )

        if len(candidate_summaries) != topic_count:
            raise ValueError(
                f"Expected {topic_count} topic summaries for "
                f"k={topic_count}, found "
                f"{len(candidate_summaries)}."
            )

        for _, summary in candidate_summaries.iterrows():
            topic_number = int(
                summary["topic_number"]
            )

            topic_index = topic_number - 1

            topic_mask = (
                dominant_topic == topic_index
            )

            topic_indices = np.flatnonzero(
                topic_mask
            )

            topic_corpus = corpus.iloc[
                topic_indices
            ]

            dominant_count = int(
                topic_mask.sum()
            )

            dominant_share = (
                dominant_count
                / len(corpus)
            )

            topic_relative_loading = (
                relative_dominant_loading[
                    topic_mask
                ]
            )

            top_10_terms = [
                term.strip()
                for term in str(
                    summary["top_10_terms"]
                ).split("|")
                if term.strip()
            ]

            generic_overlap = sorted(
                set(
                    term.lower()
                    for term in top_10_terms
                )
                & GENERIC_SOCIAL_TERMS
            )

            video_counts = (
                topic_corpus["video_id"]
                .astype(str)
                .value_counts()
            )

            largest_video_share = (
                float(
                    video_counts.iloc[0]
                    / dominant_count
                )
                if dominant_count
                else 0.0
            )

            if channel_column is not None:
                channel_counts = (
                    topic_corpus[
                        channel_column
                    ]
                    .fillna("Missing")
                    .astype(str)
                    .value_counts()
                )

                largest_channel_share = (
                    float(
                        channel_counts.iloc[0]
                        / dominant_count
                    )
                    if dominant_count
                    else 0.0
                )

                leading_channels = (
                    top_distribution(
                        topic_corpus[
                            channel_column
                        ],
                        maximum_items=3,
                    )
                )
            else:
                largest_channel_share = np.nan
                leading_channels = ""

            profile = {
                "candidate_topic_count": topic_count,
                "topic_number": topic_number,
                "topic_npmi_top_10": float(
                    summary["topic_npmi_top_10"]
                ),
                "dominant_comment_count": dominant_count,
                "dominant_comment_share": dominant_share,
                "median_relative_dominant_loading": (
                    float(
                        np.median(
                            topic_relative_loading
                        )
                    )
                    if dominant_count
                    else np.nan
                ),
                "mean_relative_dominant_loading": (
                    float(
                        np.mean(
                            topic_relative_loading
                        )
                    )
                    if dominant_count
                    else np.nan
                ),
                "unique_videos": int(
                    topic_corpus[
                        "video_id"
                    ].nunique()
                ),
                "largest_single_video_share": (
                    largest_video_share
                ),
                "channel_column_used": (
                    channel_column or ""
                ),
                "largest_single_channel_share": (
                    largest_channel_share
                ),
                "top_10_terms": summary[
                    "top_10_terms"
                ],
                "top_20_terms": summary[
                    "top_20_terms"
                ],
                "generic_social_terms_in_top_10": (
                    " | ".join(
                        generic_overlap
                    )
                ),
                "retrofit_topic_distribution": (
                    top_distribution(
                        topic_corpus[
                            "retrofit_topic"
                        ]
                    )
                    if "retrofit_topic"
                    in topic_corpus.columns
                    else ""
                ),
                "creator_type_distribution": (
                    top_distribution(
                        topic_corpus[
                            "creator_type"
                        ]
                    )
                    if "creator_type"
                    in topic_corpus.columns
                    else ""
                ),
                "video_type_distribution": (
                    top_distribution(
                        topic_corpus[
                            "video_type"
                        ]
                    )
                    if "video_type"
                    in topic_corpus.columns
                    else ""
                ),
                "primary_theme_distribution": (
                    top_distribution(
                        topic_corpus[
                            "primary_theme"
                        ]
                    )
                    if "primary_theme"
                    in topic_corpus.columns
                    else ""
                ),
                "leading_channels": leading_channels,
            }

            profile_rows.append(profile)

            reviewer_rows.append(
                {
                    **profile,
                    "provisional_topic_label": "",
                    "coherence_rating_1_to_5": "",
                    "distinctiveness_rating_1_to_5": "",
                    "research_relevance_rating_1_to_5": "",
                    "possible_overlap_with_topic": "",
                    "retain_merge_or_reject": "",
                    "reviewer_notes": "",
                }
            )

            report_lines.extend(
                [
                    (
                        f"Topic {topic_number:02d} "
                        f"| NPMI="
                        f"{summary['topic_npmi_top_10']:.4f} "
                        f"| dominant comments="
                        f"{dominant_count:,} "
                        f"({100 * dominant_share:.2f}%)"
                    ),
                    (
                        "Top terms: "
                        + str(
                            summary["top_10_terms"]
                        )
                    ),
                    (
                        "Leading retrofit areas: "
                        + profile[
                            "retrofit_topic_distribution"
                        ]
                    ),
                    (
                        "Leading creator types: "
                        + profile[
                            "creator_type_distribution"
                        ]
                    ),
                    (
                        "Leading existing themes: "
                        + profile[
                            "primary_theme_distribution"
                        ]
                    ),
                    (
                        "Largest single-video share: "
                        f"{100 * largest_video_share:.2f}%"
                    ),
                ]
            )

            if channel_column is not None:
                report_lines.append(
                    (
                        "Largest single-channel share: "
                        f"{100 * largest_channel_share:.2f}%"
                    )
                )

            if generic_overlap:
                report_lines.append(
                    (
                        "Generic/social terms among top 10: "
                        + ", ".join(
                            generic_overlap
                        )
                    )
                )

            report_lines.append(
                "Representative comments:"
            )

            topic_representatives = (
                representatives.loc[
                    (
                        representatives[
                            "candidate_topic_count"
                        ]
                        == topic_count
                    )
                    & (
                        representatives[
                            "topic_number"
                        ]
                        == topic_number
                    )
                ]
                .sort_values(
                    "representative_rank"
                )
                .head(
                    REPRESENTATIVE_COMMENTS_IN_REPORT
                )
            )

            for _, representative in (
                topic_representatives.iterrows()
            ):
                text = clean_display_text(
                    representative[
                        "comment_text"
                    ]
                )

                report_lines.append(
                    (
                        f"  {int(representative['representative_rank'])}. "
                        f"[weight="
                        f"{representative['topic_weight']:.4f}; "
                        f"relative="
                        f"{representative['relative_topic_loading']:.3f}] "
                        f"{text}"
                    )
                )

            report_lines.append("")

        report_lines.extend(
            [
                "",
                "=" * 57,
                "",
            ]
        )

    profile_table = pd.DataFrame(
        profile_rows
    )

    reviewer_table = pd.DataFrame(
        reviewer_rows
    )

    profile_path = (
        REVIEW_DIR
        / "06_candidate_topic_profiles.csv"
    )

    reviewer_path = (
        REVIEW_DIR
        / "06_candidate_topic_reviewer_sheet.csv"
    )

    report_path = (
        REVIEW_DIR
        / "06_candidate_interpretability_review.txt"
    )

    profile_table.to_csv(
        profile_path,
        index=False,
        encoding="utf-8-sig",
    )

    reviewer_table.to_csv(
        reviewer_path,
        index=False,
        encoding="utf-8-sig",
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    audit_lines = [
        "YOUTUBE RETROFIT CANDIDATE REVIEW PREPARATION",
        "=" * 46,
        "",
        "Overall status: PASS",
        "",
        (
            "Candidate topic counts prepared: "
            + ", ".join(
                str(value)
                for value in CANDIDATE_TOPIC_COUNTS
            )
        ),
        (
            "Total candidate topics reviewed: "
            f"{len(profile_table):,}"
        ),
        (
            "Channel concentration available: "
            f"{channel_column is not None}"
        ),
        (
            "Channel field used: "
            f"{channel_column or 'None available'}"
        ),
        "",
        "Outputs",
        "-------",
        str(profile_path),
        str(reviewer_path),
        str(report_path),
        "",
        "Next decision",
        "-------------",
        (
            "Review topic coherence, distinctiveness, source "
            "concentration and representative-comment meaning "
            "before selecting topic counts for repeated "
            "stability analysis."
        ),
    ]

    audit_path = (
        AUDIT_DIR
        / "06_candidate_review_preparation_report.txt"
    )

    audit_path.write_text(
        "\n".join(audit_lines),
        encoding="utf-8",
    )

    print("\n".join(audit_lines))


if __name__ == "__main__":
    main()