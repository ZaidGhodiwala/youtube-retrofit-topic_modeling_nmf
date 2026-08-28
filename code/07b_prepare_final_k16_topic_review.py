from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
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

VECTORIZER_FILE = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "02_tfidf_vectorizer_conservative.joblib"
)

MODEL_FILE = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "05_inclusive_candidates"
    / "inclusive_nmf_k16.joblib"
)

DOCUMENT_TOPIC_FILE = (
    NMF_ROOT
    / "outputs"
    / "models"
    / "05_inclusive_candidates"
    / "inclusive_document_topic_k16.npz"
)

STABILITY_FILE = (
    NMF_ROOT
    / "outputs"
    / "tables"
    / "06b_reference_topic_stability.csv"
)

PREPROCESSING_FILE = (
    NMF_ROOT
    / "outputs"
    / "tables"
    / "07_preprocessing_topic_matching.csv"
)

REVIEW_DIR = (
    NMF_ROOT
    / "outputs"
    / "review"
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
# Selected model
# ---------------------------------------------------------------------

TOPIC_COUNT = 16

TOP_TERMS = 20

REPRESENTATIVE_COMMENTS_PER_TOPIC = 20

REPRESENTATIVES_IN_TEXT_REPORT = 10

MAX_REPRESENTATIVES_PER_VIDEO = 2


# ---------------------------------------------------------------------
# Provisional interpretation
#
# These are starting labels for researcher review, not automatically
# accepted final topic names.
# ---------------------------------------------------------------------

PROVISIONAL_TOPICS = {
    1: {
        "label": (
            "Household retrofit experience, older homes "
            "and general problems"
        ),
        "type": "mixed",
        "confidence": "moderate",
    },
    2: {
        "label": "Spray foam and foam insulation",
        "type": "technical_subject",
        "confidence": "high",
    },
    3: {
        "label": (
            "What, which and specification questions"
        ),
        "type": "knowledge_sharing_behaviour",
        "confidence": "moderate",
    },
    4: {
        "label": (
            "How-to, implementation and cost questions"
        ),
        "type": "knowledge_sharing_behaviour",
        "confidence": "moderate",
    },
    5: {
        "label": "Heat pumps and heating systems",
        "type": "technical_subject",
        "confidence": "high",
    },
    6: {
        "label": (
            "Praise, gratitude and social endorsement"
        ),
        "type": "social_interaction",
        "confidence": "high",
    },
    7: {
        "label": (
            "Why questions, alternatives and challenges"
        ),
        "type": "knowledge_sharing_behaviour",
        "confidence": "moderate",
    },
    8: {
        "label": (
            "Solar PV, batteries and grid integration"
        ),
        "type": "technical_subject",
        "confidence": "high",
    },
    9: {
        "label": (
            "Does it work? Performance and troubleshooting"
        ),
        "type": "mixed",
        "confidence": "moderate",
    },
    10: {
        "label": (
            "Reported actions and implementation experience"
        ),
        "type": "knowledge_sharing_behaviour",
        "confidence": "low",
    },
    11: {
        "label": (
            "Finding, buying and locating products or resources"
        ),
        "type": "knowledge_sharing_behaviour",
        "confidence": "moderate",
    },
    12: {
        "label": (
            "Roof and attic ventilation, airflow and heat"
        ),
        "type": "technical_subject",
        "confidence": "high",
    },
    13: {
        "label": (
            "Wall, cavity and floor insulation"
        ),
        "type": "technical_subject",
        "confidence": "high",
    },
    14: {
        "label": (
            "Thermostat wiring and smart heating controls"
        ),
        "type": "technical_subject",
        "confidence": "high",
    },
    15: {
        "label": (
            "Vapour barriers and moisture control"
        ),
        "type": "technical_subject",
        "confidence": "high",
    },
    16: {
        "label": (
            "Recommendations, hypothetical choices "
            "and advice-seeking"
        ),
        "type": "knowledge_sharing_behaviour",
        "confidence": "moderate",
    },
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def top_indices(
    components: np.ndarray,
    number_of_terms: int,
) -> np.ndarray:
    """Return descending top-term indices."""

    return np.argsort(
        components,
        axis=1,
    )[:, -number_of_terms:][:, ::-1]


def clean_text(
    value: object,
    maximum_length: int = 900,
) -> str:
    """Prepare a comment for human-readable output."""

    text = re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()

    if len(text) > maximum_length:
        return (
            text[: maximum_length - 3]
            + "..."
        )

    return text


def distribution_string(
    values: pd.Series,
    maximum_items: int = 5,
) -> str:
    """Summarise the largest categories within a topic."""

    cleaned = (
        values.fillna("Missing")
        .astype(str)
        .str.strip()
        .replace("", "Missing")
    )

    counts = cleaned.value_counts(
        dropna=False
    )

    total = int(
        counts.sum()
    )

    if total == 0:
        return ""

    pieces = []

    for label, count in counts.head(
        maximum_items
    ).items():
        pieces.append(
            f"{label} "
            f"({int(count):,}; "
            f"{100 * count / total:.1f}%)"
        )

    return " | ".join(
        pieces
    )


def select_representatives(
    corpus: pd.DataFrame,
    document_topic: np.ndarray,
    topic_number: int,
) -> list[dict[str, object]]:
    """
    Select high-loading comments while limiting repeated
    representation of the same video.
    """

    topic_index = (
        topic_number - 1
    )

    topic_weights = (
        document_topic[
            :,
            topic_index,
        ]
    )

    total_topic_weight = (
        document_topic.sum(
            axis=1
        )
    )

    relative_loading = np.divide(
        topic_weights,
        total_topic_weight,
        out=np.zeros_like(
            topic_weights,
            dtype=np.float64,
        ),
        where=(
            total_topic_weight > 0
        ),
    )

    ordered_rows = np.argsort(
        topic_weights
    )[::-1]

    text_column = (
        "comment_text_for_coding"
        if "comment_text_for_coding"
        in corpus.columns
        else "nmf_text_conservative"
    )

    selected = []

    video_counts: dict[str, int] = {}

    for row_index in ordered_rows:
        if (
            len(selected)
            >= REPRESENTATIVE_COMMENTS_PER_TOPIC
        ):
            break

        video_id = str(
            corpus.iloc[
                row_index
            ]["video_id"]
        )

        current_video_count = (
            video_counts.get(
                video_id,
                0,
            )
        )

        if (
            current_video_count
            >= MAX_REPRESENTATIVES_PER_VIDEO
        ):
            continue

        record = {
            "topic_number": (
                topic_number
            ),
            "representative_rank": (
                len(selected) + 1
            ),
            "tfidf_row_index": (
                int(row_index)
            ),
            "topic_weight": float(
                topic_weights[
                    row_index
                ]
            ),
            "relative_topic_loading": float(
                relative_loading[
                    row_index
                ]
            ),
            "video_id": video_id,
            "comment_id": corpus.iloc[
                row_index
            ]["comment_id"],
            "comment_text": (
                corpus.iloc[
                    row_index
                ][text_column]
            ),
        }

        for column in [
            "retrofit_topic",
            "creator_type",
            "video_type",
            "primary_theme",
        ]:
            if column in corpus.columns:
                record[column] = (
                    corpus.iloc[
                        row_index
                    ][column]
                )

        selected.append(
            record
        )

        video_counts[
            video_id
        ] = (
            current_video_count + 1
        )

    return selected


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    for directory in (
        REVIEW_DIR,
        AUDIT_DIR,
        CONFIG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    required_files = [
        CORPUS_FILE,
        VECTORIZER_FILE,
        MODEL_FILE,
        DOCUMENT_TOPIC_FILE,
        STABILITY_FILE,
        PREPROCESSING_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                "Required file not found:\n"
                f"{file_path}"
            )

    # -----------------------------------------------------------------
    # Load selected k=16 solution and validation evidence
    # -----------------------------------------------------------------

    corpus = pd.read_csv(
        CORPUS_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    document_topic = (
        sparse.load_npz(
            DOCUMENT_TOPIC_FILE
        )
        .toarray()
    )

    stability = pd.read_csv(
        STABILITY_FILE,
        encoding="utf-8-sig",
    )

    preprocessing = pd.read_csv(
        PREPROCESSING_FILE,
        encoding="utf-8-sig",
    )

    feature_names = np.asarray(
        vectorizer.get_feature_names_out()
    )

    if (
        model.components_.shape[0]
        != TOPIC_COUNT
    ):
        raise ValueError(
            "Selected model is not a "
            "16-topic model."
        )

    if document_topic.shape != (
        len(corpus),
        TOPIC_COUNT,
    ):
        raise ValueError(
            "Document-topic matrix does not align "
            "with the corpus."
        )

    stability = stability.loc[
        stability[
            "candidate_topic_count"
        ]
        == TOPIC_COUNT
    ].copy()

    if len(stability) != TOPIC_COUNT:
        raise ValueError(
            "Expected 16 topic-level stability "
            f"rows but found {len(stability)}."
        )

    if len(preprocessing) != TOPIC_COUNT:
        raise ValueError(
            "Expected 16 preprocessing-comparison "
            f"rows but found {len(preprocessing)}."
        )

    # -----------------------------------------------------------------
    # Topic terms and dominant assignments
    # -----------------------------------------------------------------

    leading_indices = top_indices(
        model.components_,
        TOP_TERMS,
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

    topic_counts = np.bincount(
        dominant_topic,
        minlength=TOPIC_COUNT,
    )

    topic_shares = (
        topic_counts
        / len(corpus)
    )

    # -----------------------------------------------------------------
    # Build final topic-review table
    # -----------------------------------------------------------------

    review_rows = []
    representative_rows = []

    for topic_number in range(
        1,
        TOPIC_COUNT + 1,
    ):

        topic_index = (
            topic_number - 1
        )

        topic_mask = (
            dominant_topic
            == topic_index
        )

        topic_rows = np.flatnonzero(
            topic_mask
        )

        topic_corpus = (
            corpus.iloc[
                topic_rows
            ]
        )

        top_terms = (
            feature_names[
                leading_indices[
                    topic_index
                ]
            ]
        )

        topic_stability = (
            stability.loc[
                stability[
                    "reference_topic_number"
                ]
                == topic_number
            ]
        )

        if len(topic_stability) != 1:
            raise ValueError(
                "Could not uniquely identify "
                f"stability statistics for "
                f"topic {topic_number}."
            )

        topic_stability = (
            topic_stability.iloc[0]
        )

        preprocessing_match = (
            preprocessing.loc[
                preprocessing[
                    "inclusive_topic_number"
                ]
                == topic_number
            ]
        )

        if len(preprocessing_match) != 1:
            raise ValueError(
                "Could not uniquely identify "
                f"preprocessing sensitivity for "
                f"topic {topic_number}."
            )

        preprocessing_match = (
            preprocessing_match.iloc[0]
        )

        video_counts = (
            topic_corpus[
                "video_id"
            ]
            .astype(str)
            .value_counts()
        )

        largest_video_share = (
            float(
                video_counts.iloc[0]
                / len(topic_corpus)
            )
            if len(topic_corpus)
            else np.nan
        )

        relative_values = (
            relative_dominant_loading[
                topic_mask
            ]
        )

        provisional = (
            PROVISIONAL_TOPICS[
                topic_number
            ]
        )

        # Review cue only.
        # This is not an automatic accept/reject rule.
        review_cues = []

        mean_stability = float(
            topic_stability[
                "mean_cosine_similarity"
            ]
        )

        preprocessing_cosine = float(
            preprocessing_match[
                "shared_vocabulary_cosine_similarity"
            ]
        )

        if mean_stability < 0.90:
            review_cues.append(
                "inspect_sampling_stability"
            )

        if preprocessing_cosine < 0.80:
            review_cues.append(
                "inspect_preprocessing_dependence"
            )

        if (
            topic_number == 6
        ):
            review_cues.append(
                "expected_social_topic_preprocessing_change"
            )

        review_rows.append(
            {
                "topic_number": (
                    topic_number
                ),

                "provisional_label": (
                    provisional[
                        "label"
                    ]
                ),

                "provisional_topic_type": (
                    provisional[
                        "type"
                    ]
                ),

                "provisional_label_confidence": (
                    provisional[
                        "confidence"
                    ]
                ),

                "top_10_terms": (
                    " | ".join(
                        top_terms[:10]
                    )
                ),

                "top_20_terms": (
                    " | ".join(
                        top_terms
                    )
                ),

                "dominant_comment_count": (
                    int(
                        topic_counts[
                            topic_index
                        ]
                    )
                ),

                "dominant_comment_share": (
                    float(
                        topic_shares[
                            topic_index
                        ]
                    )
                ),

                "unique_videos": int(
                    topic_corpus[
                        "video_id"
                    ].nunique()
                ),

                "largest_single_video_share": (
                    largest_video_share
                ),

                "median_relative_dominant_loading": (
                    float(
                        np.median(
                            relative_values
                        )
                    )
                    if len(
                        relative_values
                    )
                    else np.nan
                ),

                "mean_relative_dominant_loading": (
                    float(
                        np.mean(
                            relative_values
                        )
                    )
                    if len(
                        relative_values
                    )
                    else np.nan
                ),

                "retrofit_topic_distribution": (
                    distribution_string(
                        topic_corpus[
                            "retrofit_topic"
                        ]
                    )
                    if "retrofit_topic"
                    in topic_corpus.columns
                    else ""
                ),

                "creator_type_distribution": (
                    distribution_string(
                        topic_corpus[
                            "creator_type"
                        ]
                    )
                    if "creator_type"
                    in topic_corpus.columns
                    else ""
                ),

                "video_type_distribution": (
                    distribution_string(
                        topic_corpus[
                            "video_type"
                        ]
                    )
                    if "video_type"
                    in topic_corpus.columns
                    else ""
                ),

                "existing_primary_theme_distribution": (
                    distribution_string(
                        topic_corpus[
                            "primary_theme"
                        ]
                    )
                    if "primary_theme"
                    in topic_corpus.columns
                    else ""
                ),

                # Sampling stability
                "sampling_mean_cosine": (
                    mean_stability
                ),

                "sampling_median_cosine": float(
                    topic_stability[
                        "median_cosine_similarity"
                    ]
                ),

                "sampling_minimum_cosine": float(
                    topic_stability[
                        "minimum_cosine_similarity"
                    ]
                ),

                "sampling_mean_top20_jaccard": float(
                    topic_stability[
                        "mean_top_20_jaccard"
                    ]
                ),

                "sampling_minimum_top20_jaccard": float(
                    topic_stability[
                        "minimum_top_20_jaccard"
                    ]
                ),

                "sampling_mean_absolute_prevalence_difference": float(
                    topic_stability[
                        "mean_absolute_prevalence_difference"
                    ]
                ),

                # Preprocessing robustness
                "matched_sensitivity_topic_number": int(
                    preprocessing_match[
                        "matched_sensitivity_topic_number"
                    ]
                ),

                "preprocessing_cosine_similarity": (
                    preprocessing_cosine
                ),

                "preprocessing_top20_jaccard": float(
                    preprocessing_match[
                        "top_20_term_jaccard"
                    ]
                ),

                "preprocessing_absolute_prevalence_difference": float(
                    preprocessing_match[
                        "absolute_prevalence_difference"
                    ]
                ),

                "sensitivity_top_20_terms": (
                    preprocessing_match[
                        "sensitivity_top_20_terms"
                    ]
                ),

                "review_cues": (
                    " | ".join(
                        review_cues
                    )
                ),

                # Researcher completion fields
                "researcher_final_label": "",
                "researcher_topic_type": "",
                "researcher_interpretation": "",
                "researcher_confidence_1_to_5": "",
                "retain_as_distinct_topic": "",
                "possible_overlap_with_topic": "",
                "researcher_notes": "",
            }
        )

        representative_rows.extend(
            select_representatives(
                corpus=corpus,
                document_topic=document_topic,
                topic_number=topic_number,
            )
        )

    review_table = pd.DataFrame(
        review_rows
    )

    representatives = pd.DataFrame(
        representative_rows
    )

    # -----------------------------------------------------------------
    # Save CSV review materials
    # -----------------------------------------------------------------

    review_path = (
        REVIEW_DIR
        / "07b_final_k16_topic_review.csv"
    )

    representative_path = (
        REVIEW_DIR
        / "07b_final_k16_representative_comments.csv"
    )

    review_table.to_csv(
        review_path,
        index=False,
        encoding="utf-8-sig",
    )

    representatives.to_csv(
        representative_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Human-readable review report
    # -----------------------------------------------------------------

    report_lines = [
        "YOUTUBE RETROFIT FINAL k=16 NMF TOPIC REVIEW",
        "=" * 49,
        "",
        "Model status",
        "------------",
        (
            "Selected model: inclusive primary "
            "TF-IDF, k=16 NMF"
        ),
        (
            f"Comments represented: "
            f"{len(corpus):,}"
        ),
        (
            f"Videos represented: "
            f"{corpus['video_id'].nunique():,}"
        ),
        "",
        "Purpose",
        "-------",
        (
            "This report supports final human "
            "interpretation and naming of the "
            "selected 16-topic NMF solution."
        ),
        (
            "Provisional labels are review aids. "
            "They are not treated as final topic "
            "labels until explicitly approved by "
            "the researcher."
        ),
        "",
    ]

    for _, row in review_table.iterrows():

        topic_number = int(
            row[
                "topic_number"
            ]
        )

        report_lines.extend(
            [
                "=" * 70,
                (
                    f"TOPIC "
                    f"{topic_number:02d}"
                ),
                "=" * 70,
                "",
                (
                    "Provisional label: "
                    f"{row['provisional_label']}"
                ),
                (
                    "Provisional type: "
                    f"{row['provisional_topic_type']}"
                ),
                (
                    "Provisional confidence: "
                    f"{row['provisional_label_confidence']}"
                ),
                "",
                (
                    "Top 20 terms:"
                ),
                str(
                    row[
                        "top_20_terms"
                    ]
                ),
                "",
                (
                    "Dominant comments: "
                    f"{int(row['dominant_comment_count']):,} "
                    f"("
                    f"{100 * row['dominant_comment_share']:.2f}%"
                    f")"
                ),
                (
                    "Unique videos: "
                    f"{int(row['unique_videos']):,}"
                ),
                (
                    "Largest single-video share: "
                    f"{100 * row['largest_single_video_share']:.2f}%"
                ),
                "",
                "Corpus context:",
                (
                    "  Retrofit areas: "
                    f"{row['retrofit_topic_distribution']}"
                ),
                (
                    "  Creator types: "
                    f"{row['creator_type_distribution']}"
                ),
                (
                    "  Video types: "
                    f"{row['video_type_distribution']}"
                ),
                (
                    "  Existing rule-based themes: "
                    f"{row['existing_primary_theme_distribution']}"
                ),
                "",
                "Sampling stability:",
                (
                    "  Mean cosine: "
                    f"{row['sampling_mean_cosine']:.4f}"
                ),
                (
                    "  Median cosine: "
                    f"{row['sampling_median_cosine']:.4f}"
                ),
                (
                    "  Minimum cosine: "
                    f"{row['sampling_minimum_cosine']:.4f}"
                ),
                (
                    "  Mean top-20 Jaccard: "
                    f"{row['sampling_mean_top20_jaccard']:.4f}"
                ),
                "",
                "Preprocessing sensitivity:",
                (
                    "  Matched sensitivity topic: "
                    f"{int(row['matched_sensitivity_topic_number']):02d}"
                ),
                (
                    "  Cosine similarity: "
                    f"{row['preprocessing_cosine_similarity']:.4f}"
                ),
                (
                    "  Top-20 Jaccard: "
                    f"{row['preprocessing_top20_jaccard']:.4f}"
                ),
            ]
        )

        if str(
            row[
                "review_cues"
            ]
        ).strip():
            report_lines.extend(
                [
                    (
                        "Review cues: "
                        f"{row['review_cues']}"
                    ),
                ]
            )

        report_lines.extend(
            [
                "",
                (
                    "Representative comments "
                    f"(first "
                    f"{REPRESENTATIVES_IN_TEXT_REPORT} "
                    "of "
                    f"{REPRESENTATIVE_COMMENTS_PER_TOPIC}):"
                ),
            ]
        )

        topic_representatives = (
            representatives.loc[
                representatives[
                    "topic_number"
                ]
                == topic_number
            ]
            .sort_values(
                "representative_rank"
            )
            .head(
                REPRESENTATIVES_IN_TEXT_REPORT
            )
        )

        for _, representative in (
            topic_representatives.iterrows()
        ):

            report_lines.append(
                (
                    f"  "
                    f"{int(representative['representative_rank'])}. "
                    f"[weight="
                    f"{representative['topic_weight']:.4f}; "
                    f"relative="
                    f"{representative['relative_topic_loading']:.3f}] "
                    f"{clean_text(representative['comment_text'])}"
                )
            )

        report_lines.extend(
            [
                "",
                "Researcher review:",
                "  Final label:",
                "  Topic type:",
                "  Interpretation:",
                "  Confidence (1-5):",
                "  Retain as distinct topic:",
                "  Possible overlap:",
                "  Notes:",
                "",
            ]
        )

    report_path = (
        REVIEW_DIR
        / "07b_final_k16_topic_review.txt"
    )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Configuration / audit
    # -----------------------------------------------------------------

    configuration = {
        "analysis_stage": (
            "Final human interpretation "
            "of selected k=16 NMF"
        ),
        "selected_model": (
            "inclusive_primary_k16"
        ),
        "topic_count": TOPIC_COUNT,
        "top_terms_per_topic": (
            TOP_TERMS
        ),
        "representative_comments_per_topic": (
            REPRESENTATIVE_COMMENTS_PER_TOPIC
        ),
        "maximum_representatives_per_video": (
            MAX_REPRESENTATIVES_PER_VIDEO
        ),
        "topic_types_available": [
            "technical_subject",
            "knowledge_sharing_behaviour",
            "social_interaction",
            "mixed",
        ],
        "interpretation_rule": (
            "Final labels require human review "
            "of terms, representative comments, "
            "topic distributions, sampling stability "
            "and preprocessing sensitivity."
        ),
    }

    config_path = (
        CONFIG_DIR
        / "07b_final_k16_topic_review_config.json"
    )

    config_path.write_text(
        json.dumps(
            configuration,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    weak_sampling = (
        review_table.loc[
            review_table[
                "sampling_mean_cosine"
            ]
            < 0.90,
            "topic_number",
        ]
        .astype(int)
        .tolist()
    )

    preprocessing_sensitive = (
        review_table.loc[
            review_table[
                "preprocessing_cosine_similarity"
            ]
            < 0.80,
            "topic_number",
        ]
        .astype(int)
        .tolist()
    )

    audit_lines = [
        "YOUTUBE RETROFIT FINAL k=16 REVIEW PREPARATION",
        "=" * 49,
        "",
        "Overall status: PASS",
        "",
        (
            "Selected model: "
            "inclusive k=16"
        ),
        (
            "Topics prepared for review: "
            f"{len(review_table)}"
        ),
        (
            "Representative comments prepared: "
            f"{len(representatives):,}"
        ),
        (
            "Representative comments per topic: "
            f"{REPRESENTATIVE_COMMENTS_PER_TOPIC}"
        ),
        "",
        "Review flags",
        "------------",
        (
            "Topics with mean sampling cosine < 0.90: "
            + (
                ", ".join(
                    str(value)
                    for value
                    in weak_sampling
                )
                if weak_sampling
                else "None"
            )
        ),
        (
            "Topics with preprocessing cosine < 0.80: "
            + (
                ", ".join(
                    str(value)
                    for value
                    in preprocessing_sensitive
                )
                if preprocessing_sensitive
                else "None"
            )
        ),
        "",
        "Important",
        "---------",
        (
            "These flags are review cues only. "
            "They are not automatic thresholds "
            "for rejecting a topic."
        ),
        (
            "Topic 6 is expected to change strongly "
            "under the content-focused sensitivity "
            "condition because praise, gratitude and "
            "platform vocabulary were deliberately "
            "removed."
        ),
        "",
        "Outputs",
        "-------",
        str(review_path),
        str(representative_path),
        str(report_path),
        str(config_path),
    ]

    audit_path = (
        AUDIT_DIR
        / "07b_final_k16_review_preparation_report.txt"
    )

    audit_path.write_text(
        "\n".join(
            audit_lines
        ),
        encoding="utf-8",
    )

    print(
        "\n".join(
            audit_lines
        )
    )


if __name__ == "__main__":
    main()