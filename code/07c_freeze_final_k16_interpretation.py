from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


NMF_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    NMF_ROOT
    / "outputs"
    / "review"
    / "07b_final_k16_topic_review.csv"
)

OUTPUT_FILE = (
    NMF_ROOT
    / "outputs"
    / "review"
    / "07c_final_k16_topic_interpretation_locked.csv"
)

REPORT_FILE = (
    NMF_ROOT
    / "outputs"
    / "audit"
    / "07c_final_k16_interpretation_lock_report.txt"
)

CONFIG_FILE = (
    NMF_ROOT
    / "config"
    / "07c_final_k16_interpretation.json"
)


FINAL_TOPICS = {
    1: {
        "label": (
            "Situated household retrofit experience "
            "and building context"
        ),
        "type": "mixed",
        "interpretation": (
            "Household- and property-specific accounts of retrofit, "
            "heating, damp, insulation, windows and building age. "
            "The component captures situated experience and the way "
            "viewers qualify retrofit advice through climate, dwelling "
            "condition, occupancy, previous interventions and personal "
            "experience rather than one discrete technology."
        ),
        "confidence": 4,
        "overlap": "5 | 12 | 13 | 15 | 16",
        "notes": (
            "Broad cross-cutting component. Sampling stability is "
            "acceptable but preprocessing correspondence is weaker "
            "than for the sharply technical topics."
        ),
    },

    2: {
        "label": "Spray foam and foam insulation",
        "type": "technical_subject",
        "interpretation": (
            "Discussion of spray foam, closed-cell foam, foam board, "
            "installation, sealing and alternatives, including strong "
            "claims about cost, risk, mortgages, suitability and "
            "material choice."
        ),
        "confidence": 5,
        "overlap": "13 | 15",
        "notes": (
            "Exceptionally stable under both subsampling and "
            "preprocessing sensitivity."
        ),
    },

    3: {
        "label": "Clarification and specification questions",
        "type": "knowledge_sharing_behaviour",
        "interpretation": (
            "Requests to identify, define or clarify products, brands, "
            "sizes, types, specifications, costs, components and the "
            "meaning or purpose of demonstrated measures."
        ),
        "confidence": 4,
        "overlap": "4 | 7 | 11 | 16",
        "notes": (
            "A discourse-function topic rather than a retrofit "
            "technology domain."
        ),
    },

    4: {
        "label": "Implementation, cost and how-to questions",
        "type": "knowledge_sharing_behaviour",
        "interpretation": (
            "Procedural help-seeking concerning how work is carried "
            "out, how much it costs, how long it takes, installation "
            "steps, dimensions, performance in particular conditions "
            "and practical implementation."
        ),
        "confidence": 4,
        "overlap": "3 | 10 | 11 | 16",
        "notes": (
            "Highly stable cross-cutting help-seeking component."
        ),
    },

    5: {
        "label": "Heat pumps and heating systems",
        "type": "technical_subject",
        "interpretation": (
            "Heat-pump and heating-system discussion involving gas "
            "boilers, hot water, temperatures, emitters, electricity, "
            "heat loss, installation, operating cost and system "
            "performance."
        ),
        "confidence": 5,
        "overlap": "1 | 9 | 14",
        "notes": (
            "Strong correspondence with the predefined heat-pump "
            "retrofit area but discovered independently by NMF."
        ),
    },

    6: {
        "label": "Praise, gratitude and social endorsement",
        "type": "social_interaction",
        "interpretation": (
            "Positive social reinforcement expressed through thanks, "
            "praise, statements of usefulness, approval and recognition "
            "of creators or videos. Some comments combine endorsement "
            "with subsequent technical questions."
        ),
        "confidence": 5,
        "overlap": "",
        "notes": (
            "Its disappearance under content-focused preprocessing is "
            "expected because the sensitivity condition deliberately "
            "removed its defining praise, gratitude and platform terms. "
            "Social endorsement must not be interpreted as evidence of "
            "technical accuracy."
        ),
    },

    7: {
        "label": (
            "Alternative methods, challenges and "
            "\"why\" questions"
        ),
        "type": "knowledge_sharing_behaviour",
        "interpretation": (
            "Questions and challenges concerning why a demonstrated "
            "method, material or sequence was chosen and why an "
            "alternative was not used. This component captures "
            "comparison, challenge and informal contestation."
        ),
        "confidence": 4,
        "overlap": "3 | 4 | 16",
        "notes": (
            "Important for analysing how disagreement and alternatives "
            "contribute to informal learning."
        ),
    },

    8: {
        "label": "Solar PV, batteries and power systems",
        "type": "technical_subject",
        "interpretation": (
            "Discussion of solar panels, batteries, inverters, grid "
            "interaction, electrical capacity, system configuration, "
            "cost, storage and installation."
        ),
        "confidence": 5,
        "overlap": "4 | 11 | 16",
        "notes": (
            "Highly stable substantive technical component."
        ),
    },

    9: {
        "label": "Performance checking and troubleshooting",
        "type": "knowledge_sharing_behaviour",
        "interpretation": (
            "Questions and reports concerning whether an intervention "
            "or device works, why it does not work, compatibility, "
            "observed failure and attempts to diagnose performance "
            "problems."
        ),
        "confidence": 4,
        "overlap": "3 | 5 | 10 | 14",
        "notes": (
            "Cross-cutting troubleshooting behaviour rather than one "
            "technical technology."
        ),
    },

    10: {
        "label": (
            "Implementation follow-up, outcomes and "
            "retrospective questions"
        ),
        "type": "knowledge_sharing_behaviour",
        "interpretation": (
            "Retrospective questions and reports about what was done, "
            "what happened afterwards, omitted implementation details, "
            "costs, sourcing, outcomes and whether a demonstrated "
            "method was subsequently successful."
        ),
        "confidence": 3,
        "overlap": "3 | 4 | 9 | 11",
        "notes": (
            "Retained cautiously. Median sampling stability is very "
            "high and preprocessing robustness is strong, but one or "
            "more subsamples reorganised this component substantially, "
            "producing a low mean and minimum stability score."
        ),
    },

    11: {
        "label": "Product sourcing, access and location questions",
        "type": "knowledge_sharing_behaviour",
        "interpretation": (
            "Requests about where products, tools or components can be "
            "bought or found, where creators or projects are located, "
            "and where viewers can access related resources."
        ),
        "confidence": 4,
        "overlap": "3 | 4 | 10",
        "notes": (
            "Very stable lexical behaviour, although some individual "
            "high-loading comments are outside substantive retrofit."
        ),
    },

    12: {
        "label": "Roof and attic ventilation and airflow",
        "type": "technical_subject",
        "interpretation": (
            "Discussion of roof and attic ventilation, warm and cold "
            "roof construction, soffit and ridge vents, air gaps, "
            "airflow, condensation and interactions between insulation "
            "and roof ventilation."
        ),
        "confidence": 5,
        "overlap": "1 | 13 | 15",
        "notes": (
            "Retained despite one anomalous subsampling match. Median "
            "sampling similarity is extremely high and preprocessing "
            "robustness is strong."
        ),
    },

    13: {
        "label": "Wall, cavity and floor insulation",
        "type": "technical_subject",
        "interpretation": (
            "Discussion of insulation in walls, cavities and floors, "
            "including internal and external approaches, boards, "
            "brickwork, damp interactions and concerns about cavity "
            "wall insulation."
        ),
        "confidence": 5,
        "overlap": "2 | 12 | 15",
        "notes": (
            "Highly stable substantive insulation component."
        ),
    },

    14: {
        "label": "Thermostat wiring and smart heating controls",
        "type": "technical_subject",
        "interpretation": (
            "Detailed troubleshooting and installation questions "
            "about thermostat wiring, C-wires, terminals, boilers, "
            "furnaces and smart-control products such as Nest, "
            "Honeywell and Hive."
        ),
        "confidence": 5,
        "overlap": "5 | 9",
        "notes": (
            "One of the clearest examples of practical technical "
            "problem-solving recovered independently by NMF."
        ),
    },

    15: {
        "label": "Vapour barriers and moisture control",
        "type": "technical_subject",
        "interpretation": (
            "Questions, recommendations and disagreement concerning "
            "vapour barriers, vapour control layers, moisture movement, "
            "material placement, breathability and interfaces with "
            "floors, roofs and insulation."
        ),
        "confidence": 5,
        "overlap": "2 | 12 | 13",
        "notes": (
            "Retained despite one anomalous subsampling result. Median "
            "sampling stability and preprocessing robustness are both "
            "extremely high."
        ),
    },

    16: {
        "label": (
            "Prospective choices, recommendations "
            "and alternatives"
        ),
        "type": "knowledge_sharing_behaviour",
        "interpretation": (
            "Prospective and hypothetical decision-making expressed "
            "through questions about what would work, what should be "
            "used, what an intervention would cost, who should perform "
            "the work, and which alternative might be preferable."
        ),
        "confidence": 3,
        "overlap": "3 | 4 | 7 | 9 | 11",
        "notes": (
            "Diffuse cross-cutting component. Retained because it "
            "captures a recognisable prospective decision-making "
            "function, but interpretation must acknowledge lower "
            "sampling stability and substantial preprocessing "
            "dependence."
        ),
    },
}


def sha256(path: Path) -> str:
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


def main() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFIG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Final k=16 review file not found:\n"
            f"{INPUT_FILE}"
        )

    review = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )
        # -----------------------------------------------------------------
    # Explicit dtypes for previously blank researcher-review columns
    #
    # These columns are blank in the Stage 07b CSV, so pandas may infer
    # them as float64 because they contain only NaN values. They must be
    # explicitly initialised before writing the locked interpretation.
    # -----------------------------------------------------------------

    TEXT_REVIEW_COLUMNS = [
        "researcher_final_label",
        "researcher_topic_type",
        "researcher_interpretation",
        "retain_as_distinct_topic",
        "possible_overlap_with_topic",
        "researcher_notes",
    ]

    for column in TEXT_REVIEW_COLUMNS:
        if column not in review.columns:
            review[column] = pd.Series(
                "",
                index=review.index,
                dtype="string",
            )
        else:
            review[column] = (
                review[column]
                .astype("string")
                .fillna("")
            )

    if "researcher_confidence_1_to_5" not in review.columns:
        review["researcher_confidence_1_to_5"] = pd.Series(
            pd.NA,
            index=review.index,
            dtype="Int64",
        )
    else:
        review["researcher_confidence_1_to_5"] = (
            pd.to_numeric(
                review["researcher_confidence_1_to_5"],
                errors="coerce",
            )
            .astype("Int64")
        )
    expected_topics = set(
        range(1, 17)
    )

    observed_topics = set(
        review[
            "topic_number"
        ]
        .astype(int)
        .tolist()
    )

    if observed_topics != expected_topics:
        raise ValueError(
            "Expected exactly topics 1-16.\n"
            f"Observed: "
            f"{sorted(observed_topics)}"
        )

    if len(review) != 16:
        raise ValueError(
            "Expected exactly 16 rows in "
            "the final review table."
        )

    for row_index, row in review.iterrows():
        topic_number = int(
            row["topic_number"]
        )

        decision = FINAL_TOPICS[
            topic_number
        ]

        review.at[
            row_index,
            "researcher_final_label",
        ] = decision["label"]

        review.at[
            row_index,
            "researcher_topic_type",
        ] = decision["type"]

        review.at[
            row_index,
            "researcher_interpretation",
        ] = decision[
            "interpretation"
        ]

        review.at[
            row_index,
            "researcher_confidence_1_to_5",
        ] = decision[
            "confidence"
        ]

        review.at[
            row_index,
            "retain_as_distinct_topic",
        ] = "yes"

        review.at[
            row_index,
            "possible_overlap_with_topic",
        ] = decision[
            "overlap"
        ]

        review.at[
            row_index,
            "researcher_notes",
        ] = decision[
            "notes"
        ]

    if review[
        "researcher_final_label"
    ].isna().any():
        raise ValueError(
            "At least one final label is missing."
        )

    if (
        review[
            "researcher_final_label"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "At least one final label is blank."
        )

    review.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    source_hash = sha256(
        INPUT_FILE
    )

    output_hash = sha256(
        OUTPUT_FILE
    )

    type_counts = {
    str(key): int(value)
    for key, value in (
        review[
            "researcher_topic_type"
        ]
        .value_counts()
        .items()
    )
}

    confidence_counts = {
        str(int(key)): int(value)
        for key, value in (
            review[
                "researcher_confidence_1_to_5"
            ]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    config = {
        "analysis_stage": (
            "Final interpretation lock "
            "for selected inclusive k=16 NMF"
        ),
        "source_review_file": str(
            INPUT_FILE
        ),
        "source_review_sha256": (
            source_hash
        ),
        "locked_output_file": str(
            OUTPUT_FILE
        ),
        "locked_output_sha256": (
            output_hash
        ),
        "topic_count": 16,
        "all_topics_retained": True,
        "interpretive_status": (
            "human-reviewed and locked"
        ),
        "topic_type_counts": (
            type_counts
        ),
        "confidence_counts": (
            confidence_counts
        ),
        "important_methodological_note": (
            "NMF is a soft-membership model. "
            "Final labels describe latent components; "
            "they are not mutually exclusive manual "
            "categories and do not imply that each "
            "comment belongs exclusively to one topic."
        ),
    }

    CONFIG_FILE.write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "YOUTUBE RETROFIT FINAL k=16 INTERPRETATION LOCK",
        "=" * 51,
        "",
        "Overall status: PASS",
        "",
        "Model interpretation",
        "--------------------",
        "Selected representation: inclusive primary TF-IDF",
        "Selected topic count: 16",
        "Topics human-reviewed: 16",
        "Topics retained: 16",
        "",
        "Topic-type structure",
        "--------------------",
        (
            "Technical subjects: "
            f"{type_counts.get('technical_subject', 0)}"
        ),
        (
            "Knowledge-sharing behaviours: "
            f"{type_counts.get('knowledge_sharing_behaviour', 0)}"
        ),
        (
            "Social-interaction topics: "
            f"{type_counts.get('social_interaction', 0)}"
        ),
        (
            "Mixed/contextual topics: "
            f"{type_counts.get('mixed', 0)}"
        ),
        "",
        "Confidence",
        "----------",
    ]

    for confidence in sorted(
    confidence_counts,
    key=int,
    ):
        report_lines.append(
            (
                f"Confidence {confidence}/5: "
                f"{confidence_counts[confidence]} topics"
            )
        )

    report_lines.extend(
        [
            "",
            "Locked topic labels",
            "-------------------",
        ]
    )

    for _, row in review.sort_values(
        "topic_number"
    ).iterrows():
        report_lines.append(
            (
                f"{int(row['topic_number']):02d}. "
                f"{row['researcher_final_label']} "
                f"[{row['researcher_topic_type']}; "
                f"confidence "
                f"{int(row['researcher_confidence_1_to_5'])}/5]"
            )
        )

    report_lines.extend(
        [
            "",
            "Interpretive caution",
            "--------------------",
            (
                "The 16 topics are latent NMF components, "
                "not mutually exclusive manual categories."
            ),
            (
                "Dominant-topic assignments are used for "
                "descriptive prevalence summaries only; "
                "comments can carry weight on multiple topics."
            ),
            (
                "Topics 10 and 16 remain lower-confidence "
                "cross-cutting discourse components."
            ),
            (
                "Topic 6 captures social endorsement, not "
                "evidence of technical correctness."
            ),
            (
                "Topics 12 and 15 were retained despite "
                "isolated unstable subsampling runs because "
                "their median stability, substantive coherence "
                "and preprocessing robustness were strong."
            ),
            "",
            "Hashes",
            "------",
            (
                "Source review SHA-256: "
                f"{source_hash}"
            ),
            (
                "Locked output SHA-256: "
                f"{output_hash}"
            ),
            "",
            "Created",
            "-------",
            str(OUTPUT_FILE),
            str(CONFIG_FILE),
        ]
    )

    REPORT_FILE.write_text(
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