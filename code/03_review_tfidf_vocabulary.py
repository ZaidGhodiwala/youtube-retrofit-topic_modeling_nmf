from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


NMF_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    NMF_ROOT
    / "outputs"
    / "tables"
    / "02_tfidf_feature_inventory.csv"
)

ZERO_VECTOR_FILE = (
    NMF_ROOT
    / "outputs"
    / "audit"
    / "02_zero_vector_comments.csv"
)

AUDIT_DIR = NMF_ROOT / "outputs" / "audit"
TABLE_DIR = NMF_ROOT / "outputs" / "tables"


# These terms are not automatically removed. They are reviewed because
# they may represent platform language rather than retrofit knowledge.
GENERIC_PLATFORM_TERMS = {
    "video",
    "videos",
    "youtube",
    "channel",
    "subscribe",
    "subscribed",
    "subscriber",
    "subscribers",
    "watch",
    "watching",
    "watched",
    "content",
    "comment",
    "comments",
    "thanks",
    "thank",
    "great",
    "good",
    "nice",
    "awesome",
    "excellent",
    "amazing",
    "love",
    "liked",
}


# These anchors provide a basic check that technically meaningful
# vocabulary survived the preprocessing and document-frequency filters.
TECHNICAL_ANCHORS = {
    "air source",
    "air tightness",
    "airflow",
    "airtightness",
    "battery",
    "boiler",
    "cavity wall",
    "cold bridge",
    "condensation",
    "damp",
    "draught",
    "external wall",
    "flow temperature",
    "heat loss",
    "heat pump",
    "heating system",
    "insulation",
    "internal wall",
    "mechanical ventilation",
    "moisture",
    "radiator",
    "retrofit",
    "roof insulation",
    "solar",
    "solar panels",
    "thermal bridge",
    "underfloor heating",
    "u-value",
    "vapour barrier",
    "ventilation",
    "wall insulation",
    "window",
}


def format_feature_rows(
    dataframe: pd.DataFrame,
    limit: int = 40,
) -> list[str]:
    """Format a feature table for the audit report."""

    lines: list[str] = []

    selected = dataframe.head(limit)

    for position, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):
        lines.append(
            f"{position:>2}. "
            f"{row['feature']:<35} "
            f"DF={int(row['document_frequency']):>6,} "
            f"({row['document_frequency_percent']:>6.2f}%)"
        )

    return lines


def classify_artifact(feature: str) -> str | None:
    """Flag features that may reflect processing artefacts."""

    feature_lower = feature.lower()

    html_residues = {
        "amp",
        "quot",
        "nbsp",
        "apos",
        "ldquo",
        "rdquo",
        "rsquo",
    }

    url_residues = {
        "http",
        "https",
        "www",
        "com",
        "youtu",
        "youtube com",
    }

    feature_tokens = set(feature_lower.split())

    if feature_tokens & html_residues:
        return "Possible HTML-entity residue"

    if feature_tokens & url_residues:
        return "Possible URL or web-address residue"

    if re.search(r"(.)\1{4,}", feature_lower):
        return "Five or more repeated characters"

    if any(len(token) > 30 for token in feature_lower.split()):
        return "Unusually long token"

    if re.search(
        r"\b(?:[a-z]{2,}\d{4,}|\d{4,}[a-z]{2,})\b",
        feature_lower,
    ):
        return "Long alphanumeric token"

    return None


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            "TF-IDF feature inventory not found:\n"
            f"{FEATURE_FILE}"
        )

    features = pd.read_csv(
        FEATURE_FILE,
        encoding="utf-8-sig",
    )

    required_columns = {
        "feature",
        "ngram_size",
        "document_frequency",
        "document_frequency_percent",
        "idf",
        "mean_tfidf_all_comments",
        "mean_tfidf_when_present",
    }

    missing_columns = sorted(
        required_columns - set(features.columns)
    )

    if missing_columns:
        raise ValueError(
            "Feature inventory is missing required columns:\n"
            + "\n".join(missing_columns)
        )

    features["feature"] = (
        features["feature"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    unigrams = (
        features.loc[features["ngram_size"] == 1]
        .sort_values(
            by=[
                "document_frequency",
                "mean_tfidf_all_comments",
                "feature",
            ],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    bigrams = (
        features.loc[features["ngram_size"] == 2]
        .sort_values(
            by=[
                "document_frequency",
                "mean_tfidf_all_comments",
                "feature",
            ],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    generic_features = features.loc[
        features["feature"]
        .str.lower()
        .isin(GENERIC_PLATFORM_TERMS)
    ].copy()

    generic_features = generic_features.sort_values(
        by=[
            "document_frequency",
            "mean_tfidf_all_comments",
        ],
        ascending=False,
        kind="stable",
    )

    technical_features = features.loc[
        features["feature"]
        .str.lower()
        .isin(TECHNICAL_ANCHORS)
    ].copy()

    technical_features = technical_features.sort_values(
        by=[
            "document_frequency",
            "mean_tfidf_all_comments",
        ],
        ascending=False,
        kind="stable",
    )

    technical_anchor_status = pd.DataFrame(
        {
            "technical_anchor": sorted(TECHNICAL_ANCHORS)
        }
    )

    feature_lookup = (
        features.set_index(
            features["feature"].str.lower()
        )
        .to_dict(orient="index")
    )

    technical_anchor_status["present_in_vocabulary"] = (
        technical_anchor_status["technical_anchor"]
        .map(lambda term: term in feature_lookup)
    )

    technical_anchor_status["document_frequency"] = (
        technical_anchor_status["technical_anchor"]
        .map(
            lambda term: (
                feature_lookup[term]["document_frequency"]
                if term in feature_lookup
                else pd.NA
            )
        )
    )

    technical_anchor_status[
        "document_frequency_percent"
    ] = (
        technical_anchor_status["technical_anchor"]
        .map(
            lambda term: (
                feature_lookup[term][
                    "document_frequency_percent"
                ]
                if term in feature_lookup
                else pd.NA
            )
        )
    )

    artifact_rows: list[dict[str, object]] = []

    for _, row in features.iterrows():
        reason = classify_artifact(row["feature"])

        if reason is not None:
            artifact_rows.append(
                {
                    "feature": row["feature"],
                    "ngram_size": row["ngram_size"],
                    "document_frequency": (
                        row["document_frequency"]
                    ),
                    "document_frequency_percent": (
                        row["document_frequency_percent"]
                    ),
                    "reason": reason,
                }
            )

    artifact_features = pd.DataFrame(
        artifact_rows,
        columns=[
            "feature",
            "ngram_size",
            "document_frequency",
            "document_frequency_percent",
            "reason",
        ],
    )

    if not artifact_features.empty:
        artifact_features = artifact_features.sort_values(
            by=[
                "document_frequency",
                "feature",
            ],
            ascending=[False, True],
            kind="stable",
        )

    generic_path = (
        TABLE_DIR
        / "03_candidate_generic_platform_features.csv"
    )

    anchor_path = (
        TABLE_DIR
        / "03_technical_anchor_review.csv"
    )

    artifact_path = (
        TABLE_DIR
        / "03_suspected_artifact_features.csv"
    )

    top_unigrams_path = (
        TABLE_DIR
        / "03_top_100_unigrams_by_document_frequency.csv"
    )

    top_bigrams_path = (
        TABLE_DIR
        / "03_top_100_bigrams_by_document_frequency.csv"
    )

    generic_features.to_csv(
        generic_path,
        index=False,
        encoding="utf-8-sig",
    )

    technical_anchor_status.to_csv(
        anchor_path,
        index=False,
        encoding="utf-8-sig",
    )

    artifact_features.to_csv(
        artifact_path,
        index=False,
        encoding="utf-8-sig",
    )

    unigrams.head(100).to_csv(
        top_unigrams_path,
        index=False,
        encoding="utf-8-sig",
    )

    bigrams.head(100).to_csv(
        top_bigrams_path,
        index=False,
        encoding="utf-8-sig",
    )

    if ZERO_VECTOR_FILE.exists():
        zero_vectors = pd.read_csv(
            ZERO_VECTOR_FILE,
            encoding="utf-8-sig",
            low_memory=False,
        )
        zero_vector_count = len(zero_vectors)
    else:
        zero_vector_count = 0

    technical_present = int(
        technical_anchor_status[
            "present_in_vocabulary"
        ].sum()
    )

    technical_total = len(technical_anchor_status)

    report_lines = [
        "YOUTUBE RETROFIT TF-IDF VOCABULARY REVIEW",
        "=" * 44,
        "",
        "Vocabulary dimensions",
        "---------------------",
        f"Total features: {len(features):,}",
        f"Unigrams: {len(unigrams):,}",
        f"Bigrams: {len(bigrams):,}",
        f"Zero-vector comments: {zero_vector_count:,}",
        "",
        "Technical-anchor check",
        "----------------------",
        (
            "Technical anchors present: "
            f"{technical_present:,} of {technical_total:,}"
        ),
        "",
        "Top 40 unigrams by document frequency",
        "-------------------------------------",
        *format_feature_rows(unigrams, limit=40),
        "",
        "Top 40 bigrams by document frequency",
        "------------------------------------",
        *format_feature_rows(bigrams, limit=40),
        "",
        "Candidate generic platform terms",
        "--------------------------------",
    ]

    if generic_features.empty:
        report_lines.append(
            "No listed generic platform terms were found."
        )
    else:
        report_lines.extend(
            format_feature_rows(
                generic_features,
                limit=len(generic_features),
            )
        )

    report_lines.extend(
        [
            "",
            "Suspected processing artefacts",
            "------------------------------",
            (
                f"Flagged features: "
                f"{len(artifact_features):,}"
            ),
        ]
    )

    if artifact_features.empty:
        report_lines.append(
            "No obvious processing artefacts were detected."
        )
    else:
        for position, (_, row) in enumerate(
            artifact_features.head(30).iterrows(),
            start=1,
        ):
            report_lines.append(
                f"{position:>2}. "
                f"{row['feature']:<35} "
                f"DF={int(row['document_frequency']):>6,} "
                f"Reason={row['reason']}"
            )

    report_lines.extend(
        [
            "",
            "Decision rule",
            "-------------",
            (
                "No feature is removed automatically at this "
                "stage. Generic or malformed-looking terms must "
                "be reviewed in context before the final "
                "vectorizer configuration is frozen."
            ),
        ]
    )

    report_path = (
        AUDIT_DIR
        / "03_tfidf_vocabulary_review.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    print()
    print("Created:")
    print(f"  {report_path}")
    print(f"  {generic_path}")
    print(f"  {anchor_path}")
    print(f"  {artifact_path}")
    print(f"  {top_unigrams_path}")
    print(f"  {top_bigrams_path}")


if __name__ == "__main__":
    main()