from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths and expected corpus properties
# ---------------------------------------------------------------------

NMF_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    NMF_ROOT
    / "data"
    / "input"
    / "core_comments_substantive_ge3_words.csv"
)

OUTPUT_DIR = NMF_ROOT / "outputs" / "audit"

EXPECTED_ROWS = 42_487
EXPECTED_VIDEOS = 1_159

TEXT_COLUMN = "comment_text_for_coding"


def calculate_sha256(file_path: Path) -> str:
    """Calculate a reproducibility hash for the input file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def format_value(value: object) -> str:
    """Format audit values for the text report."""

    if isinstance(value, float):
        return f"{value:,.3f}"

    if isinstance(value, int):
        return f"{value:,}"

    return str(value)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "The substantive comment corpus was not found at:\n"
            f"{INPUT_FILE}"
        )

    print(f"Reading: {INPUT_FILE}")

    dataframe = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8",
        low_memory=False,
    )

    required_columns = {
        "video_id",
        "comment_id",
        TEXT_COLUMN,
        "comment_word_count",
        "retrofit_topic",
        "creator_type",
        "video_type",
        "primary_theme",
    }

    missing_columns = sorted(required_columns - set(dataframe.columns))

    if missing_columns:
        raise ValueError(
            "The input file is missing required columns:\n"
            + "\n".join(missing_columns)
        )

    text = dataframe[TEXT_COLUMN].fillna("").astype(str).str.strip()
    word_counts = text.str.split().str.len()
    character_counts = text.str.len()

    duplicate_comment_ids = int(
        dataframe["comment_id"].duplicated(keep=False).sum()
    )

    duplicate_text_rows = int(
        text.duplicated(keep=False).sum()
    )

    duplicate_within_video_rows = int(
        dataframe.assign(_audit_text=text)
        .duplicated(
            subset=["video_id", "_audit_text"],
            keep=False,
        )
        .sum()
    )

    summary = {
        "input_file": str(INPUT_FILE),
        "input_sha256": calculate_sha256(INPUT_FILE),
        "text_column_selected": TEXT_COLUMN,
        "column_count": int(dataframe.shape[1]),
        "row_count": int(len(dataframe)),
        "expected_row_count": EXPECTED_ROWS,
        "row_count_matches_expected": len(dataframe) == EXPECTED_ROWS,
        "unique_video_count": int(dataframe["video_id"].nunique()),
        "expected_video_count": EXPECTED_VIDEOS,
        "video_count_matches_expected": (
            dataframe["video_id"].nunique() == EXPECTED_VIDEOS
        ),
        "unique_comment_id_count": int(
            dataframe["comment_id"].nunique(dropna=True)
        ),
        "missing_comment_ids": int(
            dataframe["comment_id"].isna().sum()
        ),
        "duplicate_comment_id_rows": duplicate_comment_ids,
        "missing_selected_text": int(
            dataframe[TEXT_COLUMN].isna().sum()
        ),
        "blank_selected_text": int(text.eq("").sum()),
        "unique_selected_text_count": int(text.nunique()),
        "duplicate_text_rows_across_corpus": duplicate_text_rows,
        "duplicate_text_rows_within_video": (
            duplicate_within_video_rows
        ),
        "minimum_word_count": int(word_counts.min()),
        "median_word_count": float(word_counts.median()),
        "mean_word_count": float(word_counts.mean()),
        "maximum_word_count": int(word_counts.max()),
        "minimum_character_count": int(character_counts.min()),
        "median_character_count": float(
            character_counts.median()
        ),
        "mean_character_count": float(
            character_counts.mean()
        ),
        "maximum_character_count": int(
            character_counts.max()
        ),
    }

    summary_table = pd.DataFrame(
        {
            "measure": list(summary.keys()),
            "value": list(summary.values()),
        }
    )

    summary_path = OUTPUT_DIR / "01_corpus_audit_summary.csv"

    summary_table.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    percentiles = [
        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    length_table = pd.DataFrame(
        {
            "percentile": percentiles,
            "word_count": [
                word_counts.quantile(percentile)
                for percentile in percentiles
            ],
            "character_count": [
                character_counts.quantile(percentile)
                for percentile in percentiles
            ],
        }
    )

    length_path = (
        OUTPUT_DIR
        / "01_comment_length_percentiles.csv"
    )

    length_table.to_csv(
        length_path,
        index=False,
        encoding="utf-8-sig",
    )

    duplicate_mask = dataframe.assign(
        _audit_text=text
    ).duplicated(
        subset=["video_id", "_audit_text"],
        keep=False,
    )

    duplicate_examples = dataframe.loc[
        duplicate_mask,
        [
            "video_id",
            "comment_id",
            "retrofit_topic",
            "creator_type",
            "video_type",
            TEXT_COLUMN,
        ],
    ].copy()

    duplicate_examples = duplicate_examples.sort_values(
        by=["video_id", TEXT_COLUMN, "comment_id"],
        kind="stable",
    )

    duplicate_path = (
        OUTPUT_DIR
        / "01_duplicate_text_examples.csv"
    )

    duplicate_examples.head(200).to_csv(
        duplicate_path,
        index=False,
        encoding="utf-8-sig",
    )

    group_summary = (
        dataframe.groupby(
            [
                "retrofit_topic",
                "creator_type",
                "video_type",
            ],
            dropna=False,
        )
        .agg(
            comment_count=("comment_id", "size"),
            video_count=("video_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            by=["comment_count", "video_count"],
            ascending=False,
            kind="stable",
        )
    )

    group_path = (
        OUTPUT_DIR
        / "01_corpus_group_distribution.csv"
    )

    group_summary.to_csv(
        group_path,
        index=False,
        encoding="utf-8-sig",
    )

    status_checks = {
        "Expected row count": (
            len(dataframe) == EXPECTED_ROWS
        ),
        "Expected video count": (
            dataframe["video_id"].nunique()
            == EXPECTED_VIDEOS
        ),
        "No missing comment IDs": (
            dataframe["comment_id"].isna().sum() == 0
        ),
        "No duplicate comment IDs": (
            duplicate_comment_ids == 0
        ),
        "No missing modelling text": (
            dataframe[TEXT_COLUMN].isna().sum() == 0
        ),
        "No blank modelling text": (
            text.eq("").sum() == 0
        ),
        "Minimum three words": (
            word_counts.min() >= 3
        ),
    }

    overall_status = (
        "PASS"
        if all(status_checks.values())
        else "REVIEW REQUIRED"
    )

    report_lines = [
        "YOUTUBE RETROFIT NMF CORPUS AUDIT",
        "=" * 42,
        "",
        f"Overall status: {overall_status}",
        "",
        "Input",
        "-----",
        f"File: {INPUT_FILE}",
        f"SHA-256: {summary['input_sha256']}",
        f"Selected text column: {TEXT_COLUMN}",
        "",
        "Corpus dimensions",
        "-----------------",
        f"Rows: {len(dataframe):,}",
        (
            "Unique videos: "
            f"{dataframe['video_id'].nunique():,}"
        ),
        (
            "Unique comment IDs: "
            f"{dataframe['comment_id'].nunique():,}"
        ),
        "",
        "Text diagnostics",
        "----------------",
        (
            "Missing selected text: "
            f"{summary['missing_selected_text']:,}"
        ),
        (
            "Blank selected text: "
            f"{summary['blank_selected_text']:,}"
        ),
        (
            "Unique selected texts: "
            f"{summary['unique_selected_text_count']:,}"
        ),
        (
            "Rows involved in repeated text across corpus: "
            f"{duplicate_text_rows:,}"
        ),
        (
            "Rows involved in repeated text within the same video: "
            f"{duplicate_within_video_rows:,}"
        ),
        "",
        "Comment length",
        "--------------",
        (
            "Words — minimum / median / mean / maximum: "
            f"{word_counts.min():,.0f} / "
            f"{word_counts.median():,.1f} / "
            f"{word_counts.mean():,.1f} / "
            f"{word_counts.max():,.0f}"
        ),
        (
            "Characters — minimum / median / mean / maximum: "
            f"{character_counts.min():,.0f} / "
            f"{character_counts.median():,.1f} / "
            f"{character_counts.mean():,.1f} / "
            f"{character_counts.max():,.0f}"
        ),
        "",
        "Validation checks",
        "-----------------",
    ]

    for check_name, passed in status_checks.items():
        report_lines.append(
            f"{'PASS' if passed else 'FAIL'}: {check_name}"
        )

    report_lines.extend(
        [
            "",
            "Interpretive note",
            "-----------------",
            (
                "Repeated text is reported but is not automatically "
                "removed. A repeated comment may represent genuine "
                "repeated posting rather than extraction duplication. "
                "Unique comment IDs are therefore the primary integrity "
                "check."
            ),
        ]
    )

    report_path = OUTPUT_DIR / "01_corpus_audit_report.txt"

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("\n".join(report_lines))
    print()
    print("Audit outputs created:")
    print(f"  {summary_path}")
    print(f"  {length_path}")
    print(f"  {duplicate_path}")
    print(f"  {group_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()