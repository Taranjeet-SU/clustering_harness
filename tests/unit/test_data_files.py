"""File formats accepted by Dataset.from_file."""
import pandas as pd
import pytest

from clustering_harness import Dataset

FRAME = pd.DataFrame({"id": [10, 2], "text": ["login fails", "card declined"], "subject": ["Login", "Card"]})


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".jsonl", ".parquet", ".xlsx"])
def test_from_file_formats(tmp_path, suffix):
    p = tmp_path / f"tickets{suffix}"
    if suffix == ".csv":
        FRAME.to_csv(p, index=False)
    elif suffix == ".tsv":
        FRAME.to_csv(p, index=False, sep="\t")
    elif suffix == ".jsonl":
        FRAME.to_json(p, orient="records", lines=True)
    elif suffix == ".parquet":
        pytest.importorskip("pyarrow")
        FRAME.to_parquet(p)
    else:
        pytest.importorskip("openpyxl")
        FRAME.to_excel(p, index=False)
    ds = Dataset.from_file(p, id_col="id", text_col="text", title_col="subject")
    assert sorted(ds.ids) == ["10", "2"] and set(ds.texts) == {"login fails", "card declined"} and ds.name == "tickets"


def test_unsupported_suffix_names_the_options(tmp_path):
    with pytest.raises(ValueError, match="xlsx"):
        Dataset.from_file(tmp_path / "x.txt")
