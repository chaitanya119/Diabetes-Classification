"""Fetch and verify the documented Pima Indians diabetes CSV mirror."""
import csv
from pathlib import Path
from urllib.request import Request, urlopen

from model import DEFAULT_DATA, FEATURES

SOURCE_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
EXPECTED_ROWS = 768


def fetch_dataset(output_path: Path = DEFAULT_DATA) -> Path:
    """Download the 768-row source file and add the project's column header."""
    request = Request(SOURCE_URL, headers={"User-Agent": "diabetes-classification-practical/1.0"})
    with urlopen(request, timeout=30) as response:
        raw_text = response.read().decode("utf-8")
    records = list(csv.reader(raw_text.splitlines()))
    if len(records) != EXPECTED_ROWS or any(len(row) != 9 for row in records):
        raise ValueError(f"Dataset validation failed: expected {EXPECTED_ROWS} rows with 9 columns.")
    for row_number, record in enumerate(records, start=1):
        try:
            [float(value) for value in record]
        except ValueError as error:
            raise ValueError(f"Invalid number on source row {row_number}.") from error
        if record[-1] not in {"0", "1"}:
            raise ValueError(f"Unexpected outcome label on source row {row_number}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([*FEATURES, "Outcome"])
        writer.writerows(records)
    return output_path


if __name__ == "__main__":
    print(f"Saved {EXPECTED_ROWS} verified records to {fetch_dataset()}")
