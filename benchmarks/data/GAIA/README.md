# GAIA Dataset

Place the GAIA dataset files here:

```
benchmarks/data/GAIA/
├── GAIA_validation.json
└── GAIA_test.json
```

## Download

```bash
# Option 1: Hugging Face datasets library
pip install datasets
python - <<'EOF'
from datasets import load_dataset
import json

ds = load_dataset("gaia-benchmark/GAIA", "2023_all", trust_remote_code=True)

for split in ["validation", "test"]:
    records = []
    for row in ds[split]:
        records.append({
            "task_id":      row["task_id"],
            "question":     row["Question"],
            "final_answer": row.get("Final answer", ""),
            "level":        int(row["Level"]),
            "file_name":    row.get("file_name", ""),
        })
    with open(f"benchmarks/data/GAIA/GAIA_{split}.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} {split} records")
EOF
```

## Format

Each record must have:

| Field          | Type   | Description                        |
|----------------|--------|------------------------------------|
| `task_id`      | str    | Unique question ID                 |
| `question`     | str    | The question text                  |
| `final_answer` | str    | Ground truth answer                |
| `level`        | int    | Difficulty: 1 (easy) → 3 (hard)   |
| `file_name`    | str    | Optional file attachment           |
