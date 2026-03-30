# BIRD Dataset

BIRD (Big Bench for Large-scale Database Grounded Text-to-SQL Evaluation)

## Why BIRD for DyFlow-T tool comparison?

Unlike Spider (5 rows avg per table), BIRD has:
- **Real databases** with up to 33,000 rows
- **Noisy, domain-specific data** — finance, sports, medical, geography
- **Complex aggregations** — the LLM genuinely cannot hallucinate a correct COUNT or SUM over thousands of rows
- **Evidence hints** — extra domain knowledge needed per question

This makes the DyFlow vs DyFlow-T gap much clearer:
- DyFlow (no tool): must guess what aggregate values would be → unreliable
- DyFlow-T (SQL tool): executes query, gets real numbers → always grounded

## Setup

```bash
# Via HuggingFace (requires accepting terms + HF_TOKEN in .env)
python scripts/install_bird.py

# Via direct URL (no token needed, ~1.5 GB)
python scripts/install_bird.py --direct
```

## Run Comparison

```bash
# DyFlow vs DyFlow-T on BIRD dev, 50 questions
python scripts/compare_spider.py --dataset bird --mode dev --size 50 --workers 3

# BIRD-only evaluation
python scripts/run_spider.py --dataset bird --mode dev --size 50
```

## Directory Layout (after install)

```
benchmarks/data/BIRD/
├── databases/              # SQLite .db files (real data, up to 33k rows)
│   ├── financial/
│   │   └── financial.db
│   ├── california_schools/
│   │   └── california_schools.db
│   └── ...
├── bird_dev.json           # 1534 dev questions (simple/moderate/challenging)
└── bird_train.json         # 9428 train questions
```

## Dataset Format

| Field | Description |
|---|---|
| `db_id` | Database name |
| `question` | Natural language question |
| `query` | Gold SQL query |
| `hardness` | `simple` / `moderate` / `challenging` |
| `evidence` | Domain hint for the question (BIRD-specific) |

## Metric

**Execution Accuracy (EX)**: predicted SQL result set matches gold SQL result set
(order-insensitive comparison of returned rows).

Paper: https://arxiv.org/abs/2305.03111
Leaderboard: https://bird-bench.github.io/
