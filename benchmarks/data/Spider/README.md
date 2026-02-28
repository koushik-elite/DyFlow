# Spider / BIRD Dataset

## Quick Start (sample DB — no download needed)

```bash
# Build the sample e_commerce SQLite database + 10 test questions
python scripts/install_spider.py --sample-only

# Run evaluation on sample
python scripts/run_spider.py --sample
```

## Full Spider Dataset

```bash
# Download Spider + build all SQLite databases (~100 MB)
python scripts/install_spider.py

# Run on dev split (20 questions, 5 workers)
python scripts/run_spider.py --mode dev --size 20 --workers 5

# Full dev evaluation
python scripts/run_spider.py --mode dev
```

## Directory Layout (after install)

```
benchmarks/data/Spider/
├── databases/                   # SQLite .db files (one per database)
│   ├── world_1/
│   │   └── world_1.db
│   ├── concert_singer/
│   │   └── concert_singer.db
│   └── ...
├── sample/                      # Sample e_commerce DB (offline testing)
│   ├── e_commerce/
│   │   └── e_commerce.db
│   ├── spider_sample.json       # 10 NL→SQL questions
│   ├── e_commerce_schema.sql    # CREATE TABLE statements
│   └── e_commerce_data.sql      # INSERT statements
├── spider_dev.json              # 1034 dev questions
└── spider_train.json            # 7000 train questions
```

## Sample Database Schema

The `e_commerce` sample database includes:

| Table         | Columns                                              |
|---------------|------------------------------------------------------|
| `customers`   | customer_id, name, email, country, signup_date       |
| `products`    | product_id, name, category, price, stock             |
| `orders`      | order_id, customer_id, order_date, status, total_amount |
| `order_items` | item_id, order_id, product_id, quantity, unit_price  |
| `reviews`     | review_id, product_id, customer_id, rating, comment, review_date |

## Metric

**Execution Accuracy (EX)**: predicted SQL is executed against the database;
result set is compared to gold SQL result set (order-insensitive).
