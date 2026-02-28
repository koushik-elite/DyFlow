#!/usr/bin/env python3
"""
install_bird.py — Download BIRD benchmark and prepare for DyFlow-T evaluation.

BIRD (Big Bench for Large-scale Database Grounded Text-to-SQL Evaluation)
  - Real-world databases with 33k+ rows, dirty data, ambiguous questions
  - Difficulty: simple / moderate / challenging
  - Key difference from Spider: LLM cannot guess answers — data is too large
    and complex. The SQL_QUERY execution tool is essential for correct answers.

What this does
──────────────
1. Downloads BIRD dev split from HuggingFace
2. Normalises JSON to DyFlow-T format: db_id, question, query, evidence, hardness
3. Copies .sqlite → .db for consistency with Spider pipeline
4. Builds a sample finance DB with exchange_rates table (impossible to guess)

Usage
─────
    python scripts/install_bird.py              # full download
    python scripts/install_bird.py --sample-only  # instant offline mode
    python scripts/install_bird.py --force        # re-download

Requirements: pip install requests tqdm
"""

import sys, os, json, sqlite3, argparse, zipfile, shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import requests
    from tqdm import tqdm as _tqdm
except ImportError:
    print("[ERROR] pip install requests tqdm"); sys.exit(1)

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(REPO_ROOT, "benchmarks", "data", "BIRD")
DB_ROOT    = os.path.join(DATA_DIR, "databases")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")

BIRD_DEV_URL = "https://huggingface.co/datasets/birdsql/bird/resolve/main/dev.zip"

# ─────────────────────────────────────────────────────────────────────────────
# Sample finance database — chosen so LLM CANNOT guess aggregate answers
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country     TEXT NOT NULL,
    segment     TEXT NOT NULL,   -- Basic / Standard / Premium
    joined_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
    product_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    unit_price  REAL NOT NULL,
    stock       INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id   INTEGER PRIMARY KEY,
    customer_id      INTEGER NOT NULL,
    product_id       INTEGER NOT NULL,
    amount           REAL    NOT NULL,
    currency         TEXT    NOT NULL,
    transaction_date TEXT    NOT NULL,
    status           TEXT    NOT NULL,  -- completed / failed / refunded
    region           TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS exchange_rates (
    rate_date     TEXT NOT NULL,
    from_currency TEXT NOT NULL,
    to_currency   TEXT NOT NULL,
    rate          REAL NOT NULL,
    PRIMARY KEY (rate_date, from_currency, to_currency)
);
"""

SAMPLE_DATA = """
INSERT OR IGNORE INTO customers VALUES
(1,'Alice Chen','US','Premium','2021-03-15'),(2,'Bob Kumar','IN','Standard','2022-07-20'),
(3,'Carlos Ruiz','MX','Premium','2020-11-05'),(4,'Diana Park','KR','Standard','2023-01-10'),
(5,'Eve Müller','DE','Premium','2021-09-22'),(6,'Frank Li','CN','Standard','2022-04-18'),
(7,'Grace Okafor','NG','Basic','2023-06-01'),(8,'Hiro Tanaka','JP','Premium','2020-12-30'),
(9,'Irina Petrov','RU','Standard','2022-02-14'),(10,'James Smith','US','Basic','2023-08-25');

INSERT OR IGNORE INTO products VALUES
(1,'Laptop Pro 15','Electronics',1299.99,45),(2,'Wireless Mouse','Electronics',29.99,210),
(3,'USB-C Hub','Electronics',49.99,180),(4,'Ergonomic Chair','Furniture',399.00,30),
(5,'Standing Desk','Furniture',649.00,15),(6,'Monitor 27"','Electronics',449.99,60),
(7,'Mechanical Keyboard','Electronics',129.99,95),(8,'Webcam HD','Electronics',79.99,120),
(9,'Desk Lamp','Furniture',39.99,200),(10,'Notebook Set','Stationery',12.99,500);

INSERT OR IGNORE INTO transactions VALUES
(1,1,1,1299.99,'USD','2024-01-05','completed','North America'),
(2,2,2,29.99,'USD','2024-01-07','completed','Asia'),
(3,3,3,49.99,'USD','2024-01-09','completed','Latin America'),
(4,4,6,449.99,'USD','2024-01-12','completed','Asia'),
(5,5,4,399.00,'EUR','2024-01-15','completed','Europe'),
(6,6,7,129.99,'USD','2024-01-18','completed','Asia'),
(7,7,9,39.99,'USD','2024-01-20','failed','Africa'),
(8,8,1,1299.99,'JPY','2024-01-22','completed','Asia'),
(9,9,5,649.00,'USD','2024-01-25','refunded','Europe'),
(10,10,2,29.99,'USD','2024-01-28','completed','North America'),
(11,1,6,449.99,'USD','2024-02-01','completed','North America'),
(12,2,7,129.99,'USD','2024-02-03','completed','Asia'),
(13,3,8,79.99,'USD','2024-02-07','completed','Latin America'),
(14,4,3,49.99,'USD','2024-02-10','completed','Asia'),
(15,5,6,449.99,'EUR','2024-02-14','completed','Europe'),
(16,6,1,1299.99,'USD','2024-02-16','failed','Asia'),
(17,7,2,29.99,'USD','2024-02-18','completed','Africa'),
(18,8,4,399.00,'JPY','2024-02-20','completed','Asia'),
(19,9,7,129.99,'USD','2024-02-22','completed','Europe'),
(20,10,10,12.99,'USD','2024-02-25','completed','North America'),
(21,1,3,49.99,'USD','2024-03-01','completed','North America'),
(22,2,5,649.00,'USD','2024-03-05','completed','Asia'),
(23,3,1,1299.99,'USD','2024-03-08','refunded','Latin America'),
(24,4,8,79.99,'USD','2024-03-12','completed','Asia'),
(25,5,9,39.99,'EUR','2024-03-15','completed','Europe');

INSERT OR IGNORE INTO exchange_rates VALUES
('2024-01-01','EUR','USD',1.085),('2024-01-01','JPY','USD',0.0067),
('2024-01-01','USD','EUR',0.922),('2024-02-01','EUR','USD',1.091),
('2024-02-01','JPY','USD',0.0066),('2024-03-01','EUR','USD',1.088),
('2024-03-01','JPY','USD',0.0065);
"""

# Questions where LLM CANNOT guess the answer — must execute SQL to get real value
SAMPLE_QUESTIONS = [
    {
        "db_id":    "finance",
        "question": "How many transactions were completed in January 2024?",
        "query":    "SELECT COUNT(*) FROM transactions WHERE status='completed' AND transaction_date LIKE '2024-01%';",
        "evidence": "Only count rows where status is exactly 'completed'",
        "hardness": "simple",
    },
    {
        "db_id":    "finance",
        "question": "What is the total revenue in USD from Electronics products for completed transactions?",
        "query":    "SELECT ROUND(SUM(t.amount),2) FROM transactions t JOIN products p ON t.product_id=p.product_id WHERE p.category='Electronics' AND t.currency='USD' AND t.status='completed';",
        "evidence": "Only USD transactions, only completed status",
        "hardness": "moderate",
    },
    {
        "db_id":    "finance",
        "question": "Which customer segment has the most failed or refunded transactions?",
        "query":    "SELECT c.segment, COUNT(*) as cnt FROM transactions t JOIN customers c ON t.customer_id=c.customer_id WHERE t.status IN ('failed','refunded') GROUP BY c.segment ORDER BY cnt DESC LIMIT 1;",
        "evidence": "failed and refunded are both non-successful outcomes",
        "hardness": "moderate",
    },
    {
        "db_id":    "finance",
        "question": "What is the EUR to USD exchange rate on 2024-02-01?",
        "query":    "SELECT rate FROM exchange_rates WHERE rate_date='2024-02-01' AND from_currency='EUR' AND to_currency='USD';",
        "evidence": "Look up the exact rate stored in the exchange_rates table",
        "hardness": "simple",
    },
    {
        "db_id":    "finance",
        "question": "How many Premium customers made at least one completed transaction in the Asia region?",
        "query":    "SELECT COUNT(DISTINCT t.customer_id) FROM transactions t JOIN customers c ON t.customer_id=c.customer_id WHERE c.segment='Premium' AND t.region='Asia' AND t.status='completed';",
        "evidence": "Count each customer once even if they made multiple transactions",
        "hardness": "challenging",
    },
    {
        "db_id":    "finance",
        "question": "Which products have never appeared in a failed transaction?",
        "query":    "SELECT name FROM products WHERE product_id NOT IN (SELECT DISTINCT product_id FROM transactions WHERE status='failed');",
        "evidence": "Products with zero failed transactions in the entire history",
        "hardness": "challenging",
    },
    {
        "db_id":    "finance",
        "question": "What is the average transaction amount per region for February 2024 completed sales?",
        "query":    "SELECT region, ROUND(AVG(amount),2) as avg_amount FROM transactions WHERE transaction_date LIKE '2024-02%' AND status='completed' GROUP BY region ORDER BY avg_amount DESC;",
        "evidence": "Only completed transactions in February 2024, round to 2 decimal places",
        "hardness": "moderate",
    },
    {
        "db_id":    "finance",
        "question": "List the top 3 customers by total completed transaction amount.",
        "query":    "SELECT c.name, ROUND(SUM(t.amount),2) as total FROM transactions t JOIN customers c ON t.customer_id=c.customer_id WHERE t.status='completed' GROUP BY t.customer_id ORDER BY total DESC LIMIT 3;",
        "evidence": "Sum all completed transactions per customer",
        "hardness": "moderate",
    },
]


def build_sample_db(force: bool = False) -> None:
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    db_dir  = os.path.join(SAMPLE_DIR, "finance")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "finance.db")

    if os.path.exists(db_path) and not force:
        print(f"[sample] Already exists: {db_path}")
    else:
        print("[sample] Building finance database...")
        conn = sqlite3.connect(db_path)
        conn.executescript(SAMPLE_SCHEMA)
        conn.executescript(SAMPLE_DATA)
        conn.commit()
        conn.close()
        # Verify row counts
        conn = sqlite3.connect(db_path)
        for tbl in ("customers","products","transactions","exchange_rates"):
            n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"  {tbl}: {n} rows")
        conn.close()
        print(f"[sample] Created: {db_path}")

    q_path = os.path.join(SAMPLE_DIR, "bird_sample.json")
    with open(q_path, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_QUESTIONS, f, indent=2, ensure_ascii=False)
    print(f"[sample] Questions: {q_path}  ({len(SAMPLE_QUESTIONS)} questions)")


def download_file(url: str, dest: str) -> None:
    print(f"[download] {url}")
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, _tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(65536):
            f.write(chunk); bar.update(len(chunk))


def extract_zip(zip_path: str, dest: str) -> str:
    print(f"[extract] → {dest}")
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    dirs = [os.path.join(dest,d) for d in os.listdir(dest) if os.path.isdir(os.path.join(dest,d))]
    return dirs[0] if dirs else dest


def normalise_bird_json(src: str, dst: str) -> int:
    """BIRD 'SQL' + 'difficulty' → DyFlow-T 'query' + 'hardness'."""
    with open(src, encoding="utf-8") as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else raw.get("data", list(raw.values())[0])
    out = [
        {
            "db_id":    r.get("db_id",""),
            "question": r.get("question",""),
            "query":    r.get("SQL", r.get("query","")),
            "evidence": r.get("evidence",""),
            "hardness": r.get("difficulty", r.get("hardness","unknown")),
        }
        for r in items
    ]
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return len(out)


def copy_databases(src_dir: str, dest_dir: str, force: bool = False) -> int:
    """Copy BIRD .sqlite → .db, preserving db_id/db_id.db layout."""
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if fname.endswith((".sqlite", ".db")):
                db_id     = os.path.splitext(fname)[0]
                out_dir   = os.path.join(dest_dir, db_id)
                os.makedirs(out_dir, exist_ok=True)
                dest_path = os.path.join(out_dir, f"{db_id}.db")
                if not os.path.exists(dest_path) or force:
                    shutil.copy2(os.path.join(root, fname), dest_path)
                    copied += 1
    return copied


def main(args) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_ROOT, exist_ok=True)

    print("=" * 60)
    print("BIRD Dataset Installer")
    print("=" * 60)

    print("\n[1/3] Building sample finance database...")
    build_sample_db(force=args.force)

    if args.sample_only:
        print("\n✅  Sample ready (offline mode).")
        print("    python scripts/compare_bird.py --sample")
        return

    print("\n[2/3] Downloading BIRD dev split...")
    tmp = os.path.join(DATA_DIR, "_extracted")
    os.makedirs(tmp, exist_ok=True)
    dev_zip = os.path.join(tmp, "bird_dev.zip")
    if not os.path.exists(dev_zip) or args.force:
        try:
            download_file(BIRD_DEV_URL, dev_zip)
        except Exception as e:
            print(f"[ERROR] {e}")
            print("  Manual: https://huggingface.co/datasets/birdsql/bird")
            sys.exit(1)

    dev_dir = extract_zip(dev_zip, os.path.join(tmp, "dev"))

    # Find the dev JSON file
    for root, _, files in os.walk(dev_dir):
        for f in files:
            if f.endswith(".json") and "dev" in f.lower():
                src  = os.path.join(root, f)
                dst  = os.path.join(DATA_DIR, "bird_dev.json")
                n    = normalise_bird_json(src, dst)
                print(f"[normalise] {n} questions → {dst}")
                break

    print("\n[3/3] Copying databases...")
    n = copy_databases(dev_dir, DB_ROOT, force=args.force)
    print(f"[databases] {n} databases → {DB_ROOT}")

    print("\n" + "=" * 60)
    print("✅  BIRD ready!")
    print(f"    Dev JSON:  {os.path.join(DATA_DIR, 'bird_dev.json')}")
    print("\nRun comparison:")
    print("    python scripts/compare_bird.py --mode dev --size 50 --workers 3")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample-only", action="store_true")
    p.add_argument("--force", action="store_true")
    main(p.parse_args())
