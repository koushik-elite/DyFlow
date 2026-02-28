#!/usr/bin/env python3
"""
install_spider.py — Download Spider dataset and build SQLite databases.

What this does
──────────────
1. Downloads the Spider Text-to-SQL dataset (Yale NLP)
2. For each database in Spider, creates a SQLite .db file from the
   provided SQL schema + INSERT statements
3. Saves the dev/train split JSON in DyFlow-T's expected format
4. (Optional) Creates a sample offline database for quick testing
   without downloading the full dataset

Usage
─────
    # Full Spider download + build
    python scripts/install_spider.py

    # Build sample offline DB only (no download needed)
    python scripts/install_spider.py --sample-only

    # Re-build databases even if they exist
    python scripts/install_spider.py --force

Requirements
────────────
    pip install requests tqdm
"""

import sys
import os
import json
import sqlite3
import argparse
import zipfile
import shutil

REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = os.path.join(REPO_ROOT, "benchmarks", "data", "Spider")
DB_ROOT     = os.path.join(DATA_DIR, "databases")
SAMPLE_DIR  = os.path.join(DATA_DIR, "sample")

# Spider download URL (official Yale NLP Google Drive mirror via direct link)
SPIDER_URL  = "https://huggingface.co/datasets/xlangai/spider/resolve/main/data/spider.zip"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Sample database (offline, no download needed)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_SCHEMA_SQL = """
-- ============================================================
-- DyFlow-T Sample Database: e_commerce
-- A small SQLite schema for offline SQL benchmark testing
-- ============================================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    email         TEXT    UNIQUE NOT NULL,
    country       TEXT    NOT NULL,
    signup_date   TEXT    NOT NULL   -- ISO 8601 date
);

CREATE TABLE IF NOT EXISTS products (
    product_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    category      TEXT    NOT NULL,
    price         REAL    NOT NULL,
    stock         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS orders (
    order_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date    TEXT    NOT NULL,
    status        TEXT    NOT NULL CHECK(status IN ('pending','shipped','delivered','cancelled')),
    total_amount  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    rating        INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    comment       TEXT,
    review_date   TEXT    NOT NULL
);
"""

SAMPLE_DATA_SQL = """
INSERT INTO customers (name, email, country, signup_date) VALUES
    ('Alice Johnson',  'alice@example.com',  'USA',    '2022-01-15'),
    ('Bob Smith',      'bob@example.com',    'UK',     '2022-03-22'),
    ('Carlos Rivera',  'carlos@example.com', 'Mexico', '2022-06-10'),
    ('Diana Chen',     'diana@example.com',  'China',  '2023-01-05'),
    ('Eva Müller',     'eva@example.com',    'Germany','2023-04-18'),
    ('Frank Okafor',   'frank@example.com',  'Nigeria','2023-07-30'),
    ('Grace Kim',      'grace@example.com',  'Korea',  '2023-09-12'),
    ('Hiro Tanaka',    'hiro@example.com',   'Japan',  '2024-01-01');

INSERT INTO products (name, category, price, stock) VALUES
    ('Laptop Pro 15',       'Electronics',  1299.99, 45),
    ('Wireless Mouse',      'Electronics',    29.99, 200),
    ('Standing Desk',       'Furniture',     499.00, 30),
    ('USB-C Hub',           'Electronics',    49.99, 150),
    ('Ergonomic Chair',     'Furniture',     799.00, 20),
    ('Noise-Cancel Headphones','Electronics',249.99, 80),
    ('Python Cookbook',     'Books',          39.99, 500),
    ('Mechanical Keyboard', 'Electronics',   129.99, 60),
    ('Monitor 27 inch',     'Electronics',   399.99, 35),
    ('Office Lamp',         'Furniture',      59.99, 90);

INSERT INTO orders (customer_id, order_date, status, total_amount) VALUES
    (1, '2023-02-10', 'delivered', 1329.98),
    (1, '2023-11-05', 'delivered',  249.99),
    (2, '2023-03-15', 'delivered',  529.99),
    (3, '2023-05-20', 'shipped',    179.97),
    (4, '2023-08-01', 'delivered',  449.98),
    (5, '2023-09-14', 'cancelled',  799.00),
    (6, '2024-01-10', 'pending',    169.98),
    (7, '2024-02-22', 'delivered', 1699.97),
    (8, '2024-03-05', 'shipped',    399.99);

INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
    (1, 1, 1, 1299.99),
    (1, 2, 1,   29.99),
    (2, 6, 1,  249.99),
    (3, 3, 1,  499.00),
    (3, 2, 1,   29.99),
    (4, 7, 2,   39.99),
    (4, 4, 2,   49.99),
    (5, 8, 1,  129.99),
    (5, 10,1,   59.99),
    (6, 5, 1,  799.00),
    (7, 2, 1,   29.99),
    (7, 7, 1,   39.99),
    (7, 10,1,   59.99),
    (7, 4, 1,   49.99),  -- Note: missing in original, added for completeness
    (8, 1, 1, 1299.99),
    (8, 9, 1,  399.99),
    (9, 9, 1,  399.99);

INSERT INTO reviews (product_id, customer_id, rating, comment, review_date) VALUES
    (1, 1, 5, 'Excellent laptop, very fast!',           '2023-02-20'),
    (2, 1, 4, 'Good mouse, comfortable.',               '2023-02-21'),
    (6, 2, 5, 'Best headphones I have owned.',          '2023-03-25'),
    (3, 2, 3, 'Desk is ok, assembly was tricky.',       '2023-03-18'),
    (7, 3, 5, 'Great book, learned a lot.',             '2023-05-28'),
    (8, 4, 4, 'Solid keyboard, good feel.',             '2023-08-10'),
    (9, 7, 5, 'Beautiful monitor, crisp display.',      '2024-03-01'),
    (1, 7, 4, 'Great performance, a bit heavy.',        '2024-03-02'),
    (2, 6, 3, 'Does the job but nothing special.',      '2024-01-20'),
    (4, 3, 5, 'Very handy hub, charges fast.',          '2023-06-01');
"""

SAMPLE_QUESTIONS = [
    {
        "db_id":    "e_commerce",
        "question": "How many customers are from the USA?",
        "query":    "SELECT COUNT(*) FROM customers WHERE country = 'USA'",
        "hardness": "easy",
    },
    {
        "db_id":    "e_commerce",
        "question": "What are the names and prices of all Electronics products, ordered by price descending?",
        "query":    "SELECT name, price FROM products WHERE category = 'Electronics' ORDER BY price DESC",
        "hardness": "easy",
    },
    {
        "db_id":    "e_commerce",
        "question": "What is the total revenue from delivered orders?",
        "query":    "SELECT SUM(total_amount) FROM orders WHERE status = 'delivered'",
        "hardness": "medium",
    },
    {
        "db_id":    "e_commerce",
        "question": "Which customers have placed more than one order? Return their names and order counts.",
        "query":    "SELECT c.name, COUNT(o.order_id) AS order_count FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id HAVING COUNT(o.order_id) > 1",
        "hardness": "medium",
    },
    {
        "db_id":    "e_commerce",
        "question": "What is the average rating for each product category?",
        "query":    "SELECT p.category, AVG(r.rating) AS avg_rating FROM products p JOIN reviews r ON p.product_id = r.product_id GROUP BY p.category",
        "hardness": "medium",
    },
    {
        "db_id":    "e_commerce",
        "question": "List the top 3 best-selling products by total quantity sold.",
        "query":    "SELECT p.name, SUM(oi.quantity) AS total_sold FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.product_id ORDER BY total_sold DESC LIMIT 3",
        "hardness": "hard",
    },
    {
        "db_id":    "e_commerce",
        "question": "Find customers who have reviewed every product they ordered.",
        "query":    "SELECT DISTINCT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id WHERE NOT EXISTS (SELECT 1 FROM order_items oi2 JOIN orders o2 ON oi2.order_id = o2.order_id WHERE o2.customer_id = c.customer_id AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.product_id = oi2.product_id AND r.customer_id = c.customer_id))",
        "hardness": "extra",
    },
    {
        "db_id":    "e_commerce",
        "question": "Which country has the highest total spending across all orders?",
        "query":    "SELECT c.country, SUM(o.total_amount) AS total_spend FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.country ORDER BY total_spend DESC LIMIT 1",
        "hardness": "hard",
    },
    {
        "db_id":    "e_commerce",
        "question": "Show all products that have never been reviewed.",
        "query":    "SELECT name FROM products WHERE product_id NOT IN (SELECT product_id FROM reviews)",
        "hardness": "medium",
    },
    {
        "db_id":    "e_commerce",
        "question": "What is the month with the highest number of orders in 2023?",
        "query":    "SELECT strftime('%m', order_date) AS month, COUNT(*) AS cnt FROM orders WHERE order_date LIKE '2023%' GROUP BY month ORDER BY cnt DESC LIMIT 1",
        "hardness": "hard",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Build SQLite DB from SQL text
# ─────────────────────────────────────────────────────────────────────────────

def build_sqlite_db(db_path: str, schema_sql: str, data_sql: str = "") -> bool:
    """Create (or recreate) a SQLite database from schema + data SQL."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        conn.executescript(schema_sql)
        if data_sql:
            conn.executescript(data_sql)
        conn.commit()
        conn.close()
        return True
    except Exception as exc:
        print(f"  [ERROR] Failed to build {db_path}: {exc}")
        return False


def build_sample_db(force: bool = False) -> str:
    """Build the sample e_commerce SQLite DB and return its path."""
    db_path = os.path.join(SAMPLE_DIR, "e_commerce", "e_commerce.db")
    if os.path.exists(db_path) and not force:
        print(f"[Sample] DB already exists: {db_path}")
        return db_path

    print("[Sample] Building e_commerce sample database...")
    ok = build_sqlite_db(db_path, SAMPLE_SCHEMA_SQL, SAMPLE_DATA_SQL)
    if ok:
        print(f"  ✅  Created: {db_path}")
    return db_path


def save_sample_questions() -> str:
    """Save sample Q&A pairs to JSON."""
    path = os.path.join(SAMPLE_DIR, "spider_sample.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_QUESTIONS, f, indent=2, ensure_ascii=False)
    print(f"  ✅  Sample questions: {path}  ({len(SAMPLE_QUESTIONS)} items)")
    return path


def export_schema_sql() -> str:
    """Save the sample schema SQL to a file for reference."""
    path = os.path.join(SAMPLE_DIR, "e_commerce_schema.sql")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_SCHEMA_SQL.strip() + "\n")
    print(f"  ✅  Schema SQL: {path}")
    return path


def export_data_sql() -> str:
    """Save the sample data SQL to a file for reference."""
    path = os.path.join(SAMPLE_DIR, "e_commerce_data.sql")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_DATA_SQL.strip() + "\n")
    print(f"  ✅  Data SQL: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build Spider DBs from downloaded SQL files
# ─────────────────────────────────────────────────────────────────────────────

def build_spider_dbs(spider_dir: str, force: bool = False) -> int:
    """
    Build SQLite .db files from Spider's SQL schema files.

    Spider provides each database as:
      database/<db_id>/schema.sql    (CREATE TABLE statements)
      database/<db_id>/<db_id>.sql   (INSERT statements)

    We combine them into: databases/<db_id>/<db_id>.db
    """
    src_db_root = os.path.join(spider_dir, "database")
    if not os.path.isdir(src_db_root):
        print(f"[Spider] database/ folder not found in {spider_dir}")
        return 0

    db_names = [d for d in os.listdir(src_db_root) if os.path.isdir(os.path.join(src_db_root, d))]
    print(f"\n[Spider] Building {len(db_names)} SQLite databases...")

    built = 0
    failed = []
    for db_id in db_names:
        out_path = os.path.join(DB_ROOT, db_id, f"{db_id}.db")
        if os.path.exists(out_path) and not force:
            built += 1
            continue

        src_dir    = os.path.join(src_db_root, db_id)
        schema_sql = ""
        data_sql   = ""

        # Read schema
        schema_path = os.path.join(src_dir, "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8", errors="replace") as f:
                schema_sql = f.read()

        # Read data inserts (same name as db_id.sql)
        data_path = os.path.join(src_dir, f"{db_id}.sql")
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8", errors="replace") as f:
                data_sql = f.read()

        # Some Spider DBs already ship as .sqlite or .db files
        existing_db = None
        for ext in [".sqlite", ".db", ".sqlite3"]:
            candidate = os.path.join(src_dir, f"{db_id}{ext}")
            if os.path.exists(candidate):
                existing_db = candidate
                break

        if existing_db:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copy2(existing_db, out_path)
            built += 1
        elif schema_sql:
            ok = build_sqlite_db(out_path, schema_sql, data_sql)
            if ok:
                built += 1
            else:
                failed.append(db_id)
        else:
            failed.append(db_id)

    print(f"  ✅  Built {built}/{len(db_names)} databases")
    if failed:
        print(f"  ⚠️   Failed: {', '.join(failed[:10])}{'...' if len(failed)>10 else ''}")
    return built


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Download Spider
# ─────────────────────────────────────────────────────────────────────────────

from typing import Optional

def download_spider(force: bool = False) -> Optional[str]:
    """Download and extract the Spider dataset. Returns extraction dir."""
    try:
        import requests
        from tqdm import tqdm as _tqdm
    except ImportError:
        print("[ERROR] Missing: pip install requests tqdm")
        return None

    zip_path = os.path.join(DATA_DIR, "spider_raw.zip")
    extract_dir = os.path.join(DATA_DIR, "_spider_extracted")

    if os.path.exists(extract_dir) and not force:
        print(f"[Spider] Already extracted at {extract_dir}")
        return extract_dir

    print(f"\n[Spider] Downloading from {SPIDER_URL}")
    print("         (This may take a few minutes — ~100 MB)\n")

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        response = requests.get(SPIDER_URL, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(zip_path, "wb") as f, _tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading"
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        print("\n[Spider] Extracting...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        os.remove(zip_path)
        print(f"[Spider] Extracted to {extract_dir}")
        return extract_dir

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("\nManual download:")
        print("  1. Get spider.zip from https://yale-lily.github.io/spider")
        print(f"  2. Extract to: {extract_dir}")
        return None


def normalise_spider_record(row: dict) -> dict:
    """Map Spider JSON fields → DyFlow-T format."""
    return {
        "db_id":    row.get("db_id", ""),
        "question": row.get("question", ""),
        "query":    row.get("query", row.get("sql", "")),
        "hardness": row.get("difficulty", row.get("hardness", "unknown")),
    }


def copy_spider_json(extract_dir: str) -> int:
    """Copy and normalise Spider dev/train JSON to data/Spider/."""
    count = 0
    # Spider has dev.json and train_spider.json
    file_map = {
        "dev.json":          "spider_dev.json",
        "train_spider.json": "spider_train.json",
        "train_others.json": "spider_train_others.json",
    }
    for src_name, dst_name in file_map.items():
        src = os.path.join(extract_dir, "spider", src_name)
        if not os.path.exists(src):
            # Try one level deeper
            src = os.path.join(extract_dir, src_name)
        if not os.path.exists(src):
            continue
        with open(src, "r", encoding="utf-8") as f:
            records = json.load(f)
        normalised = [normalise_spider_record(r) for r in records]
        dst = os.path.join(DATA_DIR, dst_name)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(normalised, f, indent=2, ensure_ascii=False)
        print(f"  ✅  {dst_name}: {len(normalised)} records")
        count += len(normalised)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Verify
# ─────────────────────────────────────────────────────────────────────────────

def verify_sample():
    print("\n[Verify] Sample setup...")
    db_path = os.path.join(SAMPLE_DIR, "e_commerce", "e_commerce.db")
    json_path = os.path.join(SAMPLE_DIR, "spider_sample.json")
    all_ok = True

    # Verify DB
    if not os.path.exists(db_path):
        print(f"  [FAIL] DB not found: {db_path}")
        all_ok = False
    else:
        try:
            conn = sqlite3.connect(db_path)
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
            conn.close()
            print(f"  [OK]   DB: {db_path}  ({len(tables)} tables: {[t[0] for t in tables]})")
        except Exception as e:
            print(f"  [FAIL] DB error: {e}")
            all_ok = False

    # Verify JSON
    if not os.path.exists(json_path):
        print(f"  [FAIL] Questions not found: {json_path}")
        all_ok = False
    else:
        with open(json_path) as f:
            qs = json.load(f)
        hardness_counts = {}
        for q in qs:
            h = q.get("hardness", "unknown")
            hardness_counts[h] = hardness_counts.get(h, 0) + 1
        print(f"  [OK]   Questions: {json_path}  ({len(qs)} items)")
        for h in ["easy", "medium", "hard", "extra"]:
            if h in hardness_counts:
                print(f"           {h}: {hardness_counts[h]}")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("Spider Dataset Installer for DyFlow-T")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_ROOT,  exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # ── Always build the sample DB ────────────────────────────────────────────
    print("\n[1/3] Building sample e_commerce database...")
    build_sample_db(force=args.force)
    save_sample_questions()
    export_schema_sql()
    export_data_sql()

    if args.sample_only:
        verify_sample()
        print_summary(sample_only=True)
        return

    # ── Download Spider ───────────────────────────────────────────────────────
    print("\n[2/3] Downloading Spider dataset...")
    extract_dir = download_spider(force=args.force)

    if extract_dir is None:
        print("\n⚠️  Spider download failed — only sample DB is available.")
        print("   You can still run:  python scripts/run_spider.py --sample")
        verify_sample()
        return

    # ── Build Spider SQLite DBs ───────────────────────────────────────────────
    print("\n[3/3] Building Spider SQLite databases from SQL files...")
    copy_spider_json(extract_dir)
    build_spider_dbs(os.path.join(extract_dir, "spider"), force=args.force)

    verify_sample()
    print_summary(sample_only=False)


def print_summary(sample_only: bool):
    print("\n" + "=" * 60)
    if sample_only:
        print("✅  Sample database ready!")
        print(f"    DB:        {os.path.join(SAMPLE_DIR, 'e_commerce', 'e_commerce.db')}")
        print(f"    Questions: {os.path.join(SAMPLE_DIR, 'spider_sample.json')}")
        print(f"    Schema:    {os.path.join(SAMPLE_DIR, 'e_commerce_schema.sql')}")
        print(f"    Data:      {os.path.join(SAMPLE_DIR, 'e_commerce_data.sql')}")
        print("\nRun sample evaluation:")
        print("    python scripts/run_spider.py --sample")
    else:
        print("✅  Spider dataset ready!")
        print(f"    DBs:       {DB_ROOT}")
        print(f"    Dev JSON:  {os.path.join(DATA_DIR, 'spider_dev.json')}")
        print("\nRun evaluation:")
        print("    python scripts/run_spider.py --mode dev --size 20 --workers 5")
        print("    python scripts/run_spider.py --sample   # sample DB only")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Spider dataset and build SQLite databases for DyFlow-T"
    )
    parser.add_argument(
        "--sample-only", action="store_true",
        help="Only build the sample e_commerce DB (no Spider download)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download and rebuild even if files already exist",
    )
    args = parser.parse_args()
    main(args)
