import sqlite3
import os
import logging
from typing import Generator, Iterable

from app_config import get_app_settings

DB_PATH = get_app_settings().db_path

SQL_LOGGER = logging.getLogger("sql")



def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    conn.execute("PRAGMA busy_timeout=30000")
    if os.getenv("SQL_TRACE", "") == "1":
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        SQL_LOGGER.setLevel(logging.INFO)
        conn.set_trace_callback(lambda stmt: SQL_LOGGER.info("SQL: %s", stmt))
    return conn


def get_db_dep() -> Generator[sqlite3.Connection, None, None]:
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            link TEXT,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            folder_path TEXT NOT NULL,
            screenshot_path TEXT,
            art_style TEXT,
            tags_json TEXT,
            full_project_copy INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_dir TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            type TEXT,
            project_id INTEGER,
            hash_main_blake3 TEXT,
            hash_main_sha256 TEXT,
            hash_full_blake3 TEXT,
            tags_json TEXT,
            meta_json TEXT,
            embedding_json TEXT,
            images_json TEXT,
            thumb_image TEXT,
            detail_image TEXT,
            full_image TEXT,
            anim_thumb TEXT,
            anim_detail TEXT,
            zip_path TEXT,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS asset_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id INTEGER,
            hash_main_blake3 TEXT,
            hash_full_blake3 TEXT,
            tags_original_json TEXT,
            tags_translated_json TEXT,
            translated_language TEXT,
            asset_created_at TEXT,
            tags_done_at TEXT,
            name_tags_done_at TEXT,
            name_translate_tags_done_at TEXT,
            translate_tags_done_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(asset_id) REFERENCES assets(id)
        )
        """
    )

    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_asset_tags_hash_full
        ON asset_tags(hash_full_blake3)
        WHERE hash_full_blake3 IS NOT NULL AND hash_full_blake3 != ''
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_asset_tags_hash_full_lookup ON asset_tags(hash_full_blake3)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_asset_tags_asset_id ON asset_tags(asset_id)")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_assets_project_id ON assets(project_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_assets_created_at ON assets(created_at)")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            target_id INTEGER,
            progress_json TEXT,
            message TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            cancel_flag INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS openai_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flow TEXT NOT NULL,
            provider TEXT,
            batch_id TEXT NOT NULL UNIQUE,
            task_id INTEGER,
            project_id INTEGER,
            request_total INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            output_file_id TEXT,
            error_text TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            processed_at TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS openai_batch_results_applied (
            batch_id TEXT PRIMARY KEY,
            flow TEXT,
            task_id INTEGER,
            rows_done INTEGER NOT NULL DEFAULT 0,
            rows_error INTEGER NOT NULL DEFAULT 0,
            applied_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS assets_fts
        USING fts5(name, description, tags)
        """
    )

    cur.execute(
        """
        CREATE TRIGGER IF NOT EXISTS assets_ai AFTER INSERT ON assets BEGIN
          INSERT INTO assets_fts(rowid, name, description, tags)
          VALUES (new.id, new.name, new.description, new.tags_json);
        END;
        """
    )

    cur.execute(
        """
        CREATE TRIGGER IF NOT EXISTS assets_au AFTER UPDATE ON assets BEGIN
          UPDATE assets_fts
          SET name = new.name,
              description = new.description,
              tags = new.tags_json
          WHERE rowid = new.id;
        END;
        """
    )

    cur.execute(
        """
        CREATE TRIGGER IF NOT EXISTS assets_ad AFTER DELETE ON assets BEGIN
          DELETE FROM assets_fts WHERE rowid = old.id;
        END;
        """
    )

    cur.execute(
        """
        INSERT INTO assets_fts(rowid, name, description, tags)
        SELECT id, name, description, tags_json
        FROM assets
        WHERE id NOT IN (SELECT rowid FROM assets_fts)
        """
    )

    ensure_column(conn, "projects", "screenshot_path", "TEXT")
    ensure_column(conn, "projects", "art_style", "TEXT")
    ensure_column(conn, "projects", "source_path", "TEXT")
    ensure_column(conn, "projects", "source_folder", "TEXT")
    ensure_column(conn, "projects", "source_size_bytes", "INTEGER")
    ensure_column(conn, "projects", "source_size_updated_at", "TEXT")
    ensure_column(conn, "projects", "export_include_json", "TEXT")
    ensure_column(conn, "projects", "export_exclude_json", "TEXT")
    ensure_column(conn, "projects", "full_project_copy", "INTEGER")
    ensure_column(conn, "projects", "reimported_once", "INTEGER")
    ensure_column(conn, "projects", "project_era", "TEXT")
    ensure_column(conn, "projects", "description", "TEXT")
    ensure_column(conn, "projects", "category_name", "TEXT")
    ensure_column(conn, "projects", "is_ai_generated", "INTEGER")
    ensure_column(conn, "assets", "hash_main_blake3", "TEXT")
    ensure_column(conn, "assets", "hash_main_sha256", "TEXT")
    ensure_column(conn, "assets", "hash_full_blake3", "TEXT")
    ensure_column(conn, "assets", "thumb_image", "TEXT")
    ensure_column(conn, "assets", "detail_image", "TEXT")
    ensure_column(conn, "assets", "full_image", "TEXT")
    ensure_column(conn, "assets", "anim_thumb", "TEXT")
    ensure_column(conn, "assets", "anim_detail", "TEXT")
    ensure_column(conn, "assets", "zip_path", "TEXT")
    ensure_column(conn, "asset_tags", "asset_id", "INTEGER")
    ensure_column(conn, "asset_tags", "hash_main_blake3", "TEXT")
    ensure_column(conn, "asset_tags", "hash_full_blake3", "TEXT")
    ensure_column(conn, "asset_tags", "tags_original_json", "TEXT")
    ensure_column(conn, "asset_tags", "tags_translated_json", "TEXT")
    ensure_column(conn, "asset_tags", "translated_language", "TEXT")
    ensure_column(conn, "asset_tags", "asset_created_at", "TEXT")
    ensure_column(conn, "asset_tags", "updated_at", "TEXT")
    ensure_column(conn, "asset_tags", "tags_done_at", "TEXT")
    ensure_column(conn, "asset_tags", "name_tags_done_at", "TEXT")
    ensure_column(conn, "asset_tags", "name_translate_tags_done_at", "TEXT")
    ensure_column(conn, "asset_tags", "translate_tags_done_at", "TEXT")
    ensure_column(conn, "openai_batches", "flow", "TEXT")
    ensure_column(conn, "openai_batches", "provider", "TEXT")
    ensure_column(conn, "openai_batches", "batch_id", "TEXT")
    ensure_column(conn, "openai_batches", "task_id", "INTEGER")
    ensure_column(conn, "openai_batches", "project_id", "INTEGER")
    ensure_column(conn, "openai_batches", "request_total", "INTEGER")
    ensure_column(conn, "openai_batches", "status", "TEXT")
    ensure_column(conn, "openai_batches", "output_file_id", "TEXT")
    ensure_column(conn, "openai_batches", "error_text", "TEXT")
    ensure_column(conn, "openai_batches", "updated_at", "TEXT")
    ensure_column(conn, "openai_batches", "processed_at", "TEXT")
    ensure_column(conn, "openai_batches", "processing_owner", "TEXT")
    ensure_column(conn, "openai_batches", "processing_started_at", "TEXT")
    ensure_column(conn, "openai_batches", "processing_heartbeat_at", "TEXT")
    ensure_column(conn, "openai_batch_results_applied", "batch_id", "TEXT")
    ensure_column(conn, "openai_batch_results_applied", "flow", "TEXT")
    ensure_column(conn, "openai_batch_results_applied", "task_id", "INTEGER")
    ensure_column(conn, "openai_batch_results_applied", "rows_done", "INTEGER")
    ensure_column(conn, "openai_batch_results_applied", "rows_error", "INTEGER")
    ensure_column(conn, "openai_batch_results_applied", "applied_at", "TEXT")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_openai_batches_pending ON openai_batches(processed_at, status, updated_at)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_openai_batches_task_pending ON openai_batches(task_id, processed_at)"
    )

    cur.execute(
        """
        DELETE FROM assets
        WHERE hash_full_blake3 IS NOT NULL
          AND hash_full_blake3 != ''
          AND id NOT IN (
            SELECT MIN(id)
            FROM assets
            WHERE hash_full_blake3 IS NOT NULL AND hash_full_blake3 != ''
            GROUP BY project_id, hash_full_blake3
          )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_assets_project_hash_full_blake3
        ON assets(project_id, hash_full_blake3)
        WHERE hash_full_blake3 IS NOT NULL AND hash_full_blake3 != ''
        """
    )

    conn.commit()
    conn.close()


def ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    columns = {row["name"] for row in rows}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def fetch_all(conn: sqlite3.Connection, query: str, params: Iterable = ()):
    cur = conn.execute(query, params)
    return [dict(row) for row in cur.fetchall()]


def fetch_one(conn: sqlite3.Connection, query: str, params: Iterable = ()):
    cur = conn.execute(query, params)
    row = cur.fetchone()
    return dict(row) if row else None
