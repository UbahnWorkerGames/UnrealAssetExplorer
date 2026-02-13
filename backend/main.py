from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
import queue
import shutil
import sqlite3
import subprocess
import shlex
import tempfile
import threading
import time
import webbrowser
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

import httpx
import logging
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

from app_config import get_app_settings
from db import DB_PATH, fetch_all, fetch_one, get_db, get_db_dep, init_db
from services.asset_processing import process_asset_zip
from services.embeddings import cosine_similarity, embed_text, embed_texts
from services.llm_tags import (
    DEFAULT_TEMPLATE,
    TRANSLATE_TEMPLATE,
    _extract_tags_from_content,
    _extract_tags_and_era,
    generate_tags,
    generate_tags_debug,
    render_template,
    translate_tags,
    translate_tags_debug,
)

APP_SETTINGS = get_app_settings()
BASE_DIR = APP_SETTINGS.base_dir
DATA_DIR = APP_SETTINGS.data_dir
ASSETS_DIR = APP_SETTINGS.assets_dir
PROJECTS_DIR = APP_SETTINGS.projects_dir
UPLOADS_DIR = APP_SETTINGS.uploads_dir
BATCH_OUTPUT_DIR = APP_SETTINGS.batch_output_dir
STARTUP_JOBS_DIR = APP_SETTINGS.startup_jobs_dir

app = FastAPI(title="Asset Explorer API")
logger = logging.getLogger("uvicorn.error")
SQL_LOGGER = logging.getLogger("sql")
SQL_LOGGER.setLevel(logging.INFO)

COPY_PROGRESS: Dict[int, Dict[str, Any]] = {}
COPY_LOCK = threading.Lock()
MIGRATE_PROGRESS: Dict[int, Dict[str, Any]] = {}
MIGRATE_LOCK = threading.Lock()
TAG_PROGRESS: Dict[int, Dict[str, Any]] = {}
TAG_LOCK = threading.Lock()
ERA_PENDING: Dict[int, str] = {}
ERA_LOCK = threading.Lock()
EMBED_PROGRESS: Dict[str, Dict[str, Any]] = {}
LAST_UPLOAD_TS = 0.0
LAST_UPLOAD_LOCK = threading.Lock()
EMBED_LOCK = threading.Lock()

TASK_QUEUE: "queue.Queue[int]" = queue.Queue()
TASK_LOCK = threading.Lock()
TASK_WORKER_STARTED = False
TASK_ACTIVE_ID: Optional[int] = None
OPENAI_RECOVERY_LOCK = threading.Lock()
OPENAI_RECOVERY_STARTED = False
STARTUP_IMPORT_LOCK = threading.Lock()
STARTUP_IMPORT_STATUS: Dict[str, Any] = {
    "running": False,
    "total": 0,
    "done": 0,
    "processed": 0,
    "failed": 0,
    "skipped": 0,
    "current_flow": "",
    "started_at": None,
    "finished_at": None,
}

_event_queues: List["queue.Queue[str]"] = []
_event_lock = threading.Lock()


def _broadcast_event(payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False)
    with _event_lock:
        for q in list(_event_queues):
            try:
                q.put_nowait(data)
            except queue.Full:
                pass


def _archive_batch_output(
    flow: str,
    provider: str,
    batch_id: str,
    output_text: str,
    task_id: Optional[int] = None,
    project_id: Optional[int] = None,
) -> str:
    safe_flow = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(flow or "unknown")).strip("_") or "unknown"
    safe_provider = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(provider or "unknown")).strip("_") or "unknown"
    safe_batch = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(batch_id or "batch")).strip("_") or "batch"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    folder = BATCH_OUTPUT_DIR / safe_flow
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / f"{ts}_{safe_provider}_{safe_batch}.json"
    payload = {
        "flow": flow,
        "provider": provider,
        "batch_id": batch_id,
        "task_id": task_id,
        "project_id": project_id,
        "archived_at": now_iso(),
        "output_text": output_text,
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return str(json_path)


def _startup_import_set(**fields: Any) -> None:
    with STARTUP_IMPORT_LOCK:
        STARTUP_IMPORT_STATUS.update(fields)


def _startup_import_snapshot() -> Dict[str, Any]:
    with STARTUP_IMPORT_LOCK:
        return dict(STARTUP_IMPORT_STATUS)


def _fmt_seconds(seconds: float) -> str:
    secs = max(0, int(seconds))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _count_archived_batch_files() -> int:
    if not BATCH_OUTPUT_DIR.exists():
        return 0
    total = 0
    for flow_dir in BATCH_OUTPUT_DIR.iterdir():
        if not flow_dir.is_dir() or flow_dir.name in {"_processed", "_failed"}:
            continue
        total += sum(1 for _ in flow_dir.glob("*.json"))
    return total


def _write_startup_job(kind: str, payload: Optional[Dict[str, Any]] = None) -> str:
    STARTUP_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    safe_kind = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(kind or "job")).strip("_") or "job"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    nonce = uuid.uuid4().hex[:8]
    path = STARTUP_JOBS_DIR / f"{ts}_{safe_kind}_{nonce}.json"
    data = {
        "kind": kind,
        "payload": payload or {},
        "created_at": now_iso(),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    return str(path)


def _run_startup_jobs(settings: Dict[str, str]) -> Dict[str, int]:
    if not STARTUP_JOBS_DIR.exists():
        return {"total": 0, "processed": 0, "failed": 0, "skipped": 0}
    files = sorted(STARTUP_JOBS_DIR.glob("*.json"))
    if not files:
        return {"total": 0, "processed": 0, "failed": 0, "skipped": 0}
    processed_root = STARTUP_JOBS_DIR / "_processed"
    failed_root = STARTUP_JOBS_DIR / "_failed"
    processed_root.mkdir(parents=True, exist_ok=True)
    failed_root.mkdir(parents=True, exist_ok=True)
    total = len(files)
    processed = 0
    failed = 0
    skipped = 0
    logger.info("Startup jobs: found %s deferred job(s)", total)
    for idx, path in enumerate(files, start=1):
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            kind = str(data.get("kind") or "").strip()
            logger.info("Startup jobs: %s/%s kind=%s file=%s", idx, total, kind or "unknown", path.name)
            if kind == "embeddings_all":
                _regenerate_embeddings(None, task_id=None)
                processed += 1
            else:
                skipped += 1
            dest = processed_root / path.name
            path.replace(dest)
        except Exception as exc:
            failed += 1
            logger.warning("Startup jobs failed file=%s err=%s", path, exc)
            try:
                dest = failed_root / path.name
                path.replace(dest)
            except Exception:
                pass
    logger.info(
        "Startup jobs: processed=%s failed=%s skipped=%s total=%s",
        processed,
        failed,
        skipped,
        total,
    )
    return {"total": total, "processed": processed, "failed": failed, "skipped": skipped}


def _import_archived_batch_outputs_on_startup(settings: Dict[str, str]) -> Dict[str, int]:
    processed = 0
    failed = 0
    skipped = 0
    done_total = 0
    error_total = 0
    if not BATCH_OUTPUT_DIR.exists():
        return {"processed": 0, "failed": 0, "skipped": 0, "done": 0, "errors": 0}

    processed_root = BATCH_OUTPUT_DIR / "_processed"
    failed_root = BATCH_OUTPUT_DIR / "_failed"
    processed_root.mkdir(parents=True, exist_ok=True)
    failed_root.mkdir(parents=True, exist_ok=True)

    flow_dirs = [p for p in BATCH_OUTPUT_DIR.iterdir() if p.is_dir() and p.name not in {"_processed", "_failed"}]
    total_files = 0
    for flow_dir in flow_dirs:
        total_files += sum(1 for _ in flow_dir.glob("*.json"))
    _startup_import_set(
        running=True,
        total=int(total_files),
        done=0,
        processed=0,
        failed=0,
        skipped=0,
        current_flow="",
        started_at=now_iso(),
        finished_at=None,
    )

    if total_files > 0:
        logger.info("Startup batch import: found %s archived batch file(s)", total_files)

    seen = 0
    for flow_dir in flow_dirs:
        for path in sorted(flow_dir.glob("*.json")):
            seen += 1
            _startup_import_set(done=int(seen), current_flow=str(flow_dir.name or ""))
            if seen == 1 or seen % 25 == 0 or seen == total_files:
                logger.info("Startup batch import progress: %s/%s (%s)", seen, total_files, flow_dir.name)
            try:
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                flow = str(payload.get("flow") or flow_dir.name or "").strip()
                output_text = payload.get("output_text")
                if not flow or not isinstance(output_text, str) or not output_text.strip():
                    skipped += 1
                    dest = processed_root / flow_dir.name / path.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(dest)
                    continue

                project_id_raw = payload.get("project_id")
                project_id = int(project_id_raw) if project_id_raw is not None else None
                task_id_raw = payload.get("task_id")
                task_id = int(task_id_raw) if task_id_raw is not None else None
                provider = str(payload.get("provider") or "").strip().lower() or None
                batch_id = str(payload.get("batch_id") or "").strip()
                def _startup_progress(done_count: int, total_count: int) -> None:
                    if total_count <= 0:
                        return
                    if done_count == total_count or done_count == 0 or done_count % 100 == 0:
                        logger.info(
                            "Startup batch import assets: %s/%s (%s)",
                            done_count,
                            total_count,
                            flow,
                        )

                stats = _apply_batch_output_for_flow(
                    flow,
                    output_text,
                    settings,
                    project_id,
                    progress_cb=_startup_progress,
                )
                done = int(stats.get("done") or 0)
                errs = int(stats.get("errors") or 0)
                done_total += done
                error_total += errs

                if batch_id:
                    _openai_batch_upsert(
                        flow=flow,
                        batch_id=batch_id,
                        provider=provider,
                        task_id=task_id,
                        project_id=project_id,
                        status="completed",
                    )
                    _openai_batch_mark_applied(
                        batch_id=batch_id,
                        flow=flow,
                        task_id=task_id,
                        rows_done=done,
                        rows_error=errs,
                    )
                    _openai_batch_mark_processed(batch_id)

                processed += 1
                _startup_import_set(processed=int(processed))
                dest = processed_root / flow_dir.name / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                path.replace(dest)
                logger.info(
                    "Startup batch import file done: %s (%s/%s)",
                    path.name,
                    seen,
                    total_files,
                )
            except Exception as exc:
                failed += 1
                _startup_import_set(failed=int(failed))
                logger.warning("Startup batch import failed file=%s err=%s", path, exc)
                try:
                    dest = failed_root / flow_dir.name / path.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(dest)
                except Exception:
                    pass

    stats = {
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
        "done": done_total,
        "errors": error_total,
    }
    _startup_import_set(
        running=False,
        total=int(total_files),
        done=int(total_files),
        processed=int(processed),
        failed=int(failed),
        skipped=int(skipped),
        finished_at=now_iso(),
        current_flow="",
    )
    return stats


def _set_last_upload() -> None:
    with LAST_UPLOAD_LOCK:
        global LAST_UPLOAD_TS
        LAST_UPLOAD_TS = time.time()


def _get_last_upload_age() -> Dict[str, Any]:
    with LAST_UPLOAD_LOCK:
        ts = LAST_UPLOAD_TS
    now = time.time()
    age = now - ts if ts else None
    return {"last_upload_ts": ts, "age_seconds": age}

def _db_retry(fn, attempts: int = 10, delay: float = 1.0):
    for i in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            time.sleep(delay * (i + 1))
    raise sqlite3.OperationalError("database is locked")


def _retry_http(fn, attempts: int = 5, delay: float = 2.0):
    for i in range(attempts):
        try:
            resp = fn()
            if resp is not None and resp.status_code in {502, 503, 504}:
                raise httpx.HTTPStatusError(
                    f"Retryable status {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            return resp
        except httpx.HTTPError:
            if i >= attempts - 1:
                raise
            time.sleep(delay * (i + 1))

def _flush_embedding_batch(batch: List[tuple]) -> None:
    if not batch:
        return
    def _write():
        conn = get_db()
        conn.execute("BEGIN")
        conn.executemany(
            "UPDATE assets SET embedding_json = ? WHERE id = ?",
            batch,
        )
        conn.commit()
        conn.close()
    _db_retry(_write)


def _upsert_asset_tags_bulk(conn, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cur = conn.cursor()
    # Keep the last payload per hash if duplicates exist in the same batch.
    with_hash_by_key: Dict[str, tuple] = {}
    without_hash: List[tuple] = []

    for row in rows:
        payload = (
            row["asset_id"],
            row["hash_main_blake3"],
            row["hash_full_blake3"],
            row["tags_original_json"],
            row["tags_translated_json"],
            row["translated_language"],
            row["asset_created_at"],
            row["tags_done_at"],
            row["name_tags_done_at"],
            row["name_translate_tags_done_at"],
            row["translate_tags_done_at"],
            row["created_at"],
            row["updated_at"],
        )
        if row["hash_full_blake3"]:
            with_hash_by_key[str(row["hash_full_blake3"])] = payload
        else:
            without_hash.append(payload)

    with_hash: List[tuple] = list(with_hash_by_key.values())
    if with_hash:
        hashes = [row[2] for row in with_hash]
        placeholders = ",".join(["?"] * len(hashes))
        existing_rows = fetch_all(
            conn,
            f"SELECT id, hash_full_blake3 FROM asset_tags WHERE hash_full_blake3 IN ({placeholders})",
            tuple(hashes),
        )
        existing_by_hash = {str(r["hash_full_blake3"]): int(r["id"]) for r in existing_rows}
        update_rows: List[tuple] = []
        insert_rows: List[tuple] = []
        for row in with_hash:
            existing_id = existing_by_hash.get(str(row[2]))
            if existing_id is not None:
                update_rows.append(
                    (
                        row[0],  # asset_id
                        row[1],  # hash_main_blake3
                        row[3],  # tags_original_json
                        row[4],  # tags_translated_json
                        row[5],  # translated_language
                        row[6],  # asset_created_at
                        row[7],  # tags_done_at
                        row[8],  # name_tags_done_at
                        row[9],  # name_translate_tags_done_at
                        row[10],  # translate_tags_done_at
                        row[12],  # updated_at
                        existing_id,
                    )
                )
            else:
                insert_rows.append(row)

        if update_rows:
            cur.executemany(
                """
                UPDATE asset_tags
                SET
                    asset_id = ?,
                    hash_main_blake3 = ?,
                    tags_original_json = ?,
                    tags_translated_json = ?,
                    translated_language = ?,
                    asset_created_at = ?,
                    tags_done_at = COALESCE(?, tags_done_at),
                    name_tags_done_at = COALESCE(?, name_tags_done_at),
                    name_translate_tags_done_at = COALESCE(?, name_translate_tags_done_at),
                    translate_tags_done_at = COALESCE(?, translate_tags_done_at),
                    updated_at = ?
                WHERE id = ?
                """,
                update_rows,
            )

        if insert_rows:
            cur.executemany(
                """
                INSERT INTO asset_tags (
                    asset_id,
                    hash_main_blake3,
                    hash_full_blake3,
                    tags_original_json,
                    tags_translated_json,
                    translated_language,
                    asset_created_at,
                    tags_done_at,
                    name_tags_done_at,
                    name_translate_tags_done_at,
                    translate_tags_done_at,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )

    if without_hash:
        # Fallback path for rows without stable full hash.
        delete_ids = [(row[0],) for row in without_hash]
        cur.executemany("DELETE FROM asset_tags WHERE asset_id = ?", delete_ids)
        cur.executemany(
            """
            INSERT INTO asset_tags (
                asset_id,
                hash_main_blake3,
                hash_full_blake3,
                tags_original_json,
                tags_translated_json,
                translated_language,
                asset_created_at,
                tags_done_at,
                name_tags_done_at,
                name_translate_tags_done_at,
                translate_tags_done_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            without_hash,
        )


def _flush_tag_batch(
    batch: List[Dict[str, Any]],
    settings: Dict[str, str],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    if not batch:
        return
    def _write():
        conn = get_db()
        conn.execute("BEGIN")
        rows = []
        embed_rows = []
        for item in batch:
            merged_tags = _merge_tags_for_asset(item["tags"], item.get("translated_tags") or [])
            rows.append((json.dumps(merged_tags), item["id"]))
            if item.get("embedding") is not None:
                embed_rows.append((json.dumps(item["embedding"]), item["id"]))
        conn.executemany(
            "UPDATE assets SET tags_json = ? WHERE id = ?",
            rows,
        )
        if embed_rows:
            conn.executemany(
                "UPDATE assets SET embedding_json = ? WHERE id = ?",
                embed_rows,
            )
        now = now_iso()
        tag_language = settings.get("tag_language") or ""
        tag_rows: List[Dict[str, Any]] = []
        for item in batch:
            tags_done_at = now if bool(item.get("mark_tags_done")) else None
            name_tags_done_at = now if bool(item.get("mark_name_tags_done")) else None
            name_translate_done_at = now if bool(item.get("mark_name_translate_done")) else None
            translate_done_at = now if bool(item.get("mark_translate_done")) else None
            tag_rows.append(
                {
                    "asset_id": int(item["id"]),
                    "hash_main_blake3": item.get("hash_main") or "",
                    "hash_full_blake3": item.get("hash_full") or "",
                    "tags_original_json": json.dumps(item["tags"]),
                    "tags_translated_json": json.dumps(item["translated_tags"]) if item["translated_tags"] else None,
                    "translated_language": tag_language or None,
                    "asset_created_at": item.get("created_at") or now,
                    "tags_done_at": tags_done_at,
                    "name_tags_done_at": name_tags_done_at,
                    "name_translate_tags_done_at": name_translate_done_at,
                    "translate_tags_done_at": translate_done_at,
                    "created_at": now,
                    "updated_at": now,
                }
            )

        _upsert_asset_tags_bulk(conn, tag_rows)
        if progress_cb is not None:
            progress_cb(len(batch), len(batch))
        _flush_project_eras(conn)
        conn.commit()
        conn.close()
    _db_retry(_write)


def _flush_tag_batch_chunked(
    batch: List[Dict[str, Any]],
    settings: Dict[str, str],
    chunk_size: int = 2000,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_label: Optional[str] = None,
) -> None:
    if not batch:
        return
    size = max(1, int(chunk_size or 1))
    total = len(batch)
    written = 0
    started_at = time.perf_counter()
    for i in range(0, len(batch), size):
        chunk = batch[i : i + size]
        chunk_base = written
        chunk_started_at = time.perf_counter()

        def _on_chunk_progress(done_in_chunk: int, total_in_chunk: int) -> None:
            current_written = chunk_base + done_in_chunk
            if log_label and (current_written >= total or current_written % 100 == 0):
                logger.info("Batch apply DB write: %s/%s (%s)", current_written, total, log_label)
            if progress_cb is not None:
                progress_cb(current_written, total)

        _flush_tag_batch(chunk, settings, progress_cb=_on_chunk_progress)
        written = chunk_base + len(chunk)
        if log_label:
            chunk_elapsed = max(0.0001, time.perf_counter() - chunk_started_at)
            total_elapsed = max(0.0001, time.perf_counter() - started_at)
            rate = written / total_elapsed
            remaining = max(0, total - written)
            eta_seconds = (remaining / rate) if rate > 0 else 0.0
            logger.info(
                "Batch apply chunk done: %s/%s (%s) chunk=%s took=%.2fs rate=%.1f rows/s elapsed=%s eta=%s",
                written,
                total,
                log_label,
                len(chunk),
                chunk_elapsed,
                rate,
                _fmt_seconds(total_elapsed),
                _fmt_seconds(eta_seconds),
            )


def _task_update(task_id: int, **fields: Any) -> None:
    if not fields:
        return
    def _write() -> None:
        conn = get_db()
        cols = []
        params = []
        for key, value in fields.items():
            cols.append(f"{key} = ?")
            params.append(value)
        params.append(task_id)
        conn.execute(f"UPDATE tasks SET {', '.join(cols)} WHERE id = ?", params)
        conn.commit()
        conn.close()
    _db_retry(_write)


def _task_get(task_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = fetch_one(conn, "SELECT * FROM tasks WHERE id = ?", (task_id,))
    conn.close()
    return row


def _task_cancelled(task_id: int) -> bool:
    row = _task_get(task_id)
    if not row:
        return True
    return bool(row.get("cancel_flag"))


_OPENAI_BATCH_TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _openai_batch_upsert(
    flow: str,
    batch_id: str,
    provider: Optional[str] = None,
    task_id: Optional[int] = None,
    project_id: Optional[int] = None,
    request_total: Optional[int] = None,
    status: Optional[str] = None,
    output_file_id: Optional[str] = None,
    error_text: Optional[str] = None,
) -> None:
    if not flow or not batch_id:
        return

    now = now_iso()

    def _write() -> None:
        conn = get_db()
        row = fetch_one(conn, "SELECT id FROM openai_batches WHERE batch_id = ?", (batch_id,))
        if row:
            sets: List[str] = ["updated_at = ?"]
            params: List[Any] = [now]
            if provider is not None:
                sets.append("provider = ?")
                params.append((provider or "").strip().lower())
            if task_id is not None:
                sets.append("task_id = ?")
                params.append(task_id)
            if project_id is not None:
                sets.append("project_id = ?")
                params.append(project_id)
            if request_total is not None:
                sets.append("request_total = ?")
                params.append(int(request_total))
            if status is not None:
                sets.append("status = ?")
                params.append(status)
            if output_file_id is not None:
                sets.append("output_file_id = ?")
                params.append(output_file_id)
            if error_text is not None:
                sets.append("error_text = ?")
                params.append(error_text)
            params.append(batch_id)
            conn.execute(f"UPDATE openai_batches SET {', '.join(sets)} WHERE batch_id = ?", params)
        else:
            conn.execute(
                """
                INSERT INTO openai_batches (
                    flow, provider, batch_id, task_id, project_id, request_total, status,
                    output_file_id, error_text, created_at, updated_at, processed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    flow,
                    (provider or "").strip().lower() or None,
                    batch_id,
                    task_id,
                    project_id,
                    int(request_total or 0),
                    status or "submitted",
                    output_file_id,
                    error_text,
                    now,
                    now,
                ),
            )
        conn.commit()
        conn.close()

    _db_retry(_write)


def _openai_batch_is_applied(batch_id: str) -> bool:
    if not batch_id:
        return False

    def _read() -> bool:
        conn = get_db()
        row = fetch_one(conn, "SELECT batch_id FROM openai_batch_results_applied WHERE batch_id = ?", (batch_id,))
        conn.close()
        return bool(row)

    return _db_retry(_read)


def _openai_batch_mark_applied(
    batch_id: str,
    flow: Optional[str] = None,
    task_id: Optional[int] = None,
    rows_done: int = 0,
    rows_error: int = 0,
) -> None:
    if not batch_id:
        return
    now = now_iso()

    def _write() -> None:
        conn = get_db()
        conn.execute(
            """
            INSERT INTO openai_batch_results_applied(batch_id, flow, task_id, rows_done, rows_error, applied_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(batch_id) DO UPDATE SET
                flow=excluded.flow,
                task_id=excluded.task_id,
                rows_done=excluded.rows_done,
                rows_error=excluded.rows_error,
                applied_at=excluded.applied_at
            """,
            (batch_id, flow, task_id, int(rows_done or 0), int(rows_error or 0), now),
        )
        conn.commit()
        conn.close()

    _db_retry(_write)


def _openai_batch_claim(batch_id: str, owner: str, lease_seconds: int = 120) -> bool:
    if not batch_id or not owner:
        return False
    now = now_iso()
    cutoff = datetime.utcfromtimestamp(time.time() - max(10, int(lease_seconds))).isoformat()
    changed: Dict[str, int] = {"n": 0}

    def _write() -> None:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE openai_batches
            SET processing_owner = ?,
                processing_started_at = COALESCE(processing_started_at, ?),
                processing_heartbeat_at = ?,
                updated_at = ?
            WHERE batch_id = ?
              AND processed_at IS NULL
              AND (
                    processing_owner IS NULL
                    OR processing_owner = ?
                    OR processing_heartbeat_at IS NULL
                    OR processing_heartbeat_at < ?
              )
            """,
            (owner, now, now, now, batch_id, owner, cutoff),
        )
        changed["n"] = int(cur.rowcount or 0)
        conn.commit()
        conn.close()

    _db_retry(_write)
    return changed["n"] > 0


def _openai_batch_heartbeat(batch_id: str, owner: str) -> None:
    if not batch_id or not owner:
        return
    now = now_iso()

    def _write() -> None:
        conn = get_db()
        conn.execute(
            """
            UPDATE openai_batches
            SET processing_heartbeat_at = ?, updated_at = ?
            WHERE batch_id = ? AND processing_owner = ? AND processed_at IS NULL
            """,
            (now, now, batch_id, owner),
        )
        conn.commit()
        conn.close()

    _db_retry(_write)


def _openai_batch_release(batch_id: str, owner: str) -> None:
    if not batch_id or not owner:
        return
    now = now_iso()

    def _write() -> None:
        conn = get_db()
        conn.execute(
            """
            UPDATE openai_batches
            SET processing_owner = NULL, processing_started_at = NULL, processing_heartbeat_at = NULL, updated_at = ?
            WHERE batch_id = ? AND processing_owner = ? AND processed_at IS NULL
            """,
            (now, batch_id, owner),
        )
        conn.commit()
        conn.close()

    _db_retry(_write)


def _openai_batch_mark_processed(batch_id: str) -> None:
    if not batch_id:
        return
    finished = now_iso()

    def _write() -> None:
        conn = get_db()
        conn.execute(
            """
            UPDATE openai_batches
            SET processed_at = ?, updated_at = ?,
                processing_owner = NULL, processing_started_at = NULL, processing_heartbeat_at = NULL
            WHERE batch_id = ?
            """,
            (finished, finished, batch_id),
        )
        conn.commit()
        conn.close()

    _db_retry(_write)


def _openai_list_pending_batches(flow: str, project_id: Optional[int]) -> List[Dict[str, Any]]:
    def _read() -> List[Dict[str, Any]]:
        conn = get_db()
        if project_id is None:
            rows = fetch_all(
                conn,
                """
                SELECT *
                FROM openai_batches
                WHERE flow = ? AND project_id IS NULL AND processed_at IS NULL
                  AND (status IS NULL OR status NOT IN ('failed', 'expired', 'cancelled'))
                ORDER BY id ASC
                """,
                (flow,),
            )
        else:
            rows = fetch_all(
                conn,
                """
                SELECT *
                FROM openai_batches
                WHERE flow = ? AND project_id = ? AND processed_at IS NULL
                  AND (status IS NULL OR status NOT IN ('failed', 'expired', 'cancelled'))
                ORDER BY id ASC
                """,
                (flow, project_id),
            )
        conn.close()
        return rows

    return _db_retry(_read)


def _openai_pending_snapshot(pending: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    ready = 0
    in_progress = 0
    finalizing = 0
    waiting = 0
    for info in pending.values():
        status = str(info.get("last_status") or "").strip().lower()
        output_ready = bool(info.get("last_output_ready"))
        if output_ready:
            ready += 1
            continue
        if status == "in_progress":
            in_progress += 1
            continue
        if status == "finalizing":
            finalizing += 1
            continue
        waiting += 1
    return {
        "ready": ready,
        "in_progress": in_progress,
        "finalizing": finalizing,
        "waiting": waiting,
        "pending": len(pending),
    }


def _openai_list_unprocessed_batches(
    limit: int = 100,
    flow: Optional[str] = None,
    task_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    def _read() -> List[Dict[str, Any]]:
        conn = get_db()
        lease_cutoff = datetime.utcfromtimestamp(time.time() - 120).isoformat()
        where = [
            "processed_at IS NULL",
            "(status IS NULL OR status NOT IN ('failed', 'expired', 'cancelled'))",
            "(processing_owner IS NULL OR processing_heartbeat_at IS NULL OR processing_heartbeat_at < ?)",
        ]
        params: List[Any] = [lease_cutoff]
        if flow:
            where.append("flow = ?")
            params.append(flow)
        if task_id is not None:
            where.append("task_id = ?")
            params.append(int(task_id))
        rows = fetch_all(
            conn,
            f"""
            SELECT *
            FROM openai_batches
            WHERE {' AND '.join(where)}
            ORDER BY id ASC
            LIMIT ?
            """,
            tuple(params + [limit]),
        )
        conn.close()
        return rows
    return _db_retry(_read)


def _has_active_tasks() -> bool:
    def _read() -> bool:
        conn = get_db()
        row = fetch_one(conn, "SELECT COUNT(1) AS c FROM tasks WHERE status IN ('queued','running')")
        conn.close()
        return bool(int((row or {}).get("c") or 0))
    return _db_retry(_read)


def _load_assets_for_ids(asset_ids: List[int]) -> Dict[str, Dict[str, Any]]:
    if not asset_ids:
        return {}
    placeholders = ",".join(["?"] * len(asset_ids))
    conn = get_db()
    rows = fetch_all(
        conn,
        f"""
        SELECT a.id, a.project_id, a.name, a.description, a.tags_json, a.hash_main_blake3, a.hash_full_blake3, a.created_at,
               t.tags_translated_json
        FROM assets a
        LEFT JOIN asset_tags t ON t.hash_full_blake3 = a.hash_full_blake3
        WHERE a.id IN ({placeholders})
        """,
        tuple(asset_ids),
    )
    conn.close()
    return {str(r["id"]): r for r in rows}


def _apply_batch_output_for_flow(
    flow: str,
    output_text: str,
    settings: Dict[str, str],
    default_project_id: Optional[int],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, int]:
    parse_started = time.perf_counter()
    parsed_items: List[Dict[str, Any]] = []
    asset_ids: List[int] = []
    for line in output_text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        custom_id = str(item.get("custom_id") or "")
        if not custom_id:
            continue
        try:
            aid = int(custom_id)
            asset_ids.append(aid)
        except Exception:
            pass
        parsed_items.append(item)
    parse_elapsed = max(0.0001, time.perf_counter() - parse_started)
    logger.info(
        "Batch apply parse done: items=%s unique_assets=%s flow=%s took=%.2fs",
        len(parsed_items),
        len(set(asset_ids)),
        flow,
        parse_elapsed,
    )

    load_started = time.perf_counter()
    rows_by_id = _load_assets_for_ids(asset_ids)
    load_elapsed = max(0.0001, time.perf_counter() - load_started)
    logger.info(
        "Batch apply asset load done: rows=%s flow=%s took=%.2fs",
        len(rows_by_id),
        flow,
        load_elapsed,
    )
    batch_rows: List[Dict[str, Any]] = []
    done = 0
    errors = 0
    total = len(parsed_items)
    last_reported_step = -1
    map_started = time.perf_counter()

    def _report(force: bool = False) -> None:
        nonlocal last_reported_step
        if progress_cb is None:
            return
        if total <= 0:
            if force:
                progress_cb(done, total)
            return
        step = done // 100
        if force or step != last_reported_step:
            last_reported_step = step
            progress_cb(done, total)

    for item in parsed_items:
        custom_id = str(item.get("custom_id") or "")
        if not custom_id:
            continue
        row = rows_by_id.get(custom_id)
        if not row:
            done += 1
            errors += 1
            _report()
            continue
        if item.get("error"):
            done += 1
            errors += 1
            _report()
            continue

        resp = item.get("response") or {}
        body = resp.get("body") if isinstance(resp, dict) else None
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception:
                body = None
        payload = body if isinstance(body, dict) else resp
        text = _extract_output_text_from_response(payload)

        if flow == "translate_name_tags":
            translated_raw = _extract_tags_from_content(text)
            translated = _normalize_tags(translated_raw)
            if not translated:
                done += 1
                errors += 1
                _report()
                continue
            existing_tags = json.loads(row.get("tags_json") or "[]")
            existing_translated = json.loads(row.get("tags_translated_json") or "[]")
            batch_rows.append(
                {
                    "id": row["id"],
                    "tags": _normalize_tags(existing_tags + translated),
                    "translated_tags": _normalize_tags(existing_translated + translated),
                    "embedding": None,
                    "hash_main": row.get("hash_main_blake3") or "",
                    "hash_full": row.get("hash_full_blake3") or "",
                    "created_at": row.get("created_at") or now_iso(),
                    "mark_name_translate_done": True,
                }
            )
            done += 1
            _report()
            continue

        if flow == "translate_tags":
            translated_raw = _extract_tags_from_content(text)
            translated = _normalize_tags(translated_raw)
            if not translated:
                done += 1
                errors += 1
                _report()
                continue
            existing_tags = json.loads(row.get("tags_json") or "[]")
            existing_translated = json.loads(row.get("tags_translated_json") or "[]")
            batch_rows.append(
                {
                    "id": row["id"],
                    "tags": _normalize_tags(existing_tags + translated),
                    "translated_tags": _normalize_tags(existing_translated + translated),
                    "embedding": None,
                    "hash_main": row.get("hash_main_blake3") or "",
                    "hash_full": row.get("hash_full_blake3") or "",
                    "created_at": row.get("created_at") or now_iso(),
                    "mark_translate_done": True,
                }
            )
            done += 1
            _report()
            continue

        if flow == "tag_assets":
            tags_raw, era = _extract_tags_and_era(text)
            tags = _normalize_tags(tags_raw)
            if not tags:
                done += 1
                errors += 1
                _report()
                continue
            project_id = int(row.get("project_id") or default_project_id or 0) or None
            if project_id is not None:
                _maybe_set_project_era(project_id, era)
            translated_tags = _translate_tags_if_enabled(settings, tags)
            batch_rows.append(
                {
                    "id": row["id"],
                    "tags": tags,
                    "translated_tags": translated_tags,
                    "embedding": None,
                    "hash_main": row.get("hash_main_blake3") or "",
                    "hash_full": row.get("hash_full_blake3") or "",
                    "created_at": row.get("created_at") or now_iso(),
                    "mark_tags_done": True,
                }
            )
            done += 1
            _report()
            continue

        done += 1
        errors += 1
        _report()

    if batch_rows:
        next_write_log = 1000
        map_elapsed = max(0.0001, time.perf_counter() - map_started)
        map_rate = done / map_elapsed if map_elapsed > 0 else 0.0
        logger.info(
            "Batch apply map done: done=%s errors=%s queued_writes=%s flow=%s took=%.2fs rate=%.1f rows/s",
            done,
            errors,
            len(batch_rows),
            flow,
            map_elapsed,
            map_rate,
        )
        logger.info("Batch apply DB write start: %s row(s) (%s)", len(batch_rows), flow)
        write_started = time.perf_counter()

        def _write_progress(written: int, total_write: int) -> None:
            nonlocal next_write_log
            # Parsing progress is already complete here; keep startup logs focused on DB write.
            if total_write > 0 and (written >= next_write_log or written == total_write):
                while written >= next_write_log:
                    next_write_log += 1000

        _flush_tag_batch_chunked(batch_rows, settings, progress_cb=_write_progress, log_label=flow)
        write_elapsed = max(0.0001, time.perf_counter() - write_started)
        write_rate = len(batch_rows) / write_elapsed if write_elapsed > 0 else 0.0
        logger.info(
            "Batch apply DB write done: %s row(s) (%s) took=%.2fs rate=%.1f rows/s",
            len(batch_rows),
            flow,
            write_elapsed,
            write_rate,
        )
    else:
        map_elapsed = max(0.0001, time.perf_counter() - map_started)
        map_rate = done / map_elapsed if map_elapsed > 0 else 0.0
        logger.info(
            "Batch apply map done: done=%s errors=%s queued_writes=0 flow=%s took=%.2fs rate=%.1f rows/s",
            done,
            errors,
            flow,
            map_elapsed,
            map_rate,
        )

    _report(force=True)
    return {"done": done, "errors": errors}


def _recover_openai_batches_once(
    limit: int = 25,
    flow: Optional[str] = None,
    task_id: Optional[int] = None,
    stale_minutes: int = 180,
) -> Dict[str, int]:
    def _db_retry_call(fn: Callable[[], Any]) -> Any:
        # Recovery may compete with long-running write transactions.
        # Use a longer retry window so completed batches do not remain stuck.
        return _db_retry(fn, attempts=30, delay=0.75)

    settings_conn = get_db()
    settings = get_settings(settings_conn)
    settings_conn.close()
    batches = _openai_list_unprocessed_batches(limit=limit, flow=flow, task_id=task_id)
    if not batches:
        return {"processed_batches": 0, "done": 0, "errors": 0}

    processed_batches = 0
    total_done = 0
    total_errors = 0
    timeout = httpx.Timeout(30.0, connect=30.0, read=30.0)
    owner = f"recover:{uuid.uuid4().hex}"

    with httpx.Client(timeout=timeout) as client:
        for row in batches:
            batch_id = str(row.get("batch_id") or "")
            flow = str(row.get("flow") or "")
            preferred_provider = str(row.get("provider") or "").strip().lower()
            active_provider = str(settings.get("provider") or "").strip().lower()
            provider_candidates = [p for p in [preferred_provider, active_provider, "openai", "groq", "openrouter"] if p]
            seen_candidates: set[str] = set()
            provider_candidates = [p for p in provider_candidates if not (p in seen_candidates or seen_candidates.add(p))]
            if not batch_id or not flow:
                continue
            if not _db_retry_call(lambda: _openai_batch_claim(batch_id, owner)):
                continue
            try:
                _db_retry_call(lambda: _openai_batch_heartbeat(batch_id, owner))
                payload: Optional[Dict[str, Any]] = None
                row_provider = provider_candidates[0] if provider_candidates else "openai"
                api_base = _provider_base_url(settings, row_provider).rstrip("/")
                headers = {"Authorization": f"Bearer {_provider_api_key(settings, row_provider)}"} if _provider_api_key(settings, row_provider) else {}
                last_poll_exc: Optional[Exception] = None
                for candidate in provider_candidates:
                    candidate_api_base = _provider_base_url(settings, candidate).rstrip("/")
                    candidate_key = _provider_api_key(settings, candidate)
                    candidate_headers = {"Authorization": f"Bearer {candidate_key}"} if candidate_key else {}
                    try:
                        poll = client.get(f"{candidate_api_base}/batches/{batch_id}", headers=candidate_headers)
                        poll.raise_for_status()
                        payload = poll.json()
                        row_provider = candidate
                        api_base = candidate_api_base
                        headers = candidate_headers
                        break
                    except Exception as exc:
                        last_poll_exc = exc
                        continue
                if payload is None:
                    if last_poll_exc is not None:
                        raise last_poll_exc
                    raise RuntimeError(f"Batch poll failed for {batch_id}")
                status = payload.get("status") or ""
                output_file_id = payload.get("output_file_id")
                _db_retry_call(
                    lambda: _openai_batch_upsert(
                        flow=flow,
                        batch_id=batch_id,
                        provider=row_provider,
                        task_id=row.get("task_id"),
                        project_id=row.get("project_id"),
                        request_total=row.get("request_total"),
                        status=status,
                        output_file_id=output_file_id,
                    )
                )

                if status not in _OPENAI_BATCH_TERMINAL:
                    # Avoid permanent queue block by stale non-terminal batches.
                    try:
                        updated_at_raw = row.get("updated_at") or row.get("created_at")
                        updated_at = datetime.fromisoformat(str(updated_at_raw))
                        age_seconds = (datetime.utcnow() - updated_at).total_seconds()
                    except Exception:
                        age_seconds = 0
                    if age_seconds > max(60, int(stale_minutes)) * 60:
                        _db_retry_call(
                            lambda: _openai_batch_upsert(
                                flow=flow,
                                batch_id=batch_id,
                                provider=row_provider,
                                task_id=row.get("task_id"),
                                project_id=row.get("project_id"),
                                request_total=row.get("request_total"),
                                status=status or "stale",
                                error_text=f"stale non-terminal batch (age={int(age_seconds)}s)",
                            )
                        )
                        _db_retry_call(lambda: _openai_batch_mark_processed(batch_id))
                        processed_batches += 1
                    continue
                if status != "completed" or not output_file_id:
                    _db_retry_call(
                        lambda: _openai_batch_upsert(
                            flow=flow,
                            batch_id=batch_id,
                            provider=row_provider,
                            task_id=row.get("task_id"),
                            project_id=row.get("project_id"),
                            request_total=row.get("request_total"),
                            status=status,
                            error_text=f"terminal status={status}",
                        )
                    )
                    _db_retry_call(lambda: _openai_batch_mark_processed(batch_id))
                    processed_batches += 1
                    continue

                if _openai_batch_is_applied(batch_id):
                    _db_retry_call(lambda: _openai_batch_mark_processed(batch_id))
                    processed_batches += 1
                    continue

                _db_retry_call(lambda: _openai_batch_heartbeat(batch_id, owner))
                result = client.get(f"{api_base}/files/{output_file_id}/content", headers=headers)
                result.raise_for_status()
                output_text = result.text
                archive_path = _archive_batch_output(
                    flow=flow,
                    provider=row_provider,
                    batch_id=batch_id,
                    output_text=output_text,
                    task_id=int(row.get("task_id") or 0) if row.get("task_id") is not None else None,
                    project_id=int(row.get("project_id") or 0) if row.get("project_id") is not None else None,
                )
                is_all_scope = row.get("project_id") is None
                if is_all_scope and flow in {"translate_name_tags", "translate_tags", "tag_assets"}:
                    estimated_done = int(row.get("request_total") or 0)
                    stats = {"done": estimated_done, "errors": 0}
                    logger.info(
                        "Recovered OpenAI batch id=%s flow=%s archived=%s",
                        batch_id,
                        flow,
                        archive_path,
                    )
                else:
                    stats = _apply_batch_output_for_flow(flow, output_text, settings, row.get("project_id"))
                total_done += int(stats.get("done") or 0)
                total_errors += int(stats.get("errors") or 0)
                _db_retry_call(
                    lambda: _openai_batch_mark_applied(
                        batch_id=batch_id,
                        flow=flow,
                        task_id=int(row.get("task_id") or 0) if row.get("task_id") is not None else None,
                        rows_done=int(stats.get("done") or 0),
                        rows_error=int(stats.get("errors") or 0),
                    )
                )
                _db_retry_call(lambda: _openai_batch_mark_processed(batch_id))
                processed_batches += 1
                logger.info(
                    "Recovered OpenAI batch id=%s flow=%s processed done=%s errors=%s",
                    batch_id,
                    flow,
                    stats.get("done"),
                    stats.get("errors"),
                )
            except Exception as exc:
                try:
                    _db_retry_call(
                        lambda: _openai_batch_upsert(
                            flow=flow,
                            batch_id=batch_id,
                            provider=row_provider,
                            task_id=row.get("task_id"),
                            project_id=row.get("project_id"),
                            request_total=row.get("request_total"),
                            status=row.get("status") or "error",
                            error_text=str(exc),
                        )
                    )
                except Exception:
                    pass
                logger.warning("OpenAI batch recovery failed id=%s flow=%s err=%s", batch_id, flow, exc)
                continue
            finally:
                try:
                    _db_retry_call(lambda: _openai_batch_release(batch_id, owner))
                except Exception:
                    pass

    return {"processed_batches": processed_batches, "done": total_done, "errors": total_errors}


def _openai_recovery_worker() -> None:
    while True:
        try:
            has_active = _has_active_tasks()
            # Keep recovery running even while tasks are active, just with lower pressure.
            limit = 6 if has_active else 20
            stats = _recover_openai_batches_once(limit=limit)
            if stats.get("processed_batches", 0) > 0:
                logger.info(
                    "OpenAI recovery pass processed batches=%s done=%s errors=%s",
                    stats.get("processed_batches"),
                    stats.get("done"),
                    stats.get("errors"),
                )
                time.sleep(2)
            else:
                time.sleep(14 if has_active else 12)
        except Exception as exc:
            logger.warning("OpenAI recovery worker error: %s", exc)
            time.sleep(10)


def _ensure_openai_recovery_worker() -> None:
    global OPENAI_RECOVERY_STARTED
    with OPENAI_RECOVERY_LOCK:
        if OPENAI_RECOVERY_STARTED:
            return
        thread = threading.Thread(target=_openai_recovery_worker, daemon=True)
        thread.start()
        OPENAI_RECOVERY_STARTED = True


def _run_startup_import_worker(settings: Dict[str, str]) -> None:
    try:
        startup_jobs_stats = _run_startup_jobs(settings)
        if any(startup_jobs_stats.values()):
            logger.info(
                "Startup jobs summary: total=%s processed=%s failed=%s skipped=%s",
                startup_jobs_stats.get("total", 0),
                startup_jobs_stats.get("processed", 0),
                startup_jobs_stats.get("failed", 0),
                startup_jobs_stats.get("skipped", 0),
            )
        import_stats = _import_archived_batch_outputs_on_startup(settings)
        if any(import_stats.values()):
            logger.info(
                "Startup batch import: processed=%s failed=%s skipped=%s done=%s errors=%s",
                import_stats.get("processed", 0),
                import_stats.get("failed", 0),
                import_stats.get("skipped", 0),
                import_stats.get("done", 0),
                import_stats.get("errors", 0),
            )
    except Exception as exc:
        _startup_import_set(running=False, finished_at=now_iso())
        logger.warning("Startup batch import worker failed: %s", exc)
    finally:
        _ensure_task_worker()
        _ensure_openai_recovery_worker()


def _task_progress(
    task_id: int,
    status: str,
    total: int,
    done: int,
    errors: int = 0,
    message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {"status": status, "total": total, "done": done, "errors": errors}
    if extra:
        payload.update(extra)
    fields = {"progress_json": json.dumps(payload)}
    if message is not None:
        fields["message"] = message
    _task_update(task_id, **fields)
    # Always emit task progress to normal app logs (independent of SQL trace),
    # so long-running jobs remain visible in server output.
    if message:
        logger.info(
            "Task progress id=%s status=%s done=%s/%s errors=%s message=%s",
            task_id,
            status,
            done,
            total,
            errors,
            message,
        )
    else:
        logger.info(
            "Task progress id=%s status=%s done=%s/%s errors=%s",
            task_id,
            status,
            done,
            total,
            errors,
        )


def _enqueue_task(kind: str, target_id: Optional[int] = None, message: Optional[str] = None) -> int:
    now = now_iso()
    task_id_box: Dict[str, int] = {"id": 0}

    def _write() -> None:
        conn = get_db()
        conn.execute(
            "INSERT INTO tasks (kind, status, target_id, progress_json, message, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (kind, "queued", target_id, json.dumps({"status": "queued"}), message, now),
        )
        row = conn.execute("SELECT last_insert_rowid() as id").fetchone()
        task_id_box["id"] = int(row["id"]) if row and row["id"] is not None else 0
        conn.commit()
        conn.close()

    _db_retry(_write)
    task_id = task_id_box["id"]
    if task_id <= 0:
        raise RuntimeError("Failed to enqueue task")
    TASK_QUEUE.put(task_id)
    return int(task_id)


def _task_worker() -> None:
    global TASK_ACTIVE_ID
    while True:
        task_id = TASK_QUEUE.get()
        if task_id is None:
            continue
        row = _task_get(task_id)
        if not row:
            continue
        if row.get("cancel_flag"):
            _task_update(task_id, status="canceled", finished_at=now_iso())
            continue
        _task_update(task_id, status="running", started_at=now_iso())
        with TASK_LOCK:
            TASK_ACTIVE_ID = task_id
        kind = row.get("kind") or ""
        target_id = row.get("target_id")
        try:
            if kind == "embeddings_all":
                _regenerate_embeddings(None, task_id=task_id)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "embeddings_all_deferred":
                job_path = _write_startup_job(
                    "embeddings_all",
                    {"requested_at": now_iso(), "task_id": int(task_id)},
                )
                _set_embed_progress("all", {"status": "queued_restart", "done": 0, "total": 0, "errors": 0})
                _task_progress(
                    task_id,
                    "done",
                    1,
                    1,
                    0,
                    message=f"Scheduled for next restart ({job_path})",
                )
                _task_update(task_id, status="done", finished_at=now_iso(), message=f"Deferred to startup: {job_path}")
            elif kind == "embeddings_project":
                _regenerate_embeddings(int(target_id) if target_id is not None else None, task_id=task_id)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "tag_project_missing":
                _tag_project_assets(int(target_id), "missing", task_id=task_id)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "tag_project_retag":
                _tag_project_assets(int(target_id), "retag", task_id=task_id)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "tag_missing_all":
                _tag_all_projects("missing", task_id=task_id)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "name_tags_all":
                _translate_name_tags(None, task_id=task_id, only_missing=False)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "name_tags_project":
                _translate_name_tags(int(target_id) if target_id is not None else None, task_id=task_id, only_missing=False)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "name_tags_all_missing":
                _translate_name_tags(None, task_id=task_id, only_missing=True)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "name_tags_project_missing":
                _translate_name_tags(int(target_id) if target_id is not None else None, task_id=task_id, only_missing=True)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "name_tags_all_simple":
                _name_tags_simple(None, task_id=task_id, only_missing=False)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "name_tags_project_simple":
                _name_tags_simple(int(target_id) if target_id is not None else None, task_id=task_id, only_missing=False)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "name_tags_all_simple_missing":
                _name_tags_simple(None, task_id=task_id, only_missing=True)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "name_tags_project_simple_missing":
                _name_tags_simple(int(target_id) if target_id is not None else None, task_id=task_id, only_missing=True)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "tags_translate_all":
                _translate_tags_only(None, task_id=task_id, only_missing=False)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "tags_translate_project":
                _translate_tags_only(int(target_id) if target_id is not None else None, task_id=task_id, only_missing=False)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "tags_translate_all_missing":
                _translate_tags_only(None, task_id=task_id, only_missing=True)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "tags_translate_project_missing":
                _translate_tags_only(int(target_id) if target_id is not None else None, task_id=task_id, only_missing=True)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "projects_import":
                _run_projects_import_task(task_id)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "openai_recover":
                recover_limit = 200
                recover_flow: Optional[str] = None
                recover_task_id: Optional[int] = None
                recover_stale_minutes = 180
                try:
                    payload = json.loads(row.get("message") or "{}")
                    recover_limit = max(1, min(int(payload.get("limit") or recover_limit), 1000))
                    flow_raw = str(payload.get("flow") or "").strip()
                    recover_flow = flow_raw or None
                    task_raw = payload.get("openai_task_id")
                    if task_raw is not None and str(task_raw).strip() != "":
                        recover_task_id = int(task_raw)
                    recover_stale_minutes = max(10, min(int(payload.get("stale_minutes") or recover_stale_minutes), 1440))
                except Exception:
                    pass
                stats = _recover_openai_batches_once(
                    limit=recover_limit,
                    flow=recover_flow,
                    task_id=recover_task_id,
                    stale_minutes=recover_stale_minutes,
                )
                _task_progress(
                    task_id,
                    "done",
                    int(stats.get("processed_batches") or 0),
                    int(stats.get("processed_batches") or 0),
                    int(stats.get("errors") or 0),
                    message=f"recovered batches={int(stats.get('processed_batches') or 0)} done={int(stats.get('done') or 0)}",
                )
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "tags_import":
                _run_tags_import_task(task_id)
                _task_update(task_id, status="done", finished_at=now_iso())
            elif kind == "tags_clear":
                _clear_all_tags(task_id=task_id)
                _task_update(task_id, status="done", finished_at=now_iso())
            else:
                _task_update(task_id, status="error", message=f"Unknown task kind: {kind}", finished_at=now_iso())
        except Exception as exc:
            logger.error("Task %s failed: %s", task_id, exc)
            _task_update(task_id, status="error", message=str(exc), finished_at=now_iso())
        finally:
            with TASK_LOCK:
                if TASK_ACTIVE_ID == task_id:
                    TASK_ACTIVE_ID = None


def _ensure_task_worker() -> None:
    global TASK_WORKER_STARTED
    with TASK_LOCK:
        if TASK_WORKER_STARTED:
            return
        thread = threading.Thread(target=_task_worker, daemon=True)
        thread.start()
        TASK_WORKER_STARTED = True


def _queue_status_snapshot() -> Dict[str, Any]:
    with TASK_LOCK:
        active_id = TASK_ACTIVE_ID
    conn = get_db()
    counts = fetch_one(
        conn,
        """
        SELECT
            SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) AS queued,
            SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS running,
            SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) AS done,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error,
            SUM(CASE WHEN status = 'canceled' THEN 1 ELSE 0 END) AS canceled
        FROM tasks
        """,
    ) or {}
    active = None
    if active_id is not None:
        row = fetch_one(
            conn,
            "SELECT id, kind, target_id, status, created_at, started_at, message FROM tasks WHERE id = ?",
            (active_id,),
        )
        if row:
            active = dict(row)
    openai = fetch_one(
        conn,
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN processed_at IS NULL THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN processed_at IS NULL AND output_file_id IS NOT NULL THEN 1 ELSE 0 END) AS ready,
            SUM(CASE WHEN processed_at IS NULL AND status = 'in_progress' THEN 1 ELSE 0 END) AS in_progress,
            SUM(CASE WHEN processed_at IS NULL AND status = 'finalizing' THEN 1 ELSE 0 END) AS finalizing,
            SUM(CASE WHEN processed_at IS NULL AND (status IS NULL OR status IN ('validating','queued','running')) THEN 1 ELSE 0 END) AS waiting
        FROM openai_batches
        """,
    ) or {}
    conn.close()
    return {
        "server_time": now_iso(),
        "worker_active_task_id": active_id,
        "worker_busy": active_id is not None,
        "queue_buffer_size": TASK_QUEUE.qsize(),
        "tasks": {
            "queued": int(counts.get("queued") or 0),
            "running": int(counts.get("running") or 0),
            "done": int(counts.get("done") or 0),
            "error": int(counts.get("error") or 0),
            "canceled": int(counts.get("canceled") or 0),
        },
        "openai_batches": {
            "total": int(openai.get("total") or 0),
            "pending": int(openai.get("pending") or 0),
            "ready": int(openai.get("ready") or 0),
            "in_progress": int(openai.get("in_progress") or 0),
            "finalizing": int(openai.get("finalizing") or 0),
            "waiting": int(openai.get("waiting") or 0),
        },
        "active_task": active,
        "startup_import": _startup_import_snapshot(),
    }


def _event_stream():
    q: "queue.Queue[str]" = queue.Queue(maxsize=200)
    with _event_lock:
        _event_queues.append(q)
    try:
        while True:
            try:
                data = q.get(timeout=15)
            except queue.Empty:
                yield ":\n\n"
                continue
            event_name = "upload"
            try:
                payload = json.loads(data)
                event_name = str(payload.get("type") or "upload")
            except Exception:
                event_name = "upload"
            yield f"event: {event_name}\ndata: {data}\n\n"
    finally:
        with _event_lock:
            if q in _event_queues:
                _event_queues.remove(q)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=DATA_DIR), name="media")

UI_ENABLED = False
UI_DIST_DIR: Optional[Path] = None


class UploadEventPayload(BaseModel):
    batch_id: Optional[int] = None
    current: Optional[int] = None
    total: Optional[int] = None
    percent: Optional[int] = None
    name: Optional[str] = None
    source: Optional[str] = None


class ProjectCreate(BaseModel):
    name: str
    link: Optional[str] = None
    tags: Optional[List[str]] = None
    art_style: Optional[str] = None
    project_era: Optional[str] = None
    description: Optional[str] = None
    category_name: Optional[str] = None
    is_ai_generated: Optional[bool] = None
    source_path: Optional[str] = None
    source_folder: Optional[str] = None
    full_project_copy: Optional[bool] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    link: Optional[str] = None
    tags: Optional[List[str]] = None
    art_style: Optional[str] = None
    project_era: Optional[str] = None
    description: Optional[str] = None
    category_name: Optional[str] = None
    is_ai_generated: Optional[bool] = None
    source_path: Optional[str] = None
    source_folder: Optional[str] = None
    full_project_copy: Optional[bool] = None


class ProjectReimport(BaseModel):
    source_path: Optional[str] = None
    source_folder: Optional[str] = None
    full_project_copy: Optional[bool] = None


class ProjectExportCmd(BaseModel):
    game_path: Optional[str] = None


class SettingsUpdate(BaseModel):
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openrouter_base_url: Optional[str] = None
    groq_base_url: Optional[str] = None
    ollama_base_url: Optional[str] = None
    model: Optional[str] = None
    tag_model: Optional[str] = None
    translate_model: Optional[str] = None
    openai_model: Optional[str] = None
    openai_translate_model: Optional[str] = None
    openrouter_translate_model: Optional[str] = None
    groq_translate_model: Optional[str] = None
    ollama_translate_model: Optional[str] = None
    openrouter_model: Optional[str] = None
    groq_model: Optional[str] = None
    ollama_model: Optional[str] = None
    import_base_url: Optional[str] = None
    export_overwrite_zips: Optional[bool] = None
    export_default_image_count: Optional[int] = None
    export_static_mesh_image_count: Optional[int] = None
    export_skeletal_mesh_image_count: Optional[int] = None
    export_material_image_count: Optional[int] = None
    export_blueprint_image_count: Optional[int] = None
    export_niagara_image_count: Optional[int] = None
    export_anim_sequence_image_count: Optional[int] = None
    export_capture360_discard_frames: Optional[int] = None
    export_upload_after_export: Optional[bool] = None
    export_upload_path_template: Optional[str] = None
    export_check_path_template: Optional[str] = None
    skip_export_if_on_server: Optional[bool] = None
    ue_cmd_path: Optional[str] = None
    ue_cmd_extra_args: Optional[str] = None
    serve_frontend: Optional[bool] = None
    frontend_dist_path: Optional[str] = None
    asset_type_catalog: Optional[str] = None
    export_include_types: Optional[str] = None
    export_exclude_types: Optional[str] = None
    tag_language: Optional[str] = None
    tag_prompt_template: Optional[str] = None
    tag_prompt_template_openai: Optional[str] = None
    tag_prompt_template_openrouter: Optional[str] = None
    tag_prompt_template_groq: Optional[str] = None
    tag_prompt_template_ollama: Optional[str] = None
    tag_image_size: Optional[int] = None
    tag_image_quality: Optional[int] = None
    tag_include_types: Optional[str] = None
    tag_exclude_types: Optional[str] = None
    tag_use_batch_mode: Optional[bool] = None
    tag_batch_max_assets: Optional[int] = None
    tag_batch_project_concurrency: Optional[int] = None
    tag_translate_enabled: Optional[bool] = None
    tag_display_limit: Optional[int] = None
    generate_embeddings_on_import: Optional[bool] = None
    default_full_project_copy: Optional[bool] = None
    sidebar_width: Optional[int] = None
    purge_assets_on_startup: Optional[bool] = None
    use_temperature: Optional[bool] = None
    temperature: Optional[float] = None


class AssetTagUpdate(BaseModel):
    tags: List[str]

class AssetTagBulkMerge(BaseModel):
    asset_ids: List[int]
    tags: List[str]


class AssetMigrateRequest(BaseModel):
    dest_path: str
    overwrite: bool = False


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def _build_image_data_url(image_path: Path, size: int, quality: int) -> str:
    if size <= 0:
        raise ValueError("Invalid image size")
    if quality < 30 or quality > 95:
        quality = 80
    with Image.open(image_path) as img:
        copy = img.convert("RGB")
        copy.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        offset = ((size - copy.width) // 2, (size - copy.height) // 2)
        canvas.paste(copy, offset)
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


def _provider_base_url(settings: Dict[str, str], provider: str) -> str:
    provider = (provider or "").strip().lower()
    if provider == "groq":
        base = settings.get("groq_base_url") or "https://api.groq.com/openai"
    elif provider == "openrouter":
        base = settings.get("openrouter_base_url") or "https://openrouter.ai/api"
    elif provider == "ollama":
        base = settings.get("ollama_base_url") or "http://127.0.0.1:11434"
    else:
        base = settings.get("openai_base_url") or settings.get("base_url") or "https://api.openai.com"
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _provider_api_key(settings: Dict[str, str], provider: str) -> str:
    provider = (provider or "").strip().lower()
    if provider == "groq":
        return settings.get("groq_api_key") or settings.get("api_key") or ""
    if provider == "openrouter":
        return settings.get("openrouter_api_key") or settings.get("api_key") or ""
    if provider == "ollama":
        return ""
    return settings.get("openai_api_key") or settings.get("api_key") or ""


def _extract_output_text_from_response(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else None
        if choice:
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in {"text", "output_text"} and isinstance(part.get("text"), str):
                        return part.get("text") or ""
            text = choice.get("text")
            if isinstance(text, str) and text.strip():
                return text
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = payload.get("output")
    if isinstance(output, list):
        for block in output:
            if not isinstance(block, dict):
                continue
            content = block.get("content") or []
            if isinstance(content, dict):
                content = [content]
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
                    return part.get("text") or ""
    return ""


def _run_batch_tagging(
    project_id: int,
    candidates: List[Dict[str, Any]],
    settings: Dict[str, str],
    mode: str,
    image_size: int,
    image_quality: int,
    provider: str,
    task_id: Optional[int] = None,
    progress_hook: Optional[Callable[[int, int, Optional[str]], None]] = None,
    total_hook: Optional[Callable[[int], None]] = None,
) -> None:
    flow = "tag_assets"
    api_key = _provider_api_key(settings, provider)
    if not api_key and provider != "ollama":
        raise RuntimeError(f"Missing {provider} API key for batch tagging")
    base_url = _provider_base_url(settings, provider)
    api_base = base_url.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"
    model = (
        settings.get("tag_model")
        or settings.get(f"{provider}_model")
        or settings.get("openai_model")
        or settings.get("model")
        or "gpt-5-mini"
    )
    if provider == "openrouter":
        template = settings.get("tag_prompt_template_openrouter") or settings.get("tag_prompt_template") or DEFAULT_TEMPLATE
    elif provider == "groq":
        template = settings.get("tag_prompt_template_groq") or settings.get("tag_prompt_template") or DEFAULT_TEMPLATE
    elif provider == "ollama":
        template = settings.get("tag_prompt_template_ollama") or settings.get("tag_prompt_template") or DEFAULT_TEMPLATE
    else:
        template = settings.get("tag_prompt_template_openai") or settings.get("tag_prompt_template") or DEFAULT_TEMPLATE
    max_assets = _normalized_batch_size(settings.get("tag_batch_max_assets"))

    total = len(candidates)
    if total_hook is not None:
        total_hook(total)
    done = 0
    errors = 0

    id_to_row: Dict[str, Dict[str, Any]] = {str(row["id"]): row for row in candidates}
    requests: List[Dict[str, Any]] = []

    for row in candidates:
        existing = [] if mode == "retag" else json.loads(row.get("tags_json") or "[]")
        prompt = render_template(
            template,
            row.get("name") or "",
            row.get("description") or "",
            existing,
            settings.get("tag_language") or "",
            row.get("type") or "",
        )
        prompt = (
            f"{prompt}\n\nReturn only JSON exactly like "
            f'{{"tags":["tag1","tag2"],"era":"medieval"}} with at most 10 tags. No extra text.'
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_data_url = None
        image_path = _pick_asset_image(row)
        if image_path and image_path.exists():
            image_data_url = _build_image_data_url(image_path, image_size, image_quality)
        if image_data_url:
            content.append({"type": "image_url", "image_url": {"url": image_data_url}})
        system_prompt = (
            "You are a tagging engine. Respond with JSON only. "
            "No reasoning, no markdown, no extra text."
        )
        body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": 400,
        }
        if provider == "openai":
            body["response_format"] = {"type": "json_object"}
        if _bool_from_setting(settings.get("use_temperature")):
            temp = settings.get("temperature")
            try:
                body["temperature"] = float(temp) if temp is not None else 1.0
            except (TypeError, ValueError):
                body["temperature"] = 1.0
        requests.append(
            {
                "custom_id": str(row["id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client_timeout = httpx.Timeout(30.0, connect=30.0, read=30.0)

    submitted: List[Dict[str, Any]] = []
    existing_pending = _openai_list_pending_batches(flow, project_id)
    if existing_pending:
        offset_seed = 0
        for row in existing_pending:
            req_count = int(row.get("request_total") or 0)
            submitted.append(
                {
                    "batch_id": str(row.get("batch_id") or ""),
                    "chunk": [{}] * max(0, req_count),
                    "offset": offset_seed,
                }
            )
            offset_seed += max(0, req_count)
        msg = f"Resuming {len(submitted)} existing OpenAI batches; no new submit"
        logger.info(msg)
        if task_id is not None:
            _task_progress(task_id, "running", total, done, errors, message=msg)
        if progress_hook is not None:
            progress_hook(0, 0, msg)
    else:
        for offset in range(0, len(requests), max_assets):
            chunk = requests[offset : offset + max_assets]
            if task_id is not None and _task_cancelled(task_id):
                _set_tag_progress(project_id, {"status": "canceled", "total": total, "done": done, "errors": errors})
                _task_progress(task_id, "canceled", total, done, errors)
                return

            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl", encoding="utf-8") as tmp:
                for entry in chunk:
                    tmp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                tmp_path = Path(tmp.name)

            try:
                with httpx.Client(timeout=client_timeout) as client:
                    def _upload_batch_file():
                        with tmp_path.open("rb") as fp:
                            return client.post(
                                f"{api_base}/files",
                                data={"purpose": "batch"},
                                files={"file": (tmp_path.name, fp, "application/jsonl")},
                                headers=headers,
                            )
                    upload = _retry_http(_upload_batch_file)
                    upload.raise_for_status()
                    file_id = upload.json().get("id")
                    if not file_id:
                        raise RuntimeError("OpenAI batch upload missing file id")

                    def _create_batch():
                        return client.post(
                            f"{api_base}/batches",
                            json={
                                "input_file_id": file_id,
                                "endpoint": "/v1/chat/completions",
                                "completion_window": "24h",
                            },
                            headers=headers,
                        )
                    batch = _retry_http(_create_batch)
                    batch.raise_for_status()
                    batch_id = batch.json().get("id")
                    if not batch_id:
                        raise RuntimeError("OpenAI batch create missing batch id")
                    logger.info("OpenAI batch submitted id=%s requests=%s", batch_id, len(chunk))
                    _openai_batch_upsert(
                        flow=flow,
                        batch_id=str(batch_id),
                        provider=provider,
                        task_id=task_id,
                        project_id=project_id,
                        request_total=len(chunk),
                        status="submitted",
                    )
                    submitted.append({"batch_id": batch_id, "chunk": chunk, "offset": offset})
                    if task_id is not None:
                        _task_progress(
                            task_id,
                            "running",
                            total,
                            done,
                            errors,
                            message=f"OpenAI batch submitted ({len(chunk)} requests): {batch_id}",
                        )
                    if progress_hook is not None:
                        progress_hook(0, 0, f"OpenAI batch submitted: {batch_id}")
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    if submitted:
        if task_id is not None:
            _task_progress(
                task_id,
                "running",
                total,
                done,
                errors,
                message=f"OpenAI batches submitted: {len(submitted)}",
            )
        if progress_hook is not None:
            progress_hook(0, 0, f"OpenAI batches submitted: {len(submitted)}")

    pending: Dict[str, Dict[str, Any]] = {
        str(info["batch_id"]): {**info, "last_counts": None, "last_status": None, "last_output_ready": False}
        for info in submitted
    }
    last_queue_msg: Optional[str] = None
    failed_batches: List[str] = []
    with httpx.Client(timeout=client_timeout) as client:
        while pending:
            if task_id is not None and _task_cancelled(task_id):
                _set_tag_progress(project_id, {"status": "canceled", "total": total, "done": done, "errors": errors})
                _task_progress(task_id, "canceled", total, done, errors)
                return
            progressed = False
            for batch_id in list(pending.keys()):
                info = pending[batch_id]
                chunk = info["chunk"]
                offset = int(info.get("offset") or 0)
                try:
                    poll = client.get(f"{api_base}/batches/{batch_id}", headers=headers)
                    poll.raise_for_status()
                    payload = poll.json()
                except httpx.HTTPError as exc:
                    status_code = getattr(exc.response, "status_code", None)
                    if status_code in {502, 503, 504}:
                        logger.warning("OpenAI batch poll retryable error id=%s status=%s", batch_id, status_code)
                        continue
                    raise

                status = payload.get("status") or ""
                output_file_id = payload.get("output_file_id")
                info["last_output_ready"] = bool(output_file_id)
                counts = payload.get("request_counts") or {}
                completed = int(counts.get("completed") or 0)
                failed = int(counts.get("failed") or 0)
                total_batch = int(counts.get("total") or len(chunk))
                progress_done = min(total, offset + completed)
                progress_errors = errors + failed
                counts_tuple = (completed, total_batch, failed)
                if counts_tuple != info.get("last_counts"):
                    msg = f"OpenAI batch {status}: {completed}/{total_batch} (failed {failed})"
                    _set_tag_progress(
                        project_id,
                        {
                            "status": "running",
                            "total": total,
                            "done": progress_done,
                            "errors": progress_errors,
                            "message": msg,
                        },
                    )
                    if task_id is not None:
                        _task_progress(task_id, "running", total, progress_done, progress_errors, message=msg)
                    if progress_hook is not None:
                        progress_hook(0, 0, msg)
                    info["last_counts"] = counts_tuple
                if status and status != info.get("last_status"):
                    logger.info("OpenAI batch status id=%s status=%s", batch_id, status)
                    info["last_status"] = status

                _openai_batch_upsert(
                    flow=flow,
                    batch_id=batch_id,
                    provider=provider,
                    task_id=task_id,
                    project_id=project_id,
                    request_total=len(chunk),
                    status=status,
                    output_file_id=output_file_id,
                )

                is_failure_terminal = status in {"failed", "expired", "cancelled"}
                can_process_output = bool(output_file_id)
                if not is_failure_terminal and not can_process_output:
                    continue

                if is_failure_terminal:
                    pending.pop(batch_id, None)
                    progressed = True
                    _openai_batch_upsert(
                        flow=flow,
                        batch_id=batch_id,
                        provider=provider,
                        task_id=task_id,
                        project_id=project_id,
                        request_total=len(chunk),
                        status=status,
                        error_text=f"terminal status={status}",
                    )
                    _openai_batch_mark_processed(batch_id)
                    failed_batches.append(f"{batch_id}:{status}")
                    continue

                try:
                    if task_id is not None:
                        _task_progress(
                            task_id,
                            "running",
                            total,
                            progress_done,
                            progress_errors,
                            message=f"OpenAI batch completed; downloading output: {batch_id}",
                        )
                    if progress_hook is not None:
                        progress_hook(0, 0, f"OpenAI batch completed; downloading output: {batch_id}")
                    result = client.get(f"{api_base}/files/{output_file_id}/content", headers=headers)
                    result.raise_for_status()
                    output_text = result.text
                    archive_path = _archive_batch_output(
                        flow=flow,
                        provider=provider,
                        batch_id=batch_id,
                        output_text=output_text,
                        task_id=task_id,
                        project_id=project_id,
                    )
                    if project_id is None:
                        done = min(total, done + len(chunk))
                        pending.pop(batch_id, None)
                        progressed = True
                        _openai_batch_mark_processed(batch_id)
                        if task_id is not None:
                            _task_progress(
                                task_id,
                                "running",
                                total,
                                done,
                                errors,
                                message=f"Archived batch output: {Path(archive_path).name}",
                            )
                        if progress_hook is not None:
                            progress_hook(0, 0, f"Archived batch output: {Path(archive_path).name}")
                        continue

                    batch_rows: List[Dict[str, Any]] = []
                    seen_ids: set[str] = set()
                    if task_id is not None:
                        _task_progress(
                            task_id,
                            "running",
                            total,
                            progress_done,
                            progress_errors,
                            message=f"Applying OpenAI batch output: {batch_id}",
                        )
                    if progress_hook is not None:
                        progress_hook(0, 0, f"Applying OpenAI batch output: {batch_id}")
                    for line in output_text.splitlines():
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        custom_id = str(item.get("custom_id") or "")
                        if not custom_id:
                            continue
                        seen_ids.add(custom_id)
                        if item.get("error"):
                            errors += 1
                            done += 1
                            if progress_hook is not None:
                                progress_hook(1, 1, None)
                            continue
                        resp = item.get("response") or {}
                        body = resp.get("body") if isinstance(resp, dict) else None
                        if isinstance(body, str):
                            try:
                                body = json.loads(body)
                            except Exception:
                                body = None
                        payload = body if isinstance(body, dict) else resp
                        text = _extract_output_text_from_response(payload)
                        tags_raw, era = _extract_tags_and_era(text)
                        tags = _normalize_tags(tags_raw)
                        row = id_to_row.get(custom_id)
                        if not row:
                            errors += 1
                            done += 1
                            if progress_hook is not None:
                                progress_hook(1, 1, None)
                            continue
                        _maybe_set_project_era(project_id, era)
                        if not tags:
                            errors += 1
                            done += 1
                            if progress_hook is not None:
                                progress_hook(1, 1, None)
                            continue
                        translated_tags = _translate_tags_if_enabled(settings, tags)
                        embedding = None
                        batch_rows.append(
                            {
                                "id": row["id"],
                                "tags": tags,
                                "translated_tags": translated_tags,
                                "embedding": embedding,
                                "hash_main": row.get("hash_main_blake3") or "",
                                "hash_full": row.get("hash_full_blake3") or "",
                                "created_at": row.get("created_at") or now_iso(),
                                "mark_tags_done": True,
                            }
                        )
                        done += 1
                        if progress_hook is not None:
                            progress_hook(1, 0, None)
                        if done % 5 == 0 or done == total:
                            _set_tag_progress(project_id, {"status": "running", "total": total, "done": done, "errors": errors})
                            if task_id is not None:
                                _task_progress(
                                    task_id,
                                    "running",
                                    total,
                                    done,
                                    errors,
                                    message=f"Applying OpenAI batch output: {batch_id}",
                                )

                    missing = len(chunk) - len(seen_ids)
                    if missing:
                        errors += missing
                        done += missing
                        if progress_hook is not None:
                            progress_hook(missing, missing, None)
                    if batch_rows:
                        if task_id is not None:
                            _task_progress(
                                task_id,
                                "running",
                                total,
                                done,
                                errors,
                                message=f"Writing tag updates: {batch_id}",
                            )
                    if progress_hook is not None:
                        progress_hook(0, 0, f"Writing tag updates: {batch_id}")
                        _flush_tag_batch_chunked(batch_rows, settings)

                    pending.pop(batch_id, None)
                    progressed = True
                    _openai_batch_mark_processed(batch_id)
                except Exception as exc:
                    _openai_batch_upsert(
                        flow=flow,
                        batch_id=batch_id,
                        provider=provider,
                        task_id=task_id,
                        project_id=project_id,
                        request_total=len(chunk),
                        status=status,
                        output_file_id=output_file_id,
                        error_text=f"processing error: {exc}",
                    )
                    logger.warning("OpenAI tag batch processing error id=%s err=%s", batch_id, exc)
                    continue

            if pending and not progressed:
                snap = _openai_pending_snapshot(pending)
                queue_msg = (
                    f"OpenAI queue: ready {snap['ready']} | in_progress {snap['in_progress']} | "
                    f"finalizing {snap['finalizing']} | waiting {snap['waiting']} | pending {snap['pending']}"
                )
                if queue_msg != last_queue_msg:
                    last_queue_msg = queue_msg
                    if task_id is not None:
                        _task_progress(task_id, "running", total, done, errors, message=queue_msg)
                    if progress_hook is not None:
                        progress_hook(0, 0, queue_msg)
                time.sleep(5)

    if failed_batches:
        raise RuntimeError(f"OpenAI batches failed: {', '.join(failed_batches[:10])}")

    _set_tag_progress(project_id, {"status": "done", "total": total, "done": done, "errors": errors})
    if task_id is not None:
        _task_progress(task_id, "done", total, done, errors)


def _run_batch_translate_names(
    rows: List[Dict[str, Any]],
    settings: Dict[str, str],
    language: str,
    task_id: Optional[int] = None,
    project_id: Optional[int] = None,
) -> None:
    flow = "translate_name_tags"
    provider = (settings.get("provider") or "").strip().lower()
    api_key = settings.get("api_key") or ""
    if provider == "openai":
        api_key = settings.get("openai_api_key") or api_key
    elif provider == "openrouter":
        api_key = settings.get("openrouter_api_key") or api_key
    elif provider == "groq":
        api_key = settings.get("groq_api_key") or api_key

    base_url = settings.get(f"{provider}_base_url") or settings.get("base_url") or ""
    api_base = base_url.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"
    model = (
        settings.get(f"{provider}_translate_model")
        or settings.get("translate_model")
        or settings.get(f"{provider}_model")
        or settings.get("model")
        or "gpt-5-mini"
    )
    if not base_url or not model:
        raise RuntimeError("LLM settings are incomplete")
    if not language:
        raise RuntimeError("Translation language is required")

    total = len(rows)
    done = 0
    errors = 0

    id_to_row: Dict[str, Dict[str, Any]] = {str(row["id"]): row for row in rows}
    requests: List[Dict[str, Any]] = []
    for row in rows:
        base_name = _clean_asset_name(row.get("name") or "")
        if not base_name:
            continue
        tokens = [t for t in re.split(r"[\s\-]+", base_name) if t]
        combined = []
        for i in range(len(tokens) - 1):
            combined.append(f"{tokens[i]}_{tokens[i+1]}")
        llm_tokens = tokens + combined
        tag_text = ", ".join([str(t).strip() for t in llm_tokens if str(t).strip()])
        if not tag_text:
            continue
        prompt = TRANSLATE_TEMPLATE.replace("{language}", language).replace("{tags}", tag_text)
        if "json" not in prompt.lower():
            prompt = f"{prompt}\n\nReturn ONLY a JSON object with a 'tags' array of strings."
        body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You translate tags for 3D/asset libraries."},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": 400,
        }
        if provider == "openai":
            body["response_format"] = {"type": "json_object"}
        if _bool_from_setting(settings.get("use_temperature")):
            temp = settings.get("temperature")
            try:
                body["temperature"] = float(temp) if temp is not None else 1.0
            except (TypeError, ValueError):
                body["temperature"] = 1.0
        requests.append(
            {
                "custom_id": str(row["id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client_timeout = httpx.Timeout(30.0, connect=30.0, read=30.0)

    submitted: List[Dict[str, Any]] = []
    max_assets = _normalized_batch_size(settings.get("tag_batch_max_assets"))
    chunks_total = max(1, (len(requests) + max_assets - 1) // max_assets)
    existing_pending = _openai_list_pending_batches(flow, project_id)
    if existing_pending:
        for idx, row in enumerate(existing_pending, start=1):
            req_count = int(row.get("request_total") or 0)
            submitted.append(
                {
                    "batch_id": str(row.get("batch_id") or ""),
                    "chunk": [{}] * max(0, req_count),
                    "offset": 0,
                    "chunk_idx": idx,
                    "chunks_total": len(existing_pending),
                }
            )
        if task_id is not None:
            _task_progress(task_id, "running", total, done, errors, message=f"Translate name-tags: resume {len(submitted)} existing batches")
    else:
        for chunk_idx, offset in enumerate(range(0, len(requests), max_assets), start=1):
            chunk = requests[offset : offset + max_assets]
            if task_id is not None and _task_cancelled(task_id):
                _task_progress(task_id, "canceled", total, done, errors)
                return
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl", encoding="utf-8") as tmp:
                for entry in chunk:
                    tmp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                tmp_path = Path(tmp.name)
            try:
                with httpx.Client(timeout=client_timeout) as client:
                    def _upload_batch_file():
                        with tmp_path.open("rb") as fp:
                            return client.post(
                                f"{api_base}/files",
                                data={"purpose": "batch"},
                                files={"file": (tmp_path.name, fp, "application/jsonl")},
                                headers=headers,
                            )
                    upload = _retry_http(_upload_batch_file)
                    upload.raise_for_status()
                    file_id = upload.json().get("id")
                    if not file_id:
                        raise RuntimeError("OpenAI batch upload missing file id")
                    def _create_batch():
                        return client.post(
                            f"{api_base}/batches",
                            json={
                                "input_file_id": file_id,
                                "endpoint": "/v1/chat/completions",
                                "completion_window": "24h",
                            },
                            headers=headers,
                        )
                    batch = _retry_http(_create_batch)
                    batch.raise_for_status()
                    batch_id = batch.json().get("id")
                    if not batch_id:
                        raise RuntimeError("OpenAI batch create missing batch id")
                    logger.info("OpenAI name-translate batch submitted id=%s requests=%s", batch_id, len(chunk))
                    _openai_batch_upsert(
                        flow=flow,
                        batch_id=str(batch_id),
                        provider=provider,
                        task_id=task_id,
                        project_id=project_id,
                        request_total=len(chunk),
                        status="submitted",
                    )
                    submitted.append({"batch_id": batch_id, "chunk": chunk, "offset": offset, "chunk_idx": chunk_idx, "chunks_total": chunks_total})
                    if task_id is not None:
                        _task_progress(
                            task_id,
                            "running",
                            total,
                            done,
                            errors,
                            message=f"Translate name-tags: submitted chunk {chunk_idx}/{chunks_total} ({len(chunk)} req)",
                        )
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    report_step = done // 100
    stall_seconds = 45 * 60
    now_monotonic = time.monotonic()
    process_owner = f"task:{task_id or 0}:{uuid.uuid4().hex}"
    pending: Dict[str, Dict[str, Any]] = {
        str(info["batch_id"]): {
            **info,
            "last_counts": None,
            "last_status": None,
            "last_output_ready": False,
            "last_change_mono": now_monotonic,
        }
        for info in submitted
    }
    last_queue_msg: Optional[str] = None
    failed_batches: List[str] = []
    with httpx.Client(timeout=client_timeout) as client:
        while pending:
            if task_id is not None and _task_cancelled(task_id):
                _task_progress(task_id, "canceled", total, done, errors)
                return
            progressed = False
            for batch_id in list(pending.keys()):
                info = pending[batch_id]
                chunk = info["chunk"]
                chunk_idx = int(info.get("chunk_idx") or 0)
                chunks_total = int(info.get("chunks_total") or len(submitted) or 1)
                try:
                    poll = client.get(f"{api_base}/batches/{batch_id}", headers=headers)
                    poll.raise_for_status()
                    payload = poll.json()
                except httpx.HTTPError as exc:
                    status_code = getattr(exc.response, "status_code", None)
                    if status_code in {502, 503, 504}:
                        logger.warning(
                            "OpenAI translate-name poll retryable error id=%s status=%s",
                            batch_id,
                            status_code,
                        )
                        continue
                    logger.warning("OpenAI translate-name poll error id=%s err=%s", batch_id, exc)
                    continue
                status = payload.get("status") or ""
                output_file_id = payload.get("output_file_id")
                info["last_output_ready"] = bool(output_file_id)
                counts = payload.get("request_counts") or {}
                completed = int(counts.get("completed") or 0)
                failed = int(counts.get("failed") or 0)
                total_batch = int(counts.get("total") or len(chunk))
                counts_tuple = (completed, total_batch, failed)
                prev_counts = info.get("last_counts")
                prev_status = info.get("last_status")
                prev_output_ready = bool(info.get("last_output_ready"))
                if counts_tuple != info.get("last_counts") and task_id is not None:
                    msg = f"Translate name-tags chunk {chunk_idx}/{chunks_total}: {status} {completed}/{total_batch} (failed {failed})"
                    _task_progress(task_id, "running", total, done, errors, message=msg)
                    info["last_counts"] = counts_tuple
                if status and status != info.get("last_status"):
                    logger.info("OpenAI translate-name batch status id=%s status=%s", batch_id, status)
                    info["last_status"] = status
                if (
                    counts_tuple != prev_counts
                    or status != prev_status
                    or bool(output_file_id) != prev_output_ready
                ):
                    info["last_change_mono"] = time.monotonic()

                _openai_batch_upsert(
                    flow=flow,
                    batch_id=batch_id,
                    provider=provider,
                    task_id=task_id,
                    project_id=project_id,
                    request_total=len(chunk),
                    status=status,
                    output_file_id=output_file_id,
                )

                is_failure_terminal = status in {"failed", "expired", "cancelled"}
                can_process_output = bool(output_file_id)
                if not is_failure_terminal and not can_process_output:
                    idle_for = time.monotonic() - float(info.get("last_change_mono") or 0.0)
                    if idle_for >= stall_seconds:
                        stalled_msg = f"stalled batch timeout ({int(idle_for)}s) status={status or 'unknown'}"
                        _openai_batch_upsert(
                            flow=flow,
                            batch_id=batch_id,
                            provider=provider,
                            task_id=task_id,
                            project_id=project_id,
                            request_total=len(chunk),
                            status=status or "stale",
                            error_text=stalled_msg,
                        )
                        _openai_batch_mark_processed(batch_id)
                        pending.pop(batch_id, None)
                        progressed = True
                        failed_batches.append(f"{batch_id}:stalled")
                        if task_id is not None:
                            _task_progress(task_id, "running", total, done, errors, message=f"Translate name-tags chunk {chunk_idx}/{chunks_total}: {stalled_msg}")
                        continue
                if not is_failure_terminal and not can_process_output:
                    continue

                if not _openai_batch_claim(batch_id, process_owner):
                    continue

                if is_failure_terminal:
                    pending.pop(batch_id, None)
                    progressed = True
                    _openai_batch_upsert(
                        flow=flow,
                        batch_id=batch_id,
                        provider=provider,
                        task_id=task_id,
                        project_id=project_id,
                        request_total=len(chunk),
                        status=status,
                        error_text=f"terminal status={status}",
                    )
                    _openai_batch_mark_processed(batch_id)
                    _openai_batch_release(batch_id, process_owner)
                    failed_batches.append(f"{batch_id}:{status}")
                    continue

                if _openai_batch_is_applied(batch_id):
                    _openai_batch_mark_processed(batch_id)
                    _openai_batch_release(batch_id, process_owner)
                    pending.pop(batch_id, None)
                    progressed = True
                    continue

                if task_id is not None:
                    _task_progress(task_id, "running", total, done, errors, message=f"Translate name-tags chunk {chunk_idx}/{chunks_total}: downloading output")
                _openai_batch_heartbeat(batch_id, process_owner)
                result = client.get(f"{api_base}/files/{output_file_id}/content", headers=headers)
                result.raise_for_status()
                output_text = result.text
                archive_path = _archive_batch_output(
                    flow=flow,
                    provider=provider,
                    batch_id=batch_id,
                    output_text=output_text,
                    task_id=task_id,
                    project_id=project_id,
                )
                if project_id is None:
                    done = min(total, done + len(chunk))
                    _openai_batch_mark_applied(
                        batch_id=batch_id,
                        flow=flow,
                        task_id=task_id,
                        rows_done=len(chunk),
                        rows_error=0,
                    )
                    pending.pop(batch_id, None)
                    progressed = True
                    _openai_batch_mark_processed(batch_id)
                    _openai_batch_release(batch_id, process_owner)
                    if task_id is not None:
                        _task_progress(
                            task_id,
                            "running",
                            total,
                            done,
                            errors,
                            message=f"Archived batch output: {Path(archive_path).name}",
                        )
                    continue

                batch_rows: List[Dict[str, Any]] = []
                seen_ids: set[str] = set()
                for line in output_text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    custom_id = str(item.get("custom_id") or "")
                    if not custom_id:
                        continue
                    seen_ids.add(custom_id)
                    if item.get("error"):
                        errors += 1
                        done += 1
                        continue
                    resp = item.get("response") or {}
                    body = resp.get("body") if isinstance(resp, dict) else None
                    if isinstance(body, str):
                        try:
                            body = json.loads(body)
                        except Exception:
                            body = None
                    payload = body if isinstance(body, dict) else resp
                    text = _extract_output_text_from_response(payload)
                    translated_raw = _extract_tags_from_content(text)
                    translated = _normalize_tags(translated_raw)
                    row = id_to_row.get(custom_id)
                    if not row:
                        errors += 1
                        done += 1
                        continue
                    if not translated:
                        errors += 1
                        done += 1
                        continue
                    existing_tags = json.loads(row.get("tags_json") or "[]")
                    existing_translated = json.loads(row.get("tags_translated_json") or "[]")
                    merged_tags = _normalize_tags(existing_tags + translated)
                    merged_translated = _normalize_tags(existing_translated + translated)
                    batch_rows.append(
                        {
                            "id": row["id"],
                            "tags": merged_tags,
                            "translated_tags": merged_translated,
                            "embedding": None,
                            "hash_main": row.get("hash_main_blake3") or "",
                            "hash_full": row.get("hash_full_blake3") or "",
                            "created_at": row.get("created_at") or now_iso(),
                            "mark_name_tags_done": True,
                        }
                    )
                    done += 1
                    if task_id is not None:
                        step = done // 100
                        if done == total or step != report_step:
                            report_step = step
                            _task_progress(
                                task_id,
                                "running",
                                total,
                                done,
                                errors,
                                message=f"Translate name-tags: {done}/{total} • chunk {chunk_idx}/{chunks_total}",
                            )

                missing = len(chunk) - len(seen_ids)
                if missing:
                    errors += missing
                    done += missing
                    if task_id is not None:
                        step = done // 100
                        if done == total or step != report_step:
                            report_step = step
                            _task_progress(
                                task_id,
                                "running",
                                total,
                                done,
                                errors,
                                message=f"Translate name-tags: {done}/{total} • chunk {chunk_idx}/{chunks_total}",
                            )
                if batch_rows:
                    _flush_tag_batch_chunked(batch_rows, settings)
                _openai_batch_mark_applied(
                    batch_id=batch_id,
                    flow=flow,
                    task_id=task_id,
                    rows_done=max(0, len(seen_ids) - max(0, missing)),
                    rows_error=max(0, missing),
                )
                pending.pop(batch_id, None)
                progressed = True
                _openai_batch_mark_processed(batch_id)
                _openai_batch_release(batch_id, process_owner)

            if pending and not progressed:
                snap = _openai_pending_snapshot(pending)
                queue_msg = (
                    f"OpenAI queue: ready {snap['ready']} | in_progress {snap['in_progress']} | "
                    f"finalizing {snap['finalizing']} | waiting {snap['waiting']} | pending {snap['pending']}"
                )
                if queue_msg != last_queue_msg and task_id is not None:
                    last_queue_msg = queue_msg
                    _task_progress(task_id, "running", total, done, errors, message=queue_msg)
                time.sleep(5)

    if failed_batches:
        raise RuntimeError(f"Translate batches failed: {', '.join(failed_batches[:10])}")

    if task_id is not None:
        _task_progress(task_id, "done", total, done, errors)

def slugify(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    value = value.lower().strip().replace(" ", "-")
    return "".join([c for c in value if c in allowed]) or "project"


def ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STARTUP_JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _set_copy_progress(project_id: int, data: Dict[str, Any]) -> None:
    with COPY_LOCK:
        COPY_PROGRESS[project_id] = data


def _set_migrate_progress(asset_id: int, data: Dict[str, Any]) -> None:
    with MIGRATE_LOCK:
        MIGRATE_PROGRESS[asset_id] = data


def _set_tag_progress(project_id: int, data: Dict[str, Any]) -> None:
    with TAG_LOCK:
        TAG_PROGRESS[project_id] = data


def _normalize_era(value: str) -> str:
    return str(value or "").strip().lower()


def _maybe_set_project_era(project_id: int, era: str) -> None:
    era_value = _normalize_era(era)
    if not era_value:
        return
    with ERA_LOCK:
        if project_id not in ERA_PENDING:
            ERA_PENDING[project_id] = era_value


def _flush_project_eras(conn: sqlite3.Connection, batch_size: int = 900) -> None:
    with ERA_LOCK:
        if not ERA_PENDING:
            return
        items = list(ERA_PENDING.items())
        ERA_PENDING.clear()
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        ids = [pid for pid, _ in chunk]
        params: List[Any] = list(ids)
        where = f"WHERE id IN ({','.join(['?'] * len(ids))})"
        existing_rows = fetch_all(conn, f"SELECT id, project_era FROM projects {where}", params)
        existing_map = {row["id"]: (row.get("project_era") or "").strip() for row in existing_rows}
        updates = []
        for pid, era_value in chunk:
            if existing_map.get(pid):
                continue
            updates.append((era_value, pid))
        if updates:
            conn.executemany("UPDATE projects SET project_era = ? WHERE id = ?", updates)
            _db_retry(conn.commit)


def _set_embed_progress(key: str, data: Dict[str, Any]) -> None:
    with EMBED_LOCK:
        EMBED_PROGRESS[key] = data


def _pick_asset_image(row: Dict[str, Any]) -> Optional[Path]:
    rel = (
        row.get("detail_image")
        or row.get("full_image")
        or row.get("thumb_image")
        or row.get("anim_detail")
        or row.get("anim_thumb")
    )
    if not rel:
        return None
    return ASSETS_DIR / row["asset_dir"] / rel




def _generate_project_setcard(project: Dict[str, Any], width: int = 1920, height: int = 1080) -> Optional[str]:
    try:
        project_id = project["id"]
        conn = get_db()
        rows = fetch_all(
            conn,
            "SELECT name, type, asset_dir, detail_image, full_image, thumb_image, anim_detail, anim_thumb, meta_json "
            "FROM assets WHERE project_id = ?",
            (project_id,),
        )
        conn.close()

        images: List[Path] = []
        type_counts: Dict[str, int] = {}
        for row in rows:
            asset_type = (row.get("type") or "").strip()
            if asset_type:
                type_counts[asset_type] = type_counts.get(asset_type, 0) + 1
            rel = (
                row.get("detail_image")
                or row.get("full_image")
                or row.get("thumb_image")
                or row.get("anim_detail")
                or row.get("anim_thumb")
            )
            if not rel:
                meta_raw = row.get("meta_json") or "{}"
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    meta = {}
                preview_files = meta.get("preview_files") or meta.get("frames") or []
                if isinstance(preview_files, list) and preview_files:
                    # frames might be list of dicts
                    if isinstance(preview_files[0], dict):
                        rel = preview_files[0].get("file")
                    else:
                        rel = preview_files[0]
            if not rel:
                continue
            img_path = ASSETS_DIR / row["asset_dir"] / rel
            if img_path.exists():
                images.append(img_path)

        if not images:
            return None

        canvas = Image.new("RGB", (width, height), (16, 16, 18))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        # tight grid over full canvas
        gap = 2
        count = len(images)
        cols = max(1, math.ceil(math.sqrt(count * width / height)))
        rows_grid = max(1, math.ceil(count / cols))
        cell_w = max(1, (width - (cols - 1) * gap) // cols)
        cell_h = max(1, (height - (rows_grid - 1) * gap) // rows_grid)

        for idx, img_path in enumerate(images):
            r = idx // cols
            c = idx % cols
            if r >= rows_grid:
                break
            x0 = c * (cell_w + gap)
            y0 = r * (cell_h + gap)
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((x1 - x0, y1 - y0), Image.LANCZOS)
                    tile = Image.new("RGB", (x1 - x0, y1 - y0), (24, 24, 26))
                    tx = (tile.width - im.width) // 2
                    ty = (tile.height - im.height) // 2
                    tile.paste(im, (tx, ty))
                    canvas.paste(tile, (x0, y0))
            except Exception:
                continue

        # optional small title overlay (top-left)
        title = project.get("name") or "Project"
        draw.text((8, 8), title, fill=(240, 240, 240), font=font)

        # overlay logo (bottom-right)
        logo_path = BASE_DIR / "frontend" / "src" / "assets" / "logo64.png"
        if logo_path.exists():
            try:
                with Image.open(logo_path) as logo:
                    logo = logo.convert("RGBA")
                    size = 64
                    logo = logo.resize((size, size), Image.LANCZOS)
                    x = width - size - 12
                    y = height - size - 12
                    canvas.paste(logo, (x, y), logo)
            except Exception:
                pass

        folder_path = Path(project.get("folder_path") or "")
        if not folder_path:
            return None
        out_path = folder_path / "setcard.png"
        canvas.save(out_path, "PNG")
        return str(out_path)
    except Exception as exc:
        logger.error("Setcard generation failed: %s", exc)
        return None



def _queue_preview_generation(project_id: int) -> None:
    try:
        conn = get_db()
        rows = fetch_all(
            conn,
            "SELECT id, asset_dir FROM assets WHERE project_id = ?",
            (project_id,),
        )
        conn.close()
        for row in rows:
            asset_dir = ASSETS_DIR / row["asset_dir"]
            if not any((asset_dir / name).exists() for name in ["0.webp", "1.webp", "thumb.webp"]):
                marker = asset_dir / "_needs_preview"
                try:
                    marker.write_text("1", encoding="utf-8")
                except Exception:
                    pass
    except Exception as exc:
        logger.error("Queue preview generation failed: %s", exc)
def _parse_type_filter(value: Optional[str]) -> List[str]:
    if not value:
        return []
    tokens = [t.strip().lower() for t in value.replace("|", ",").replace(";", ",").split(",")]
    return [t for t in tokens if t]


def _clean_asset_name(name: str) -> str:
    value = (name or "").strip()
    if not value:
        return ""
    prefixes = ["SM_", "SK_", "MI_", "M_", "BP_", "ABP_", "ANIM_", "MF_", "T_", "NS_", "S_", "P_", "FX_"]
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    value = value.replace("_", " ").strip()
    return value


def _translate_name_tags(
    project_id: Optional[int],
    task_id: Optional[int] = None,
    only_missing: bool = False,
) -> None:
    conn = get_db()
    params = []
    where = ""
    if project_id is not None:
        where = "WHERE a.project_id = ?"
        params.append(project_id)
    rows = fetch_all(
        conn,
        "SELECT a.id, a.name, a.description, a.tags_json, a.hash_main_blake3, a.hash_full_blake3, a.created_at, "
        "t.tags_translated_json, t.name_translate_tags_done_at "
        "FROM assets a LEFT JOIN asset_tags t ON t.hash_full_blake3 = a.hash_full_blake3 "
        f"{where}",
        params,
    )
    conn.close()
    settings_conn = get_db()
    settings = get_settings(settings_conn)
    settings_conn.close()
    if only_missing:
        rows = [row for row in rows if not row.get("name_translate_tags_done_at")]
    total = len(rows)
    done = 0
    errors = 0
    batch: List[Dict[str, Any]] = []
    batch_size = min(_normalized_batch_size(settings.get("tag_batch_max_assets")), 2000)
    use_batch_mode = _bool_from_setting(settings.get("tag_use_batch_mode"))
    provider = (settings.get("provider") or "").strip().lower()
    language = (settings.get("tag_language") or "").strip()
    if use_batch_mode and provider in {"openai", "groq"}:
        _run_batch_translate_names(rows, settings, language, task_id=task_id, project_id=project_id)
        return
    for row in rows:
        if task_id is not None and _task_cancelled(task_id):
            _task_progress(task_id, "canceled", total, done, errors)
            return
        try:
            base_name = _clean_asset_name(row.get("name") or "")
            if not base_name:
                done += 1
                continue
            existing_tags = json.loads(row.get("tags_json") or "[]")
            existing_translated = json.loads(row.get("tags_translated_json") or "[]")
            model = settings.get(f"{provider}_translate_model") or settings.get(f"{provider}_model") or settings.get("model") or ""
            base_url = settings.get(f"{provider}_base_url") or settings.get("base_url") or ""
            logger.info("Name->tags LLM request: name=%s provider=%s model=%s base_url=%s language=%s", base_name, provider, model, base_url, language)
            tokens = [t for t in re.split(r"[\s\-]+", base_name) if t]
            combined = []
            for i in range(len(tokens) - 1):
                combined.append(f"{tokens[i]}_{tokens[i+1]}")
            llm_tokens = tokens + combined
            logger.info("Name->tags LLM tokens: %s", llm_tokens)
            debug = translate_tags_debug(settings, llm_tokens, settings.get("tag_language") or "")
            translated = _normalize_tags(debug.get("tags") or [])
            logger.info("Name->tags LLM prompt: %s", json.dumps({"prompt": debug.get("prompt")}, ensure_ascii=False))
            logger.info("Name->tags LLM response json: %s", json.dumps(debug.get("response_json"), ensure_ascii=False))
            logger.info("Name->tags LLM response text: %s", debug.get("response_text"))
            logger.info("Name->tags LLM result: name=%s tags=%s", base_name, translated)
            merged_tags = _normalize_tags(existing_tags + translated)
            merged_translated = _normalize_tags(existing_translated + translated)
            embedding = None
            batch.append({
                "id": row["id"],
                "tags": merged_tags,
                "translated_tags": merged_translated,
                "embedding": embedding,
                "hash_main": row.get("hash_main_blake3") or "",
                "hash_full": row.get("hash_full_blake3") or "",
                "created_at": row.get("created_at") or now_iso(),
                "mark_name_translate_done": True,
            })
            if len(batch) >= batch_size:
                _flush_tag_batch(batch, settings)
                batch = []
        except Exception as exc:
            errors += 1
            logger.error("Name->tags failed for asset %s: %s", row.get("id"), exc)
        done += 1
        if task_id is not None and (done % 100 == 0 or done == total):
            _task_progress(task_id, "running", total, done, errors)
    if batch:
        _flush_tag_batch(batch, settings)
    if task_id is not None:
        _task_progress(task_id, "done", total, done, errors)


def _name_tags_simple(
    project_id: Optional[int],
    task_id: Optional[int] = None,
    only_missing: bool = False,
) -> None:
    conn = get_db()
    params = []
    where = ""
    if project_id is not None:
        where = "WHERE a.project_id = ?"
        params.append(project_id)
    rows = fetch_all(
        conn,
        "SELECT a.id, a.name, a.tags_json, a.hash_main_blake3, a.hash_full_blake3, a.created_at, "
        "t.tags_translated_json, t.name_tags_done_at "
        "FROM assets a LEFT JOIN asset_tags t ON t.hash_full_blake3 = a.hash_full_blake3 "
        f"{where}",
        params,
    )
    conn.close()
    settings_conn = get_db()
    settings = get_settings(settings_conn)
    settings_conn.close()
    if only_missing:
        rows = [row for row in rows if not row.get("name_tags_done_at")]
    total = len(rows)
    done = 0
    errors = 0
    batch: List[Dict[str, Any]] = []
    batch_size = min(_normalized_batch_size(settings.get("tag_batch_max_assets")), 2000)
    for row in rows:
        if task_id is not None and _task_cancelled(task_id):
            _task_progress(task_id, "canceled", total, done, errors)
            return
        try:
            base_name = _clean_asset_name(row.get("name") or "")
            if not base_name:
                done += 1
                continue
            tokens = [t for t in re.split(r"[\s\-]+", base_name) if t]
            combined = []
            for i in range(len(tokens) - 1):
                combined.append(f"{tokens[i]}_{tokens[i+1]}")
            name_tags = _normalize_tags(tokens + combined)
            if not name_tags:
                done += 1
                continue
            existing_tags = json.loads(row.get("tags_json") or "[]")
            existing_translated = json.loads(row.get("tags_translated_json") or "[]")
            merged_tags = _normalize_tags(existing_tags + name_tags)
            batch.append({
                "id": row["id"],
                "tags": merged_tags,
                "translated_tags": existing_translated,
                "embedding": None,
                "hash_main": row.get("hash_main_blake3") or "",
                "hash_full": row.get("hash_full_blake3") or "",
                "created_at": row.get("created_at") or now_iso(),
                "mark_name_tags_done": True,
            })
            if len(batch) >= batch_size:
                _flush_tag_batch(batch, settings)
                batch = []
        except Exception as exc:
            errors += 1
            logger.error("Name->tags (simple) failed for asset %s: %s", row.get("id"), exc)
        done += 1
        if task_id is not None and (done % 100 == 0 or done == total):
            _task_progress(task_id, "running", total, done, errors)
    if batch:
        _flush_tag_batch(batch, settings)
    if task_id is not None:
        _task_progress(task_id, "done", total, done, errors)


def _run_batch_translate_tags(
    rows: List[Dict[str, Any]],
    settings: Dict[str, str],
    language: str,
    task_id: Optional[int] = None,
    project_id: Optional[int] = None,
) -> None:
    flow = "translate_tags"
    provider = (settings.get("provider") or "").strip().lower()
    api_key = settings.get("api_key") or ""
    if provider == "openai":
        api_key = settings.get("openai_api_key") or api_key
    elif provider == "openrouter":
        api_key = settings.get("openrouter_api_key") or api_key
    elif provider == "groq":
        api_key = settings.get("groq_api_key") or api_key

    base_url = settings.get(f"{provider}_base_url") or settings.get("base_url") or ""
    api_base = base_url.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"
    model = (
        settings.get(f"{provider}_translate_model")
        or settings.get("translate_model")
        or settings.get(f"{provider}_model")
        or settings.get("model")
        or "gpt-5-mini"
    )
    if not base_url or not model:
        raise RuntimeError("LLM settings are incomplete")
    if not language:
        raise RuntimeError("Translation language is required")

    total = len(rows)
    done = 0
    errors = 0
    id_to_row: Dict[str, Dict[str, Any]] = {str(row["id"]): row for row in rows}
    requests: List[Dict[str, Any]] = []

    for row in rows:
        existing_tags = json.loads(row.get("tags_json") or "[]")
        if not existing_tags:
            continue
        tag_text = ", ".join([str(t).strip() for t in existing_tags if str(t).strip()])
        if not tag_text:
            continue
        prompt = TRANSLATE_TEMPLATE.replace("{language}", language).replace("{tags}", tag_text)
        if "json" not in prompt.lower():
            prompt = f"{prompt}\n\nReturn ONLY a JSON object with a 'tags' array of strings."
        body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You translate tags for 3D/asset libraries."},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": 400,
        }
        if provider == "openai":
            body["response_format"] = {"type": "json_object"}
        if _bool_from_setting(settings.get("use_temperature")):
            temp = settings.get("temperature")
            try:
                body["temperature"] = float(temp) if temp is not None else 1.0
            except (TypeError, ValueError):
                body["temperature"] = 1.0
        requests.append(
            {
                "custom_id": str(row["id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client_timeout = httpx.Timeout(30.0, connect=30.0, read=30.0)
    submitted: List[Dict[str, Any]] = []
    max_assets = _normalized_batch_size(settings.get("tag_batch_max_assets"))
    chunks_total = max(1, (len(requests) + max_assets - 1) // max_assets)
    existing_pending = _openai_list_pending_batches(flow, project_id)
    if existing_pending:
        for idx, row in enumerate(existing_pending, start=1):
            req_count = int(row.get("request_total") or 0)
            submitted.append(
                {
                    "batch_id": str(row.get("batch_id") or ""),
                    "chunk": [{}] * max(0, req_count),
                    "offset": 0,
                    "chunk_idx": idx,
                    "chunks_total": len(existing_pending),
                }
            )
        if task_id is not None:
            _task_progress(task_id, "running", total, done, errors, message=f"Translate tags: resume {len(submitted)} existing batches")
    else:
        for chunk_idx, offset in enumerate(range(0, len(requests), max_assets), start=1):
            chunk = requests[offset : offset + max_assets]
            if task_id is not None and _task_cancelled(task_id):
                _task_progress(task_id, "canceled", total, done, errors)
                return
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl", encoding="utf-8") as tmp:
                for entry in chunk:
                    tmp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                tmp_path = Path(tmp.name)
            try:
                with httpx.Client(timeout=client_timeout) as client:
                    def _upload_batch_file():
                        with tmp_path.open("rb") as fp:
                            return client.post(
                                f"{api_base}/files",
                                data={"purpose": "batch"},
                                files={"file": (tmp_path.name, fp, "application/jsonl")},
                                headers=headers,
                            )
                    upload = _retry_http(_upload_batch_file)
                    upload.raise_for_status()
                    file_id = upload.json().get("id")
                    if not file_id:
                        raise RuntimeError("OpenAI batch upload missing file id")
                    def _create_batch():
                        return client.post(
                            f"{api_base}/batches",
                            json={
                                "input_file_id": file_id,
                                "endpoint": "/v1/chat/completions",
                                "completion_window": "24h",
                            },
                            headers=headers,
                        )
                    batch = _retry_http(_create_batch)
                    batch.raise_for_status()
                    batch_id = batch.json().get("id")
                    if not batch_id:
                        raise RuntimeError("OpenAI batch create missing batch id")
                    logger.info("OpenAI tag-translate batch submitted id=%s requests=%s", batch_id, len(chunk))
                    _openai_batch_upsert(
                        flow=flow,
                        batch_id=str(batch_id),
                        provider=provider,
                        task_id=task_id,
                        project_id=project_id,
                        request_total=len(chunk),
                        status="submitted",
                    )
                    submitted.append({"batch_id": batch_id, "chunk": chunk, "offset": offset, "chunk_idx": chunk_idx, "chunks_total": chunks_total})
                    if task_id is not None:
                        _task_progress(
                            task_id,
                            "running",
                            total,
                            done,
                            errors,
                            message=f"Translate tags: submitted chunk {chunk_idx}/{chunks_total} ({len(chunk)} req)",
                        )
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    report_step = done // 100
    pending: Dict[str, Dict[str, Any]] = {
        str(info["batch_id"]): {**info, "last_counts": None, "last_status": None, "last_output_ready": False}
        for info in submitted
    }
    last_queue_msg: Optional[str] = None
    failed_batches: List[str] = []
    with httpx.Client(timeout=client_timeout) as client:
        while pending:
            if task_id is not None and _task_cancelled(task_id):
                _task_progress(task_id, "canceled", total, done, errors)
                return
            progressed = False
            for batch_id in list(pending.keys()):
                info = pending[batch_id]
                chunk = info["chunk"]
                chunk_idx = int(info.get("chunk_idx") or 0)
                chunks_total = int(info.get("chunks_total") or len(submitted) or 1)
                try:
                    poll = client.get(f"{api_base}/batches/{batch_id}", headers=headers)
                    poll.raise_for_status()
                    payload = poll.json()
                except httpx.HTTPError as exc:
                    status_code = getattr(exc.response, "status_code", None)
                    if status_code in {502, 503, 504}:
                        logger.warning(
                            "OpenAI translate-tags poll retryable error id=%s status=%s",
                            batch_id,
                            status_code,
                        )
                        continue
                    logger.warning("OpenAI translate-tags poll error id=%s err=%s", batch_id, exc)
                    continue
                status = payload.get("status") or ""
                output_file_id = payload.get("output_file_id")
                info["last_output_ready"] = bool(output_file_id)
                counts = payload.get("request_counts") or {}
                completed = int(counts.get("completed") or 0)
                failed = int(counts.get("failed") or 0)
                total_batch = int(counts.get("total") or len(chunk))
                counts_tuple = (completed, total_batch, failed)
                if counts_tuple != info.get("last_counts") and task_id is not None:
                    msg = f"Translate tags chunk {chunk_idx}/{chunks_total}: {status} {completed}/{total_batch} (failed {failed})"
                    _task_progress(task_id, "running", total, done, errors, message=msg)
                    info["last_counts"] = counts_tuple
                if status and status != info.get("last_status"):
                    logger.info("OpenAI translate-tags batch status id=%s status=%s", batch_id, status)
                    info["last_status"] = status

                _openai_batch_upsert(
                    flow=flow,
                    batch_id=batch_id,
                    provider=provider,
                    task_id=task_id,
                    project_id=project_id,
                    request_total=len(chunk),
                    status=status,
                    output_file_id=output_file_id,
                )

                is_failure_terminal = status in {"failed", "expired", "cancelled"}
                can_process_output = bool(output_file_id)
                if not is_failure_terminal and not can_process_output:
                    continue

                if is_failure_terminal:
                    pending.pop(batch_id, None)
                    progressed = True
                    _openai_batch_upsert(
                        flow=flow,
                        batch_id=batch_id,
                        provider=provider,
                        task_id=task_id,
                        project_id=project_id,
                        request_total=len(chunk),
                        status=status,
                        error_text=f"terminal status={status}",
                    )
                    _openai_batch_mark_processed(batch_id)
                    failed_batches.append(f"{batch_id}:{status}")
                    continue

                if task_id is not None:
                    _task_progress(task_id, "running", total, done, errors, message=f"Translate tags chunk {chunk_idx}/{chunks_total}: downloading output")
                result = client.get(f"{api_base}/files/{output_file_id}/content", headers=headers)
                result.raise_for_status()
                output_text = result.text
                archive_path = _archive_batch_output(
                    flow=flow,
                    provider=provider,
                    batch_id=batch_id,
                    output_text=output_text,
                    task_id=task_id,
                    project_id=project_id,
                )
                if project_id is None:
                    done = min(total, done + len(chunk))
                    pending.pop(batch_id, None)
                    progressed = True
                    _openai_batch_mark_processed(batch_id)
                    if task_id is not None:
                        _task_progress(
                            task_id,
                            "running",
                            total,
                            done,
                            errors,
                            message=f"Archived batch output: {Path(archive_path).name}",
                        )
                    continue

                batch_rows: List[Dict[str, Any]] = []
                seen_ids: set[str] = set()
                for line in output_text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    custom_id = str(item.get("custom_id") or "")
                    if not custom_id:
                        continue
                    seen_ids.add(custom_id)
                    if item.get("error"):
                        errors += 1
                        done += 1
                        continue
                    resp = item.get("response") or {}
                    body = resp.get("body") if isinstance(resp, dict) else None
                    if isinstance(body, str):
                        try:
                            body = json.loads(body)
                        except Exception:
                            body = None
                    payload = body if isinstance(body, dict) else resp
                    text = _extract_output_text_from_response(payload)
                    translated_raw = _extract_tags_from_content(text)
                    translated = _normalize_tags(translated_raw)
                    row = id_to_row.get(custom_id)
                    if not row:
                        errors += 1
                        done += 1
                        continue
                    if not translated:
                        errors += 1
                        done += 1
                        continue
                    existing_tags = json.loads(row.get("tags_json") or "[]")
                    existing_translated = json.loads(row.get("tags_translated_json") or "[]")
                    merged_tags = _normalize_tags(existing_tags + translated)
                    merged_translated = _normalize_tags(existing_translated + translated)
                    batch_rows.append(
                        {
                            "id": row["id"],
                            "tags": merged_tags,
                            "translated_tags": merged_translated,
                            "embedding": None,
                            "hash_main": row.get("hash_main_blake3") or "",
                            "hash_full": row.get("hash_full_blake3") or "",
                            "created_at": row.get("created_at") or now_iso(),
                            "mark_translate_done": True,
                        }
                    )
                    done += 1
                    if task_id is not None:
                        step = done // 100
                        if done == total or step != report_step:
                            report_step = step
                            _task_progress(
                                task_id,
                                "running",
                                total,
                                done,
                                errors,
                                message=f"Translate tags: {done}/{total} • chunk {chunk_idx}/{chunks_total}",
                            )

                missing = len(chunk) - len(seen_ids)
                if missing:
                    errors += missing
                    done += missing
                    if task_id is not None:
                        step = done // 100
                        if done == total or step != report_step:
                            report_step = step
                            _task_progress(
                                task_id,
                                "running",
                                total,
                                done,
                                errors,
                                message=f"Translate tags: {done}/{total} • chunk {chunk_idx}/{chunks_total}",
                            )
                if batch_rows:
                    _flush_tag_batch_chunked(batch_rows, settings)
                pending.pop(batch_id, None)
                progressed = True
                _openai_batch_mark_processed(batch_id)

            if pending and not progressed:
                snap = _openai_pending_snapshot(pending)
                queue_msg = (
                    f"OpenAI queue: ready {snap['ready']} | in_progress {snap['in_progress']} | "
                    f"finalizing {snap['finalizing']} | waiting {snap['waiting']} | pending {snap['pending']}"
                )
                if queue_msg != last_queue_msg and task_id is not None:
                    last_queue_msg = queue_msg
                    _task_progress(task_id, "running", total, done, errors, message=queue_msg)
                time.sleep(5)

    if failed_batches:
        raise RuntimeError(f"Translate tag batches failed: {', '.join(failed_batches[:10])}")

    if task_id is not None:
        _task_progress(task_id, "done", total, done, errors)


def _translate_tags_only(
    project_id: Optional[int],
    task_id: Optional[int] = None,
    only_missing: bool = False,
) -> None:
    conn = get_db()
    params = []
    where = ""
    if project_id is not None:
        where = "WHERE a.project_id = ?"
        params.append(project_id)
    rows = fetch_all(
        conn,
        "SELECT a.id, a.name, a.description, a.tags_json, a.hash_main_blake3, a.hash_full_blake3, a.created_at, "
        "t.tags_translated_json, t.translated_language, t.translate_tags_done_at "
        "FROM assets a LEFT JOIN asset_tags t ON t.hash_full_blake3 = a.hash_full_blake3 "
        f"{where}",
        params,
    )
    conn.close()
    settings_conn = get_db()
    settings = get_settings(settings_conn)
    settings_conn.close()
    if only_missing:
        target_language = (settings.get("tag_language") or "").strip()
        filtered_rows: List[Dict[str, Any]] = []
        for row in rows:
            if not row.get("translate_tags_done_at"):
                filtered_rows.append(row)
                continue
            row_language = (row.get("translated_language") or "").strip()
            if target_language and row_language.lower() != target_language.lower():
                filtered_rows.append(row)
        rows = filtered_rows
    total = len(rows)
    done = 0
    errors = 0
    batch: List[Dict[str, Any]] = []
    batch_size = min(_normalized_batch_size(settings.get("tag_batch_max_assets")), 2000)
    use_batch_mode = _bool_from_setting(settings.get("tag_use_batch_mode"))
    provider = (settings.get("provider") or "").strip().lower()
    language = (settings.get("tag_language") or "").strip()
    if use_batch_mode and provider in {"openai", "groq"}:
        _run_batch_translate_tags(rows, settings, language, task_id=task_id, project_id=project_id)
        return
    for row in rows:
        if task_id is not None and _task_cancelled(task_id):
            _task_progress(task_id, "canceled", total, done, errors)
            return
        try:
            existing_tags = json.loads(row.get("tags_json") or "[]")
            if not existing_tags:
                done += 1
                continue
            translated = _translate_tags_if_enabled(settings, existing_tags)
            if not translated:
                done += 1
                continue
            existing_translated = json.loads(row.get("tags_translated_json") or "[]")
            merged_tags = _normalize_tags(existing_tags + translated)
            merged_translated = _normalize_tags(existing_translated + translated)
            batch.append({
                "id": row["id"],
                "tags": merged_tags,
                "translated_tags": merged_translated,
                "embedding": None,
                "hash_main": row.get("hash_main_blake3") or "",
                "hash_full": row.get("hash_full_blake3") or "",
                "created_at": row.get("created_at") or now_iso(),
                "mark_translate_done": True,
            })
            if len(batch) >= batch_size:
                _flush_tag_batch(batch, settings)
                batch = []
        except Exception as exc:
            errors += 1
            logger.error("Translate tags failed for asset %s: %s", row.get("id"), exc)
        done += 1
        if task_id is not None and (done % 100 == 0 or done == total):
            _task_progress(task_id, "running", total, done, errors)
    if batch:
        _flush_tag_batch(batch, settings)
    if task_id is not None:
        _task_progress(task_id, "done", total, done, errors)

def _tag_all_projects(mode: str, task_id: Optional[int] = None) -> None:
    conn = get_db()
    rows = fetch_all(conn, "SELECT id, name FROM projects")
    settings = get_settings(conn)
    conn.close()
    total = len(rows)
    done = 0
    errors = 0
    global_total = 0
    global_done = 0
    global_errors = 0
    progress_lock = threading.Lock()

    def _push_progress(message: Optional[str] = None) -> None:
        if task_id is None:
            return
        _task_progress(task_id, "running", global_total, global_done, global_errors, message=message)

    def _make_total_hook(project_label: str):
        def _hook(count: int) -> None:
            nonlocal global_total
            with progress_lock:
                global_total += count
                msg = f"Tag missing all: {global_done}/{global_total}"
                if project_label:
                    msg = f"{msg} ({project_label})"
            _push_progress(msg)
        return _hook

    def _make_progress_hook(project_label: str):
        def _hook(done_inc: int, errors_inc: int, message: Optional[str]) -> None:
            nonlocal global_done, global_errors
            with progress_lock:
                global_done += done_inc
                global_errors += errors_inc
                msg = message or f"Tag missing all: {global_done}/{global_total}"
                if project_label and msg.startswith("Tag missing all"):
                    msg = f"{msg} ({project_label})"
            _push_progress(msg)
        return _hook
    try:
        max_workers = int(settings.get("tag_batch_project_concurrency") or 3)
    except (TypeError, ValueError):
        max_workers = 3
    if mode == "missing":
        max_workers = 1
    if max_workers <= 0:
        max_workers = 1
    max_workers = min(max_workers, 10)

    if max_workers == 1:
        for row in rows:
            if task_id is not None and _task_cancelled(task_id):
                _task_progress(task_id, "canceled", global_total, global_done, global_errors)
                return
            try:
                project_id = int(row["id"])
                project_name = row.get("name") or f"project {project_id}"
                _tag_project_assets(
                    project_id,
                    mode,
                    task_id=task_id,
                    progress_hook=_make_progress_hook(project_name),
                    total_hook=_make_total_hook(project_name),
                )
            except Exception as exc:
                errors += 1
                logger.error("Tag all failed for project %s: %s", row.get("id"), exc)
            done += 1
            if task_id is not None:
                _push_progress()
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for row in rows:
            if task_id is not None and _task_cancelled(task_id):
                _task_progress(task_id, "canceled", global_total, global_done, global_errors)
                return
            project_id = int(row["id"])
            project_name = row.get("name") or f"project {project_id}"
            futures.append(
                executor.submit(
                    _tag_project_assets,
                    project_id,
                    mode,
                    task_id,
                    _make_progress_hook(project_name),
                    _make_total_hook(project_name),
                )
            )
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors += 1
                logger.error("Tag all failed for project: %s", exc)
            done += 1
            if task_id is not None:
                _push_progress()

def _tag_project_assets(
    project_id: int,
    mode: str,
    task_id: Optional[int] = None,
    progress_hook: Optional[Callable[[int, int, Optional[str]], None]] = None,
    total_hook: Optional[Callable[[int], None]] = None,
) -> None:
    conn = get_db()
    settings = get_settings(conn)
    rows = fetch_all(
        conn,
        "SELECT a.id, a.name, a.description, a.tags_json, a.type, a.asset_dir, a.detail_image, a.full_image, a.thumb_image, a.anim_detail, a.anim_thumb, "
        "a.hash_main_blake3, a.hash_full_blake3, a.created_at, t.tags_done_at "
        "FROM assets a LEFT JOIN asset_tags t ON t.hash_full_blake3 = a.hash_full_blake3 "
        "WHERE a.project_id = ?",
        (project_id,),
    )
    conn.close()

    include_types = set(_parse_type_filter(settings.get("tag_include_types")))
    exclude_types = set(_parse_type_filter(settings.get("tag_exclude_types")))
    if include_types or exclude_types:
        filtered = []
        for row in rows:
            asset_type = (row.get("type") or "").strip().lower()
            if include_types and asset_type not in include_types:
                continue
            if exclude_types and asset_type in exclude_types:
                continue
            filtered.append(row)
        rows = filtered

    if mode == "missing":
        candidates = [row for row in rows if not row.get("tags_done_at")]
    else:
        candidates = rows

    total = len(candidates)
    if total_hook is not None:
        total_hook(total)
    _set_tag_progress(project_id, {"status": "running", "total": total, "done": 0, "errors": 0})

    try:
        size_setting = settings.get("tag_image_size") or "512"
        image_size = int(size_setting)
    except ValueError:
        image_size = 512
    try:
        quality_setting = settings.get("tag_image_quality") or "80"
        image_quality = int(quality_setting)
    except ValueError:
        image_quality = 80

    done = 0
    errors = 0
    batch: List[Dict[str, Any]] = []
    batch_size = _normalized_batch_size(settings.get("tag_batch_max_assets"))
    use_batch_mode = _bool_from_setting(settings.get("tag_use_batch_mode"))
    provider = (settings.get("provider") or "").strip().lower()
    if use_batch_mode and provider in {"openai", "groq"}:
        _run_batch_tagging(
            project_id,
            candidates,
            settings,
            mode,
            image_size,
            image_quality,
            provider,
            task_id=task_id if progress_hook is None else None,
            progress_hook=progress_hook,
            total_hook=None,
        )
        return
    if use_batch_mode and provider not in {"openai", "groq"}:
        logger.info("Batch mode requested but provider=%s; falling back to per-asset tagging.", provider)
    for row in candidates:
        if task_id is not None and _task_cancelled(task_id):
            _set_tag_progress(project_id, {"status": "canceled", "total": total, "done": done, "errors": errors})
            if progress_hook is None:
                _task_progress(task_id, "canceled", total, done, errors)
            elif progress_hook is not None:
                progress_hook(0, 0, "Canceled")
            return
        err_inc = 0
        try:
            image_data_url = None
            image_path = _pick_asset_image(row)
            if image_path and image_path.exists():
                image_data_url = _build_image_data_url(image_path, image_size, image_quality)

            existing = [] if mode == "retag" else json.loads(row["tags_json"] or "[]")
            tags, era = generate_tags(
                settings,
                row["name"],
                row["description"] or "",
                existing,
                image_data_url,
                row.get("type") or "",
                return_era=True,
            )
            tags = _normalize_tags(tags)
            _maybe_set_project_era(project_id, era)
            translated_tags = _translate_tags_if_enabled(settings, tags)
            embedding_text = _build_embedding_text(row["name"], row["description"] or "", tags, translated_tags)
            embedding = embed_text(embedding_text)
            batch.append({
                "id": row["id"],
                "tags": tags,
                "translated_tags": translated_tags,
                "embedding": embedding,
                "hash_main": row.get("hash_main_blake3") or "",
                "hash_full": row.get("hash_full_blake3") or "",
                "created_at": row.get("created_at") or now_iso(),
                "mark_tags_done": True,
            })
            if len(batch) >= batch_size:
                _flush_tag_batch(batch, settings)
                batch = []
        except Exception:
            errors += 1
            err_inc = 1
        done += 1
        if progress_hook is not None:
            progress_hook(1, err_inc, None)
        if done % 5 == 0 or done == total:
            _set_tag_progress(project_id, {"status": "running", "total": total, "done": done, "errors": errors})

    if batch:
        _flush_tag_batch(batch, settings)

    _set_tag_progress(project_id, {"status": "done", "total": total, "done": done, "errors": errors})
    if task_id is not None:
        if progress_hook is None:
            _task_progress(task_id, "done", total, done, errors)


def _normalize_meta_rel_path(raw_path: Any) -> Optional[str]:
    rel = str(raw_path or "").strip().replace("\\", "/")
    if not rel:
        return None
    # Prevent absolute/drive-qualified paths from escaping destination roots.
    rel = re.sub(r"^[A-Za-z]:/+", "", rel)
    rel = rel.lstrip("/")
    if rel.lower().startswith("content/"):
        rel = rel[8:]
    parts = [p for p in rel.split("/") if p and p != "."]
    if not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


def _migrate_asset_files(
    asset_id: int,
    source_root: Path,
    dest_root: Path,
    files: List[str],
    overwrite: bool,
    source_fallback_root: Optional[Path] = None,
    source_folder: Optional[str] = None,
) -> None:
    try:
        total = len(files)
        _set_migrate_progress(asset_id, {"status": "running", "copied": 0, "total": total})
        copied = 0
        normalized_source_folder = (source_folder or "").strip().replace("\\", "/").strip("/")
        for rel_path in files:
            rel_norm = _normalize_meta_rel_path(rel_path)
            if not rel_norm:
                copied += 1
                continue

            src_candidates = [source_root / rel_norm]
            if normalized_source_folder and rel_norm.lower().startswith((normalized_source_folder + "/").lower()):
                trimmed = rel_norm[len(normalized_source_folder) + 1 :]
                if trimmed:
                    src_candidates.append(source_root / trimmed)
            if source_fallback_root is not None:
                src_candidates.append(source_fallback_root / rel_norm)
                if normalized_source_folder and rel_norm.lower().startswith((normalized_source_folder + "/").lower()):
                    trimmed = rel_norm[len(normalized_source_folder) + 1 :]
                    if trimmed:
                        src_candidates.append(source_fallback_root / trimmed)

            src = next((c for c in src_candidates if c.exists()), src_candidates[0])
            dest = dest_root / rel_norm
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                copied += 1
                continue
            if dest.exists() and not overwrite:
                copied += 1
                continue
            shutil.copy2(src, dest)
            copied += 1
            if copied % 20 == 0 or copied == total:
                _set_migrate_progress(asset_id, {"status": "running", "copied": copied, "total": total})
        _set_migrate_progress(asset_id, {"status": "done", "copied": total, "total": total})
    except Exception as exc:
        _set_migrate_progress(asset_id, {"status": "error", "error": str(exc), "copied": 0, "total": 0})


def _build_snapshot_zip(asset: Dict[str, Any], project: Dict[str, Any], include_content_root: bool = False) -> Path:
    meta = json.loads(asset.get("meta_json") or "{}")
    files = meta.get("files_on_disk") or []
    if not files:
        raise HTTPException(status_code=400, detail="No files_on_disk in asset meta")

    content_root = Path(project["folder_path"]) / "Content"
    source_content_root = _resolve_source_content_path(project)
    logger.info(
        "Download zip request asset_id=%s hash=%s project_id=%s content_root=%s",
        asset.get("id"),
        asset.get("hash_main_blake3") or "",
        project.get("id"),
        str(content_root),
    )
    if not content_root.exists():
        if not source_content_root or not source_content_root.exists():
            raise HTTPException(status_code=400, detail="Project Content folder not found")

    temp_name = f"download_{asset['id']}_{int(time.time())}.zip"
    zip_path = UPLOADS_DIR / temp_name
    missing: List[str] = []

    logger.info(
        "Download zip build asset_id=%s hash=%s content_root=%s files=%d",
        asset.get("id"),
        asset.get("hash_main_blake3") or "",
        str(content_root),
        len(files),
    )

    def normalize_rel_path(value: str) -> str:
        rel_str = str(value).strip().replace("\\", "/").lstrip("/")
        if not rel_str:
            return ""
        path = Path(value)
        if path.is_absolute():
            try:
                rel_str = str(path.resolve().relative_to(content_root.resolve())).replace("\\", "/")
            except ValueError:
                rel_str = str(path).replace("\\", "/").lstrip("/")
        if rel_str.lower().startswith("content/"):
            rel_str = rel_str[8:]
        return rel_str.lstrip("/")

    resolved_sources = _resolve_source_paths(project.get("source_path"), project.get("source_folder"))
    source_project_root = resolved_sources.get("project_root")
    source_content_root = _resolve_source_content_path(project)
    extra_roots: List[Path] = []
    if source_content_root and source_content_root.exists():
        if source_content_root.resolve() != content_root.resolve():
            extra_roots.append(source_content_root)
    if source_project_root:
        extra_root = source_project_root / "Content"
        if extra_root.exists() and extra_root.resolve() != content_root.resolve():
            if not source_content_root or extra_root.resolve() != source_content_root.resolve():
                extra_roots.append(extra_root)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for rel_path in files:
            rel_norm = normalize_rel_path(rel_path)
            if not rel_norm:
                continue
            arc_rel = rel_norm
            src = content_root / rel_norm
            matched_root = content_root
            parts = rel_norm.split("/", 1)
            alt_rel = parts[1] if len(parts) == 2 else ""
            if not src.exists() and alt_rel:
                alt_src = content_root / alt_rel
                if alt_src.exists():
                    src = alt_src
            if not src.exists():
                for root in extra_roots:
                    candidate = root / rel_norm
                    if candidate.exists():
                        src = candidate
                        matched_root = root
                        break
                    if alt_rel:
                        candidate = root / alt_rel
                        if candidate.exists():
                            src = candidate
                            matched_root = root
                            break
            if not src.exists():
                missing.append(rel_norm)
                continue
            rel_from_root = src.relative_to(matched_root)
            arc = arc_rel
            if not arc:
                arc = rel_from_root.as_posix()
            if include_content_root:
                if matched_root == content_root:
                    arc = f"Content/{arc}"
                else:
                    prefix = None
                    if source_project_root:
                        try:
                            content_base = (source_project_root / "Content").resolve()
                            prefix = matched_root.resolve().relative_to(content_base).as_posix()
                        except Exception:
                            prefix = None
                    if prefix is None:
                        prefix = matched_root.name
                    prefix_norm = str(prefix or "").replace("\\", "/").strip("/")
                    arc_norm = str(arc or "").replace("\\", "/").lstrip("/")
                    if prefix_norm:
                        if arc_norm.lower() == prefix_norm.lower():
                            arc_norm = ""
                        elif arc_norm.lower().startswith((prefix_norm + "/").lower()):
                            arc_norm = arc_norm[len(prefix_norm) + 1 :]
                        arc = f"Content/{prefix_norm}/{arc_norm}" if arc_norm else f"Content/{prefix_norm}"
                    else:
                        arc = f"Content/{arc_norm}"
            zf.write(src, arc)

    if missing:
        logger.info("zip missing files (%d): %s", len(missing), ", ".join(missing))
        zip_path.unlink(missing_ok=True)
        raise HTTPException(status_code=404, detail=f"Missing files: {', '.join(missing[:10])}")

    return zip_path


def _copy_project_content(project_id: int, source_path: str, dest_root: Path) -> None:
    try:
        source = Path(source_path).expanduser()
        content_dir = source / "Content"
        dest_content = dest_root / "Content"
        dest_content.mkdir(parents=True, exist_ok=True)
        if not content_dir.exists() or not content_dir.is_dir():
            _set_copy_progress(project_id, {"status": "done", "copied": 0, "total": 0})
            return

        files = [p for p in content_dir.rglob("*") if p.is_file()]
        total = len(files)
        _set_copy_progress(project_id, {"status": "running", "copied": 0, "total": total})

        copied = 0
        for file_path in files:
            rel = file_path.relative_to(content_dir)
            target = dest_content / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target)
            copied += 1
            if copied % 20 == 0 or copied == total:
                _set_copy_progress(project_id, {"status": "running", "copied": copied, "total": total})

        _set_copy_progress(project_id, {"status": "done", "copied": total, "total": total})
    except Exception as exc:
        _set_copy_progress(project_id, {"status": "error", "error": str(exc), "copied": 0, "total": 0})


def _resolve_source_paths(source_path: Optional[str], source_folder: Optional[str]) -> Dict[str, Optional[Path]]:
    resolved_source = Path(source_path).expanduser() if source_path else None
    if source_path is None and source_folder:
        sf_path = Path(source_folder).expanduser()
        parts = [p for p in sf_path.parts]
        if 'Content' in parts:
            idx = parts.index('Content')
            if idx > 0 and idx + 1 < len(parts):
                source_path = str(Path(*parts[:idx]))
                source_folder = Path(*parts[idx+1:]).name
                resolved_source = Path(source_path).expanduser()

    if source_folder and ('\\' in source_folder or '/' in source_folder):
        source_folder = Path(source_folder).name
    resolved_folder = Path(source_folder).expanduser() if source_folder else None

    if resolved_source and resolved_source.exists():
        if (resolved_source / "Content").is_dir():
            # source_path points to a UE project root
            if resolved_folder is None:
                return {"project_root": resolved_source, "source_folder": None}
            if not resolved_folder.exists():
                # allow source_folder to be a name; resolve inside Content
                candidate = resolved_source / "Content" / str(source_folder)
                if candidate.exists():
                    return {"project_root": resolved_source, "source_folder": candidate}
            return {"project_root": resolved_source, "source_folder": resolved_folder}

        if resolved_source.name.lower() == "content":
            if resolved_folder is None:
                return {"project_root": resolved_source.parent, "source_folder": None}
            if not resolved_folder.exists():
                candidate = resolved_source / str(source_folder)
                if candidate.exists():
                    return {"project_root": resolved_source.parent, "source_folder": candidate}
            return {"project_root": resolved_source.parent, "source_folder": resolved_folder}

        if resolved_source.parent.name.lower() == "content":
            # source_path points to a pack folder inside Content
            return {"project_root": resolved_source.parent.parent, "source_folder": resolved_source}

    if resolved_folder and resolved_folder.exists():
        if resolved_folder.parent.name.lower() == "content":
            return {"project_root": resolved_folder.parent.parent, "source_folder": resolved_folder}
        return {"project_root": None, "source_folder": resolved_folder}

    return {"project_root": resolved_source, "source_folder": resolved_folder}

def _collect_extra_content_roots(project_id: int, primary_root: Optional[str]) -> List[str]:
    conn = get_db()
    rows = fetch_all(conn, "SELECT meta_json FROM assets WHERE project_id = ?", (project_id,))
    conn.close()

    roots: set[str] = set()
    for row in rows:
        try:
            meta = json.loads(row.get("meta_json") or "{}")
        except json.JSONDecodeError:
            meta = {}
        files = meta.get("files_on_disk") or []
        for raw in files:
            rel = str(raw).strip().replace("\\", "/").lstrip("/")
            if not rel:
                continue
            if rel.lower().startswith("content/"):
                rel = rel[8:]
            top = rel.split("/", 1)[0]
            if not top:
                continue
            if primary_root and top.lower() == primary_root.lower():
                continue
            roots.add(top)
    return sorted(roots)


def _sync_tree(project_id: int, source_root: Path, dest_root: Path) -> None:
    files = [p for p in source_root.rglob("*") if p.is_file()]
    total = len(files)
    _set_copy_progress(project_id, {"status": "running", "copied": 0, "total": total})

    copied = 0
    for file_path in files:
        rel = file_path.relative_to(source_root)
        target = dest_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        do_copy = True
        if target.exists():
            try:
                src_stat = file_path.stat()
                dst_stat = target.stat()
                do_copy = src_stat.st_size != dst_stat.st_size or int(src_stat.st_mtime) != int(dst_stat.st_mtime)
            except OSError:
                do_copy = True
        if do_copy:
            shutil.copy2(file_path, target)
        copied += 1
        if copied % 50 == 0 or copied == total:
            _set_copy_progress(project_id, {"status": "running", "copied": copied, "total": total})

    _set_copy_progress(project_id, {"status": "done", "copied": total, "total": total})


def _reimport_project(
    project_id: int,
    project_name: str,
    source_path: Optional[str],
    source_folder: Optional[str],
    dest_root: Path,
    full_project_copy: bool = False,
) -> None:
    try:
        resolved = _resolve_source_paths(source_path, source_folder)
        project_root = resolved.get("project_root")
        folder_path = resolved.get("source_folder")
        if full_project_copy and project_root:
            folder_path = None

        _ensure_uproject_file(project_name, dest_root, source_path)

        if project_root and (project_root / "Content").is_dir() and not folder_path:
            dest_content = dest_root / "Content"
            dest_content.mkdir(parents=True, exist_ok=True)
            _sync_tree(project_id, project_root / "Content", dest_content)
        elif folder_path:
            _set_copy_progress(project_id, {"status": "done", "copied": 0, "total": 0})
        else:
            raise ValueError("Source path not found")

        if folder_path and folder_path.exists():
            dest_folder = dest_root / "Content" / folder_path.name
            dest_folder.mkdir(parents=True, exist_ok=True)
            _sync_tree(project_id, folder_path, dest_folder)
            if project_root and (project_root / "Content").is_dir():
                extra_roots = _collect_extra_content_roots(project_id, folder_path.name)
                for root in extra_roots:
                    source_extra = project_root / "Content" / root
                    if not source_extra.exists():
                        continue
                    dest_extra = dest_root / "Content" / root
                    dest_extra.mkdir(parents=True, exist_ok=True)
                    logger.info("Reimport copying extra content root: %s", root)
                    _sync_tree(project_id, source_extra, dest_extra)
    except Exception as exc:
        _set_copy_progress(project_id, {"status": "error", "error": str(exc), "copied": 0, "total": 0})


def _ensure_uproject_file(project_name: str, dest_root: Path, source_path: Optional[str]) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    if source_path:
        source = Path(source_path).expanduser()
        candidates = list(source.glob("*.uproject"))
        if candidates:
            dest_name = candidates[0].name
            shutil.copy2(candidates[0], dest_root / dest_name)
            return

    filename = f"{slugify(project_name)}.uproject"
    target = dest_root / filename
    if target.exists():
        return
    payload = {
        "FileVersion": 3,
        "EngineAssociation": "",
        "Category": "",
        "Description": "",
        "Modules": [],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_export_game_path(project_row: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        raw = override.strip()
        if not raw.startswith("/"):
            raw = f"/{raw}"
        if not raw.lower().startswith("/game"):
            raw = f"/Game{raw}"
        return raw
    source_folder = (project_row.get("source_folder") or "").strip().strip("/\\")
    if source_folder:
        return f"/Game/{source_folder}"
    return "/Game"


def _find_uproject_path(project_name: str, dest_root: Path, source_path: Optional[str]) -> Path:
    _ensure_uproject_file(project_name, dest_root, source_path)
    candidates = list(dest_root.glob("*.uproject"))
    if candidates:
        non_import = [p for p in candidates if p.name.lower() != "import.uproject"]
        if non_import:
            return non_import[0]
    if candidates:
        return candidates[0]
    return dest_root / f"{slugify(project_name)}.uproject"


def _pick_uproject_in_root(root: Path) -> Optional[Path]:
    if not root or not root.exists():
        return None
    candidates = list(root.glob("*.uproject"))
    if not candidates:
        return None
    non_import = [p for p in candidates if p.name.lower() != "import.uproject"]
    return non_import[0] if non_import else candidates[0]


def _prefer_import_uproject(root: Path) -> Path:
    import_candidate = root / "import.uproject"
    if import_candidate.exists():
        return import_candidate
    _ensure_uproject_file("import", root, "")
    return root / "import.uproject"


def get_settings(conn) -> Dict[str, str]:
    rows = fetch_all(conn, "SELECT key, value FROM settings")
    return {row["key"]: row["value"] for row in rows}


def _bool_from_setting(value: Optional[str]) -> bool:
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _normalized_batch_size(value: Any, default: int = 500, upper: int = 50000) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        parsed = default
    if parsed < 1:
        return default
    return min(parsed, upper)


def _should_generate_embeddings_on_import(settings: Dict[str, str]) -> bool:
    raw = settings.get("generate_embeddings_on_import")
    if raw is None or str(raw).strip() == "":
        return True
    return _bool_from_setting(raw)


def _dir_size_bytes(path: Optional[str]) -> int:
    if not path:
        return 0
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for base, _, files in os.walk(root):
        for name in files:
            try:
                total += (Path(base) / name).stat().st_size
            except OSError:
                continue
    return total


def _resolve_source_content_path(row: Dict[str, Any]) -> Optional[Path]:
    source_path = (row.get("source_path") or "").strip()
    if not source_path:
        return None
    source_folder = (row.get("source_folder") or "").strip()
    base = Path(source_path)
    if source_folder:
        folder_path = Path(source_folder)
        if folder_path.is_absolute():
            return folder_path
        return base / "Content" / source_folder
    return base


def _project_screenshot_url_from_path(path_value: Optional[str]) -> str:
    if not path_value:
        return ""
    try:
        rel = Path(path_value).resolve().relative_to(DATA_DIR.resolve())
        return f"/media/{str(rel).replace('\\', '/')}"
    except ValueError:
        return ""


def _normalize_legacy_project_folder_path(path_value: Optional[str]) -> str:
    raw = (path_value or "").strip()
    if not raw:
        return ""
    try:
        p = Path(raw)
        if p.exists():
            return str(p)
    except Exception:
        return raw
    marker = f"{os.sep}Plugins{os.sep}AssetMetaExplorerBridge{os.sep}open{os.sep}data{os.sep}"
    if marker in raw:
        candidate = raw.replace(marker, f"{os.sep}open{os.sep}data{os.sep}")
        try:
            if Path(candidate).exists():
                return str(Path(candidate))
        except Exception:
            pass
    return raw


def _pick_project_image_from_folder(folder_path: Optional[str]) -> Optional[str]:
    if not folder_path:
        return None
    try:
        base = Path(folder_path)
    except Exception:
        return None
    if not base.exists() or not base.is_dir():
        return None
    # Prefer screenshot.* first, then setcard.*
    candidates = sorted(base.glob("screenshot.*")) + sorted(base.glob("setcard.*"))
    for c in candidates:
        if c.is_file() and c.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return str(c)
    return None


def _pick_local_project_screenshot(source_dir: Optional[Path]) -> Optional[Path]:
    if not source_dir or not source_dir.exists() or not source_dir.is_dir():
        return None
    for entry in source_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return entry
    return None


def _copy_local_project_screenshot(source_dir: Optional[Path], folder_path: Path) -> Optional[str]:
    candidate = _pick_local_project_screenshot(source_dir)
    if not candidate:
        return None
    dest = folder_path / f"screenshot{candidate.suffix.lower()}"
    try:
        shutil.copy2(candidate, dest)
    except OSError:
        return None
    return str(dest)


def _try_download_screenshot(url: str, folder_path: Path) -> Optional[str]:
    url = (url or "").strip()
    if not url:
        return None
    dest = folder_path / "screenshot.jpg"
    try:
        if url.startswith("/media/"):
            rel = url[len("/media/") :]
            source = DATA_DIR / rel
            if source.exists():
                shutil.copy2(source, dest)
                return str(dest)
        local_path = Path(url)
        if not local_path.is_absolute():
            local_path = Path(os.path.expanduser(url))
        if local_path.exists():
            shutil.copy2(local_path, dest)
            return str(dest)
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            dest.write_bytes(response.content)
        return str(dest)
    except Exception as exc:
        logger.error("Project import screenshot download failed: %s", exc)
        return None


def _normalize_tags(tags: List[str]) -> List[str]:
    cleaned = []
    for tag in tags:
        value = str(tag).strip().lower()
        if not value:
            continue
        # Drop mojibake / invalid control chars (e.g. "weiã")
        if "\ufffd" in value or any(0x80 <= ord(ch) <= 0x9F for ch in value):
            continue
        if value not in cleaned:
            cleaned.append(value)
    return cleaned


def _merge_tags_for_asset(tags: List[str], translated_tags: List[str]) -> List[str]:
    if not translated_tags:
        return tags
    return _normalize_tags(list(tags) + list(translated_tags))


def _translate_tags_if_enabled(settings: Dict[str, str], tags: List[str]) -> List[str]:
    if not tags:
        return []
    enabled = _bool_from_setting(settings.get("tag_translate_enabled"))
    language = (settings.get("tag_language") or "").strip()
    if not enabled or not language:
        return []
    try:
        return translate_tags(settings, tags, language)
    except Exception as exc:
        logger.error("Tag translation failed: %s", exc)
        return []


def _build_embedding_text(
    name: str,
    description: str,
    tags: List[str],
    translated_tags: List[str],
) -> str:
    parts = [name or "", description or "", " ".join(tags)]
    if translated_tags:
        parts.append(" ".join(translated_tags))
    return " ".join([p for p in parts if p])


def _normalize_path_value(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        resolved = Path(value).expanduser().resolve()
        return str(resolved).replace("\\", "/").lower()
    except Exception:
        return str(value).replace("\\", "/").lower()


def _get_cached_source_size(
    conn: sqlite3.Connection,
    row: Dict[str, Any],
    max_age_seconds: int = 600,
    force: bool = False,
) -> int:
    source_root = _resolve_source_content_path(row)
    if not source_root:
        return 0
    if not source_root.exists():
        return 0

    stored_size = row.get("source_size_bytes")
    updated_at = row.get("source_size_updated_at")
    if not force and stored_size is not None and updated_at:
        try:
            age = (datetime.utcnow() - datetime.fromisoformat(str(updated_at))).total_seconds()
            if age < max_age_seconds:
                return int(stored_size or 0)
        except Exception:
            pass
    elif not force and stored_size is not None and updated_at is None:
        return int(stored_size or 0)

    size = _dir_size_bytes(str(source_root))
    try:
        conn.execute(
            "UPDATE projects SET source_size_bytes = ?, source_size_updated_at = ? WHERE id = ?",
            (int(size), now_iso(), row.get("id")),
        )
        _db_retry(conn.commit)
    except Exception:
        pass
    return int(size)


def _project_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    screenshot_url = _project_screenshot_url_from_path(row.get("screenshot_path"))
    folder_size_bytes = _dir_size_bytes(row.get("folder_path"))
    conn = get_db()
    try:
        source_size_bytes = _get_cached_source_size(conn, row, max_age_seconds=600)
    finally:
        conn.close()
    return {
        "id": row["id"],
        "name": row["name"],
        "link": row.get("link"),
        "size_bytes": row.get("size_bytes") or 0,
        "folder_size_bytes": folder_size_bytes,
        "source_size_bytes": source_size_bytes,
        "folder_path": row.get("folder_path"),
        "screenshot_url": screenshot_url,
        "art_style": row.get("art_style") or "",
        "tags": json.loads(row.get("tags_json") or "[]"),
        "source_path": row.get("source_path") or "",
        "source_folder": (Path(row.get("source_folder")).name if row.get("source_folder") else ""),
        "full_project_copy": bool(row.get("full_project_copy") or 0),
        "created_at": row.get("created_at") or now_iso(),
        "copy_started": False,
    }


def _find_project_by_source(
    conn,
    source_path: Optional[str],
    source_folder: Optional[str],
) -> Optional[Dict[str, Any]]:
    if source_folder:
        cleaned = re.split(r"[\/]+", source_folder.strip().strip("/\\"))
        source_folder = cleaned[-1] if cleaned else source_folder
    resolved = _resolve_source_paths(source_path, source_folder)
    include_project_root = False
    resolved_source = None
    try:
        resolved_source = Path(source_path).expanduser() if source_path else None
    except Exception:
        resolved_source = None
    if resolved_source:
        if (resolved_source / "Content").is_dir() or resolved_source.name.lower() == "content":
            include_project_root = True
    candidates = [source_path, source_folder, resolved.get("source_folder")]
    if include_project_root:
        candidates.append(resolved.get("project_root"))
    normalized = {val for val in (_normalize_path_value(c) for c in candidates) if val}
    if not normalized:
        return None
    rows = fetch_all(conn, "SELECT * FROM projects")
    for row in rows:
        row_values = [
            _normalize_path_value(row.get("source_path")),
            _normalize_path_value(row.get("source_folder")),
        ]
        if any(val and val in normalized for val in row_values):
            return row
    return None


def _create_project_from_root(root_name: str) -> int:
    logger.info("Auto-create project: root=%s", root_name)

    name = root_name.strip() or "project"
    conn = get_db()
    existing = fetch_one(
        conn,
        "SELECT id FROM projects WHERE lower(source_folder) = lower(?)",
        (name,),
    )
    if existing:
        logger.info("Auto-create project: existing match id=%s for root=%s", existing["id"], name)
        conn.close()
        return existing["id"]

    name_slug = slugify(name)[:16] or "project"
    folder_name = f"{name_slug}-{int(time.time())}"
    folder_path = PROJECTS_DIR / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    (folder_path / "Content").mkdir(parents=True, exist_ok=True)
    _ensure_uproject_file(name, folder_path, "")

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO projects (
            name, link, size_bytes, folder_path, screenshot_path, art_style, project_era, tags_json, description, category_name, is_ai_generated, created_at,
            source_path, source_folder, full_project_copy
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            None,
            0,
            str(folder_path),
            None,
            None,
            None,
            json.dumps([]),
            None,
            None,
            None,
            now_iso(),
            None,
            name,
            0,
        ),
    )
    _db_retry(conn.commit)
    project_id = cur.lastrowid
    conn.close()
    logger.info("Auto-create project: created id=%s name=%s", project_id, name)
    return project_id


def _resolve_project_id_from_meta(conn, meta: Dict[str, Any]) -> Optional[int]:
    vendor = str(meta.get("vendor") or "").strip()
    package = str(meta.get("package") or "")

    root = ""
    if vendor:
        root = vendor
    elif package.startswith("/Game/"):
        relative = package[6:]
        root = relative.split("/", 1)[0].strip()

    if not root:
        return None

    root_lc = root.lower()
    root_key = slugify(root_lc)
    rows = fetch_all(conn, "SELECT id, name, source_path, source_folder FROM projects")
    debug_samples = []
    for row in rows:
        name_raw = (row.get("name") or "").strip().lower()
        name_key = slugify(name_raw)
        name_match = name_raw == root_lc or (name_key and name_key == root_key)
        source_path = row.get("source_path") or ""
        source_folder = row.get("source_folder") or ""
        source_leaf = Path(source_path).name.lower() if source_path else ""
        folder_leaf = Path(source_folder).name.lower() if source_folder else ""
        debug_samples.append({
            "id": row.get("id"),
            "name": name_raw,
            "name_key": name_key,
            "source_leaf": source_leaf,
            "folder_leaf": folder_leaf,
        })
        source_key = slugify(source_leaf) if source_leaf else ""
        folder_key = slugify(folder_leaf) if folder_leaf else ""
        if (
            name_match
            or source_leaf == root_lc
            or folder_leaf == root_lc
            or source_key == root_key
            or folder_key == root_key
        ):
            return row.get("id")
    logger.warning("Resolve project failed for root=%s key=%s samples=%s", root_lc, root_key, debug_samples[:5])
    return None



def _regenerate_embeddings(project_id: Optional[int], task_id: Optional[int] = None) -> None:
    key = "all" if project_id is None else str(project_id)
    scope_label = "all" if project_id is None else f"project={project_id}"
    conn = get_db()
    params: List[Any] = []
    where = ""
    if project_id is not None:
        where = "WHERE a.project_id = ?"
        params.append(project_id)
    rows = fetch_all(
        conn,
        "SELECT a.id, a.name, a.description, a.tags_json, a.hash_full_blake3, t.tags_translated_json "
        "FROM assets a LEFT JOIN asset_tags t ON t.hash_full_blake3 = a.hash_full_blake3 "
        f"{where}",
        params,
    )
    total = len(rows)
    started_at = time.perf_counter()
    startup_log_step = 500
    last_startup_log_done = 0
    _set_embed_progress(key, {"status": "running", "total": total, "done": 0, "errors": 0})
    if task_id is None:
        logger.info("Embeddings rebuild start: total=%s scope=%s", total, scope_label)
    if task_id is not None:
        _task_progress(task_id, "running", total, 0, 0)
    done = 0
    errors = 0
    write_batch: List[tuple] = []
    write_batch_size = 5000
    embed_batch_ids: List[int] = []
    embed_batch_texts: List[str] = []
    embed_batch_size = 1024

    def _emit_progress() -> None:
        nonlocal last_startup_log_done
        _set_embed_progress(key, {"status": "running", "total": total, "done": done, "errors": errors})
        if task_id is not None:
            _task_progress(task_id, "running", total, done, errors)
            return
        should_log = done == total or (done - last_startup_log_done) >= startup_log_step
        if not should_log:
            return
        elapsed = max(0.0001, time.perf_counter() - started_at)
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = max(0, total - done)
        eta_seconds = (remaining / rate) if rate > 0 else 0.0
        logger.info(
            "Embeddings rebuild progress: %s/%s errors=%s scope=%s elapsed=%s rate=%.1f rows/s eta=%s",
            done,
            total,
            errors,
            scope_label,
            _fmt_seconds(elapsed),
            rate,
            _fmt_seconds(eta_seconds),
        )
        last_startup_log_done = done

    def _flush_write_batch() -> None:
        nonlocal write_batch
        if write_batch:
            _flush_embedding_batch(write_batch)
            write_batch = []

    def _consume_embed_batch() -> None:
        nonlocal done, errors, write_batch, embed_batch_ids, embed_batch_texts
        if not embed_batch_ids:
            return
        try:
            vectors = embed_texts(embed_batch_texts)
            if len(vectors) != len(embed_batch_ids):
                raise RuntimeError(f"embed_texts size mismatch: {len(vectors)} != {len(embed_batch_ids)}")
            for asset_id, embedding in zip(embed_batch_ids, vectors):
                write_batch.append((json.dumps(embedding), asset_id))
                done += 1
                if len(write_batch) >= write_batch_size:
                    _flush_write_batch()
                if done % 50 == 0 or done == total:
                    _emit_progress()
        except Exception as exc:
            logger.error("Embedding regen batch failed size=%s: %s", len(embed_batch_ids), exc)
            for asset_id, embedding_text in zip(embed_batch_ids, embed_batch_texts):
                try:
                    embedding = embed_text(embedding_text)
                    write_batch.append((json.dumps(embedding), asset_id))
                    if len(write_batch) >= write_batch_size:
                        _flush_write_batch()
                except Exception as one_exc:
                    errors += 1
                    logger.error("Embedding regen failed for asset %s: %s", asset_id, one_exc)
                done += 1
                if done % 50 == 0 or done == total:
                    _emit_progress()
        finally:
            embed_batch_ids = []
            embed_batch_texts = []

    for row in rows:
        if task_id is not None and _task_cancelled(task_id):
            _set_embed_progress(key, {"status": "canceled", "total": total, "done": done, "errors": errors})
            _task_progress(task_id, "canceled", total, done, errors)
            conn.close()
            return
        try:
            tags = json.loads(row.get("tags_json") or "[]")
            translated = json.loads(row.get("tags_translated_json") or "[]")
            embedding_text = _build_embedding_text(row["name"], row["description"] or "", tags, translated)
            embed_batch_ids.append(int(row["id"]))
            embed_batch_texts.append(embedding_text)
            if len(embed_batch_ids) >= embed_batch_size:
                _consume_embed_batch()
        except Exception as exc:
            errors += 1
            done += 1
            logger.error("Embedding regen prepare failed for asset %s: %s", row.get("id"), exc)
            if done % 50 == 0 or done == total:
                _emit_progress()

    _consume_embed_batch()
    _flush_write_batch()
    conn.close()
    _set_embed_progress(key, {"status": "done", "total": total, "done": done, "errors": errors})
    if task_id is None:
        elapsed = max(0.0001, time.perf_counter() - started_at)
        rate = done / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Embeddings rebuild done: %s/%s errors=%s scope=%s took=%s rate=%.1f rows/s",
            done,
            total,
            errors,
            scope_label,
            _fmt_seconds(elapsed),
            rate,
        )
    if task_id is not None:
        _task_progress(task_id, "done", total, done, errors)


def _upsert_asset_tags(
    conn,
    asset_id: int,
    hash_main: str,
    hash_full: str,
    asset_created_at: str,
    tags: List[str],
    translated_tags: List[str],
    translated_language: str,
    mark_tags_done: bool = False,
    mark_name_tags_done: bool = False,
    mark_name_translate_done: bool = False,
    mark_translate_done: bool = False,
) -> None:
    now = now_iso()
    tags_done_at = now if mark_tags_done else None
    name_tags_done_at = now if mark_name_tags_done else None
    name_translate_tags_done_at = now if mark_name_translate_done else None
    translate_tags_done_at = now if mark_translate_done else None
    cur = conn.cursor()
    payload = {
        "asset_id": asset_id,
        "hash_main_blake3": hash_main,
        "hash_full_blake3": hash_full,
        "tags_original_json": json.dumps(tags),
        "tags_translated_json": json.dumps(translated_tags) if translated_tags else None,
        "translated_language": translated_language or None,
        "asset_created_at": asset_created_at,
        "created_at": now,
        "updated_at": now,
        "tags_done_at": tags_done_at,
        "name_tags_done_at": name_tags_done_at,
        "name_translate_tags_done_at": name_translate_tags_done_at,
        "translate_tags_done_at": translate_tags_done_at,
    }

    if hash_full:
        cur.execute(
            """
            UPDATE asset_tags
            SET
                asset_id = ?,
                hash_main_blake3 = ?,
                tags_original_json = ?,
                tags_translated_json = ?,
                translated_language = ?,
                asset_created_at = ?,
                tags_done_at = CASE WHEN ? IS NOT NULL THEN ? ELSE tags_done_at END,
                name_tags_done_at = CASE WHEN ? IS NOT NULL THEN ? ELSE name_tags_done_at END,
                name_translate_tags_done_at = CASE WHEN ? IS NOT NULL THEN ? ELSE name_translate_tags_done_at END,
                translate_tags_done_at = CASE WHEN ? IS NOT NULL THEN ? ELSE translate_tags_done_at END,
                updated_at = ?
            WHERE hash_full_blake3 = ?
            """,
            (
                payload["asset_id"],
                payload["hash_main_blake3"],
                payload["tags_original_json"],
                payload["tags_translated_json"],
                payload["translated_language"],
                payload["asset_created_at"],
                payload["tags_done_at"],
                payload["tags_done_at"],
                payload["name_tags_done_at"],
                payload["name_tags_done_at"],
                payload["name_translate_tags_done_at"],
                payload["name_translate_tags_done_at"],
                payload["translate_tags_done_at"],
                payload["translate_tags_done_at"],
                payload["updated_at"],
                payload["hash_full_blake3"],
            ),
        )
        if cur.rowcount == 0:
            cur.execute(
                """
                INSERT INTO asset_tags (
                    asset_id,
                    hash_main_blake3,
                    hash_full_blake3,
                    tags_original_json,
                    tags_translated_json,
                    translated_language,
                    asset_created_at,
                    tags_done_at,
                    name_tags_done_at,
                    name_translate_tags_done_at,
                    translate_tags_done_at,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["asset_id"],
                    payload["hash_main_blake3"],
                    payload["hash_full_blake3"],
                    payload["tags_original_json"],
                    payload["tags_translated_json"],
                    payload["translated_language"],
                    payload["asset_created_at"],
                    payload["tags_done_at"],
                    payload["name_tags_done_at"],
                    payload["name_translate_tags_done_at"],
                    payload["translate_tags_done_at"],
                    payload["created_at"],
                    payload["updated_at"],
                ),
            )
    else:
        cur.execute("DELETE FROM asset_tags WHERE asset_id = ?", (asset_id,))
        cur.execute(
            """
            INSERT INTO asset_tags (
                asset_id,
                hash_main_blake3,
                hash_full_blake3,
                tags_original_json,
                tags_translated_json,
                translated_language,
                asset_created_at,
                tags_done_at,
                name_tags_done_at,
                name_translate_tags_done_at,
                translate_tags_done_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["asset_id"],
                payload["hash_main_blake3"],
                payload["hash_full_blake3"],
                payload["tags_original_json"],
                payload["tags_translated_json"],
                payload["translated_language"],
                payload["asset_created_at"],
                payload["tags_done_at"],
                payload["name_tags_done_at"],
                payload["name_translate_tags_done_at"],
                payload["translate_tags_done_at"],
                payload["created_at"],
                payload["updated_at"],
            ),
        )


def _resolve_ui_dist(settings: Dict[str, str]) -> Optional[Path]:
    env_path = os.getenv("ASSET_UI_DIST") or os.getenv("FRONTEND_DIST")
    if env_path:
        return Path(env_path).expanduser()
    cfg_path = settings.get("frontend_dist_path") or ""
    if cfg_path.strip():
        return Path(cfg_path).expanduser()
    return BASE_DIR / "frontend" / "dist"


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url, new=2)
    except Exception:
        pass


def _configure_frontend(settings: Dict[str, str]) -> None:
    global UI_ENABLED, UI_DIST_DIR
    env_enabled = _bool_from_setting(os.getenv("ASSET_UI") or os.getenv("SERVE_FRONTEND"))
    cfg_enabled = _bool_from_setting(settings.get("serve_frontend"))
    enabled = env_enabled or cfg_enabled
    dist_dir = _resolve_ui_dist(settings)
    if enabled and dist_dir and dist_dir.is_dir():
        UI_ENABLED = True
        UI_DIST_DIR = dist_dir
        app.mount("/ui", StaticFiles(directory=dist_dir, html=True), name="ui")
        logger.info("UI enabled at /ui (dist=%s)", dist_dir)
    else:
        UI_ENABLED = False
        UI_DIST_DIR = None


@app.on_event("startup")
def startup() -> None:
    ensure_dirs()
    init_db()
    conn = get_db()
    settings = get_settings(conn)
    # Clear any queued/running tasks from previous session.
    try:
        conn.execute(
            "UPDATE tasks SET status = 'canceled', cancel_flag = 1, finished_at = ? "
            "WHERE status IN ('queued','running')",
            (now_iso(),),
        )
    except Exception:
        pass
    # Drain in-memory queue as well.
    try:
        while True:
            TASK_QUEUE.get_nowait()
    except Exception:
        pass
    purge = str(settings.get("purge_assets_on_startup") or "").strip().lower()
    if purge in {"1", "true", "yes", "on"}:
        conn.execute("DELETE FROM assets")
    _db_retry(conn.commit)
    conn.close()
    startup_total = _count_archived_batch_files()
    _startup_import_set(
        running=True,
        total=int(startup_total),
        done=0,
        processed=0,
        failed=0,
        skipped=0,
        current_flow="scan",
        started_at=now_iso(),
        finished_at=None,
    )
    logger.info("Startup: importing archived batch outputs from %s", BATCH_OUTPUT_DIR)
    _run_startup_import_worker(dict(settings))
    if _startup_import_snapshot().get("running"):
        logger.warning("Startup import still running; frontend stays disabled")
    else:
        _configure_frontend(settings)


@app.middleware("http")
async def startup_import_write_lock(request, call_next):
    path = request.url.path or "/"
    # Do not serve UI while startup import replays archived batch files.
    snapshot = _startup_import_snapshot()
    if snapshot.get("running"):
        if path == "/" or path.startswith("/ui"):
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Startup batch import in progress",
                    "startup_import": snapshot,
                },
            )
        if request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Startup batch import in progress",
                    "startup_import": snapshot,
                },
            )
    return await call_next(request)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _delayed_process_exit(delay_seconds: float = 0.8) -> None:
    try:
        time.sleep(max(0.1, float(delay_seconds)))
    finally:
        os._exit(0)


@app.post("/server/restart")
def restart_server() -> Dict[str, Any]:
    # Intended for local tooling. With uvicorn --reload, the reloader process
    # should spawn a fresh worker after this process exits.
    threading.Thread(target=_delayed_process_exit, daemon=True).start()
    return {"status": "restarting", "message": "Server restart requested"}


@app.get("/")
def root() -> Dict[str, str]:
    if UI_ENABLED:
        return RedirectResponse(url="/ui/")
    return {"status": "ok"}


def _ui_index_response(index_path: Path) -> Response:
    html = index_path.read_text(encoding="utf-8")
    # Support dist builds created with vite base="/" by serving assets under /ui.
    html = html.replace('"/assets/', '"/ui/assets/')
    html = html.replace("'/assets/", "'/ui/assets/")
    html = html.replace('"/favicon.ico"', '"/ui/favicon.ico"')
    html = html.replace("'/favicon.ico'", "'/ui/favicon.ico'")
    return Response(content=html, media_type="text/html")


@app.get("/ui/{full_path:path}")
def ui_fallback(full_path: str):
    if not UI_ENABLED or not UI_DIST_DIR:
        raise HTTPException(status_code=404, detail="UI disabled")
    candidate = UI_DIST_DIR / full_path
    if full_path and candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    index_path = UI_DIST_DIR / "index.html"
    if index_path.exists():
        return _ui_index_response(index_path)
    raise HTTPException(status_code=404, detail="UI index missing")


@app.post("/projects")
def create_project(payload: ProjectCreate) -> Dict[str, Any]:
    if not payload.name.strip():
        raise HTTPException(status_code=400, detail="Project name is required")
    if not (payload.source_path or "").strip():
        raise HTTPException(status_code=400, detail="Source content path is required")
    if not (payload.source_folder or "").strip():
        raise HTTPException(status_code=400, detail="Source pack folder is required")

    source_path_value = (payload.source_path or "").strip()
    conn = get_db()
    existing = _find_project_by_source(conn, source_path_value, payload.source_folder)
    if existing:
        conn.close()
        return _project_row_to_dict(existing)

    name_slug = slugify(payload.name.strip())[:16] or "project"
    folder_name = f"{name_slug}-{int(time.time())}"
    folder_path = PROJECTS_DIR / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    (folder_path / "Content").mkdir(parents=True, exist_ok=True)
    _ensure_uproject_file(payload.name.strip(), folder_path, payload.source_path)

    resolved_sources = _resolve_source_paths(payload.source_path, payload.source_folder)
    source_project_root = resolved_sources.get("project_root")
    source_folder_path = resolved_sources.get("source_folder")
    full_copy = bool(payload.full_project_copy)
    if full_copy and source_project_root:
        source_folder_path = None
    screenshot_source = source_folder_path
    if not screenshot_source and payload.source_path:
        screenshot_source = Path(payload.source_path).expanduser()
    screenshot_path = _copy_local_project_screenshot(screenshot_source, folder_path)

    tags_json = json.dumps(payload.tags or [])

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO projects (
            name, link, size_bytes, folder_path, screenshot_path, art_style, project_era, tags_json, description, category_name, is_ai_generated, created_at,
            source_path, source_folder, full_project_copy
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.name.strip(),
            payload.link,
            0,
            str(folder_path),
            screenshot_path,
            payload.art_style,
            payload.project_era,
            tags_json,
            payload.description,
            payload.category_name,
            1 if payload.is_ai_generated else 0 if payload.is_ai_generated is not None else None,
            now_iso(),
            str(source_project_root) if source_project_root else (payload.source_path or None),
            (Path(source_folder_path).name if source_folder_path else (payload.source_folder or None)),
            1 if full_copy else 0,
        ),
    )
    _db_retry(conn.commit)
    project_id = cur.lastrowid
    conn.close()

    copy_started = False

    return {
        "id": project_id,
        "name": payload.name,
        "link": payload.link,
        "size_bytes": 0,
        "folder_size_bytes": _dir_size_bytes(str(folder_path)),
        "source_size_bytes": _dir_size_bytes(str(_resolve_source_content_path(payload.dict()) or "")),
        "folder_path": str(folder_path),
        "screenshot_url": _project_screenshot_url_from_path(screenshot_path),
        "art_style": payload.art_style,
        "project_era": payload.project_era or "",
        "tags": payload.tags or [],
        "source_path": str(source_project_root) if source_project_root else (payload.source_path or ""),
        "source_folder": (Path(source_folder_path).name if source_folder_path else (payload.source_folder or "")),
        "created_at": now_iso(),
        "copy_started": copy_started,
        "full_project_copy": full_copy,
    }


@app.get("/projects")
def list_projects(include_sizes: bool = False) -> List[Dict[str, Any]]:
    conn = get_db()
    rows = fetch_all(conn, "SELECT * FROM projects ORDER BY created_at DESC")
    output = []
    for row in rows:
        normalized_folder_path = _normalize_legacy_project_folder_path(row.get("folder_path"))
        screenshot_path = row.get("screenshot_path")
        screenshot_url = _project_screenshot_url_from_path(screenshot_path)
        if not screenshot_url:
            guessed = _pick_project_image_from_folder(normalized_folder_path)
            if guessed:
                screenshot_path = guessed
                screenshot_url = _project_screenshot_url_from_path(guessed)
        if normalized_folder_path != (row.get("folder_path") or "") or screenshot_path != (row.get("screenshot_path") or ""):
            conn.execute(
                "UPDATE projects SET folder_path = ?, screenshot_path = ? WHERE id = ?",
                (normalized_folder_path, screenshot_path, row["id"]),
            )
        if include_sizes:
            folder_size_bytes = _dir_size_bytes(normalized_folder_path)
            source_size_bytes = _get_cached_source_size(conn, row, max_age_seconds=600, force=True)
        else:
            folder_size_bytes = int(row.get("size_bytes") or 0)
            source_size_bytes = int(row.get("source_size_bytes") or 0)
        output.append(
            {
                "id": row["id"],
                "name": row["name"],
                "link": row["link"],
                "size_bytes": row["size_bytes"],
                "folder_size_bytes": folder_size_bytes,
                "source_size_bytes": source_size_bytes,
                "folder_path": normalized_folder_path,
                "screenshot_url": screenshot_url,
                "art_style": row["art_style"] or "",
                "project_era": row.get("project_era") or "",
                "tags": json.loads(row["tags_json"] or "[]"),
                "source_path": row.get("source_path") or "",
                "source_folder": row.get("source_folder") or "",
                "full_project_copy": bool(row.get("full_project_copy") or 0),
                "created_at": row["created_at"],
            }
    )
    conn.commit()
    conn.close()
    return output


@app.get("/events")
def stream_events() -> StreamingResponse:
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(_event_stream(), media_type="text/event-stream", headers=headers)


@app.post("/events/notify")
def notify_event(payload: UploadEventPayload) -> Dict[str, Any]:
    _broadcast_event(
        {
            "type": "upload",
            "batch_id": payload.batch_id,
            "current": payload.current,
            "total": payload.total,
            "percent": payload.percent,
            "name": payload.name,
            "source": payload.source or "plugin",
        }
    )
    return {"ok": True}


@app.get("/projects/resolve")
def resolve_project(source_path: str, auto_create: bool = True) -> Dict[str, Any]:
    if not source_path.strip():
        raise HTTPException(status_code=400, detail="source_path is required")
    target = _normalize_path_value(source_path)
    conn = get_db()
    existing = _find_project_by_source(conn, source_path, None)
    if existing:
        conn.close()
        return {"project_id": existing["id"]}
    rows = fetch_all(conn, "SELECT id, source_path, source_folder, folder_path FROM projects")
    conn.close()
    for row in rows:
        candidates = [
            _normalize_path_value(row.get("source_path")),
            _normalize_path_value(row.get("source_folder")),
            _normalize_path_value(row.get("folder_path")),
        ]
        if target in candidates:
            return {"project_id": row["id"]}
    if auto_create:
        resolved = _resolve_source_paths(source_path, None)
        project_root = resolved.get("project_root")
        source_folder_path = resolved.get("source_folder")

        candidate_path = Path(source_path).expanduser()
        if candidate_path.is_file():
            candidate_path = candidate_path.parent
        parts = list(candidate_path.parts)
        lower_parts = [p.lower() for p in parts]
        project_name = ""
        if "content" in lower_parts:
            idx = lower_parts.index("content")
            if idx + 1 < len(parts):
                project_name = parts[idx + 1]
        if not project_name:
            if candidate_path.name.lower() == "content":
                project_name = candidate_path.parent.name
            elif candidate_path.parent.name.lower() == "content":
                project_name = candidate_path.name
            else:
                project_name = candidate_path.name or "project"

        source_folder_name = Path(source_folder_path).name if source_folder_path else ""
        conn = get_db()
        if source_folder_name:
            existing = fetch_one(conn, "SELECT id FROM projects WHERE lower(source_folder) = lower(?)", (source_folder_name,))
            if existing:
                conn.close()
                return {"project_id": existing["id"]}

        folder_name = f"{slugify(project_name)}-{int(time.time())}"
        folder_path = PROJECTS_DIR / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        (folder_path / "Content").mkdir(parents=True, exist_ok=True)
        _ensure_uproject_file(project_name, folder_path, source_path)

        screenshot_source = source_folder_path or candidate_path
        screenshot_path = _copy_local_project_screenshot(screenshot_source, folder_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO projects (
                name, link, size_bytes, folder_path, screenshot_path, art_style, project_era, tags_json, created_at,
                source_path, source_folder, full_project_copy
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_name,
                None,
                0,
                str(folder_path),
                screenshot_path,
                None,
                None,
                json.dumps([]),
                now_iso(),
                str(project_root) if project_root else source_path,
                source_folder_name or None,
                0,
            ),
        )
        _db_retry(conn.commit)
        project_id = cur.lastrowid
        conn.close()
        _set_copy_progress(project_id, {"status": "done", "copied": 0, "total": 0})
        return {"project_id": project_id}
    return {"project_id": None}


@app.get("/projects/export")
def export_projects() -> Response:
    conn = get_db()
    rows = fetch_all(conn, "SELECT * FROM projects ORDER BY created_at DESC")
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "name",
            "link",
            "tags",
            "art_style",
            "project_era",
            "source_path",
            "source_folder",
            "full_project_copy",
            "folder_path",
            "created_at",
            "screenshot_url",
        ]
    )
    for row in rows:
        tags = json.loads(row["tags_json"] or "[]")
        writer.writerow(
            [
                row["id"],
                row["name"],
                row.get("link") or "",
                ",".join(tags),
                row.get("art_style") or "",
                row.get("project_era") or "",
                row.get("source_path") or "",
                row.get("source_folder") or "",
                str(int(row.get("full_project_copy") or 0)),
                row.get("folder_path") or "",
                row.get("created_at") or "",
                _project_screenshot_url_from_path(row.get("screenshot_path")),
            ]
        )
    return Response(content=output.getvalue(), media_type="text/csv")


@app.post("/projects/import")
def import_projects(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload must be a csv file")

    logger.info("Projects import queued: filename=%s", file.filename)
    ts = int(time.time() * 1000)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", file.filename)
    temp_path = UPLOADS_DIR / f"projects_import_{ts}_{safe_name}"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    with temp_path.open("wb") as dest:
        shutil.copyfileobj(file.file, dest)
    task_id = _enqueue_task("projects_import")
    _task_update(task_id, message=json.dumps({"file": str(temp_path)}))
    _ensure_task_worker()
    return {"status": "queued", "task_id": task_id, "file": file.filename}


@app.get("/projects/stats")
def project_stats(
    query: Optional[str] = None,
    tag: Optional[str] = None,
    types: Optional[str] = None,
    nanite: Optional[str] = None,
    collision: Optional[str] = None,
) -> Dict[str, Any]:
    conn = get_db()
    totals = fetch_all(
        conn,
        "SELECT project_id, COUNT(*) AS total FROM assets GROUP BY project_id",
    )
    total_map = {row["project_id"]: row["total"] for row in totals}

    filters = []
    params: List[Any] = []
    base = "FROM assets a"

    if types:
        type_list = [t.strip() for t in types.split(",") if t.strip()]
        if type_list:
            placeholders = ",".join(["?"] * len(type_list))
            filters.append(f"a.type IN ({placeholders})")
            params.extend(type_list)
    if nanite is not None and str(nanite).strip() != "":
        raw = str(nanite).strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            filters.append("a.meta_json LIKE ?")
            params.append('%"nanite_enabled": true%')
        elif raw in {"0", "false", "no", "off"}:
            filters.append("(a.meta_json NOT LIKE ? OR a.meta_json IS NULL)")
            params.append('%"nanite_enabled": true%')
    if collision is not None and str(collision).strip() != "":
        raw = str(collision).strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            filters.append("a.meta_json LIKE ?")
            params.append('%"collision_complexity":%')
            filters.append("a.meta_json NOT LIKE ?")
            params.append('%"collision_complexity": "NoCollision"%')
        elif raw in {"0", "false", "no", "off"}:
            filters.append(
                "(a.meta_json NOT LIKE ? OR a.meta_json IS NULL OR a.meta_json LIKE ?)"
            )
            params.append('%"collision_complexity":%')
            params.append('%"collision_complexity": "NoCollision"%')
    if tag:
        filters.append("a.tags_json LIKE ?")
        params.append(f"%{tag}%")
    if query:
        query_lc = query.lower()
        filters.append("(lower(a.name) LIKE ? OR lower(a.description) LIKE ? OR lower(a.tags_json) LIKE ?)")
        like_value = f"%{query_lc}%"
        params.extend([like_value, like_value, like_value])

    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    matched = fetch_all(
        conn,
        f"SELECT a.project_id, COUNT(*) AS matched {base} {where} GROUP BY a.project_id",
        params,
    )
    matched_map = {row["project_id"]: row["matched"] for row in matched}
    bytes_rows = fetch_all(
        conn,
        f"SELECT a.project_id, SUM(a.size_bytes) AS bytes_total {base} {where} GROUP BY a.project_id",
        params,
    )
    tagged_rows = fetch_all(
        conn,
        f"SELECT a.project_id, COUNT(*) AS tagged {base} {where} AND a.tags_json IS NOT NULL AND a.tags_json != '[]' GROUP BY a.project_id" if where else
        f"SELECT a.project_id, COUNT(*) AS tagged {base} WHERE a.tags_json IS NOT NULL AND a.tags_json != '[]' GROUP BY a.project_id",
        params,
    )
    tag_value_rows = fetch_all(
        conn,
        f"SELECT a.project_id, a.tags_json {base} {where} AND a.tags_json IS NOT NULL AND a.tags_json != '[]'" if where else
        f"SELECT a.project_id, a.tags_json {base} WHERE a.tags_json IS NOT NULL AND a.tags_json != '[]'",
        params,
    )
    tagged_map = {row["project_id"]: int(row["tagged"] or 0) for row in tagged_rows}
    bytes_map = {row["project_id"]: int(row["bytes_total"] or 0) for row in bytes_rows}
    tag_assignments_map: Dict[int, int] = {}
    unique_tags_map: Dict[int, set[str]] = {}
    unique_tags_global: set[str] = set()
    for row in tag_value_rows:
        project_id = int(row["project_id"])
        raw_tags = row.get("tags_json")
        if not raw_tags:
            continue
        try:
            loaded = json.loads(raw_tags)
        except Exception:
            continue
        if not isinstance(loaded, list):
            continue
        cleaned = [str(tag).strip() for tag in loaded if str(tag).strip()]
        if not cleaned:
            continue
        tag_assignments_map[project_id] = tag_assignments_map.get(project_id, 0) + len(cleaned)
        bucket = unique_tags_map.setdefault(project_id, set())
        for tag in cleaned:
            bucket.add(tag)
            unique_tags_global.add(tag)
    unique_tags_count_map = {project_id: len(tags) for project_id, tags in unique_tags_map.items()}
    type_rows = fetch_all(
        conn,
        f"SELECT a.project_id, a.type, COUNT(*) AS count {base} {where} GROUP BY a.project_id, a.type",
        params,
    )
    type_map: Dict[int, Dict[str, int]] = {}
    for row in type_rows:
        project_id = row["project_id"]
        asset_type = (row["type"] or "Unknown").strip() or "Unknown"
        bucket = type_map.setdefault(project_id, {})
        bucket[asset_type] = int(row["count"] or 0)
    conn.close()

    items = []
    for project_id, total in total_map.items():
        items.append(
            {
                "project_id": project_id,
                "total": total,
                "matched": matched_map.get(project_id, 0),
                "types": type_map.get(project_id, {}),
                "bytes_total": bytes_map.get(project_id, 0),
                "tagged": tagged_map.get(project_id, 0),
                "tag_assignments_total": tag_assignments_map.get(project_id, 0),
                "unique_tags_count": unique_tags_count_map.get(project_id, 0),
            }
        )
    total_assets = sum(int(item.get("total") or 0) for item in items)
    tagged_assets = sum(int(item.get("tagged") or 0) for item in items)
    tag_assignments_total = sum(int(item.get("tag_assignments_total") or 0) for item in items)
    summary = {
        "assets_total": total_assets,
        "assets_tagged": tagged_assets,
        "assets_without_tags": max(0, total_assets - tagged_assets),
        "tag_assignments_total": tag_assignments_total,
        "unique_tags_total": len(unique_tags_global),
        "avg_tags_per_tagged_asset": (tag_assignments_total / tagged_assets) if tagged_assets else 0.0,
    }
    return {"items": items, "summary": summary}


@app.get("/projects/{project_id}")
def get_project(project_id: int) -> Dict[str, Any]:
    conn = get_db()
    row = fetch_one(conn, "SELECT * FROM projects WHERE id = ?", (project_id,))
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")
    normalized_folder_path = _normalize_legacy_project_folder_path(row.get("folder_path"))
    screenshot_path = row.get("screenshot_path")
    screenshot_url = _project_screenshot_url_from_path(screenshot_path)
    if not screenshot_url:
        guessed = _pick_project_image_from_folder(normalized_folder_path)
        if guessed:
            screenshot_path = guessed
            screenshot_url = _project_screenshot_url_from_path(guessed)
    if normalized_folder_path != (row.get("folder_path") or "") or screenshot_path != (row.get("screenshot_path") or ""):
        conn.execute(
            "UPDATE projects SET folder_path = ?, screenshot_path = ? WHERE id = ?",
            (normalized_folder_path, screenshot_path, row["id"]),
        )
        conn.commit()
    conn.close()
    folder_size_bytes = _dir_size_bytes(normalized_folder_path)
    source_size_bytes = _dir_size_bytes(str(_resolve_source_content_path(row) or ""))
    return {
        "id": row["id"],
        "name": row["name"],
        "link": row["link"],
        "size_bytes": row["size_bytes"],
        "folder_size_bytes": folder_size_bytes,
        "source_size_bytes": source_size_bytes,
        "folder_path": normalized_folder_path,
        "screenshot_url": screenshot_url,
        "art_style": row["art_style"] or "",
        "project_era": row.get("project_era") or "",
        "tags": json.loads(row["tags_json"] or "[]"),
        "source_path": row.get("source_path") or "",
        "source_folder": row.get("source_folder") or "",
        "full_project_copy": bool(row.get("full_project_copy") or 0),
        "created_at": row["created_at"],
    }


@app.put("/projects/{project_id}")
def update_project(project_id: int, payload: ProjectUpdate) -> Dict[str, Any]:
    data = payload.dict(exclude_unset=True)
    if "tag_translation_language" in data and data.get("tag_translation_language"):
        data["tag_language"] = data.get("tag_translation_language")
        data.pop("tag_translation_language", None)
    if not data:
        return {"status": "ok"}
    if "name" in data and data["name"] is not None:
        data["name"] = data["name"].strip()
        if not data["name"]:
            raise HTTPException(status_code=400, detail="Project name is required")
    if "tags" in data:
        data["tags_json"] = json.dumps(data.pop("tags") or [])
    if "source_path" in data and data["source_path"] is not None:
        data["source_path"] = data["source_path"].strip() or None
    if "source_folder" in data and data["source_folder"] is not None:
        cleaned = re.split(r"[\/]+", str(data["source_folder"]).strip().strip("/\\"))
        data["source_folder"] = (cleaned[-1] if cleaned else data["source_folder"]).strip() or None
    if "full_project_copy" in data and data["full_project_copy"] is not None:
        data["full_project_copy"] = 1 if data["full_project_copy"] else 0

    conn = get_db()
    project = fetch_one(conn, "SELECT * FROM projects WHERE id = ?", (project_id,))
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")

    fields = []
    params: List[Any] = []
    for key, value in data.items():
        fields.append(f"{key} = ?")
        params.append(value)
    params.append(project_id)
    conn.execute(f"UPDATE projects SET {', '.join(fields)} WHERE id = ?", params)
    _db_retry(conn.commit)
    conn.close()
    return {"status": "ok"}


@app.post("/projects/{project_id}/screenshot")
def upload_project_screenshot(
    project_id: int,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
) -> Dict[str, Any]:
    conn = get_db()
    project = fetch_one(conn, "SELECT * FROM projects WHERE id = ?", (project_id,))
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")

    folder_path = Path(project["folder_path"])
    folder_path.mkdir(parents=True, exist_ok=True)
    # Always persist project screenshots inside DATA_DIR so /media can serve them
    project_slug = folder_path.name
    media_project_dir = DATA_DIR / "projects" / project_slug
    media_project_dir.mkdir(parents=True, exist_ok=True)

    if file is None and not url:
        conn.close()
        raise HTTPException(status_code=400, detail="Provide a file or url")

    filename = ""
    if file is not None:
        filename = file.filename or "screenshot"
        ext = Path(filename).suffix.lower() or ".jpg"
        dest = media_project_dir / f"screenshot{ext}"
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    else:
        url = url.strip()
        if not url:
            conn.close()
            raise HTTPException(status_code=400, detail="URL is empty")
        dest = media_project_dir / "screenshot.jpg"
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            dest.write_bytes(response.content)

    cur = conn.cursor()
    cur.execute("UPDATE projects SET screenshot_path = ? WHERE id = ?", (str(dest), project_id))
    _db_retry(conn.commit)
    conn.close()

    try:
        rel = dest.resolve().relative_to(DATA_DIR.resolve())
        screenshot_url = f"/media/{str(rel).replace('\\', '/')}"
    except ValueError:
        screenshot_url = ""

    return {"status": "ok", "screenshot_url": screenshot_url}


@app.post("/projects/{project_id}/open")
def open_project_folder(project_id: int, target: str = Query("auto")) -> Dict[str, str]:
    conn = get_db()
    project = fetch_one(conn, "SELECT * FROM projects WHERE id = ?", (project_id,))
    conn.close()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    def open_source() -> Dict[str, str]:
        source_content = _resolve_source_content_path(project)
        candidate = None
        if source_content and source_content.exists():
            candidate = source_content
        else:
            source_path = project.get("source_path")
            source_folder = project.get("source_folder")
            if source_path and Path(source_path).exists():
                candidate = Path(source_path)
            elif source_folder and Path(source_folder).exists():
                candidate = Path(source_folder)
        if not candidate or not candidate.exists():
            raise HTTPException(status_code=404, detail="Source folder not found")
        # open folder that contains a Content directory if possible
        container = candidate
        if candidate.name.lower() == "content":
            container = candidate.parent
        else:
            for parent in candidate.parents:
                if parent.name.lower() == "content":
                    container = parent.parent
                    break
        os.startfile(str(container))
        return {"status": "opened", "target": "source"}

    def open_project() -> Dict[str, str]:
        folder_path = project["folder_path"]
        if not folder_path or not Path(folder_path).exists():
            raise HTTPException(status_code=404, detail="Folder not found")
        os.startfile(folder_path)
        return {"status": "opened", "target": "project"}

    if target == "source":
        return open_source()
    if target == "project":
        return open_project()

    # auto: prefer project if Content has any files, else source
    folder_path = project.get("folder_path") or ""
    content_dir = Path(folder_path) / "Content" if folder_path else None
    if content_dir and content_dir.exists():
        try:
            if any(p.is_file() for p in content_dir.rglob("*")):
                return open_project()
        except Exception:
            pass
    return open_source()


@app.post("/projects/{project_id}/reimport")
def reimport_project(project_id: int, payload: ProjectReimport) -> Dict[str, Any]:
    conn = get_db()
    project = fetch_one(conn, "SELECT * FROM projects WHERE id = ?", (project_id,))
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")

    source_path = payload.source_path or project.get("source_path")
    source_folder = payload.source_folder or project.get("source_folder")
    if source_folder:
        cleaned = re.split(r"[\/]+", str(source_folder).strip().strip("/\\"))
        source_folder = cleaned[-1] if cleaned else source_folder
    full_copy = (
        payload.full_project_copy
        if payload.full_project_copy is not None
        else bool(project.get("full_project_copy") or 0)
    )
    if not source_path and not source_folder:
        conn.close()
        raise HTTPException(status_code=400, detail="Source path is missing")

    conn.execute(
        "UPDATE projects SET source_path = ?, source_folder = ?, full_project_copy = ?, reimported_once = 1 WHERE id = ?",
        (source_path, source_folder, 1 if full_copy else 0, project_id),
    )
    _db_retry(conn.commit)
    conn.close()

    folder_path = Path(project["folder_path"])
    _set_copy_progress(project_id, {"status": "queued", "copied": 0, "total": 0})
    thread = threading.Thread(
        target=_reimport_project,
        args=(project_id, project.get("name") or "project", source_path, source_folder, folder_path, full_copy),
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


@app.post("/projects/{project_id}/export-cmd")
def run_project_export_cmd(project_id: int, payload: ProjectExportCmd) -> Dict[str, Any]:
    conn = get_db()
    settings = get_settings(conn)
    project = fetch_one(conn, "SELECT * FROM projects WHERE id = ?", (project_id,))
    conn.close()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    ue_cmd = (settings.get("ue_cmd_path") or "").strip()
    if not ue_cmd:
        raise HTTPException(status_code=400, detail="ue_cmd_path is not configured")

    folder_path = (project.get("folder_path") or "").strip()
    if not folder_path:
        raise HTTPException(status_code=400, detail="Project folder_path is empty")

    dest_root = Path(folder_path)
    source_root = Path(project["source_path"]).expanduser() if project.get("source_path") else None
    reimported_once = bool(project.get("reimported_once") or 0)
    if reimported_once:
        work_root = dest_root
    elif source_root and source_root.exists():
        work_root = source_root
    else:
        work_root = dest_root
    uproject_path = _prefer_import_uproject(work_root)

    game_path = _resolve_export_game_path(project, payload.game_path if payload else None)

    args = [
        ue_cmd,
        str(uproject_path),
        f"-ExecCmds=aeb {game_path}",
        "-exit",
        "-nop4",
        "-nosplash",
        "-unattended",
    ]
    extra_args_raw = (settings.get("ue_cmd_extra_args") or "").strip()
    if extra_args_raw:
        try:
            args.extend(shlex.split(extra_args_raw, posix=os.name != "nt"))
        except ValueError:
            args.append(extra_args_raw)

    if os.name == "nt":
        cmdline = subprocess.list2cmdline(args)
        subprocess.Popen(
            args,
            cwd=str(work_root),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        command = cmdline
    else:
        subprocess.Popen(args, cwd=str(work_root))
        command = " ".join(args)

    logger.info("Export cmd: %s", command)
    return {"status": "started", "command": command, "uproject": str(uproject_path), "game_path": game_path}


@app.get("/projects/{project_id}/copy-status")
def project_copy_status(project_id: int) -> Dict[str, Any]:
    with COPY_LOCK:
        return COPY_PROGRESS.get(project_id, {"status": "idle", "copied": 0, "total": 0})


@app.get("/projects/{project_id}/tag-status")
def project_tag_status(project_id: int) -> Dict[str, Any]:
    with TAG_LOCK:
        return TAG_PROGRESS.get(project_id, {"status": "idle", "done": 0, "total": 0, "errors": 0})


@app.post("/projects/{project_id}/tag-missing")
def tag_missing_project_assets(project_id: int) -> Dict[str, Any]:
    _set_tag_progress(project_id, {"status": "queued", "done": 0, "total": 0, "errors": 0})
    task_id = _enqueue_task("tag_project_missing", project_id)
    return {"status": "queued", "task_id": task_id}


@app.post("/projects/{project_id}/retag-all")
def retag_project_assets(project_id: int) -> Dict[str, Any]:
    _set_tag_progress(project_id, {"status": "queued", "done": 0, "total": 0, "errors": 0})
    task_id = _enqueue_task("tag_project_retag", project_id)
    return {"status": "queued", "task_id": task_id}


@app.post("/projects/{project_id}/name-tags")
def translate_project_names_to_tags(project_id: int) -> Dict[str, Any]:
    task_id = _enqueue_task("name_tags_project", project_id)
    return {"status": "queued", "task_id": task_id}

@app.post("/projects/{project_id}/name-tags-missing")
def translate_project_names_to_tags_missing(project_id: int) -> Dict[str, Any]:
    task_id = _enqueue_task("name_tags_project_missing", project_id)
    return {"status": "queued", "task_id": task_id}

@app.post("/projects/{project_id}/name-tags-simple")
def name_tags_project_simple(project_id: int) -> Dict[str, Any]:
    task_id = _enqueue_task("name_tags_project_simple", project_id)
    return {"status": "queued", "task_id": task_id}

@app.post("/projects/{project_id}/name-tags-simple-missing")
def name_tags_project_simple_missing(project_id: int) -> Dict[str, Any]:
    task_id = _enqueue_task("name_tags_project_simple_missing", project_id)
    return {"status": "queued", "task_id": task_id}

@app.post("/projects/{project_id}/translate-tags")
def translate_tags_project(project_id: int) -> Dict[str, Any]:
    task_id = _enqueue_task("tags_translate_project", project_id)
    return {"status": "queued", "task_id": task_id}

@app.post("/projects/{project_id}/translate-tags-missing")
def translate_tags_project_missing(project_id: int) -> Dict[str, Any]:
    task_id = _enqueue_task("tags_translate_project_missing", project_id)
    return {"status": "queued", "task_id": task_id}


@app.post("/llm/test-tags")
def test_llm_tags(
    file: Optional[UploadFile] = File(None),
    settings_json: Optional[str] = Form(None),
) -> Dict[str, Any]:
    conn = get_db()
    settings = get_settings(conn)
    conn.close()
    if settings_json:
        try:
            overrides = json.loads(settings_json)
            if isinstance(overrides, dict):
                for key in ("api_key", "openai_api_key", "openrouter_api_key", "groq_api_key"):
                    if key in overrides:
                        raw = overrides.get(key)
                        if not raw or str(raw).replace('*', '').strip() == '':
                            overrides.pop(key, None)
                settings.update(overrides)
        except Exception:
            pass
    # quick key check for providers that require it
    provider = (settings.get("provider") or "").strip().lower()
    if provider in {"openai", "openrouter", "groq"}:
        key_map = {
            "openai": "openai_api_key",
            "openrouter": "openrouter_api_key",
            "groq": "groq_api_key",
        }
        key_name = key_map.get(provider)
        if key_name and not settings.get(key_name):
            raise HTTPException(status_code=400, detail="Missing API key for active provider")

    image_data_url = None
    try:
        if file is not None:
            data = file.file.read()
            b64 = base64.b64encode(data).decode("utf-8")
            mime = file.content_type or "image/png"
            image_data_url = f"data:{mime};base64,{b64}"
        else:
            sample_path = BASE_DIR.parent / "logo64.png"
            if sample_path.exists():
                data = sample_path.read_bytes()
                b64 = base64.b64encode(data).decode("utf-8")
                image_data_url = f"data:image/png;base64,{b64}"
    except Exception:
        image_data_url = None

    result = generate_tags_debug(
        settings,
        "Test image",
        "Small test image for tagging",
        [],
        image_data_url,
        "StaticMesh",
    )
    result["tags"] = _normalize_tags(result.get("tags") or [])
    return {"status": "ok", **result}


@app.post("/projects/name-tags-all")
def translate_all_names_to_tags() -> Dict[str, Any]:
    task_id = _enqueue_task("name_tags_all", None)
    return {"status": "queued", "task_id": task_id}

@app.post("/projects/name-tags-all-missing")
def translate_all_names_to_tags_missing() -> Dict[str, Any]:
    task_id = _enqueue_task("name_tags_all_missing", None)
    return {"status": "queued", "task_id": task_id}

@app.post("/projects/name-tags-all-simple")
def name_tags_all_simple() -> Dict[str, Any]:
    task_id = _enqueue_task("name_tags_all_simple", None)
    return {"status": "queued", "task_id": task_id}

@app.post("/projects/name-tags-all-simple-missing")
def name_tags_all_simple_missing() -> Dict[str, Any]:
    task_id = _enqueue_task("name_tags_all_simple_missing", None)
    return {"status": "queued", "task_id": task_id}

@app.post("/projects/translate-tags-all")
def translate_tags_all() -> Dict[str, Any]:
    task_id = _enqueue_task("tags_translate_all", None)
    return {"status": "queued", "task_id": task_id}

@app.post("/projects/translate-tags-all-missing")
def translate_tags_all_missing() -> Dict[str, Any]:
    task_id = _enqueue_task("tags_translate_all_missing", None)
    return {"status": "queued", "task_id": task_id}


@app.get("/projects/{project_id}/embedding-status")
def project_embedding_status(project_id: int) -> Dict[str, Any]:
    with EMBED_LOCK:
        return EMBED_PROGRESS.get(str(project_id), {"status": "idle", "done": 0, "total": 0, "errors": 0})

@app.post("/projects/tag-missing-all")
def tag_missing_all() -> Dict[str, Any]:
    task_id = _enqueue_task("tag_missing_all", None)
    return {"status": "queued", "task_id": task_id}



@app.post("/projects/{project_id}/embeddings/regenerate")
def regenerate_project_embeddings(project_id: int) -> Dict[str, Any]:
    _set_embed_progress(str(project_id), {"status": "queued", "done": 0, "total": 0, "errors": 0})
    task_id = _enqueue_task("embeddings_project", project_id)
    return {"status": "queued", "task_id": task_id}


@app.get("/embeddings/status")
def all_embeddings_status() -> Dict[str, Any]:
    with EMBED_LOCK:
        return EMBED_PROGRESS.get("all", {"status": "idle", "done": 0, "total": 0, "errors": 0})


@app.post("/embeddings/regenerate-all")
def regenerate_all_embeddings() -> Dict[str, Any]:
    _set_embed_progress("all", {"status": "queued_restart", "done": 0, "total": 0, "errors": 0})
    task_id = _enqueue_task("embeddings_all_deferred", None)
    return {"status": "queued_restart", "task_id": task_id}


@app.put("/settings")
def update_settings(payload: SettingsUpdate) -> Dict[str, str]:
    conn = get_db()
    cur = conn.cursor()
    data = payload.dict(exclude_unset=True)
    if "tag_translation_language" in data and data.get("tag_translation_language"):
        data["tag_language"] = data.get("tag_translation_language")
        data.pop("tag_translation_language", None)
    if "tag_batch_max_assets" in data:
        data["tag_batch_max_assets"] = _normalized_batch_size(data.get("tag_batch_max_assets"))
    for key, value in data.items():
        if value is None:
            continue
        if key in {
            "skip_export_if_on_server",
            "purge_assets_on_startup",
            "serve_frontend",
            "tag_use_batch_mode",
            "tag_translate_enabled",
            "generate_embeddings_on_import",
            "default_full_project_copy",
        }:
            if isinstance(value, bool):
                value = "true" if value else "false"
        cur.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
    _db_retry(conn.commit)
    conn.close()
    return {"status": "ok"}

@app.get("/tasks")
def list_tasks(limit: int = 200) -> Dict[str, Any]:
    conn = get_db()
    # Keep recent tasks, but always include active ones and tasks with pending OpenAI batches.
    rows = fetch_all(
        conn,
        """
        SELECT *
        FROM tasks
        WHERE id IN (
            SELECT id FROM (
                SELECT id
                FROM tasks
                WHERE status IN ('queued', 'running')
                UNION
                SELECT DISTINCT task_id
                FROM openai_batches
                WHERE processed_at IS NULL AND task_id IS NOT NULL
                UNION
                SELECT id
                FROM tasks
                ORDER BY id DESC
                LIMIT ?
            )
        )
        ORDER BY id DESC
        """,
        (limit,),
    )
    task_ids = [int(r["id"]) for r in rows if r.get("id") is not None]
    openai_stats: Dict[int, Dict[str, int]] = {}
    if task_ids:
        placeholders = ",".join(["?"] * len(task_ids))
        stat_rows = fetch_all(
            conn,
            f"""
            SELECT
                task_id,
                COUNT(*) AS total,
                SUM(CASE WHEN processed_at IS NULL THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN processed_at IS NULL AND output_file_id IS NOT NULL THEN 1 ELSE 0 END) AS ready,
                SUM(CASE WHEN processed_at IS NULL AND status = 'in_progress' THEN 1 ELSE 0 END) AS in_progress,
                SUM(CASE WHEN processed_at IS NULL AND status = 'finalizing' THEN 1 ELSE 0 END) AS finalizing,
                SUM(CASE WHEN processed_at IS NULL AND (status IS NULL OR status IN ('validating','queued','running')) THEN 1 ELSE 0 END) AS waiting,
                SUM(CASE WHEN processed_at IS NOT NULL THEN 1 ELSE 0 END) AS processed
            FROM openai_batches
            WHERE task_id IN ({placeholders})
            GROUP BY task_id
            """,
            tuple(task_ids),
        )
        for s in stat_rows:
            tid = int(s.get("task_id") or 0)
            if tid > 0:
                openai_stats[tid] = {
                    "total": int(s.get("total") or 0),
                    "pending": int(s.get("pending") or 0),
                    "ready": int(s.get("ready") or 0),
                    "in_progress": int(s.get("in_progress") or 0),
                    "finalizing": int(s.get("finalizing") or 0),
                    "waiting": int(s.get("waiting") or 0),
                    "processed": int(s.get("processed") or 0),
                }
    conn.close()
    items = []
    for row in rows:
        payload = dict(row)
        progress_raw = payload.get("progress_json")
        if progress_raw:
            try:
                payload["progress"] = json.loads(progress_raw)
            except Exception:
                payload["progress"] = None
        task_id = int(payload.get("id") or 0)
        if task_id in openai_stats:
            payload["openai_batches"] = openai_stats[task_id]
        items.append(payload)
    return {"items": items}


@app.get("/queue/status")
def queue_status() -> Dict[str, Any]:
    return _queue_status_snapshot()


@app.get("/openai/batches")
def list_openai_batches(
    flow: Optional[str] = None,
    only_open: bool = True,
    limit: int = 500,
) -> Dict[str, Any]:
    conn = get_db()
    where: List[str] = []
    params: List[Any] = []
    if flow:
        where.append("flow = ?")
        params.append(flow)
    if only_open:
        where.append("processed_at IS NULL")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = fetch_all(
        conn,
        f"""
        SELECT *
        FROM openai_batches
        {where_sql}
        ORDER BY id DESC
        LIMIT ?
        """,
        tuple(params + [limit]),
    )
    conn.close()
    return {"items": rows}


@app.post("/openai/recover-now")
def recover_openai_batches_now(
    limit: int = 100,
    flow: Optional[str] = None,
    task_id: Optional[int] = None,
    stale_minutes: int = 180,
) -> Dict[str, Any]:
    stats = _recover_openai_batches_once(
        limit=max(1, min(int(limit), 1000)),
        flow=(str(flow).strip() or None) if flow is not None else None,
        task_id=task_id,
        stale_minutes=max(10, min(int(stale_minutes), 1440)),
    )
    return {"status": "ok", **stats}


@app.post("/openai/recover-enqueue")
def recover_openai_batches_enqueue(
    limit: int = 300,
    flow: Optional[str] = None,
    task_id: Optional[int] = None,
    stale_minutes: int = 180,
) -> Dict[str, Any]:
    payload = {
        "limit": max(1, min(int(limit), 1000)),
        "flow": (str(flow).strip() or None) if flow is not None else None,
        "openai_task_id": task_id,
        "stale_minutes": max(10, min(int(stale_minutes), 1440)),
    }
    enqueued_task_id = _enqueue_task("openai_recover", None, json.dumps(payload))
    return {"status": "ok", "task_id": int(enqueued_task_id), "payload": payload}


@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> Dict[str, Any]:
    row = _task_get(task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    payload = dict(row)
    progress_raw = payload.get("progress_json")
    if progress_raw:
        try:
            payload["progress"] = json.loads(progress_raw)
        except Exception:
            payload["progress"] = None
    conn = get_db()
    stat = fetch_one(
        conn,
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN processed_at IS NULL THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN processed_at IS NULL AND output_file_id IS NOT NULL THEN 1 ELSE 0 END) AS ready,
            SUM(CASE WHEN processed_at IS NULL AND status = 'in_progress' THEN 1 ELSE 0 END) AS in_progress,
            SUM(CASE WHEN processed_at IS NULL AND status = 'finalizing' THEN 1 ELSE 0 END) AS finalizing,
            SUM(CASE WHEN processed_at IS NULL AND (status IS NULL OR status IN ('validating','queued','running')) THEN 1 ELSE 0 END) AS waiting,
            SUM(CASE WHEN processed_at IS NOT NULL THEN 1 ELSE 0 END) AS processed
        FROM openai_batches
        WHERE task_id = ?
        """,
        (task_id,),
    )
    conn.close()
    if stat and int(stat.get("total") or 0) > 0:
        payload["openai_batches"] = {
            "total": int(stat.get("total") or 0),
            "pending": int(stat.get("pending") or 0),
            "ready": int(stat.get("ready") or 0),
            "in_progress": int(stat.get("in_progress") or 0),
            "finalizing": int(stat.get("finalizing") or 0),
            "waiting": int(stat.get("waiting") or 0),
            "processed": int(stat.get("processed") or 0),
        }
    return payload


@app.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: int) -> Dict[str, Any]:
    row = _task_get(task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    status = row.get("status")
    if status not in {"queued", "running"}:
        return {"status": status, "canceled": False}
    _task_update(task_id, cancel_flag=1)
    return {"status": "canceling"}

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int) -> Dict[str, Any]:
    row = _task_get(task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    def _write() -> None:
        conn = get_db()
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        conn.close()

    _db_retry(_write)
    return {"status": "ok"}

@app.post("/tasks/cleanup")
def cleanup_tasks() -> Dict[str, Any]:
    deleted_box: Dict[str, int] = {"value": 0}

    def _write() -> None:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM tasks WHERE status IN ('done','error','canceled')")
        deleted_box["value"] = int(cur.rowcount or 0)
        conn.commit()
        conn.close()

    _db_retry(_write)
    deleted = deleted_box["value"]
    return {"status": "ok", "deleted": deleted}


@app.post("/admin/reset-db")
def reset_db() -> Dict[str, Any]:
    conn = get_db()
    backup_path = None
    try:
        backup_dir = DB_PATH.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"app_{timestamp}.db"
        backup_conn = sqlite3.connect(backup_path)
        conn.backup(backup_conn)
        backup_conn.close()
    except Exception as exc:
        logger.warning(f"DB backup failed: {exc}")
        backup_path = None

@app.post("/tasks/cancel-all")
def cancel_all_tasks() -> Dict[str, Any]:
    def _write() -> None:
        conn = get_db()
        conn.execute("UPDATE tasks SET cancel_flag = 1 WHERE status IN ('queued','running')")
        conn.commit()
        conn.close()

    _db_retry(_write)
    return {"status": "canceling"}


    conn.execute("DELETE FROM asset_tags")
    conn.execute("DELETE FROM assets")
    conn.execute("DELETE FROM projects")
    conn.execute("DELETE FROM assets_fts")
    _db_retry(conn.commit)
    conn.close()

    with COPY_LOCK:
        COPY_PROGRESS.clear()
    with MIGRATE_LOCK:
        MIGRATE_PROGRESS.clear()
    with TAG_LOCK:
        TAG_PROGRESS.clear()
    with EMBED_LOCK:
        EMBED_PROGRESS.clear()

    return {"status": "ok", "backup_path": str(backup_path) if backup_path else None}



@app.get("/settings")
def read_settings(conn: sqlite3.Connection = Depends(get_db_dep)) -> Dict[str, Any]:
    settings = get_settings(conn)
    masked = {**settings}
    if "import_base_url" not in masked:
        masked["import_base_url"] = "http://127.0.0.1:9090"
    if "ue_cmd_path" not in masked:
        masked["ue_cmd_path"] = ""
    if "skip_export_if_on_server" in masked:
        raw = str(masked.get("skip_export_if_on_server") or "").strip().lower()
        masked["skip_export_if_on_server"] = "true" if raw in {"1", "true", "yes", "on"} else "false"
    if "export_overwrite_zips" in masked:
        raw = str(masked.get("export_overwrite_zips") or "").strip().lower()
        masked["export_overwrite_zips"] = "true" if raw in {"1", "true", "yes", "on"} else "false"
    if "export_upload_after_export" in masked:
        raw = str(masked.get("export_upload_after_export") or "").strip().lower()
        masked["export_upload_after_export"] = "true" if raw in {"1", "true", "yes", "on"} else "false"
    if "serve_frontend" in masked:
        raw = str(masked.get("serve_frontend") or "").strip().lower()
        masked["serve_frontend"] = "true" if raw in {"1", "true", "yes", "on"} else "false"
    if "tag_translate_enabled" in masked:
        raw = str(masked.get("tag_translate_enabled") or "").strip().lower()
        masked["tag_translate_enabled"] = "true" if raw in {"1", "true", "yes", "on"} else "false"
    if "tag_use_batch_mode" in masked:
        raw = str(masked.get("tag_use_batch_mode") or "").strip().lower()
        masked["tag_use_batch_mode"] = "true" if raw in {"1", "true", "yes", "on"} else "false"
    if "generate_embeddings_on_import" in masked:
        raw = str(masked.get("generate_embeddings_on_import") or "").strip().lower()
        masked["generate_embeddings_on_import"] = "true" if raw in {"1", "true", "yes", "on"} else "false"
    else:
        masked["generate_embeddings_on_import"] = "true"
    if "default_full_project_copy" in masked:
        raw = str(masked.get("default_full_project_copy") or "").strip().lower()
        masked["default_full_project_copy"] = "true" if raw in {"1", "true", "yes", "on"} else "false"
    else:
        masked["default_full_project_copy"] = "false"
    if "tag_display_limit" in masked:
        raw = str(masked.get("tag_display_limit") or "").strip()
        if not raw.isdigit():
            masked["tag_display_limit"] = "0"
    else:
        masked["tag_display_limit"] = "0"
    if "sidebar_width" in masked:
        raw = str(masked.get("sidebar_width") or "").strip()
        masked["sidebar_width"] = raw if raw.isdigit() else ""
    if "export_exclude_types" not in masked:
        masked["export_exclude_types"] = "Material,MaterialInstance,MaterialInstanceConstant"
    if "api_key" in masked:
        masked["api_key"] = "***" if masked["api_key"] else ""
        masked["has_api_key"] = bool(settings.get("api_key"))
    if "openai_api_key" in masked:
        masked["openai_api_key"] = "***" if masked["openai_api_key"] else ""
        masked["has_openai_api_key"] = bool(settings.get("openai_api_key"))
    if "openrouter_api_key" in masked:
        masked["openrouter_api_key"] = "***" if masked["openrouter_api_key"] else ""
        masked["has_openrouter_api_key"] = bool(settings.get("openrouter_api_key"))
    if "groq_api_key" in masked:
        masked["groq_api_key"] = "***" if masked["groq_api_key"] else ""
        masked["has_groq_api_key"] = bool(settings.get("groq_api_key"))
    return masked


@app.get("/uploads/last")
def get_last_upload() -> Dict[str, Any]:
    return _get_last_upload_age()


@app.post("/assets/upload")
async def upload_asset(
    file: UploadFile = File(...),
    project_id: Optional[int] = Form(None),
) -> Dict[str, Any]:
    return await run_in_threadpool(_upload_asset_sync, file, project_id)


def _upload_asset_sync(
    file: UploadFile,
    project_id: Optional[int],
) -> Dict[str, Any]:
    ensure_dirs()
    if not file.filename.lower().endswith(".zip"):
        logger.warning("Upload rejected (not zip): filename=%s project_id=%s", file.filename, project_id)
        raise HTTPException(status_code=400, detail="Upload must be a zip file")

    temp_key = str(uuid.uuid4())
    zip_path = UPLOADS_DIR / f"{temp_key}.zip"
    with zip_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("meta.json") as meta_file:
                meta = json.loads(meta_file.read().decode("utf-8"))
        logger.info("Upload meta loaded: project_id=%s package=%s class=%s vendor=%s hash_full=%s", project_id, meta.get("package"), meta.get("class"), meta.get("vendor"), meta.get("hash_full_blake3"))
        _set_last_upload()
    except KeyError as exc:
        logger.warning("Upload rejected: meta.json missing in zip (file=%s)", file.filename)
        raise HTTPException(status_code=400, detail="meta.json not found in zip") from exc
    except Exception as exc:
        logger.exception("Upload rejected: failed to read meta.json (file=%s)", file.filename)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    hash_main = (meta.get("hash_main_blake3") or "").strip()
    if not hash_main:
        logger.warning("Upload rejected: hash_main_blake3 missing (file=%s)", file.filename)
        raise HTTPException(status_code=400, detail="hash_main_blake3 missing in meta")
    hash_full = (meta.get("hash_full_blake3") or "").strip()
    if not hash_full:
        logger.warning("Upload rejected: hash_full_blake3 missing (file=%s)", file.filename)
        raise HTTPException(status_code=400, detail="hash_full_blake3 missing in meta")
    hash_main_sha256 = (meta.get("hash_main_sha256") or "").strip()

    conn = get_db()
    try:
        project_row = fetch_one(conn, "SELECT id FROM projects WHERE id = ?", (project_id,)) if project_id is not None else None
        if project_id is not None and not project_row:
            logger.warning("Upload project_id %s not found in DB", project_id)
        if not project_row:
            resolved_id = _resolve_project_id_from_meta(conn, meta)
            if resolved_id:
                project_id = resolved_id
            else:
                vendor = str(meta.get("vendor") or "").strip()
                if vendor:
                    existing_vendor = fetch_one(conn, "SELECT id FROM projects WHERE lower(source_folder) = lower(?)", (vendor,))
                    if existing_vendor:
                        project_id = existing_vendor["id"]
                    else:
                        logger.warning("Upload resolve failed, auto-creating project for vendor=%s", vendor)
                        project_id = _create_project_from_root(vendor)
                else:
                    logger.warning("Upload rejected: project_id missing and could not be resolved (vendor/package=%s)", meta.get("package"))
                    raise HTTPException(status_code=400, detail="project_id missing and could not be resolved from meta")

        existing = fetch_one(
            conn,
            "SELECT id, size_bytes FROM assets WHERE project_id = ? AND hash_full_blake3 = ? LIMIT 1",
            (project_id, hash_full),
        )
    finally:
        conn.close()
    if existing:
        asset_prefix = hash_main[:3]
        asset_dir = ASSETS_DIR / asset_prefix / hash_main
        asset_dir.mkdir(parents=True, exist_ok=True)
        try:
            meta, preview_files, thumb_image, detail_image, full_image, anim_thumb, anim_detail = process_asset_zip(
                zip_path, asset_dir, hash_main, meta.get("class")
            )
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
        except Exception as exc:
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
            logger.exception("Upload rejected: process_asset_zip failed (file=%s)", file.filename)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        size_bytes = int(meta.get("disk_bytes_total") or 0)
        if size_bytes <= 0:
            size_bytes = int(existing.get("size_bytes") or 0)

        write_conn = get_db()
        cur = write_conn.cursor()
        cur.execute(
            """
            UPDATE assets
            SET images_json = ?, thumb_image = ?, detail_image = ?, full_image = ?,
                anim_thumb = ?, anim_detail = ?, meta_json = ?, size_bytes = ?
            WHERE id = ?
            """,
            (
                json.dumps(preview_files),
                thumb_image,
                detail_image,
                full_image,
                anim_thumb,
                anim_detail,
                json.dumps(meta),
                size_bytes,
                existing["id"],
            ),
        )
        _db_retry(write_conn.commit)
        write_conn.close()
        logger.info("Upload duplicate: refreshed images project_id=%s hash_full=%s", project_id, hash_full)
        return {"status": "updated_images", "id": existing["id"], "project_id": project_id}

    asset_prefix = hash_main[:3]
    asset_dir = ASSETS_DIR / asset_prefix / hash_main
    asset_dir.mkdir(parents=True, exist_ok=True)

    asset_zip_rel = ""
    asset_zip_path = asset_dir / f"{hash_main}.zip"
    try:
        meta, preview_files, thumb_image, detail_image, full_image, anim_thumb, anim_detail = process_asset_zip(
            zip_path, asset_dir, hash_main, meta.get("class")
        )
        if asset_zip_path.exists():
            asset_zip_path.unlink(missing_ok=True)
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)
    except Exception as exc:
        shutil.rmtree(asset_dir, ignore_errors=True)
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)
        logger.exception("Upload rejected: process_asset_zip failed (file=%s)", file.filename)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    package = meta.get("package") or ""
    name = package.split("/")[-1] if package else Path(file.filename).stem
    description = meta.get("description") or ""
    asset_type = meta.get("class") or meta.get("type") or meta.get("category") or ""
    if asset_type == "MaterialInstanceConstant":
        asset_type = "MaterialInstance"
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    tags = _normalize_tags(tags)
    size_bytes = int(meta.get("disk_bytes_total") or 0)

    write_conn = get_db()
    settings = get_settings(write_conn)
    translated_tags = _translate_tags_if_enabled(settings, tags)
    merged_tags = _merge_tags_for_asset(tags, translated_tags)
    embedding = None
    if _should_generate_embeddings_on_import(settings):
        embedding_text = _build_embedding_text(name, description, tags, translated_tags)
        embedding = embed_text(embedding_text)

    cur = write_conn.cursor()
    def _insert_asset():
        cur.execute(
            """
            INSERT INTO assets (
                asset_dir, name, description, type, project_id, hash_main_blake3, hash_main_sha256, hash_full_blake3,
                tags_json, meta_json, embedding_json, images_json, thumb_image, detail_image, full_image,
                anim_thumb, anim_detail, zip_path, size_bytes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(asset_dir.relative_to(ASSETS_DIR)),
                name,
                description,
                asset_type,
                project_id,
                hash_main,
                hash_main_sha256 or None,
                hash_full or None,
                json.dumps(merged_tags),
                json.dumps(meta),
                json.dumps(embedding) if embedding is not None else None,
                json.dumps([]),
                thumb_image,
                detail_image,
                full_image,
                anim_thumb,
                anim_detail,
                asset_zip_rel or None,
                size_bytes,
                now_iso(),
            ),
        )
    _db_retry(_insert_asset)
    _db_retry(write_conn.commit)
    asset_id = cur.lastrowid
    _db_retry(lambda: _upsert_asset_tags(
        write_conn,
        asset_id,
        hash_main,
        hash_full,
        now_iso(),
        tags,
        translated_tags,
        settings.get("tag_language") or "",
    ))
    _db_retry(write_conn.commit)
    write_conn.close()

    return {"id": asset_id, "name": name, "project_id": project_id}


def _parse_tags(value: str) -> List[str]:
    return [t.strip().lower() for t in value.split(",") if t.strip()]


def _parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [t.strip() for t in str(value).split(",") if t.strip()]


def _resolve_hash_column(hash_type: str) -> str:
    key = (hash_type or "").strip().lower()
    if key in {"blake3", "hash_main_blake3", "main_blake3"}:
        return "hash_main_blake3"
    if key in {"sha256", "hash_main_sha256", "main_sha256"}:
        return "hash_main_sha256"
    raise HTTPException(status_code=400, detail="hash_type must be blake3 or sha256")


def _run_projects_import_task(task_id: int) -> None:
    row = _task_get(task_id)
    if not row:
        return
    file_path = None
    try:
        payload = json.loads(row.get("message") or "{}")
        file_path = payload.get("file")
    except Exception:
        file_path = None
    if not file_path:
        raise RuntimeError("projects_import task missing file path")
    path = Path(file_path)
    if not path.exists():
        raise RuntimeError(f"projects_import file missing: {path}")

    import_id = f"projects_import_{task_id}"
    logger.info("Projects import started: file=%s task_id=%s", path, task_id)

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise HTTPException(status_code=400, detail="CSV header missing")
        total = sum(1 for _ in reader)

    _broadcast_event(
        {
            "type": "projects_import",
            "import_id": import_id,
            "status": "started",
            "current": 0,
            "total": total,
            "created": 0,
            "skipped": 0,
            "errors": 0,
        }
    )

    created = 0
    skipped = 0
    errors = 0
    block_size = 1000
    lookup_batch = 900
    pending = 0
    pending_progress: Optional[int] = None
    conn = get_db()
    cur = conn.cursor()

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, start=1):
            if _task_cancelled(task_id):
                _task_update(task_id, status="canceled", finished_at=now_iso())
                break
            try:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                link = (row.get("link") or "").strip() or None
                art_style = (row.get("art_style") or "").strip() or None
                project_era = (row.get("project_era") or row.get("era") or "").strip() or None
                description = (row.get("description") or "").strip() or None
                category_name = (row.get("category_name") or "").strip() or None
                is_ai_raw = (row.get("is_ai_generated") or "").strip().lower()
                is_ai_generated = is_ai_raw in {"1", "true", "yes", "on"}
                source_path = (row.get("source_path") or "").strip() or None
                source_folder = (row.get("source_folder") or "").strip() or None
                if not source_path:
                    raise ValueError("Source content path is required")
                full_project_copy_raw = (row.get("full_project_copy") or "").strip().lower()
                full_project_copy = full_project_copy_raw in {"1", "true", "yes", "on"}
                tags_value = (row.get("tags") or "").strip()
                tags = [t.strip() for t in tags_value.split(",") if t.strip()]

                existing = _find_project_by_source(conn, source_path, source_folder)
                if existing:
                    skipped += 1
                    continue

                source_leaf = Path(source_path).expanduser().name or name
                folder_name = f"{slugify(source_leaf)}-{int(time.time())}"
                folder_path = PROJECTS_DIR / folder_name
                folder_path.mkdir(parents=True, exist_ok=True)
                (folder_path / "Content").mkdir(parents=True, exist_ok=True)
                _ensure_uproject_file(name, folder_path, source_path)

                screenshot_url = (row.get("screenshot_url") or "").strip()
                screenshot_path = (
                    _try_download_screenshot(screenshot_url, folder_path) if screenshot_url else None
                )

                cur.execute(
                    """
                    INSERT INTO projects (
                        name, link, size_bytes, folder_path, screenshot_path, art_style, project_era, tags_json, description, category_name, is_ai_generated, created_at,
                        source_path, source_folder, full_project_copy
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        link,
                        0,
                        str(folder_path),
                        screenshot_path,
                        art_style,
                        project_era,
                        json.dumps(tags),
                        description,
                        category_name,
                        1 if is_ai_generated else 0,
                        now_iso(),
                        source_path,
                        source_folder,
                        1 if full_project_copy else 0,
                    ),
                )
                created += 1
                pending += 1
            except Exception:
                errors += 1
            if idx == 1 or idx % 50 == 0 or idx == total:
                logger.info(
                    "Projects import progress: %s/%s (created=%s skipped=%s errors=%s)",
                    idx,
                    total,
                    created,
                    skipped,
                    errors,
                )
                _broadcast_event(
                    {
                        "type": "projects_import",
                        "import_id": import_id,
                        "status": "running",
                        "current": idx,
                        "total": total,
                        "created": created,
                        "skipped": skipped,
                        "errors": errors,
                    }
                )
                pending_progress = idx
            if pending >= batch_size:
                if pending_progress is not None:
                    progress_payload = {"status": "running", "total": total, "done": pending_progress, "errors": errors}
                    cur.execute(
                        "UPDATE tasks SET progress_json = ?, message = ? WHERE id = ?",
                        (json.dumps(progress_payload), "importing", task_id),
                    )
                    pending_progress = None
                _db_retry(conn.commit)
                pending = 0
    if pending:
        if pending_progress is not None:
            progress_payload = {"status": "running", "total": total, "done": pending_progress, "errors": errors}
            cur.execute(
                "UPDATE tasks SET progress_json = ?, message = ? WHERE id = ?",
                (json.dumps(progress_payload), "importing", task_id),
            )
            pending_progress = None
        _db_retry(conn.commit)
    _flush_project_eras(conn)
    conn.close()

    logger.info(
        "Projects import done: total=%s created=%s skipped=%s errors=%s",
        total,
        created,
        skipped,
        errors,
    )
    _broadcast_event(
        {
            "type": "projects_import",
            "import_id": import_id,
            "status": "done",
            "current": total,
            "total": total,
            "created": created,
            "skipped": skipped,
            "errors": errors,
        }
    )
    _task_progress(task_id, "done", total, total, errors, message="done")
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _clear_all_tags(task_id: Optional[int] = None) -> None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE assets SET tags_json = '[]', embedding_json = NULL")
    cur.execute("DELETE FROM asset_tags")
    _db_retry(conn.commit)
    conn.close()
    if task_id is not None:
        _task_progress(task_id, "running", 1, 1, 0)


def _run_tags_import_task(task_id: int) -> None:
    row = _task_get(task_id)
    if not row:
        return
    file_path = None
    hash_type = None
    mode = "replace"
    project_id = None
    try:
        payload = json.loads(row.get("message") or "{}")
        file_path = payload.get("file")
        hash_type = payload.get("hash_type")
        mode = payload.get("mode") or "replace"
        project_id = payload.get("project_id")
    except Exception:
        file_path = None
    if not file_path:
        raise RuntimeError("tags_import task missing file path")
    path = Path(file_path)
    if not path.exists():
        raise RuntimeError(f"tags_import file missing: {path}")

    import_id = f"tags_import_{task_id}"
    logger.info("Tags import started: file=%s task_id=%s", path, task_id)

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise HTTPException(status_code=400, detail="CSV header missing")
        total = sum(1 for _ in reader)

    _broadcast_event(
        {
            "type": "tags_import",
            "import_id": import_id,
            "status": "started",
            "current": 0,
            "total": total,
            "updated": 0,
            "missing": 0,
            "errors": 0,
        }
    )

    updated = 0
    missing = 0
    errors = 0
    block_size = 1000
    lookup_batch = 900
    pending = 0
    pending_progress: Optional[int] = None
    last_heartbeat = time.time()
    conn = get_db()
    cur = conn.cursor()
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-200000")
        conn.execute("PRAGMA mmap_size=268435456")
    except Exception:
        pass
    settings = get_settings(conn)

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise HTTPException(status_code=400, detail="CSV header missing")
        if hash_type:
            column = _resolve_hash_column(hash_type)
        else:
            header = [name.strip() for name in reader.fieldnames or []]
            if "hash_main_blake3" in header:
                column = "hash_main_blake3"
            elif "hash_main_sha256" in header:
                column = "hash_main_sha256"
            else:
                raise HTTPException(status_code=400, detail="CSV must include hash_main_blake3 or hash_main_sha256")
        if "tags" not in reader.fieldnames:
            raise HTTPException(status_code=400, detail="CSV must include tags column")
        if mode not in {"replace", "merge"}:
            raise HTTPException(status_code=400, detail="mode must be replace or merge")
        idx = 0
        while True:
            chunk_rows = []
            for _ in range(block_size):
                try:
                    row = next(reader)
                except StopIteration:
                    row = None
                if row is None:
                    break
                idx += 1
                chunk_rows.append((idx, row))
            if not chunk_rows:
                break
            if _task_cancelled(task_id):
                _task_update(task_id, status="canceled", finished_at=now_iso())
                break
            hashes = []
            for _, row in chunk_rows:
                hash_value = (row.get(column) or "").strip()
                if hash_value:
                    hashes.append(hash_value)
            if not hashes:
                continue
            asset_map = {}
            for i in range(0, len(hashes), lookup_batch):
                slice_hashes = hashes[i : i + lookup_batch]
                params: List[Any] = list(slice_hashes)
                where = f"WHERE {column} IN ({','.join(['?'] * len(slice_hashes))})"
                if project_id is not None:
                    where += " AND project_id = ?"
                    params.append(int(project_id))
                asset_rows = fetch_all(
                    conn,
                    f"SELECT id, project_id, name, description, tags_json, hash_main_blake3, hash_full_blake3, created_at, {column} "
                    f"FROM assets {where}",
                    params,
                )
                asset_map.update({row[column]: row for row in asset_rows})
            for _, row in chunk_rows:
                try:
                    hash_value = (row.get(column) or "").strip()
                    if not hash_value:
                        continue
                    tags_value = row.get("tags") or ""
                    era_value = (row.get("project_era") or row.get("era") or "").strip()
                    incoming = _parse_tags(tags_value)
                    if not incoming:
                        continue
                    asset = asset_map.get(hash_value)
                    if not asset:
                        missing += 1
                        continue
                    _maybe_set_project_era(asset.get("project_id"), era_value)
                    if mode == "merge":
                        current = json.loads(asset["tags_json"] or "[]")
                        merged = []
                        for tag in current + incoming:
                            tag = str(tag).strip().lower()
                            if tag and tag not in merged:
                                merged.append(tag)
                        tags = merged
                    else:
                        tags = incoming
                    tags = _normalize_tags(tags)
                    translated_tags = _translate_tags_if_enabled(settings, tags)
                    merged_tags = _merge_tags_for_asset(tags, translated_tags)
                    if _should_generate_embeddings_on_import(settings):
                        embedding_text = _build_embedding_text(
                            asset["name"], asset["description"] or "", tags, translated_tags
                        )
                        embedding = embed_text(embedding_text)
                        cur.execute(
                            "UPDATE assets SET tags_json = ?, embedding_json = ? WHERE id = ?",
                            (json.dumps(merged_tags), json.dumps(embedding), asset["id"]),
                        )
                    else:
                        cur.execute(
                            "UPDATE assets SET tags_json = ?, embedding_json = NULL WHERE id = ?",
                            (json.dumps(merged_tags), asset["id"]),
                        )
                    _upsert_asset_tags(
                        conn,
                        asset.get("id"),
                        asset.get("hash_main_blake3") or "",
                        asset.get("hash_full_blake3") or "",
                        asset.get("created_at") or now_iso(),
                        tags,
                        translated_tags,
                        settings.get("tag_language") or "",
                    )
                    updated += 1
                    pending += 1
                except Exception:
                    errors += 1
            now = time.time()
            if idx == 1 or idx % block_size == 0 or idx >= total or (now - last_heartbeat) >= 10:
                logger.info(
                    "Tags import progress: %s/%s (updated=%s missing=%s errors=%s)",
                    idx,
                    total,
                    updated,
                    missing,
                    errors,
                )
                _broadcast_event(
                    {
                        "type": "tags_import",
                        "import_id": import_id,
                        "status": "running",
                        "current": idx,
                        "total": total,
                        "updated": updated,
                        "missing": missing,
                        "errors": errors,
                    }
                )
                pending_progress = idx
                last_heartbeat = now
            if pending >= block_size:
                if pending_progress is not None:
                    progress_payload = {"status": "running", "total": total, "done": pending_progress, "errors": errors}
                    cur.execute(
                        "UPDATE tasks SET progress_json = ?, message = ? WHERE id = ?",
                        (json.dumps(progress_payload), "importing", task_id),
                    )
                    pending_progress = None
                _db_retry(conn.commit)
                pending = 0
        if pending:
            if pending_progress is not None:
                progress_payload = {"status": "running", "total": total, "done": pending_progress, "errors": errors}
                cur.execute(
                    "UPDATE tasks SET progress_json = ?, message = ? WHERE id = ?",
                    (json.dumps(progress_payload), "importing", task_id),
                )
                pending_progress = None
            _db_retry(conn.commit)
    conn.close()

    logger.info(
        "Tags import done: total=%s updated=%s missing=%s errors=%s",
        total,
        updated,
        missing,
        errors,
    )
    _broadcast_event(
        {
            "type": "tags_import",
            "import_id": import_id,
            "status": "done",
            "current": total,
            "total": total,
            "updated": updated,
            "missing": missing,
            "errors": errors,
        }
    )
    _task_progress(task_id, "done", total, total, errors, message="done")
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


@app.get("/assets/exists")
def asset_exists(hash: str, hash_type: str = "blake3") -> Dict[str, Any]:
    if not hash:
        raise HTTPException(status_code=400, detail="hash is required")
    column = _resolve_hash_column(hash_type)
    conn = get_db()
    row = fetch_one(conn, f"SELECT id FROM assets WHERE {column} = ? LIMIT 1", (hash,))
    conn.close()
    logger.info("assets/exists hash=%s hash_type=%s exists=%s", hash, hash_type, bool(row))
    return {"exists": bool(row), "id": row["id"] if row else None}


@app.get("/tags/export")
def export_tags(hash_type: str = "blake3", project_id: Optional[int] = None) -> Response:
    column = _resolve_hash_column(hash_type)
    conn = get_db()
    params: List[Any] = []
    where = ""
    if project_id is not None:
        where = "WHERE project_id = ?"
        params.append(project_id)
    rows = fetch_all(
        conn,
        f"SELECT a.{column} AS hash_value, a.tags_json, p.project_era "
        f"FROM assets a JOIN projects p ON p.id = a.project_id {where} ORDER BY a.id",
        params,
    )
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([column, "tags", "project_era"])
    for row in rows:
        if not row["hash_value"]:
            continue
        tags = json.loads(row["tags_json"] or "[]")
        if not tags:
            continue
        writer.writerow([row["hash_value"], ",".join(tags), row.get("project_era") or ""])
    return Response(content=output.getvalue(), media_type="text/csv")


@app.post("/tags/import")
def import_tags(
    file: UploadFile = File(...),
    hash_type: Optional[str] = None,
    project_id: Optional[int] = None,
    mode: str = "replace",
) -> Dict[str, Any]:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload must be a csv file")
    if mode not in {"replace", "merge"}:
        raise HTTPException(status_code=400, detail="mode must be replace or merge")
    if hash_type:
        try:
            _resolve_hash_column(hash_type)
        except Exception:
            raise HTTPException(status_code=400, detail="hash_type must be blake3 or sha256")

    logger.info("Tags import queued: filename=%s", file.filename)
    ts = int(time.time() * 1000)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", file.filename)
    temp_path = UPLOADS_DIR / f"tags_import_{ts}_{safe_name}"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    with temp_path.open("wb") as dest:
        shutil.copyfileobj(file.file, dest)
    task_payload = json.dumps(
        {
            "file": str(temp_path),
            "hash_type": hash_type,
            "mode": mode,
            "project_id": project_id,
        }
    )
    task_id = _enqueue_task("tags_import", message=task_payload)
    _ensure_task_worker()
    return {"status": "queued", "task_id": task_id, "file": file.filename}


@app.post("/tags/clear")
def clear_all_tags() -> Dict[str, Any]:
    task_id = _enqueue_task("tags_clear", None)
    _ensure_task_worker()
    return {"status": "queued", "task_id": task_id}


def serialize_asset(
    row: Dict[str, Any],
    display_limit: int = 0,
    project_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = json.loads(row["meta_json"] or "{}")
    tags = json.loads(row["tags_json"] or "[]")
    translated = json.loads(row.get("tags_translated_json") or "[]")
    display_tags = translated if translated else tags
    if display_limit and display_limit > 0:
        display_tags = display_tags[:display_limit]
    roots: List[str] = []
    files = meta.get("files_on_disk") or []
    for entry in files:
        rel = str(entry).strip().replace("\\", "/").lstrip("/")
        if not rel:
            continue
        if rel.lower().startswith("content/"):
            rel = rel[8:]
        top = rel.split("/", 1)[0].strip()
        if top and top not in roots:
            roots.append(top)
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "type": row["type"],
        "project_id": row["project_id"],
        "project_art_style": row.get("project_art_style") or "",
        "project_era": row.get("project_era") or "",
        "project_settings": project_settings or {},
        "hash_main_blake3": row.get("hash_main_blake3") or "",
        "hash_full_blake3": row.get("hash_full_blake3") or "",
        "tags": tags,
        "display_tags": display_tags,
        "translated_tags": translated,
        "images": [f"/media/assets/{row['asset_dir']}/{p}" for p in json.loads(row["images_json"] or "[]")],
        "thumb_image": f"/media/assets/{row['asset_dir']}/{row['thumb_image']}" if row.get("thumb_image") else "",
        "detail_image": f"/media/assets/{row['asset_dir']}/{row['detail_image']}" if row.get("detail_image") else "",
        "full_image": f"/media/assets/{row['asset_dir']}/{row['full_image']}" if row.get("full_image") else "",
        "anim_thumb": f"/media/assets/{row['asset_dir']}/{row['anim_thumb']}" if row.get("anim_thumb") else "",
        "anim_detail": f"/media/assets/{row['asset_dir']}/{row['anim_detail']}" if row.get("anim_detail") else "",
        "meta": meta,
        "path_warning": len(roots) > 1,
        "path_roots": roots,
        "size_bytes": row["size_bytes"],
        "created_at": row["created_at"],
    }


@app.get("/assets")
def list_assets(
    query: Optional[str] = None,
    project_id: Optional[int] = None,
    project_ids: Optional[str] = None,
    types: Optional[str] = None,
    asset_type: Optional[str] = None,
    tag: Optional[str] = None,
    nanite: Optional[str] = None,
    collision: Optional[str] = None,
    semantic: bool = False,
    page: int = 1,
    page_size: int = 24,
) -> Dict[str, Any]:
    conn = get_db()
    settings = get_settings(conn)
    try:
        display_limit = int(settings.get("tag_display_limit") or 0)
    except ValueError:
        display_limit = 0
    filters = []
    params: List[Any] = []
    use_fts = bool((query and not semantic) or tag)

    def _fts_escape(term: str) -> str:
        cleaned = term.replace('"', '""').strip()
        if " " in cleaned:
            return f"\"{cleaned}\""
        return cleaned

    if project_ids:
        ids = [pid.strip() for pid in project_ids.split(",") if pid.strip().isdigit()]
        if ids:
            placeholders = ",".join(["?"] * len(ids))
            filters.append(f"a.project_id IN ({placeholders})")
            params.extend([int(pid) for pid in ids])
    elif project_id:
        filters.append("a.project_id = ?")
        params.append(project_id)
    if asset_type:
        filters.append("a.type = ?")
        params.append(asset_type)
    if types:
        type_list = [t.strip() for t in types.split(",") if t.strip()]
        if type_list:
            placeholders = ",".join(["?"] * len(type_list))
            filters.append(f"a.type IN ({placeholders})")
            params.extend(type_list)
    if tag and not use_fts:
        filters.append("a.tags_json LIKE ?")
        params.append(f"%{tag}%")
    if nanite is not None and str(nanite).strip() != "":
        raw = str(nanite).strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            filters.append("a.meta_json LIKE ?")
            params.append('%"nanite_enabled": true%')
        elif raw in {"0", "false", "no", "off"}:
            filters.append("(a.meta_json NOT LIKE ? OR a.meta_json IS NULL)")
            params.append('%"nanite_enabled": true%')
    if collision is not None and str(collision).strip() != "":
        raw = str(collision).strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            filters.append("a.meta_json LIKE ?")
            params.append('%"collision_complexity":%')
            filters.append("a.meta_json NOT LIKE ?")
            params.append('%"collision_complexity": "NoCollision"%')
        elif raw in {"0", "false", "no", "off"}:
            filters.append(
                "(a.meta_json NOT LIKE ? OR a.meta_json IS NULL OR a.meta_json LIKE ?)"
            )
            params.append('%"collision_complexity":%')
            params.append('%"collision_complexity": "NoCollision"%')

    if page_size > 200:
        page_size = 200
    if page < 1:
        page = 1

    base = "FROM assets a JOIN projects p ON p.id = a.project_id LEFT JOIN asset_tags t ON t.hash_full_blake3 = a.hash_full_blake3"
    if use_fts:
        base = (
            "FROM assets a JOIN assets_fts ON assets_fts.rowid = a.id "
            "JOIN projects p ON p.id = a.project_id "
            "LEFT JOIN asset_tags t ON t.hash_full_blake3 = a.hash_full_blake3"
        )
    order_by = "ORDER BY a.created_at DESC"

    if use_fts:
        fts_parts = []
        if query:
            fts_parts.append(_fts_escape(query))
        if tag:
            fts_parts.append(f"tags:{_fts_escape(tag)}")
        if fts_parts:
            filters.append("assets_fts MATCH ?")
            params.append(" AND ".join(fts_parts))
    elif query and not semantic:
        query_lc = query.lower()
        filters.append("(lower(a.name) LIKE ? OR lower(a.description) LIKE ? OR lower(a.tags_json) LIKE ?)")
        like_value = f"%{query_lc}%"
        params.extend([like_value, like_value, like_value])

    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    total_all_row = fetch_one(conn, "SELECT COUNT(*) AS count FROM assets")
    total_all = total_all_row["count"] if total_all_row else 0

    if query and semantic:
        if len(query.strip()) < 3:
            semantic = False
        max_candidates = 500
    if query and semantic:
        rows = fetch_all(
            conn,
            f"SELECT a.*, t.tags_translated_json, p.art_style AS project_art_style, p.project_era {base} {where} {order_by} LIMIT ?",
            params + [max_candidates],
        )
        query_vec = embed_text(query)
        scored = []
        for row in rows:
            try:
                emb = json.loads(row.get("embedding_json") or "[]")
            except json.JSONDecodeError:
                emb = []
            score = cosine_similarity(query_vec, emb)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        total = len(scored)
        offset = (page - 1) * page_size
        page_rows = [row for _, row in scored[offset : offset + page_size]]
        conn.close()
        return {
            "items": [serialize_asset(row, display_limit) for row in page_rows],
            "total": total,
            "total_all": total_all,
            "page": page,
            "page_size": page_size,
        }

    count_row = fetch_one(conn, f"SELECT COUNT(*) AS count {base} {where}", params)
    total = count_row["count"] if count_row else 0
    offset = (page - 1) * page_size
    rows = fetch_all(
        conn,
        f"SELECT a.*, t.tags_translated_json, p.art_style AS project_art_style, p.project_era {base} {where} {order_by} LIMIT ? OFFSET ?",
        params + [page_size, offset],
    )
    conn.close()

    return {
        "items": [serialize_asset(row, display_limit) for row in rows],
        "total": total,
        "total_all": total_all,
        "page": page,
        "page_size": page_size,
    }


@app.get("/assets/types")
def list_asset_types() -> Dict[str, Any]:
    conn = get_db()
    settings = get_settings(conn)
    rows = fetch_all(
        conn,
        "SELECT DISTINCT type FROM assets WHERE type IS NOT NULL AND type != '' ORDER BY type",
    )
    types = [row["type"] for row in rows]
    if not types:
        cur = conn.cursor()
        cur.execute("SELECT id, meta_json FROM assets WHERE type IS NULL OR type = '' LIMIT 2000")
        inferred = set()
        updates = []
        for row in cur.fetchall():
            meta = json.loads(row["meta_json"] or "{}")
            value = meta.get("type") or meta.get("category") or meta.get("class") or ""
            if value:
                inferred.add(value)
                updates.append((value, row["id"]))
        if updates:
            cur.executemany("UPDATE assets SET type = ? WHERE id = ?", updates)
            _db_retry(conn.commit)
        types = sorted(inferred)
    catalog = _parse_csv_list(settings.get("asset_type_catalog"))
    default_types = [
        "StaticMesh",
        "SkeletalMesh",
        "AnimSequence",
        "Material",
        "MaterialInstance",
        "MaterialInstanceConstant",
        "Texture2D",
        "Blueprint",
        "NiagaraSystem",
    ]
    merged = {t for t in types if t}
    merged.update(default_types)
    merged.update(catalog)
    conn.close()
    return {"items": sorted(merged)}


@app.delete("/assets/{asset_id}")
def delete_asset(asset_id: int) -> Dict[str, str]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
    _db_retry(conn.commit)
    conn.close()
    return {"status": "deleted"}


@app.delete("/projects/{project_id}/assets")
def delete_project_assets(project_id: int) -> Dict[str, Any]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS count FROM assets WHERE project_id = ?", (project_id,))
    count_row = cur.fetchone()
    cur.execute("DELETE FROM assets WHERE project_id = ?", (project_id,))
    _db_retry(conn.commit)
    conn.close()
    return {"status": "deleted", "count": count_row[0] if count_row else 0}


@app.delete("/projects/{project_id}")
def delete_project(project_id: int) -> Dict[str, Any]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT folder_path FROM projects WHERE id = ?", (project_id,))
    project_row = cur.fetchone()
    if not project_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")
    cur.execute("SELECT COUNT(*) AS count FROM assets WHERE project_id = ?", (project_id,))
    count_row = cur.fetchone()
    cur.execute("DELETE FROM assets WHERE project_id = ?", (project_id,))
    cur.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    _db_retry(conn.commit)
    conn.close()
    return {"status": "deleted", "assets_deleted": count_row[0] if count_row else 0}


@app.post("/assets/{asset_id}/migrate")
def migrate_asset(asset_id: int, payload: AssetMigrateRequest) -> Dict[str, Any]:
    dest_path = (payload.dest_path or "").strip()
    if not dest_path:
        raise HTTPException(status_code=400, detail="Destination path is required")

    conn = get_db()
    asset = fetch_one(conn, "SELECT * FROM assets WHERE id = ?", (asset_id,))
    if not asset:
        conn.close()
        raise HTTPException(status_code=404, detail="Asset not found")

    project = None
    if asset.get("project_id"):
        project = fetch_one(conn, "SELECT * FROM projects WHERE id = ?", (asset["project_id"],))
    conn.close()
    if not project:
        raise HTTPException(status_code=400, detail="Asset has no project")

    files = json.loads(asset.get("meta_json") or "{}").get("files_on_disk", [])
    if not files:
        raise HTTPException(status_code=400, detail="No files to migrate")

    source_root = Path(project["folder_path"]) / "Content"
    source_fallback_root = _resolve_source_content_path(project)
    if not source_root.exists() and (not source_fallback_root or not source_fallback_root.exists()):
        raise HTTPException(status_code=400, detail="Project Content folder not found")

    dest_root = Path(dest_path) / "Content"
    dest_root.mkdir(parents=True, exist_ok=True)

    _set_migrate_progress(asset_id, {"status": "queued", "copied": 0, "total": len(files)})
    thread = threading.Thread(
        target=_migrate_asset_files,
        args=(
            asset_id,
            source_root,
            dest_root,
            files,
            payload.overwrite,
            source_fallback_root,
            project.get("source_folder"),
        ),
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


@app.get("/assets/{asset_id}/migrate-status")
def migrate_status(asset_id: int) -> Dict[str, Any]:
    with MIGRATE_LOCK:
        return MIGRATE_PROGRESS.get(asset_id, {"status": "idle", "copied": 0, "total": 0})


@app.post("/assets/backfill-types")
def backfill_asset_types(limit: int = 10000) -> Dict[str, Any]:
    if limit < 1:
        raise HTTPException(status_code=400, detail="Limit must be positive")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, meta_json FROM assets WHERE type IS NULL OR type = '' LIMIT ?", (limit,))
    rows = cur.fetchall()
    updated = 0
    for row in rows:
        meta = json.loads(row["meta_json"] or "{}")
        inferred = meta.get("type") or meta.get("category") or meta.get("class") or ""
        if inferred:
            cur.execute("UPDATE assets SET type = ? WHERE id = ?", (inferred, row["id"]))
            updated += 1
    _db_retry(conn.commit)
    conn.close()
    return {"updated": updated, "processed": len(rows)}


@app.get("/assets/{asset_id}")
def get_asset(asset_id: str) -> Dict[str, Any]:
    conn = get_db()
    settings = get_settings(conn)
    try:
        display_limit = int(settings.get("tag_display_limit") or 0)
    except ValueError:
        display_limit = 0
    if str(asset_id).isdigit():
        row = fetch_one(
            conn,
            "SELECT a.*, t.tags_translated_json, p.art_style AS project_art_style, p.project_era "
            "FROM assets a JOIN projects p ON p.id = a.project_id "
            "LEFT JOIN asset_tags t ON t.hash_full_blake3 = a.hash_full_blake3 "
            "WHERE a.id = ?",
            (int(asset_id),),
        )
    else:
        row = fetch_one(
            conn,
            "SELECT a.*, t.tags_translated_json, p.art_style AS project_art_style, p.project_era "
            "FROM assets a JOIN projects p ON p.id = a.project_id "
            "LEFT JOIN asset_tags t ON t.hash_full_blake3 = a.hash_full_blake3 "
            "WHERE a.hash_main_blake3 = ? OR a.hash_full_blake3 = ? LIMIT 1",
            (asset_id, asset_id),
        )
    project_settings = {}
    if row and row.get("project_id"):
        project_row = fetch_one(
            conn,
            """
            SELECT id, name, link, folder_path, source_path, source_folder, full_project_copy, reimported_once,
                   export_include_json, export_exclude_json, art_style, project_era
            FROM projects
            WHERE id = ?
            """,
            (row["project_id"],),
        )
        if project_row:
            project_settings = {
                "id": project_row["id"],
                "name": project_row["name"],
                "link": project_row.get("link"),
                "folder_path": project_row.get("folder_path"),
                "source_path": project_row.get("source_path") or "",
                "source_folder": (Path(project_row.get("source_folder")).name if project_row.get("source_folder") else ""),
                "full_project_copy": bool(project_row.get("full_project_copy") or 0),
                "reimported_once": bool(project_row.get("reimported_once") or 0),
                "export_include": json.loads(project_row.get("export_include_json") or "[]"),
                "export_exclude": json.loads(project_row.get("export_exclude_json") or "[]"),
                "art_style": project_row.get("art_style") or "",
                "project_era": project_row.get("project_era") or "",
            }
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Asset not found")
    return serialize_asset(row, display_limit, project_settings)


@app.get("/download/{snapshot_id}.zip")
def download_snapshot(snapshot_id: str, background_tasks: BackgroundTasks, layout: str = Query("asset")) -> FileResponse:
    conn = get_db()
    row = None
    if snapshot_id.isdigit():
        row = fetch_one(conn, "SELECT * FROM assets WHERE id = ?", (int(snapshot_id),))
    if not row:
        row = fetch_one(conn, "SELECT * FROM assets WHERE hash_main_blake3 = ? LIMIT 1", (snapshot_id,))
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Asset not found")

    project = None
    if row.get("project_id"):
        project = fetch_one(conn, "SELECT * FROM projects WHERE id = ?", (row["project_id"],))
    conn.close()

    if project:
        include_content = layout == "project"
        zip_path = _build_snapshot_zip(row, project, include_content)
        background_tasks.add_task(zip_path.unlink, missing_ok=True)
        return FileResponse(zip_path, media_type="application/zip", filename=f"{snapshot_id}.zip")

    asset_dir = ASSETS_DIR / row["asset_dir"]
    zip_rel = row.get("zip_path") or f"{row.get('hash_main_blake3')}.zip"
    zip_path = asset_dir / zip_rel
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Zip not found")

    return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)


@app.put("/assets/{asset_id}/tags")
def update_asset_tags(asset_id: int, payload: AssetTagUpdate) -> Dict[str, Any]:
    tags = _normalize_tags([t for t in payload.tags if t.strip()])
    conn = get_db()
    settings = get_settings(conn)
    asset = fetch_one(conn, "SELECT * FROM assets WHERE id = ?", (asset_id,))
    if not asset:
        conn.close()
        raise HTTPException(status_code=404, detail="Asset not found")

    translated_tags = _translate_tags_if_enabled(settings, tags)
    merged_tags = _merge_tags_for_asset(tags, translated_tags)
    embedding_text = _build_embedding_text(asset["name"], asset["description"] or "", tags, translated_tags)
    embedding = embed_text(embedding_text)

    cur = conn.cursor()
    cur.execute(
        "UPDATE assets SET tags_json = ?, embedding_json = ? WHERE id = ?",
        (json.dumps(merged_tags), json.dumps(embedding), asset_id),
    )
    _upsert_asset_tags(
        conn,
        asset_id,
        asset.get("hash_main_blake3") or "",
        asset.get("hash_full_blake3") or "",
        asset.get("created_at") or now_iso(),
        tags,
        translated_tags,
        settings.get("tag_language") or "",
    )
    _db_retry(conn.commit)
    conn.close()
    return {"status": "ok", "tags": tags}


@app.post("/assets/tags/merge")
def merge_asset_tags(payload: AssetTagBulkMerge) -> Dict[str, Any]:
    asset_ids = [int(a) for a in payload.asset_ids if str(a).strip().isdigit()]
    if not asset_ids:
        raise HTTPException(status_code=400, detail="asset_ids required")
    incoming = _normalize_tags([t for t in payload.tags if t.strip()])
    if not incoming:
        raise HTTPException(status_code=400, detail="tags required")

    conn = get_db()
    settings = get_settings(conn)
    updated = 0
    missing = 0
    errors = 0
    batch_size = 200
    pending = 0
    cur = conn.cursor()

    for asset_id in asset_ids:
        try:
            asset = fetch_one(conn, "SELECT * FROM assets WHERE id = ?", (asset_id,))
            if not asset:
                missing += 1
                continue
            existing = json.loads(asset.get("tags_json") or "[]")
            merged = _normalize_tags(existing + incoming)
            translated_tags = _translate_tags_if_enabled(settings, merged)
            merged_tags = _merge_tags_for_asset(merged, translated_tags)
            if _should_generate_embeddings_on_import(settings):
                embedding_text = _build_embedding_text(
                    asset["name"], asset["description"] or "", merged, translated_tags
                )
                embedding = embed_text(embedding_text)
                cur.execute(
                    "UPDATE assets SET tags_json = ?, embedding_json = ? WHERE id = ?",
                    (json.dumps(merged_tags), json.dumps(embedding), asset_id),
                )
            else:
                cur.execute(
                    "UPDATE assets SET tags_json = ?, embedding_json = NULL WHERE id = ?",
                    (json.dumps(merged_tags), asset_id),
                )
            _upsert_asset_tags(
                conn,
                asset_id,
                asset.get("hash_main_blake3") or "",
                asset.get("hash_full_blake3") or "",
                asset.get("created_at") or now_iso(),
                merged,
                translated_tags,
                settings.get("tag_language") or "",
            )
            updated += 1
            pending += 1
            if pending >= batch_size:
                _db_retry(conn.commit)
                pending = 0
        except Exception:
            errors += 1
    if pending:
        _db_retry(conn.commit)
    conn.close()
    return {"status": "ok", "updated": updated, "missing": missing, "errors": errors}


@app.post("/assets/{asset_id}/generate-tags")
def generate_asset_tags(asset_id: int) -> Dict[str, Any]:
    conn = get_db()
    row = fetch_one(conn, "SELECT * FROM assets WHERE id = ?", (asset_id,))
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Asset not found")

    settings = get_settings(conn)
    conn.close()

    image_data_url = None
    image_rel = (
        row.get("detail_image")
        or row.get("full_image")
        or row.get("thumb_image")
        or row.get("anim_detail")
        or row.get("anim_thumb")
    )
    if image_rel:
        try:
            size_setting = settings.get("tag_image_size") or "512"
            image_size = int(size_setting)
        except ValueError:
            image_size = 512
        try:
            quality_setting = settings.get("tag_image_quality") or "80"
            image_quality = int(quality_setting)
        except ValueError:
            image_quality = 80
        if image_size > 0:
            image_path = ASSETS_DIR / row["asset_dir"] / image_rel
            if image_path.exists():
                image_data_url = _build_image_data_url(image_path, image_size, image_quality)

    try:
        meta = json.loads(row["meta_json"] or "{}")
        tags, era = generate_tags(
            settings,
            row["name"],
            row["description"] or "",
            json.loads(row["tags_json"] or "[]"),
            image_data_url,
            meta.get("class") or row.get("type") or "",
            return_era=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    tags = _normalize_tags(tags)
    _maybe_set_project_era(row["project_id"], era)
    translated_tags = _translate_tags_if_enabled(settings, tags)
    merged_tags = _merge_tags_for_asset(tags, translated_tags)
    embedding_text = _build_embedding_text(row["name"], row["description"] or "", tags, translated_tags)
    embedding = embed_text(embedding_text)

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE assets SET tags_json = ?, embedding_json = ? WHERE id = ?",
        (json.dumps(merged_tags), json.dumps(embedding), asset_id),
    )
    _upsert_asset_tags(
        conn,
        asset_id,
        row.get("hash_main_blake3") or "",
        row.get("hash_full_blake3") or "",
        row.get("created_at") or now_iso(),
        tags,
        translated_tags,
        settings.get("tag_language") or "",
    )
    _db_retry(conn.commit)
    conn.close()

    conn = get_db()
    updated = fetch_one(
        conn,
        "SELECT a.*, t.tags_translated_json FROM assets a "
        "LEFT JOIN asset_tags t ON t.hash_full_blake3 = a.hash_full_blake3 "
        "WHERE a.id = ?",
        (asset_id,),
    )
    conn.close()
    try:
        display_limit = int(settings.get("tag_display_limit") or 0)
    except ValueError:
        display_limit = 0
    return {"status": "ok", "tags": tags, "asset": serialize_asset(updated, display_limit) if updated else None}


@app.post("/projects/{project_id}/setcard")
def generate_project_setcard(project_id: int, force: bool = Query(False)) -> Dict[str, Any]:
    conn = get_db()
    project = fetch_one(conn, "SELECT * FROM projects WHERE id = ?", (project_id,))
    conn.close()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    folder_path = Path(project.get("folder_path") or "")
    if not folder_path or not folder_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")

    out_path = folder_path / "setcard.png"
    if out_path.exists() and not force:
        url = _project_screenshot_url_from_path(str(out_path))
        return {"status": "ok", "setcard_url": url}

    created = _generate_project_setcard(project)
    if not created:
        raise HTTPException(status_code=400, detail="Setcard generation failed: no preview images found")
    url = _project_screenshot_url_from_path(created)
    return {"status": "ok", "setcard_url": url}


@app.post("/projects/{project_id}/generate-previews")
def generate_project_previews(project_id: int) -> Dict[str, Any]:
    _queue_preview_generation(project_id)
    return {"status": "queued"}

