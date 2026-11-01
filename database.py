import json
import logging
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    import redis
except ImportError:  # The app can still run until requirements are installed.
    redis = None

DB_PATH = Path("database/assistant.db")
MESSAGE_CACHE_TTL_SECONDS = int(os.getenv("MESSAGE_CACHE_TTL_SECONDS", "120"))
MESSAGE_CACHE_PREFIX = "nova:conversation:messages:"
logger = logging.getLogger(__name__)
redis_client = (
    redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
        decode_responses=True,
        socket_connect_timeout=0.5,
        socket_timeout=0.5
    )
    if redis else None
)
_redis_warning_emitted = False


def warn_redis_unavailable(reason):
    global _redis_warning_emitted
    if not _redis_warning_emitted:
        logger.warning("Redis cache unavailable; using SQLite: %s", reason)
        _redis_warning_emitted = True


def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_message_history(conversation_id):
    """Read history from Redis when available, falling back to SQLite."""
    cache_key = f"{MESSAGE_CACHE_PREFIX}{conversation_id}"

    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except (redis.RedisError, json.JSONDecodeError) as exc:
            warn_redis_unavailable(exc)
    else:
        warn_redis_unavailable("Python redis package is not installed")

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp, message_id
            """,
            (conversation_id,)
        ).fetchall()
        messages = [dict(row) for row in rows]
    finally:
        conn.close()

    if redis_client:
        try:
            redis_client.setex(
                cache_key,
                MESSAGE_CACHE_TTL_SECONDS,
                json.dumps(messages, ensure_ascii=False)
            )
        except redis.RedisError as exc:
            warn_redis_unavailable(exc)

    return messages


def invalidate_message_history(conversation_id):
    """Remove a transcript from Redis after its messages change."""
    if redis_client:
        try:
            redis_client.delete(f"{MESSAGE_CACHE_PREFIX}{conversation_id}")
        except redis.RedisError as exc:
            warn_redis_unavailable(exc)


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(conversation_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL DEFAULT 1,
            filename TEXT NOT NULL,
            file_path TEXT,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            task_type TEXT NOT NULL,
            task_input TEXT,
            result TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    document_columns = {
        row["name"] for row in cursor.execute("PRAGMA table_info(documents)").fetchall()
    }
    if "user_id" not in document_columns:
        cursor.execute(
            "ALTER TABLE documents ADD COLUMN user_id INTEGER NOT NULL DEFAULT 1"
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully!")
