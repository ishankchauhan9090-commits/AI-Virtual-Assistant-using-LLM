
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader
from werkzeug.utils import secure_filename

from database import (
    create_tables,
    get_connection,
    get_message_history,
    invalidate_message_history
)
from llm import generate_response
from rag import (
    add_documents_to_vectorstore,
    remove_document_from_vectorstore,
    search_documents,
)
from tools import calculator


# =========================
# FastAPI Application
# =========================

app = FastAPI(
    title="AI Virtual Assistant API",
    description="Backend API for an LLM-powered virtual assistant",
    version="1.0.0"
)

create_tables()


# =========================
# Data Models
# =========================

class UserCreate(BaseModel):
    name: str
    email: str


class ConversationCreate(BaseModel):
    user_id: int
    title: str


class MessageCreate(BaseModel):
    conversation_id: int
    role: str
    message: str


class TaskCreate(BaseModel):
    user_id: int
    task_type: str
    task_input: str


class ChatRequest(BaseModel):
    user_id: int
    conversation_id: int
    message: str


MAX_DOCUMENT_SIZE = 10 * 1024 * 1024
MAX_UPLOAD_FILES = 5
MAX_UPLOAD_BATCH_SIZE = 30 * 1024 * 1024


def is_action_plan_request(message: str) -> bool:
    """Recognize explicit requests to turn document evidence into saved actions."""
    normalized = " ".join(message.lower().split())
    action_phrases = (
        "action items",
        "action plan",
        "make a checklist",
        "create a checklist",
        "make checklist",
        "create task list",
        "extract tasks",
        "to-do list",
        "todo list",
        "tasks from",
        "turn this into tasks",
        "turn these into tasks",
    )
    return any(phrase in normalized for phrase in action_phrases)


def is_comparison_request(message: str) -> bool:
    normalized = " ".join(message.lower().split())
    return any(
        phrase in normalized
        for phrase in ("compare", "differences between", "difference between", "across the pdfs")
    )


# =========================
# Home
# =========================

@app.get("/")
def home():
    return {
        "message": "AI Virtual Assistant Backend is running",
        "status": "success"
    }


# =========================
# Users
# =========================

@app.post("/users")
def create_user(user: UserCreate):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users (name, email)
            VALUES (?, ?)
            """,
            (user.name, user.email)
        )

        conn.commit()

        user_id = cursor.lastrowid

        return {
            "message": "User created successfully",
            "user_id": user_id
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    finally:
        conn.close()


@app.get("/users/{user_id}")
def get_user(user_id: int):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM users
        WHERE user_id = ?
        """,
        (user_id,)
    )

    user = cursor.fetchone()

    conn.close()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    return dict(user)


# =========================
# Conversations
# =========================

@app.post("/conversations")
def create_conversation(data: ConversationCreate):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO conversations (user_id, title)
        VALUES (?, ?)
        """,
        (data.user_id, data.title)
    )

    conn.commit()

    conversation_id = cursor.lastrowid

    conn.close()

    return {
        "message": "Conversation created successfully",
        "conversation_id": conversation_id
    }


@app.get("/conversations/{user_id}")
def get_conversations(user_id: int):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM conversations
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,)
    )

    conversations = cursor.fetchall()

    conn.close()

    return [
        dict(row)
        for row in conversations
    ]


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, user_id: int):
    conn = get_connection()
    try:
        conversation = conn.execute(
            "SELECT conversation_id FROM conversations WHERE conversation_id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        conn.execute(
            "DELETE FROM conversations WHERE conversation_id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
        conn.commit()
        invalidate_message_history(conversation_id)
        return {"success": True, "conversation_id": conversation_id}
    finally:
        conn.close()


# =========================
# Messages
# =========================

@app.post("/messages")
def create_message(data: MessageCreate):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO messages
        (conversation_id, role, message)
        VALUES (?, ?, ?)
        """,
        (
            data.conversation_id,
            data.role,
            data.message
        )
    )

    conn.commit()

    message_id = cursor.lastrowid

    invalidate_message_history(data.conversation_id)

    conn.close()

    return {
        "message": "Message saved successfully",
        "message_id": message_id
    }


@app.get("/messages/{conversation_id}")
def get_messages(conversation_id: int):
    return get_message_history(conversation_id)


# =========================
# Document upload
# =========================

@app.post("/documents/upload")
def upload_documents(
    files: list[UploadFile] = File(...),
    user_id: int = Form(1),
):
    if not files:
        raise HTTPException(status_code=400, detail="Choose one or more PDFs to upload.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload at most {MAX_UPLOAD_FILES} PDFs at a time.",
        )

    validated = []
    total_size = 0
    for uploaded in files:
        original_name = secure_filename(uploaded.filename or "")
        if not original_name or Path(original_name).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Every uploaded file must be a PDF.")
        content = uploaded.file.read(MAX_DOCUMENT_SIZE + 1)
        if len(content) > MAX_DOCUMENT_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"{original_name} exceeds the 10 MB per-file limit.",
            )
        total_size += len(content)
        if total_size > MAX_UPLOAD_BATCH_SIZE:
            raise HTTPException(status_code=413, detail="The combined upload limit is 30 MB.")
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")
            has_text = any((page.extract_text() or "").strip() for page in reader.pages)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"{original_name} could not be read as a PDF.",
            ) from exc
        if not has_text:
            raise HTTPException(
                status_code=400,
                detail=f"{original_name} has no selectable text. Scanned PDFs need OCR.",
            )
        validated.append((original_name, content))

    conn = get_connection()
    document_ids = []
    try:
        cursor = conn.cursor()
        existing_names = {
            row["filename"].casefold()
            for row in cursor.execute(
                "SELECT filename FROM documents WHERE user_id = ?", (user_id,)
            ).fetchall()
        }
        named_uploads = []
        for original_name, content in validated:
            stem = Path(original_name).stem
            suffix = Path(original_name).suffix
            unique_name = original_name
            version = 2
            while unique_name.casefold() in existing_names:
                unique_name = f"{stem} ({version}){suffix}"
                version += 1
            existing_names.add(unique_name.casefold())
            cursor.execute(
                "INSERT INTO documents (user_id, filename, file_path) VALUES (?, ?, NULL)",
                (user_id, unique_name),
            )
            document_id = cursor.lastrowid
            document_ids.append(document_id)
            named_uploads.append((content, unique_name, document_id))
        conn.commit()

        # Only extracted chunks and vectors persist. Original PDF bytes remain in
        # memory for this request and are released after indexing completes.
        chunk_count = add_documents_to_vectorstore(named_uploads)
        if not chunk_count:
            raise RuntimeError("No searchable text could be indexed.")
    except Exception as exc:
        conn.rollback()
        for document_id in document_ids:
            try:
                remove_document_from_vectorstore(document_id)
            except Exception:
                pass
        if document_ids:
            conn.executemany(
                "DELETE FROM documents WHERE document_id = ? AND user_id = ?",
                [(document_id, user_id) for document_id in document_ids],
            )
            conn.commit()
        raise HTTPException(
            status_code=500,
            detail=f"The PDFs were received but could not be indexed: {exc}",
        ) from exc
    finally:
        conn.close()

    return {
        "success": True,
        "files_processed": len(validated),
        "documents": [
            {"document_id": document_id, "filename": name}
            for document_id, (_, name, _) in zip(document_ids, named_uploads)
        ],
        "chunks_indexed": chunk_count,
    }


@app.get("/documents/{user_id}")
def get_documents(user_id: int):
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT document_id, filename, uploaded_at
            FROM documents
            WHERE user_id = ?
            ORDER BY uploaded_at DESC, document_id DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


@app.delete("/documents/{document_id}")
def delete_document(document_id: int, user_id: int):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ? AND user_id = ?",
            (document_id, user_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="PDF not found.")

        matching_name_count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE user_id = ? AND filename = ?",
            (user_id, row["filename"]),
        ).fetchone()[0]
        # Legacy chunks may lack document IDs. Source-name removal is safe only
        # when that name identifies one saved document for this user.
        legacy_source_name = row["filename"] if matching_name_count == 1 else None
        remove_document_from_vectorstore(document_id, legacy_source_name)

        # Older app versions stored original PDFs locally. Delete only the path
        # recorded for this document and only when it remains inside documents/.
        if row["file_path"]:
            project_documents = Path("documents").resolve()
            stored_path = Path(row["file_path"]).resolve()
            if project_documents in stored_path.parents and stored_path.is_file():
                stored_path.unlink()

        conn.execute(
            "DELETE FROM documents WHERE document_id = ? AND user_id = ?",
            (document_id, user_id),
        )
        conn.commit()
        return {"success": True, "document_id": document_id}
    finally:
        conn.close()


# =========================
# Tasks
# =========================

@app.post("/tasks")
def create_task(data: TaskCreate):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO tasks
        (user_id, task_type, task_input)
        VALUES (?, ?, ?)
        """,
        (
            data.user_id,
            data.task_type,
            data.task_input
        )
    )

    conn.commit()

    task_id = cursor.lastrowid

    conn.close()

    return {
        "message": "Task created successfully",
        "task_id": task_id
    }


@app.get("/tasks/{user_id}")
def get_tasks(user_id: int):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM tasks
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,)
    )

    tasks = cursor.fetchall()

    conn.close()

    return [
        dict(row)
        for row in tasks
    ]


# =========================
# Chat with LLM + RAG + Tools
# =========================

@app.post("/chat")
def chat(data: ChatRequest):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        # -------------------------
        # 1. Save user message
        # -------------------------

        cursor.execute(
            """
            INSERT INTO messages
            (conversation_id, role, message)
            VALUES (?, ?, ?)
            """,
            (
                data.conversation_id,
                "user",
                data.message
            )
        )

        conn.commit()
        invalidate_message_history(data.conversation_id)


        # Give a new conversation a useful title from its first question.
        title = " ".join(data.message.split())
        if len(title) > 48:
            title = title[:45].rstrip() + "..."

        cursor.execute(
            """
            UPDATE conversations
            SET title = ?
            WHERE conversation_id = ?
              AND user_id = ?
              AND (title IS NULL OR TRIM(title) = '' OR title = 'New Conversation')
            """,
            (title, data.conversation_id, data.user_id)
        )
        conn.commit()


        # -------------------------
        # 2. Get conversation history
        # -------------------------

        history = get_message_history(data.conversation_id)


        # -------------------------
        # 3. Check for calculator task
        # -------------------------

        task_id = None
        action_mode = is_action_plan_request(data.message)
        comparison_mode = is_comparison_request(data.message)

        if data.message.lower().startswith("calculate "):

            expression = data.message[10:].strip()

            calculation = calculator(expression)

            if calculation["success"]:

                response = (
                    f"The result of "
                    f"{calculation['expression']} "
                    f"is {calculation['result']}."
                )

            else:

                response = (
                    "I could not calculate that expression. "
                    f"Error: {calculation['error']}"
                )


        else:

            # -------------------------
            # 4. Search knowledge base
            # -------------------------

            context = search_documents(
                data.message,
                top_k=10 if action_mode or comparison_mode else 3,
            )


            # -------------------------
            # 5. Generate AI response
            # -------------------------

            response = generate_response(
                data.message,
                history,
                context,
                task_mode=action_mode,
            )

            if action_mode and context:
                cursor.execute(
                    """
                    INSERT INTO tasks (user_id, task_type, task_input, result)
                    VALUES (?, ?, ?, ?)
                    """,
                    (data.user_id, "document_action_plan", data.message, response),
                )
                task_id = cursor.lastrowid
                response = f"Saved to your task list as Task #{task_id}.\n\n{response}"
            elif action_mode:
                response = (
                    "I could not find relevant indexed PDF evidence, so I did not save a task. "
                    "Upload the document or ask a question that identifies the relevant topic."
                )


        # -------------------------
        # 6. Save AI response
        # -------------------------

        cursor.execute(
            """
            INSERT INTO messages
            (conversation_id, role, message)
            VALUES (?, ?, ?)
            """,
            (
                data.conversation_id,
                "assistant",
                response
            )
        )

        conn.commit()
        invalidate_message_history(data.conversation_id)


        # -------------------------
        # 7. Return response
        # -------------------------

        return {
            "user_message": data.message,
            "assistant_response": response,
            "task_id": task_id,
        }


    except Exception as e:

        conn.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"LLM Error: {str(e)}"
        )


    finally:

        conn.close()
