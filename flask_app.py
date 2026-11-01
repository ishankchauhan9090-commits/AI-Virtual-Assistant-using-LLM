import os

from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import requests

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "local-development-key-change-before-deployment"
)

FASTAPI_URL = os.getenv(
    "FASTAPI_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

USER_ID = int(os.getenv("USER_ID", "1"))
DEFAULT_CONVERSATION_ID = int(os.getenv("DEFAULT_CONVERSATION_ID", "1"))


def backend_is_online():
    try:
        response = requests.get(f"{FASTAPI_URL}/", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def get_conversations():
    try:
        response = requests.get(
            f"{FASTAPI_URL}/conversations/{USER_ID}",
            timeout=10
        )

        if response.status_code != 200:
            return []

        data = response.json()

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return data.get("conversations", [])

    except requests.RequestException:
        return []

    return []


def get_messages(conversation_id):
    try:
        response = requests.get(
            f"{FASTAPI_URL}/messages/{conversation_id}",
            timeout=10
        )

        if response.status_code != 200:
            return []

        data = response.json()

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return data.get("messages", [])

    except requests.RequestException:
        return []

    return []


# ---------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    conversation_id = session.get(
        "conversation_id",
        DEFAULT_CONVERSATION_ID
    )

    backend_online = backend_is_online()

    if backend_online:
        conversations = get_conversations()
        messages = get_messages(conversation_id)
        error = None
    else:
        conversations = []
        messages = []
        error = "FastAPI is not responding. Start the backend service, then refresh this page."

    return render_template(
        "index.html",
        conversations=conversations,
        messages=messages,
        conversation_id=conversation_id,
        error=error,
        backend_online=backend_online
    )


# ---------------------------------------------------------
# AJAX CHAT ENDPOINT
# ---------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():

    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "error": "No data received."
        }), 400

    user_message = data.get(
        "message",
        ""
    ).strip()

    conversation_id = data.get(
        "conversation_id"
    )

    if not user_message:
        return jsonify({
            "success": False,
            "error": "Message cannot be empty."
        }), 400

    if not conversation_id:
        conversation_id = session.get(
            "conversation_id",
            DEFAULT_CONVERSATION_ID
        )

    payload = {
        "user_id": USER_ID,
        "conversation_id": int(conversation_id),
        "message": user_message
    }

    try:

        response = requests.post(
            f"{FASTAPI_URL}/chat",
            json=payload,
            timeout=120
        )

        if response.status_code != 200:

            return jsonify({
                "success": False,
                "error": response.text
            }), 500

        result = response.json()

        return jsonify({
            "success": True,
            "user_message": result.get(
                "user_message",
                user_message
            ),
            "assistant_response": result.get(
                "assistant_response",
                ""
            ),
            "task_id": result.get("task_id"),
            "conversation_id": int(conversation_id)
        })

    except requests.Timeout:

        return jsonify({
            "success": False,
            "error": "The AI took too long to respond. Please try again."
        }), 504

    except requests.RequestException as e:

        return jsonify({
            "success": False,
            "error": f"Backend connection error: {str(e)}"
        }), 500


@app.route("/api/documents", methods=["POST"])
def api_upload_document():
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        uploaded_files = request.files.getlist("file")
    uploaded_files = [item for item in uploaded_files if item and item.filename]

    if not uploaded_files:
        return jsonify({
            "success": False,
            "error": "Choose one or more PDFs to upload."
        }), 400

    if len(uploaded_files) > 5:
        return jsonify({
            "success": False,
            "error": "Upload at most 5 PDFs at a time."
        }), 413

    try:
        response = requests.post(
            f"{FASTAPI_URL}/documents/upload",
            files=[
                (
                    "files",
                    (
                        uploaded_file.filename,
                        uploaded_file.stream,
                        uploaded_file.mimetype or "application/pdf"
                    )
                )
                for uploaded_file in uploaded_files
            ],
            data={"user_id": USER_ID},
            timeout=180
        )

        try:
            result = response.json()
        except ValueError:
            result = {}

        if response.status_code != 200:
            return jsonify({
                "success": False,
                "error": result.get(
                    "detail",
                    "The document service could not process this PDF."
                )
            }), response.status_code

        return jsonify(result)

    except requests.Timeout:
        return jsonify({
            "success": False,
            "error": "Indexing took too long. Please try a smaller PDF."
        }), 504

    except requests.RequestException:
        return jsonify({
            "success": False,
            "error": "Cannot reach the document service. Make sure FastAPI is running."
        }), 502


@app.route("/api/documents", methods=["GET"])
def api_get_documents():
    try:
        response = requests.get(
            f"{FASTAPI_URL}/documents/{USER_ID}",
            timeout=10,
        )
        if response.status_code != 200:
            return jsonify({"success": False, "error": "Could not load PDFs."}), 502
        return jsonify({"success": True, "documents": response.json()})
    except requests.RequestException:
        return jsonify({"success": False, "error": "Cannot reach the document service."}), 502


@app.route("/api/documents/<int:document_id>", methods=["DELETE"])
def api_delete_document(document_id):
    try:
        response = requests.delete(
            f"{FASTAPI_URL}/documents/{document_id}",
            params={"user_id": USER_ID},
            timeout=60,
        )
        if response.status_code != 200:
            try:
                detail = response.json().get("detail", "Could not delete the PDF.")
            except ValueError:
                detail = "Could not delete the PDF."
            return jsonify({"success": False, "error": detail}), response.status_code
        return jsonify(response.json())
    except requests.RequestException:
        return jsonify({"success": False, "error": "Cannot reach the document service."}), 502


@app.route("/api/tasks", methods=["GET"])
def api_get_tasks():
    try:
        response = requests.get(
            f"{FASTAPI_URL}/tasks/{USER_ID}",
            timeout=10,
        )
        if response.status_code != 200:
            return jsonify({"success": False, "error": "Could not load saved tasks."}), 502
        tasks = response.json()
        return jsonify({
            "success": True,
            "tasks": [task for task in tasks if task.get("task_type") == "document_action_plan"],
        })
    except requests.RequestException:
        return jsonify({"success": False, "error": "Cannot reach the task service."}), 502


@app.route("/api/conversations/<int:conversation_id>", methods=["DELETE"])
def api_delete_conversation(conversation_id):
    try:
        response = requests.delete(
            f"{FASTAPI_URL}/conversations/{conversation_id}",
            params={"user_id": USER_ID},
            timeout=15,
        )
        if response.status_code != 200:
            try:
                detail = response.json().get("detail", "Could not delete the conversation.")
            except ValueError:
                detail = "Could not delete the conversation."
            return jsonify({"success": False, "error": detail}), response.status_code

        if session.get("conversation_id") == conversation_id:
            session.pop("conversation_id", None)
            new_response = requests.post(
                f"{FASTAPI_URL}/conversations",
                json={"user_id": USER_ID, "title": "New Conversation"},
                timeout=10,
            )
            if new_response.ok:
                session["conversation_id"] = new_response.json().get("conversation_id")
        return jsonify({"success": True, "conversation_id": conversation_id})
    except requests.RequestException:
        return jsonify({"success": False, "error": "Cannot reach the conversation service."}), 502


# ---------------------------------------------------------
# NEW CHAT
# ---------------------------------------------------------

@app.route("/new-chat", methods=["POST"])
def new_chat():

    title = request.form.get(
        "title",
        "New Conversation"
    ).strip()

    if not title:
        title = "New Conversation"

    payload = {
        "user_id": USER_ID,
        "title": title
    }

    try:

        response = requests.post(
            f"{FASTAPI_URL}/conversations",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:

            data = response.json()

            conversation_id = data.get(
                "conversation_id"
            )

            if conversation_id:

                session["conversation_id"] = (
                    conversation_id
                )

    except requests.RequestException:
        pass

    return redirect(url_for("home"))


# ---------------------------------------------------------
# SELECT CONVERSATION
# ---------------------------------------------------------

@app.route("/conversation/<int:conversation_id>")
def select_conversation(conversation_id):

    session["conversation_id"] = conversation_id

    return redirect(url_for("home"))


# ---------------------------------------------------------
# RUN FLASK
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )
