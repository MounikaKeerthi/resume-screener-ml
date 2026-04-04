from flask import Flask, request, jsonify
import os
from resume_parser import parse_resume

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint — useful for debugging."""
    return jsonify({"status": "ok", "version": "2.0"})


@app.route("/parse-resume", methods=["POST"])
def parse_resume_endpoint():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded. Send as multipart/form-data with key 'resume'"}), 400

    file = request.files["resume"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        job_description = request.form.get("job_description", "")
        parsed_data = parse_resume(
            pdf_path=file_path,
            job_description=job_description
        )
        return jsonify(parsed_data)

    except Exception as e:
        return jsonify({"error": str(e), "hint": "Check server logs for full traceback"}), 500

    finally:
        # Clean up uploaded file after processing
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    app.run(debug=True)