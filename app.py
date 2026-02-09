from flask import Flask, request, jsonify
import os
from resume_parser import parse_resume

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/parse-resume", methods=["POST"])
def parse_resume_endpoint():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400

    file = request.files["resume"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

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
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
