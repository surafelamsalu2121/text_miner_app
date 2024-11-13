from flask import Blueprint, request, render_template, jsonify, current_app
import os
from werkzeug.utils import secure_filename
from app.nlp_processor import parse_document
from app.embeddings import generate_embeddings_and_store, generate_summary
from app.utils import save_json_output

routes = Blueprint('routes', __name__)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@routes.route('/upload', methods=['POST'])
def upload_files():
    current_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    if 'document' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('document')
    processed_results = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the document using NLP logic
            parsed_content = parse_document(file_path)
            embedding = generate_embeddings_and_store(parsed_content)
            summary = generate_summary(parsed_content)

            print(f"Summary for {filename}: {summary}")  # Debug print
            print(f"Embedding for {filename}: {embedding}")  # Debug print

            # Save JSON output and embeddings
            save_json_output(parsed_content, filename)
            processed_results.append({
                "filename": filename,
                "summary": summary,
                # Remove embedding if it should not be displayed
                # "embedding": embedding
            })

    return render_template('results.html', results=processed_results)
@routes.route('/about')
def about():
    return render_template('about.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
