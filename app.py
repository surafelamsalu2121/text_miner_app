from flask import Flask, render_template
from app.routes import routes  # Import the blueprint
import os
from pinecone import Pinecone, ServerlessSpec

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads/'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize Pinecone instance and set it in the app context
    app.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Check if index exists and create if necessary
    index_name = "smbap"
    if index_name not in [idx.name for idx in app.pinecone_client.list_indexes()]:
        app.pinecone_client.create_index(
            name=index_name,
            dimension=384,  # Set the dimension to match the embedding model
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")  # Change region if needed
        )

    # Register the blueprint
    app.register_blueprint(routes)

    # Define the home route
    @app.route('/')
    def home():
        return render_template('index.html')  # Render the index template for the home route

    return app

if __name__ == '__main__':
    if os.getenv("FLASK_ENV") == "development":
        app.run(debug=True, host='0.0.0.0', port=5000)
