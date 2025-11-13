import os
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
from page_analysis import process_pdf
from compression import compress_pdf_smart
from helper_function import load_metadata

app = Flask(__name__)

# Folder settings
IMAGE_SUBDIR = 'classified_pages'
UPLOAD_SUBDIR = 'uploaded_files'
COMPRESSED_SUBDIR = 'compressed_files'

app.config['IMAGE_SUBDIR'] = IMAGE_SUBDIR
app.config['UPLOAD_SUBDIR'] = UPLOAD_SUBDIR
app.config['COMPRESSED_SUBDIR'] = COMPRESSED_SUBDIR

app.config['IMAGE_FOLDER_PATH'] = os.path.join(app.root_path, 'static', IMAGE_SUBDIR)
app.config['UPLOAD_FOLDER_PATH'] = os.path.join(app.root_path, UPLOAD_SUBDIR)
app.config['COMPRESSED_FOLDER_PATH'] = os.path.join(app.root_path, COMPRESSED_SUBDIR)

app.config['ALLOWED_IMAGE_EXTENSIONS'] = ('.png',)

# Ensure directories exist
os.makedirs(app.config['IMAGE_FOLDER_PATH'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER_PATH'], exist_ok=True)
os.makedirs(app.config['COMPRESSED_FOLDER_PATH'], exist_ok=True)


def clear_image_folder(folder_path):
    """Delete all files in the specified folder."""
    print(f"Clearing folder: {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF uploads."""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return 'No PDF file selected or file is not a PDF.', 400

    clear_image_folder(app.config['IMAGE_FOLDER_PATH'])

    filename = secure_filename(file.filename)
    pdf_save_path = os.path.join(app.config['UPLOAD_FOLDER_PATH'], filename)
    file.save(pdf_save_path)
    print(f"File successfully uploaded and saved to: {pdf_save_path}")

    try:
        process_pdf(pdf_save_path, app.config['IMAGE_FOLDER_PATH'])
    except Exception as e:
        print(f"Error during PDF processing: {e}")
        return f"An error occurred during PDF processing: {e}", 500

    # Redirect to results page with filename as parameter
    return redirect(url_for('show_results', filename=filename))


@app.route('/results')
def show_results():
    """Display processed images."""
    image_paths = []
    folder_path = app.config['IMAGE_FOLDER_PATH']

    if os.path.exists(folder_path):
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(app.config['ALLOWED_IMAGE_EXTENSIONS']):
                relative_path = f"{app.config['IMAGE_SUBDIR']}/{filename}"
                image_paths.append(relative_path)

    filename = request.args.get('filename')
    return render_template('results.html', image_list=image_paths, filename=filename)


@app.route('/compress', methods=['POST'])
def compress():
    """Compress the uploaded PDF and return download link."""
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER_PATH'], filename)
    output_filename = f"compressed_{filename}"
    output_path = os.path.join(app.config['COMPRESSED_FOLDER_PATH'], output_filename)
    metadata_file = os.path.join(app.config['IMAGE_FOLDER_PATH'], "region_metadata.json")

    # Check if files exist
    if not os.path.exists(input_path):
        return jsonify({'error': 'Input PDF not found'}), 404
    
    if not os.path.exists(metadata_file):
        return jsonify({'error': 'Metadata file not found. Please process the PDF first.'}), 404

    try:
        # Load metadata
        metadata = load_metadata(metadata_file)
        print(f"✓ Loaded metadata for {len(metadata)} pages")

        # Call compression function
        compress_pdf_smart(input_path, output_path, metadata)

        # Get file sizes for stats
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        savings_mb = (original_size - compressed_size) / (1024 * 1024)
        compression_pct = (1 - compressed_size / original_size) * 100

        return jsonify({
            'message': 'Compression completed successfully.',
            'download_url': url_for('download_file', filename=output_filename),
            'stats': {
                'original_size_mb': round(original_size / (1024 * 1024), 2),
                'compressed_size_mb': round(compressed_size / (1024 * 1024), 2),
                'savings_mb': round(savings_mb, 2),
                'compression_percentage': round(compression_pct, 1)
            }
        })
    except Exception as e:
        print(f"Error during compression: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Serve compressed file for download."""
    file_path = os.path.join(app.config['COMPRESSED_FOLDER_PATH'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404


if __name__ == '__main__':
    app.run(debug=True)