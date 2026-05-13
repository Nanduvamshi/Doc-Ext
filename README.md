# Doc-Ext: AWS Textract Document Parser

This application provides a FastAPI-based backend to upload documents and extract text, tables, and forms using AWS Textract. It also integrates with the Gemini API for enhanced document analysis.

## Features

- **Document Upload:** Upload PDF or image files for processing.
- **AWS Textract Integration:** Automatically extracts structured data (tables, forms) and raw text.
- **Gemini API Integration:** Utilizes Gemini 2.0 Flash for intelligent content generation and analysis based on extracted text.
- **Static Frontend:** Includes a simple web interface for easy interaction.

## Prerequisites

- Python 3.7+
- AWS Account with Textract and S3 access.
- Gemini API Key.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nanduvamshi/Doc-Ext.git
   cd Doc-Ext
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (if applicable) or configure `main.py` with your AWS credentials and Gemini API Key.

## Usage

Start the FastAPI server:
```bash
uvicorn main:app --reload
```
Open your browser and navigate to `http://localhost:8000` to access the interface.

## Project Structure

- `main.py`: FastAPI backend logic.
- `static/`: Frontend static files (HTML, CSS, JS).
- `requirements.txt`: Python dependencies.
