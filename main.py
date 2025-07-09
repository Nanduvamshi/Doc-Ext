import boto3
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # Import StaticFiles
from pydantic import BaseModel
import io
import time
import os
import tempfile
import shutil
import traceback
import re
import csv
from io import StringIO
import httpx # Import httpx for asynchronous HTTP requests
import json # Import json for parsing API responses

# --- Configuration ---
# IMPORTANT: Replace with your actual S3 bucket name
S3_BUCKET_NAME = 'insurify-code' # <<< CHANGE THIS
AWS_REGION = 'us-east-1' # Change to your desired AWS region

# --- Gemini API Configuration ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
# API key is left as an empty string; Canvas will provide it at runtime.
GEMINI_API_KEY = "AIzaSyAedmC_y_dfv9FgF3p0NKwx0MR8zUWarM0"

# --- Initialize AWS Clients ---
textract_client = boto3.client('textract', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)

app = FastAPI(
    title="AWS Textract Document Parser",
    description="API to upload documents and extract text, tables, and forms using AWS Textract."
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount Static Files Directory ---
# This tells FastAPI to serve files from the "static" directory.
# Ensure your index.html is inside a folder named 'static' next to main.py
app.mount("/static", StaticFiles(directory="static"), name="static")

class DocumentParseRequest(BaseModel):
    extracted_text: str
    forms: list

# --- Helper Functions for AWS Textract Interaction ---

def upload_document_to_s3(file_path: str, bucket_name: str, object_name: str = None):
    """Uploads a file to an S3 bucket."""
    if object_name is None:
        object_name = os.path.basename(file_path)
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"File '{file_path}' uploaded to s3://{bucket_name}/{object_name}")
        return {
            'Bucket': bucket_name,
            'Name': object_name
        }
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return None

def start_textract_job(document_location: dict, job_type: str = 'ANALYZE_DOCUMENT'):
    """
    Starts an asynchronous Textract job.
    """
    try:
        if job_type == 'DETECT_TEXT':
            response = textract_client.start_document_text_detection(DocumentLocation=document_location)
        elif job_type == 'ANALYZE_DOCUMENT':
            response = textract_client.start_document_analysis(
                DocumentLocation=document_location,
                FeatureTypes=['TABLES', 'FORMS']
            )
        else:
            raise ValueError("Invalid job_type. Must be 'DETECT_TEXT' or 'ANALYZE_DOCUMENT'.")

        job_id = response['JobId']
        print(f"Started Textract job with JobId: {job_id}")
        return job_id
    except Exception as e:
        print(f"Error starting Textract job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start Textract job: {e}")

def get_textract_job_results(job_id: str):
    """
    Polls for Textract job results until the job is complete.
    """
    print(f"Waiting for Textract job {job_id} to complete...")
    status = None
    while status != 'SUCCEEDED' and status != 'FAILED':
        time.sleep(5)
        try:
            response = textract_client.get_document_analysis(JobId=job_id)
            status = response['JobStatus']
            print(f"Job status: {status}")
        except Exception as e:
            print(f"Error getting job status for {job_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get Textract job status: {e}")

    if status == 'SUCCEEDED':
        full_response_blocks = []
        next_token = None
        while True:
            try:
                if next_token:
                    response = textract_client.get_document_analysis(JobId=job_id, NextToken=next_token)
                else:
                    response = textract_client.get_document_analysis(JobId=job_id)

                full_response_blocks.extend(response.get('Blocks', []))
                if 'NextToken' in response:
                    next_token = response['NextToken']
                else:
                    break
            except Exception as e:
                print(f"Error fetching paginated results for {job_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to fetch Textract results: {e}")
        return {'Blocks': full_response_blocks}
    else:
        print(f"Textract job {job_id} failed with status: {status}")
        raise HTTPException(status_code=500, detail=f"Textract job failed with status: {e}")

def extract_text_from_blocks(blocks: list):
    """
    Extracts all detected text from Textract Blocks.
    """
    full_text = []
    for block in blocks:
        if block.get('BlockType') == 'LINE':
            full_text.append(block.get('Text', ''))
    return "\n".join(full_text)

def parse_textract_tables(blocks: list):
    """
    Parses table data from Textract blocks into a list of lists.
    """
    tables_data = []
    block_map = {block['Id']: block for block in blocks if 'Id' in block}

    for block in blocks:
        if block.get('BlockType') == 'TABLE':
            cell_ids = []
            relationships = block.get('Relationships')
            if relationships and isinstance(relationships, list):
                cell_ids = [rel['Id'] for rel in relationships if rel.get('Type') == 'CHILD' and 'Id' in rel]
            else:
                print(f"Debug: Table block {block.get('Id', 'N/A')} missing or malformed 'Relationships'. Skipping cell extraction.")
                continue

            table_cells = []
            for cid in cell_ids:
                cell_block = block_map.get(cid)
                if cell_block:
                    table_cells.append(cell_block)
                else:
                    print(f"Debug: Cell ID {cid} referenced in table {block.get('Id', 'N/A')} not found in block_map.")

            max_row = 0
            max_col = 0
            for cell in table_cells:
                max_row = max(max_row, cell.get('RowIndex', 0))
                max_col = max(max_col, cell.get('ColumnIndex', 0))
            
            if max_row == 0 or max_col == 0:
                print(f"Debug: Table {block.get('Id', 'N/A')} has no cells with valid RowIndex/ColumnIndex or is empty.")
                continue

            current_table_data = [['' for _ in range(max_col)] for _ in range(max_row)]

            for cell in table_cells:
                row_idx = cell.get('RowIndex', 1) - 1
                col_idx = cell.get('ColumnIndex', 1) - 1
                if 0 <= row_idx < max_row and 0 <= col_idx < max_col:
                    current_table_data[row_idx][col_idx] = cell.get('Text', '')

            tables_data.append(current_table_data)
    return tables_data

def parse_textract_forms(blocks: list):
    """
    Parses form data (key-value pairs) from Textract blocks into a list of dictionaries.
    """
    key_value_pairs = []
    block_map = {block['Id']: block for block in blocks if 'Id' in block} 

    for block in blocks:
        if block.get('BlockType') == 'KEY_VALUE_SET' and block.get('EntityType') == 'KEY':
            key_text = ''
            value_text = ''

            relationships = block.get('Relationships')
            if relationships and isinstance(relationships, list):
                for rel in relationships:
                    if rel.get('Type') == 'CHILD' and 'Ids' in rel and isinstance(rel['Ids'], list):
                        for child_id in rel['Ids']:
                            child_block = block_map.get(child_id)
                            if child_block and child_block.get('BlockType') == 'WORD':
                                key_text += child_block.get('Text', '') + ' '
            else:
                print(f"Debug: KEY_VALUE_SET (KEY) block {block.get('Id', 'N/A')} missing or malformed 'Relationships'.")
                continue

            relationships = block.get('Relationships')
            if relationships and isinstance(relationships, list):
                for rel in relationships:
                    if rel.get('Type') == 'VALUE' and 'Ids' in rel and isinstance(rel['Ids'], list):
                        for value_id in rel['Ids']:
                            value_block = block_map.get(value_id)
                            if value_block:
                                val_relationships = value_block.get('Relationships')
                                if val_relationships and isinstance(val_relationships, list):
                                    for val_rel in val_relationships:
                                        if val_rel.get('Type') == 'CHILD' and 'Ids' in val_rel and isinstance(val_rel['Ids'], list):
                                            for val_child_id in val_rel['Ids']:
                                                val_child_block = block_map.get(val_child_id)
                                                if val_child_block and val_child_block.get('BlockType') == 'WORD':
                                                    value_text += val_child_block.get('Text', '') + ' '
                                else:
                                    print(f"Debug: VALUE block {value_block.get('Id', 'N/A')} missing or malformed 'Relationships'.")
                            break

            key_value_pairs.append({'key': key_text.strip(), 'value': value_text.strip()})
    return key_value_pairs


# --- Global Configuration for Name Recognition (now primarily for context to Gemini) ---
# These lists are kept for potential future use or to help frame the prompt for Gemini,
# but the direct rule-based name extraction is now handled by the LLM.
keywords_that_are_never_names = [
    'date of birth', 'dob', 'gender', 'male', 'female', 'transgender', 'address', 
    'id', 'card', 'number', 'pan', 'income tax', 'department', 'ministry', 
    'government of india', 'unique identification authority', 'भारतीय विशिष्ट पहचान प्राधिकरण', 'भारत सरकार', 
    'p.o.', 'tal', 'dist', 'state', 'pin', 'uidai', 
    'director general', 'ceo', 'enrollment no', 'acknowledgement no', 
    'indian', 'india', 'acknowledgement', 'photograph', 'valid till', 'phone', 'mobile',
    'forhist', 'hrook', 'under with', 'dest',
    'e - permanent account number card',
    'tax department', 'govt. of india', 'permanent account number card', 'signature',
    'issued by', 'for', 'father\'s name', 'father name', 'पिता का नाम',
    '21-17', 
    'Vissue', 
    'VID :', 
    'you MALE', 
    'HRT Ran', 
    'PVT you', 
    'Domnhad', 
    'HRT 3TTETT , HA',
    'herr', 'celle', 'ade', 'areit', # Added more specific noise words for relation name filtering
    'street hear', 'offsite', 'stret', 'floof par', 'art' # Added new noise from user example
]

full_line_patterns_to_avoid = [
    r'^\s*\d{4}\s\d{4}\s\d{4}\s*$', # Aadhar number format (e.g., "3306 9998 1453")
    r'^\s*[A-Z]{5}\d{4}[A-Z]{1}\s*$', # PAN number format
    r'^\s*\d{2}[/\.\-\\]\d{2}[/\\.\-\\]\d{4}\s*$', # Simple date format
    r'^[A-Z\s]{2,}\s*:\s*.+', # General Key: Value pattern, e.g., "THE ITEM:"
    r'^\s*[\s.,\'-]*$', # Empty lines or lines with only ignored characters
    r'^\s*[A-Z]{1,2}\s*$', # Single or two uppercase letters
    r'^\s*e\s*$',
    r'^\s*a\s*$',
    r'^\s*3112100\s+FORHIST\s*$',
    r'^\s*HRR\s+HROOK\s*$',
    r'^\s*7\s+\d+\s*$',
    r'^\s*under\s+with\s*$',
    r'^\s*dest\s+\d+\s*$',
    r'^\s*root\s+\$[0-9a-zA-Z]+\s+aRe\s*$',
    r'^\s*INCOME\s+TAX\s+DEPARTMENT\s*$',
    r'^\s*GOVT\.\s+OF\s+INDIA\s*$',
    r'^\s*Permanent\s+Account\s+Number\s+Card\s*$',
    r'^\s*Signature\s*$',
    r'^\s*#\s*title\s+name\s*$',
    r'^\s*##text just after the name title is the name that have to be printed under the name column in csv\s*$',
    r'^\s*issued\s+by\s*$',
    r'^\s*for\s*$',
    r'^\s*\d+\s+\d+\s*\/\s*Father\'s\s+Name\s*$',
    r'\b(?:HIRET\s+HEAR|FATHEST|HRA|FROOT|TRUM|Ferreft\s+HRG|TTA)\b',
    r'^\s*21-17\s*$', 
    r'^\s*Vissue', 
    r'^\s*VID :', 
    r'^\s*you MALE', 
    r'^\s*HRT Ran', 
    r'^\s*PVT you', 
    r'^\s*Domnhad', 
    r'^\s*HRT 3TTETT , HA',
    'herr', 'celle', 'ade', 'areit', # Added more specific noise words for relation name filtering
    # Added patterns that might be part of an address but should not be the entire "Father's Name"
    r'^\s*p\.o\.\s*$',
    r'^\s*tal\s*$',
    r'^\s*dist\s*$',
    r'^\s*state\s*$',
    r'^\s*pin\s*$',
    r'^\s*\d{6}\s*$', # Standalone 6-digit number (PIN code)
    r'^\s*city\s*$',
    r'^\s*district\s*$',
    r'^\s*village\s*$',
    r'^\s*town\s*$',
    r'^\s*road\s*$',
    r'^\s*street\s*$',
    r'^\s*house\s+no\s*$',
    r'^\s*flat\s+no\s*$',
    r'^\s*lane\s*$',
    r'^\s*area\s*$',
    r'^\s*locality\s*$',
    r'^\s*post\s+office\s*$',
    r'^\s*police\s+station\s*$',
    r'^\s*sub\s+district\s*$',
    r'^\s*tehsil\s*$',
    r'^\s*mandal\s*$',
    r'^\s*thana\s*$',
    r'^\s*marg\s*$',
    r'^\s*gali\s*$',
    r'^\s*sector\s*$',
    r'^\s*ward\s*$',
    r'^\s*building\s*$',
    r'^\s*apartment\s*$',
    r'^\s*colony\s*$',
    r'^\s*near\s*$',
    r'^\s*opposite\s*$',
    r'^\s*backside\s*$',
    r'^\s*front\s*$',
    r'^\s*main\s*$',
    r'^\s*cross\s*$',
    r'^\s*by\s+pass\s*$',
    r'^\s*bypass\s*$',
    r'^\s*highway\s*$',
    r'^\s*roadway\s*$',
    r'^\s*industrial\s+area\s*$',
    r'^\s*estate\s*$',
    r'^\s*park\s*$',
    r'^\s*garden\s*$',
    r'^\s*sq\.\s*$',
    r'^\s*square\s*$',
    r'^\s*unit\s*$',
    r'^\s*floor\s*$',
    r'^\s*shop\s*$',
    r'^\s*office\s*$',
    r'^\s*plot\s*$',
    r'^\s*survey\s+no\s*$',
    r'^\s*khasra\s+no\s*$',
    r'^\s*ghat\s*$',
    r'^\s*bank\s*$',
    r'^\s*river\s*$',
    r'^\s*canal\s*$',
    r'^\s*dam\s*$',
    r'^\s*lake\s*$',
    r'^\s*sea\s*$',
    r'^\s*ocean\s*$',
    r'^\s*hill\s*$',
    r'^\s*mountain\s*$',
    r'^\s*forest\s*$',
    r'^\s*reserve\s*$',
    r'^\s*national\s+park\s*$',
    r'^\s*wildlife\s+sanctuary\s*$',
    r'^\s*bird\s+sanctuary\s*$',
    r'^\s*temple\s*$',
    r'^\s*mosque\s*$',
    r'^\s*church\s*$',
    r'^\s*gurudwara\s*$',
    r'^\s*ashram\s*$',
    r'^\s*math\s*$',
    r'^\s*hospital\s*$',
    r'^\s*clinic\s*$',
    r'^\s*dispensary\s*$',
    r'^\s*school\s*$',
    r'^\s*college\s*$',
    r'^\s*university\s*$',
    r'^\s*institute\s*$',
    r'^\s*coaching\s+centre\s*$',
    r'^\s*bank\s*$',
    r'^\s*atm\s*$',
    r'^\s*petrol\s+pump\s*$',
    r'^\s*police\s+station\s*$',
    r'^\s*fire\s+station\s*$',
    r'^\s*bus\s+stand\s*$',
    r'^\s*railway\s+station\s*$',
    r'^\s*airport\s*$',
    r'^\s*seaport\s*$',
    r'^\s*dockyard\s*$',
    r'^\s*port\s+trust\s*$',
    r'^\s*market\s*$',
    r'^\s*shopping\s+mall\s*$',
    r'^\s*cinema\s+hall\s*$',
    r'^\s*theatre\s*$',
    r'^\s*stadium\s*$',
    r'^\s*community\s+centre\s*$',
    r'^\s*club\s*$',
    r'^\s*resort\s*$',
    r'^\s*hotel\s*$',
    r'^\s*guest\s+house\s*$',
    r'^\s*dharamshala\s*$',
    r'^\s*lodge\s*$',
    r'^\s*resthouse\s*$',
    r'^\s*factory\s*$',
    r'^\s*industry\s*$',
    r'^\s*company\s*$',
    r'^\s*corporation\s*$',
    r'^\s*limited\s*$',
    r'^\s*private\s+limited\s*$',
    r'^\s*pvt\.\s+ltd\s*$',
    r'^\s*inc\.\s*$',
    r'^\s*llc\s*$',
    r'^\s*partnership\s*$',
    r'^\s*proprietorship\s*$',
    r'^\s*trust\s*$',
    r'^\s*society\s*$',
    r'^\s*association\s*$',
    r'^\s*federation\s*$',
    r'^\s*union\s*$',
    r'^\s*board\s*$',
    r'^\s*council\s*$',
    r'^\s*committee\s*$',
    r'^\s*commission\s*$',
    r'^\s*authority\s*$',
    r'^\s*department\s*$',
    r'^\s*ministry\s*$',
    r'^\s*government\s*$',
    r'^\s*india\s*$',
    r'^\s*indian\s*$',
    r'^\s*bharat\s*$',
    r'^\s*republic\s+of\s+india\s*$',
    r'^\s*union\s+of\s+india\s*$',
    r'^\s*state\s+of\s+\w+\s*$', # e.g., "State of Uttar Pradesh"
    r'^\s*district\s+of\s+\w+\s*$',
    r'^\s*tehsil\s+of\s+\w+\s*$',
    r'^\s*taluk\s+of\s+\w+\s*$',
    r'^\s*mandal\s+of\s+\w+\s*$',
    r'^\s*p\.s\.\s*$', # Police Station abbreviation
    r'^\s*f\.i\.r\.\s*$', # FIR
    r'^\s*case\s+no\s*$',
    r'^\s*file\s+no\s*$',
    r'^\s*document\s+no\s*$',
    r'^\s*reference\s+no\s*$',
    r'^\s*application\s+no\s*$',
    r'^\s*enrollment\s+no\s*$',
    r'^\s*acknowledgement\s+no\s*$',
    r'^\s*date\s+of\s+issue\s*$',
    r'^\s*date\s+of\s+expiry\s*$',
    r'^\s*valid\s+from\s*$',
    r'^\s*valid\s+till\s*$',
    r'^\s*date\s+of\s+effect\s*$',
    r'^\s*effective\s+date\s*$',
    r'^\s*issue\s+date\s*$',
    r'^\s*expiry\s+date\s*$',
    r'^\s*from\s+date\s*$',
    r'^\s*to\s+date\s*$',
    r'^\s*period\s+of\s+validity\s*$',
    r'^\s*duration\s*$',
    r'^\s*age\s*$',
    r'^\s*years\s*$',
    r'^\s*months\s*$',
    r'^\s*days\s*$',
    r'^\s*hours\s*$',
    r'^\s*minutes\s*$',
    r'^\s*seconds\s*$',
    r'^\s*time\s*$',
    r'^\s*am\s*$',
    r'^\s*pm\s*$',
    r'^\s*morning\s*$',
    r'^\s*afternoon\s*$',
    r'^\s*evening\s*$',
    r'^\s*night\s*$',
    r'^\s*today\s*$',
    r'^\s*tomorrow\s*$',
    r'^\s*yesterday\s*$',
    r'^\s*next\s+day\s*$',
    r'^\s*previous\s+day\s*$',
    r'^\s*current\s+date\s*$',
    r'^\s*current\s+time\s*$',
    r'^\s*signature\s*$',
    r'^\s*applicant\s*$',
    r'^\s*authorized\s+signatory\s*$',
    r'^\s*seal\s*$',
    r'^\s*stamp\s*$',
    r'^\s*photo\s*$',
    r'^\s*photograph\s*$',
    r'^\s*image\s*$',
    r'^\s*picture\s*$',
    r'^\s*qr\s+code\s*$',
    r'^\s*barcode\s*$',
    r'^\s*digital\s+signature\s*$',
    r'^\s*serial\s+no\s*$',
    r'^\s*document\s+id\s*$',
    r'^\s*transaction\s+id\s*$',
    r'^\s*payment\s+id\s*$',
    r'^\s*order\s+id\s*$',
    r'^\s*invoice\s+no\s*$',
    r'^\s*bill\s+no\s*$',
    r'^\s*receipt\s+no\s*$',
    r'^\s*challan\s+no\s*$',
    r'^\s*gstin\s*$',
    r'^\s*iec\s+code\s*$',
    r'^\s*hsn\s+code\s*$',
    r'^\s*sac\s+code\s*$',
    r'^\s*bank\s+account\s+no\s*$',
    r'^\s*ifsc\s+code\s*$',
    r'^\s*micr\s+code\s*$',
    r'^\s*swift\s+code\s*$',
    r'^\s*branch\s+name\s*$',
    r'^\s*bank\s+name\s*$',
    r'^\s*account\s+holder\s+name\s*$',
    r'^\s*account\s+type\s*$',
    r'^\s*savings\s*$',
    r'^\s*current\s*$',
    r'^\s*loan\s*$',
    r'^\s*credit\s+card\s*$',
    r'^\s*debit\s+card\s*$',
    r'^\s*card\s+number\s*$',
    r'^\s*cvv\s*$',
    r'^\s*expiry\s+date\s*$',
    r'^\s*cardholder\s+name\s*$',
    r'^\s*network\s*$',
    r'^\s*visa\s*$',
    r'^\s*mastercard\s*$',
    r'^\s*rupay\s*$',
    r'^\s*amex\s*$',
    r'^\s*discover\s*$',
    r'^\s*upi\s+id\s*$',
    r'^\s*upi\s+number\s*$',
    r'^\s*vpa\s*$',
    r'^\s*transaction\s+date\s*$',
    r'^\s*transaction\s+time\s*$',
    r'^\s*amount\s*$',
    r'^\s*currency\s*$',
    r'^\s*inr\s*$',
    r'^\s*usd\s*$',
    r'^\s*eur\s*$',
    r'^\s*gbp\s*$',
    r'^\s*jpy\s*$',
    r'^\s*cad\s*$',
    r'^\s*aud\s*$',
    r'^\s*sgd\s*$',
    r'^\s*chf\s*$',
    r'^\s*cny\s*$',
    r'^\s*total\s+amount\s*$',
    r'^\s*net\s+amount\s*$',
    r'^\s*tax\s+amount\s*$',
    r'^\s*gst\s+amount\s*$',
    r'^\s*discount\s+amount\s*$',
    r'^\s*cess\s+amount\s*$',
    r'^\s*grand\s+total\s*$',
    r'^\s*total\s+paid\s*$',
    r'^\s*balance\s+due\s*$',
    r'^\s*due\s+date\s*$',
    r'^\s*payment\s+status\s*$',
    r'^\s*paid\s*$',
    r'^\s*unpaid\s*$',
    r'^\s*partially\s+paid\s*$',
    r'^\s*refunded\s*$',
    r'^\s*cancelled\s*$',
    r'^\s*pending\s*$',
    r'^\s*status\s*$',
    r'^\s*active\s*$',
    r'^\s*inactive\s*$',
    r'^\s*suspended\s*$',
    r'^\s*closed\s*$',
    r'^\s*approved\s*$',
    r'^\s*rejected\s*$',
    r'^\s*processing\s*$',
    r'^\s*completed\s*$',
    r'^\s*failed\s*$',
    r'^\s*success\s*$',
    r'^\s*error\s*$',
    r'^\s*message\s*$',
    r'^\s*remark\s*$',
    r'^\s*note\s*$',
    r'^\s*comments\s*$',
    r'^\s*description\s*$',
    r'^\s*purpose\s*$',
    r'^\s*category\s*$',
    r'^\s*type\s*$',
    r'^\s*subtype\s*$',
    r'^\s*code\s*$',
    r'^\s*number\s*$',
    r'^\s*id\s*$',
    r'^\s*name\s*$',
    r'^\s*title\s*$',
    r'^\s*designation\s*$',
    r'^\s*department\s*$',
    r'^\s*organization\s*$',
    r'^\s*company\s*$',
    r'^\s*firm\s*$',
    r'^\s*business\s*$',
    r'^\s*shop\s*$',
    r'^\s*store\s*$',
    r'^\s*establishment\s*$',
    r'^\s*institute\s*$',
    r'^\s*hospital\s*$',
    r'^\s*clinic\s*$',
    r'^\s*school\s*$',
    r'^\s*college\s*$',
    r'^\s*university\s*$',
    r'^\s*address\s*$',
    r'^\s*contact\s+number\s*$',
    r'^\s*phone\s+number\s*$',
    r'^\s*mobile\s+number\s*$',
    r'^\s*email\s+id\s*$',
    r'^\s*website\s*$',
    r'^\s*fax\s+number\s*$',
    r'^\s*telephone\s+number\s*$',
    r'^\s*customer\s+id\s*$',
    r'^\s*vendor\s+id\s*$',
    r'^\s*employee\s+id\s*$',
    r'^\s*student\s+id\s*$',
    r'^\s*patient\s+id\s*$',
    r'^\s*user\s+id\s*$',
    r'^\s*agent\s+id\s*$',
    r'^\s*license\s+number\s*$',
    r'^\s*registration\s+number\s*$',
    r'^\s*certificate\s+number\s*$',
    r'^\s*policy\s+number\s*$',
    r'^\s*policy\s+name\s*$',
    r'^\s*insured\s+name\s*$',
    r'^\s*insurer\s+name\s*$',
    r'^\s*premium\s+amount\s*$',
    r'^\s*sum\s+insured\s*$',
    r'^\s*maturity\s+date\s*$',
    r'^\s*nominee\s+name\s*$',
    r'^\s*relation\s+to\s+nominee\s*$',
    r'^\s*relation\s+name\s*$', # Added for explicit relation name
    r'^\s*fathers\s+name\s*$', # Added for explicit father's name
    r'^\s*mother\'s\s+name\s*$',
    r'^\s*spouse\'s\s+name\s*$',
    r'^\s*guardian\'s\s+name\s*$',
    r'^\s*kin\s+name\s*$',
    r'^\s*emergency\s+contact\s+name\s*$',
    r'^\s*emergency\s+contact\s+number\s*$',
    r'^\s*blood\s+group\s*$',
    r'^\s*height\s*$',
    r'^\s*weight\s*$',
    r'^\s*occupation\s*$',
    r'^\s*profession\s*$',
    r'^\s*marital\s+status\s*$',
    r'^\s*single\s*$',
    r'^\s*married\s*$',
    r'^\s*divorced\s*$',
    r'^\s*widowed\s*$',
    r'^\s*nationality\s*$',
    r'^\s*religion\s*$',
    r'^\s*caste\s*$',
    r'^\s*sub\s+caste\s*$',
    r'^\s*category\s*$',
    r'^\s*general\s*$',
    r'^\s*obc\s*$',
    r'^\s*sc\s*$',
    r'^\s*st\s*$',
    r'^\s*pwd\s*$',
    r'^\s*quota\s*$',
    r'^\s*domicile\s*$',
    r'^\s*residence\s*$',
    r'^\s*permanent\s+address\s*$',
    r'^\s*correspondence\s+address\s*$',
    r'^\s*present\s+address\s*$',
    r'^\s*past\s+address\s*$',
    r'^\s*native\s+place\s*$',
    r'^\s*birth\s+place\s*$',
    r'^\s*date\s+of\s+joining\s*$',
    r'^\s*date\s+of\s+leaving\s*$',
    r'^\s*employment\s+period\s*$',
    r'^\s*experience\s*$',
    r'^\s*qualification\s*$',
    r'^\s*degree\s*$',
    r'^\s*diploma\s*$',
    r'^\s*certificate\s*$',
    r'^\s*university\s*$',
    r'^\s*board\s*$',
    r'^\s*institute\s*$',
    r'^\s*year\s+of\s+passing\s*$',
    r'^\s*marks\s*$',
    r'^\s*percentage\s*$',
    r'^\s*grade\s*$',
    r'^\s*division\s*$',
    r'^\s*roll\s+number\s*$',
    r'^\s*registration\s+number\s*$',
    r'^\s*enrollment\s+number\s*$',
    r'^\s*seat\s+number\s*$',
    r'^\s*exam\s+name\s*$',
    r'^\s*subject\s*$',
    r'^\s*course\s*$',
    r'^\s*program\s*$',
    r'^\s*duration\s*$',
    r'^\s*admission\s+date\s*$',
    r'^\s*completion\s+date\s*$',
    r'^\s*s/o\s*$', # Standalone S/O
    r'^\s*c/o\s*$', # Standalone C/O
    r'^\s*d/o\s*$', # Standalone D/O
    r'^\s*w/o\s*$', # Standalone W/O
    r'^\s*yob\s*$', # Year of Birth abbreviation
    r'^\s*y\.\s+o\.\s+b\.\s*$', # Year of Birth abbreviation
    r'^\s*year\s+of\s+birth\s*$', # Full phrase Year of Birth
    r'^\s*जन्म\s+का\s+वर्ष\s*$', # Hindi for Year of Birth
    r'^\s*photo\s+identity\s+card\s*$',
    r'^\s*issued\s+on\s*$',
    r'^\s*issuing\s+authority\s*$',
    r'^\s*unique\s+identity\s*$',
    r'^\s*identification\s+number\s*$',
    r'^\s*national\s+population\s+register\s*$',
    r'^\s*ministry\s+of\s+home\s+affairs\s*$',
    r'^\s*government\s+of\s+india\s*$',
    r'^\s*uidai\s*$',
    r'^\s*indian\s+citizen\s*$',
    r'^\s*resident\s+of\s+india\s*$',
    r'^\s*date\s*$',
    r'^\s*place\s*$',
    r'^\s*district\s*$',
    r'^\s*state\s*$',
    r'^\s*country\s*$',
    r'^\s*asia\s*$',
    r'^\s*india\s*$',
    r'^\s*new\s+delhi\s*$',
    r'^\s*mumbai\s*$',
    r'^\s*kolkata\s*$',
    r'^\s*chennai\s*$',
    r'^\s*bengaluru\s*$',
    r'^\s*hyderabad\s*$',
    r'^\s*ahmedabad\s*$',
    r'^\s*pune\s*$',
    r'^\s*jaipur\s*$',
    r'^\s*lucknow\s*$',
    r'^\s*kanpur\s*$',
    r'^\s*nagpur\s*$',
    r'^\s*indore\s*$',
    r'^\s*bhopal\s*$',
    r'^\s*patna\s*$',
    r'^\s*ludhiana\s*$',
    r'^\s*kochi\s*$',
    r'^\s*kozhikode\s*$',
    r'^\s*thiruvananthapuram\s*$',
    r'^\s*visakhapatnam\s*$',
    r'^\s*vadodara\s*$',
    r'^\s*ghaziabad\s*$',
    r'^\s*agra\s*$',
    r'^\s*faridabad\s*$',
    r'^\s*meerut\s*$',
    r'^\s*rajkot\s*$',
    r'^\s*varanasi\s*$',
    r'^\s*srinagar\s*$',
    r'^\s*amritsar\s*$',
    r'^\s*jamshedpur\s*$',
    r'^\s*ranchi\s*$',
    r'^\s*guwahati\s*$',
    r'^\s*chandigarh\s*$',
    r'^\s*mysore\s*$',
    r'^\s*mangalore\s*$',
    r'^\s*bhubaneswar\s*$',
    r'^\s*cuttack\s*$',
    r'^\s*pondicherry\s*$',
    r'^\s*port\s+blair\s*$',
    r'^\s*shimla\s*$',
    r'^\s*dehradun\s*$',
    r'^\s*nainital\s*$',
    r'^\s*haridwar\s*$',
    r'^\s*rishikesh\s*$',
    r'^\s*gangtok\s*$',
    r'^\s*shillong\s*$',
    r'^\s*kohima\s*$',
    r'^\s*aizawl\s*$',
    r'^\s*imphal\s*$',
    r'^\s*agartala\s*$',
    r'^\s*itanagar\s*$',
    r'^\s*gangtok\s*$',
    r'^\s*diu\s*$',
    r'^\s*daman\s*$',
    r'^\s*silvassa\s*$',
    r'^\s*kavaratti\s*$',
    r'^\s*andaman\s+and\s+nicobar\s+islands\s*$',
    r'^\s*dadra\s+and\s+nagar\s+haveli\s*$',
    r'^\s*daman\s+and\s+diu\s*$',
    r'^\s*lakshadweep\s*$',
    r'^\s*puducherry\s*$',
    r'^\s*ladakh\s*$',
    r'^\s*jammu\s+and\s+kashmir\s*$',
    r'^\s*andhra\s+pradesh\s*$',
    r'^\s*arunachal\s+pradesh\s*$',
    r'^\s*assam\s*$',
    r'^\s*bihar\s*$',
    r'^\s*chhattisgarh\s*$',
    r'^\s*goa\s*$',
    r'^\s*gujarat\s*$',
    r'^\s*haryana\s*$',
    r'^\s*himachal\s+pradesh\s*$',
    r'^\s*jharkhand\s*$',
    r'^\s*karnataka\s*$',
    r'^\s*kerala\s*$',
    r'^\s*madhya\s+pradesh\s*$',
    r'^\s*maharashtra\s*$',
    r'^\s*manipur\s*$',
    r'^\s*meghalaya\s*$',
    r'^\s*mizoram\s*$',
    r'^\s*nagaland\s*$',
    r'^\s*odisha\s*$',
    r'^\s*punjab\s*$',
    r'^\s*rajasthan\s*$',
    r'^\s*sikkim\s*$',
    r'^\s*tamil\s+nadu\s*$',
    r'^\s*telangana\s*$',
    r'^\s*tripura\s*$',
    r'^\s*uttar\s+pradesh\s*$',
    r'^\s*uttarakhand\s*$',
    r'^\s*west\s+bengal\s*$',
    r'^\s*national\s+capital\s+territory\s+of\s+delhi\s*$',
    r'^\s*andaman\s+&\s+nicobar\s+islands\s*$',
    r'^\s*dadra\s+&\s+nagar\s+haveli\s+and\s+daman\s+&\s+diu\s*$',
    r'^\s*lakshadweep\s*$',
    r'^\s*puducherry\s*$',
    r'^\s*chhattisgarh\s*$',
    r'^\s*jammu\s+&\s+kashmir\s*$',
    r'^\s*ladakh\s*$',
    r'^\s*the\s+item\s*$',
    r'^\s*\d{4}\s+\d{5}\s*$', # Added to ignore "3742 91034"
    r'^\s*:\s*$', # Added to ignore standalone ":"
    r'^\s*\d{5}\s*$', # Added to ignore standalone 5-digit numbers like "91034" (if not part of 6-digit PIN) or "48310"
    r'^\s*the\s*$', # Added to ignore lone "THE"
    r'^\s*item\s*$', # Added to ignore lone "ITEM"
    r'^\s*pareft[,.]?\s*$', # Added to ignore "Pareft,"
    r'^\s*address\s*$', # Added to ignore standalone "Address" label itself, as it's processed separately
    r'^\s*\d+cm\s+s/o:\s*.*(?:herr|celle|ade|areit).*$', # Specifically for the problematic line "4cm S/O: HERR celle ade 3. AREIT"
    r'^\s*fagr,\s*\d+\s+jr\s*\d+\s*$', # Specifically for "fagr, 4154171 JR 9RT."
    r'^\s*a\s+hight\s+print\s+start,\s*\d+\s*$', # Specifically for "a HIGHT PRINT START, 173212"
    r'^\s*help@uidai\.gov\.in\s*$', # Email address
    r'^\s*www\.uidai\.gov\.in,\s*$', # Website address
    r'^\s*unique\s+identification\s+authority\s+of\s+india\s*$', # Aadhar header
    r'^\s*aadhaar\s*$', # Aadhar header
    r'^\s*\d{4}\s*$', # For standalone year like "1947" or short numbers like "48310"
    r'^\s*[\w\.-]+@[\w\.-]+\.\w+(?:-\w+)*\s*$', # General Email pattern
    r'^\s*(?:https?:\/\/)?(?:www\.)?[\w\.-]+\.[\w\.-]+(?:\/[\w\.-]*)*\s*$', # General URL pattern
    r'^\s*STREET\s+HEAR\s*$', # Added for specific noise
    r'^\s*OFFSITE\s*$',    # Added for specific noise
    r'^\s*STRET\s*$',      # Added for specific noise
    r'^\s*Floof\s+Par\s*$', # Added for specific noise
    r'^\s*ART\s*,\s*\d+\s*$', # Added for specific noise like "ART , 46210"
]

# --- Helper Functions (Shared by extraction logic) ---
# Ensure these functions are defined before they are called by other parts of the script.

def sanitize_alpha_text(text):
    if not isinstance(text, str): return ''
    cleaned_text = re.sub(r'[^A-Za-z\s.,\'-]', '', text).strip()
    cleaned_text = re.sub(r'\s*--\s*$', '', cleaned_text)
    return cleaned_text

def clean_address_line(line_text):
    """
    Normalizes spaces. Preserves all characters in the line.
    """
    if not isinstance(line_text, str):
        return ''
    cleaned_line = re.sub(r'\s+', ' ', line_text).strip()
    return cleaned_line

def is_english_line(line_text, threshold=0.7):
    """
    Checks if a line predominantly contains English (Latin alphabet) characters.
    Args:
        line_text (str): The text line to check.
        threshold (float): The minimum ratio of English-like characters to total characters.
    Returns:
        bool: True if the line is likely English, False otherwise.
    """
    if not isinstance(line_text, str) or not line_text.strip():
        return False
    
    # Count characters that are English alphabet, numbers, or common punctuation
    latin_chars = re.findall(r'[A-Za-z0-9\s.,\'-]', line_text) 
    total_chars = len(line_text)
    
    if total_chars == 0:
        return False # Avoid division by zero
        
    ratio = len(latin_chars) / total_chars
    return ratio >= threshold


def is_plausible_name_line(line_text, allow_all_caps_as_name=False):
    """
    Checks if a given line is a plausible candidate for a name.
    
    Args:
        line_text (str): The text line to check.
        allow_all_caps_as_name (bool): If True, allows names that are entirely in uppercase.
                                       Useful for documents like PAN cards.
    Returns:
        bool: True if the line is a plausible name, False otherwise.
    """
    original_line_text = line_text 
    line_text = line_text.strip()
    
    if not line_text or len(line_text) < 3: 
        return False
    
    # Check if line contains digits
    if re.search(r'\d', line_text):
        return False

    # Check for keywords that are never names (case-insensitive, partial match within the line)
    if any(keyword in line_text.lower() for keyword in keywords_that_are_never_names):
        return False

    # Check for patterns of other known fields (full line match)
    if any(re.fullmatch(pattern, line_text, re.IGNORECASE) for pattern in full_line_patterns_to_avoid):
        return False
        
    # Check for proper noun-like structure or just general capitalized words
    words = line_text.split()
    if not words: return False 
    
    is_capitalized_properly = False
    # If allowing all-caps (for PAN), check if all words are uppercase and at least one word
    if allow_all_caps_as_name and all(word.isupper() for word in words) and len(words) >=1:
        is_capitalized_properly = True
    # Otherwise, check if at least one word starts with a capital letter
    elif sum(1 for word in words if word and word[0].isupper()) >= 1:
        is_capitalized_properly = True
    
    if not is_capitalized_properly:
        return False

    # Ensure it primarily consists of name-allowed characters
    allowed_chars_pattern = r'[A-Za-z\s.,\'-]'
    num_allowed_chars = len(re.findall(allowed_chars_pattern, line_text))
    
    # At least 70% of characters must be alphanumeric or standard name punctuation.
    if len(line_text) > 0 and (num_allowed_chars / len(line_text)) < 0.70:
         return False

    return True


def detect_document_type(text):
    text_lower = text.lower()
    # Removed Hindi keywords for Aadhar and PAN detection to only detect English
    aadhar_keywords = ["unique identification authority", "aadhaar", "government of india"] 
    pan_keywords = ["income tax department", "permanent account number", "pan card"] 
    incorporation_keywords = ["date of incorporation", "date of formation"]

    has_aadhar_keywords = any(keyword in text_lower for keyword in aadhar_keywords)
    has_pan_keywords = any(keyword in text_lower for keyword in pan_keywords)
    has_incorporation_keywords = any(keyword in text_lower for keyword in incorporation_keywords)

    if has_incorporation_keywords and has_pan_keywords:
        return "Business PAN"
    elif has_aadhar_keywords and not has_pan_keywords:
        return "Aadhar"
    elif has_pan_keywords and not has_aadhar_keywords:
        return "PAN"
    elif has_aadhar_keywords and has_pan_keywords:
        # Ambiguous case, try to be more specific or default
        if "aadhaar" in text_lower: # Removed "आधार"
            return "Aadhar"
        elif "permanent account number" in text_lower or "pan" in text_lower:
            return "PAN"
        else:
            return "Unknown"
    else:
        return "Unknown"

async def call_gemini_for_name_extraction(extracted_text: str):
    """
    Calls the Gemini API to extract the person's name from the extracted text.
    This function now expects a pre-filtered text containing primarily plausible name candidates.
    """
    prompt_text = f"""
You are an expert document parser. Your task is to identify the full name of a person or entity from the provided text.
This text has already been pre-filtered to remove most non-name lines and noise.

Based on the content below, identify the single most prominent and plausible human or entity name.
Prioritize capitalized words or phrases that clearly represent a name.

If you find multiple plausible names, choose the one that appears most like a primary personal or business name.

Return only the identified name as a JSON object with a single key "name". If no name is found that strictly adheres to these rules, return "Not Found".

Pre-filtered Text:
---
{extracted_text}
---

Expected JSON Output Format:
```json
{{
  "name": "Extracted Name"
}}
```
Or, if not found:
```json
{{
  "name": "Not Found"
}}
```
"""
    headers = {
        'Content-Type': 'application/json',
    }
    params = {'key': GEMINI_API_KEY}
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "name": { "type": "STRING" }
                },
                "required": ["name"]
            }
        }
    }

    print(f"\n--- Sending Request to Gemini API ---")
    print(f"  Prompt being sent:\n{prompt_text}\n")
    print(f"  Payload being sent:\n{json.dumps(payload, indent=2)}\n")


    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_API_URL, headers=headers, params=params, json=payload, timeout=30.0)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            result = response.json()
            
            print(f"--- Received Response from Gemini API ---")
            print(f"  Raw response text:\n{response.text}\n")
            print(f"  Parsed JSON result:\n{json.dumps(result, indent=2)}\n")

            if result and result.get('candidates') and len(result['candidates']) > 0 and \
               result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
               len(result['candidates'][0]['content']['parts']) > 0:
                
                gemini_response_text = result['candidates'][0]['content']['parts'][0]['text']
                
                # Gemini returns a string that represents a JSON object, so we need to parse it
                try:
                    parsed_gemini_json = json.loads(gemini_response_text)
                    extracted_name = parsed_gemini_json.get("name", "Not Found")
                    print(f"  [Gemini API] Extracted Name: '{extracted_name}'")
                    return extracted_name
                except json.JSONDecodeError:
                    print(f"  [Gemini API Error] Failed to parse JSON from Gemini response: {gemini_response_text}")
                    return "Not Found"
            else:
                print(f"  [Gemini API] Unexpected response structure from Gemini: {result}")
                return "Not Found"
    except httpx.HTTPStatusError as e:
        print(f"  [Gemini API Error] HTTP error calling Gemini API: {e.response.status_code} - {e.response.text}")
        return "Not Found"
    except httpx.RequestError as e:
        print(f"  [Gemini API Error] Request error calling Gemini API: {e}")
        return "Not Found"
    except Exception as e:
        print(f"  [Gemini API Error] An unexpected error occurred during Gemini API call: {e}")
        return "Not Found"

# --- Core Extraction Logic Function ---
async def _process_document_for_extraction(extracted_text: str, forms: list):
    extracted_fields = {
        'Name': '',
        'Father\'s Name': '', # Keeping this key internally for now, will map to 'Fathers Name / Relation Name' for CSV
        'Date of Birth': '', # This will store either DOB or Date of Incorporation/Formation
        'PAN': '',
        'Aadhar Number': '',
        'Gender': '',
        'Address': '',
        'Document type': '' # New field for document type
    }
    
    lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
    doc_category = detect_document_type(extracted_text) 
    extracted_fields['Document type'] = doc_category
    print(f"  [Core Extraction] Detected Document Type: '{extracted_fields['Document type']}'")

    # --- Pre-filter lines to send to Gemini for name extraction ---
    plausible_name_lines = []
    # Limit the number of lines to scan for name for efficiency and relevance
    # Name is typically found in the first few lines, up to ~15.
    lines_to_scan_for_name = lines[:15] 

    print("\n--- Pre-filtering lines for Name Extraction ---")
    for line in lines_to_scan_for_name:
        # Use allow_all_caps_as_name=True here for a broader initial filter.
        # Gemini will then use its specific rules for Aadhar/PAN capitalization.
        if is_plausible_name_line(line, allow_all_caps_as_name=True):
            plausible_name_lines.append(line)
            print(f"    [Pre-filter] Kept line for name consideration: '{line}'")
        else:
            print(f"    [Pre-filter] Discarded line for name consideration: '{line}'")

    # Join the plausible lines. If no lines remain, send the original text as a fallback
    # to give Gemini a chance to extract something, even if it's noisy.
    cleaned_text_for_gemini = "\n".join(plausible_name_lines)
    if not cleaned_text_for_gemini:
        print("    [Pre-filter] No plausible name lines found after filtering. Sending original text to Gemini as fallback.")
        cleaned_text_for_gemini = extracted_text 

    # Use Gemini for name extraction with the pre-filtered text
    extracted_name_from_gemini = await call_gemini_for_name_extraction(cleaned_text_for_gemini)
    extracted_fields['Name'] = extracted_name_from_gemini if extracted_name_from_gemini != "Not Found" else ""
    print(f"  [Core Extraction] Final Name (from Gemini after pre-filter): '{extracted_fields['Name']}'")

    print(f"\n--- Starting Other Entity Extraction ---")

    # --- Address Extraction (MOVED UP FOR PRIORITY) ---
    print("\n--- Starting Address Extraction ---")
    raw_address_lines = []
    final_address = '' # Initialize to empty string

    # Condition: Only proceed with address extraction if the document is NOT PAN
    if doc_category != "PAN":
        # Find the starting index of the "Address:" label
        start_address_index = -1
        for i, line in enumerate(lines):
            if re.search(r'^Address\s*:', line, re.IGNORECASE): # Match "Address:" only at the start of the line
                start_address_index = i
                print(f"  [Address] Detected 'Address:' label at index {i}: '{line}'")
                break
                
        # Only proceed with address extraction if "Address:" label is found
        if start_address_index != -1:
            # If the address starts on the "Address :" line, remove "Address :" and add it.
            address_start_line_content = lines[start_address_index].replace('Address :', '').strip()
            if address_start_line_content:
                raw_address_lines.append(clean_address_line(address_start_line_content))
                print(f"    Initial Address Part: '{address_start_line_content}'")
            
            # Start collecting from the line *after* the "Address :" line
            for i in range(start_address_index + 1, len(lines)):
                line_stripped = lines[i].strip()

                if not line_stripped:
                    print(f"    Skipping empty line at index {i} (empty).")
                    continue

                # Filter non-English lines for Aadhar documents
                if doc_category == "Aadhar" and not is_english_line(line_stripped):
                    print(f"    Skipping non-English line (Aadhar document): '{line_stripped}' at index {i}")
                    continue

                # Define patterns that should NOT be part of the address but should NOT stop collection
                non_address_noise_patterns = [
                    r'^\s*www\s*$', # Standalone "www"
                    r'^\s*\.\s*$', # Just a standalone dot
                    r'^\s*\d{4}\s*$', # Standalone 4-digit numbers like 1947
                    r'^\s*[\w\.-]+@[\w\.-]+\.\w+(?:-\w+)*\s*$', # General Email pattern
                    r'^\s*(?:https?:\/\/)?(?:www\.)?[\w\.-]+\.[\w\.-]+(?:\/[\w\.-]*)*\s*$', # General URL pattern
                    r'^\s*P\.O\.\s+Box\s+No\.\s+\d+\s*\.?$', # P.O. Box lines
                    r'^\s*\d{3,4}\s+\d{3}\s+\d{4}\s*$', # Phone numbers (e.g. 1800 300 1947)
                    r'^\s*Bengaluru-560\s+001\s*$', # Specific city+pin line example from noise
                    r'^\s*UNIQUE\s+IDENTIFICATION\s+AUTHORITY\s+OF\s+INDIA\s*$', # Aadhar header
                    r'^\s*311\s+are\s*$', # OCR noise example
                    r'^\s*was\s+WITH\s+THAT\s+\d+\s*$', # More OCR noise
                    r'^\s*ARd\s+also\s+\d+\s+alt\s*$', # More OCR noise
                ]
                
                # If the line matches a non-address noise pattern, skip it but CONTINUE collection
                if any(re.fullmatch(pattern, line_stripped, re.IGNORECASE) for pattern in non_address_noise_patterns):
                    print(f"    SKIPPING non-address noise line: '{line_stripped}' at index {i}")
                    continue

                # Stop if we hit a known field that is definitely not part of the address
                if re.search(r'\b(?:Date of Birth|DOB|Gender|PAN|Aadhar|Signature|Father\'s Name)\b', line_stripped, re.IGNORECASE):
                    print(f"    Stopping address collection at index {i} due to other field: '{line_stripped}'")
                    break
                
                # Check for a 6-digit PIN code. If found, add the entire line and stop.
                pin_match = re.search(r'\b(\d{6})\b', line_stripped)
                
                cleaned_address_part = clean_address_line(line_stripped)
                if cleaned_address_part:
                    raw_address_lines.append(cleaned_address_part)
                    print(f"    Adding to raw_address_lines: '{line_stripped}' (cleaned: '{cleaned_address_part}')")
                
                if pin_match:
                    print(f"    Found PIN code, stopping address collection at index {i} and including PIN: '{line_stripped}'")
                    break # Stop after including the line with the PIN

        # Join all collected raw address lines with a single space.
        final_address = " ".join(raw_address_lines).strip()
        
        # Remove any periods that are followed by a space (unless they are part of a 6-digit PIN).
        # This will remove periods used as line breaks but preserve periods in abbreviations like "P.O." or "S/O:".
        # It also handles the lone '.' line if it wasn't skipped earlier.
        final_address = re.sub(r'\s*\.\s*', ' ', final_address).strip()
        # Ensure only one space after commas
        final_address = re.sub(r'\s*,\s*', ', ', final_address).strip()

        # Append a trailing period if the address ends with a 6-digit number and is missing it.
        if final_address and re.search(r'\d{6}$', final_address) and not final_address.endswith('.'):
            final_address += '.'
            print(f"  [Address] Appended trailing period. New address: '{final_address}'")
            
    extracted_fields['Address'] = final_address # Assign the final address (will be empty if PAN or not found)
    print(f"  [Address Success] Final combined Address: '{extracted_fields['Address']}'")

    # --- Father's Name / Relation Name Extraction (Now depends on Address) ---
    print("\n--- Starting Father's Name / Relation Name Extraction ---")
    fathers_name_found = False
    relation_patterns = r'(S/O|C/O|D/O|W/O)\s*:\s*([^,]+)' # Capture relation label and name

    # Priority 1: Extract from the *already populated* Address string (if present and if it's an Aadhar document)
    if extracted_fields['Address'] and doc_category == "Aadhar":
        print("  [Relation Name] Attempt 1: Checking extracted Address for relation patterns.")
        relation_match_in_address = re.search(relation_patterns, extracted_fields['Address'], re.IGNORECASE)
        if relation_match_in_address:
            candidate_value = relation_match_in_address.group(2).strip() # Extract the name part
            
            # Apply plausibility check only to the *extracted name part*, not the whole address line.
            if is_plausible_name_line(candidate_value, allow_all_caps_as_name=True):
                extracted_fields["Father's Name"] = f"{relation_match_in_address.group(1)}: {sanitize_alpha_text(candidate_value)}"
                # If the extracted relation ends with a comma (due to non-greedy match), remove it for cleaner output
                if extracted_fields["Father's Name"].endswith(','):
                    extracted_fields["Father's Name"] = extracted_fields["Father's Name"][:-1].strip()
                print(f"""  [Relation Name Success - From Address] Found: '{extracted_fields["Father's Name"]}' from extracted address.""")
                fathers_name_found = True

    # Priority 2 (Fallback if not found in address): Search general text for explicit "Father's Name" label
    if not fathers_name_found:
        print("  [Relation Name Fallback] Not found in address. Searching full text for 'Father\'s Name' label.")
        for i, line in enumerate(lines):
            if re.search(r"Father'?s Name|पिता का नाम", line, re.IGNORECASE):
                if i + 1 < len(lines):
                    father_name_candidate_from_label = lines[i + 1].strip()
                    # Apply plausibility check only to the candidate name line
                    if is_plausible_name_line(father_name_candidate_from_label, allow_all_caps_as_name=True):
                        extracted_fields["Father's Name"] = sanitize_alpha_text(father_name_candidate_from_label)
                        print(f"""  [Relation Name Success - Fallback Label] Found: '{extracted_fields["Father's Name"]}' from line after label.""")
                        fathers_name_found = True
                        break

    # DOB / Date of Incorporation/Formation / Year of Birth Extraction
    print("\n--- Starting DOB / Date of Incorporation/Formation / Year of Birth Extraction ---")
    date_found = False

    # Priority 1: Exact date format with DOB/Date/जन्मतिथि/Date of Incorporation/Formation label
    dob_incorporation_keywords = r'(?:Date of Birth|DOB|जन्मतिथि|Date of Incorporation/Formation|Date of Incorporation|Date of Formation)'
    for i, line in enumerate(lines):
        date_line_match = re.search(fr'{dob_incorporation_keywords}\s*[:\s]*(\d{{2}}[/\\.\\-\\]\d{{2}}[/\\.\\-\\]\d{{4}})\b', line, re.IGNORECASE)
        if date_line_match:
            extracted_fields['Date of Birth'] = date_line_match.group(1)
            print(f"  [Date Success] Found on same line with label: '{extracted_fields['Date of Birth']}' from line: '{line}'")
            date_found = True
            break
        if re.search(dob_incorporation_keywords, line, re.IGNORECASE):
            for j in range(1, 3): 
                if i + j < len(lines):
                    next_line = lines[i + j]
                    date_match = re.search(r'\b(\d{2}[/\\.\\-\\]\d{2}[/\\.\\-\\]\d{4})\b', next_line)
                    if date_match:
                        extracted_fields['Date of Birth'] = date_match.group(1)
                        print(f"  [Date Success] Found on subsequent line with label: '{extracted_fields['Date of Birth']}' from line: '{next_line}'")
                        date_found = True
                        break
            if date_found: break

    # Priority 2 (if no full date found by label): "Year of Birth"
    if not date_found and doc_category == "Aadhar":
        print(f"  [Year of Birth Fallback - Aadhar] No full date found. Checking for 'Year of Birth'.")
        for i, line in enumerate(lines):
            year_of_birth_match = re.search(r'(?:Year of Birth|जन्म का वर्ष)\s*[:\s]*(\d{4})\b', line, re.IGNORECASE)
            if year_of_birth_match:
                extracted_fields['Date of Birth'] = year_of_birth_match.group(1)
                print(f"  [Year of Birth Success] Found: '{extracted_fields['Date of Birth']}' from line: '{line}'")
                date_found = True
                break
            if re.search(r'(?:Year of Birth|जन्म का वर्ष)', line, re.IGNORECASE):
                for j in range(1, 2): # Check next 1 line
                    if i + j < len(lines):
                        next_line = lines[i + j]
                        year_match = re.search(r'\b(\d{4})\b', next_line)
                        if year_match:
                            extracted_fields['Date of Birth'] = year_match.group(1)
                            print(f"  [Year of Birth Success] Found on subsequent line: '{extracted_fields['Date of Birth']}' from line: '{next_line}'")
                            date_found = True
                            break
                if date_found: break

    # Priority 3 (if no date or year found by label): General date pattern for PAN documents
    if not date_found and doc_category == "PAN":
        print(f"  [Date Fallback - PAN] No explicit DOB/Date/Year label found for PAN. Searching for any date pattern.")
        for line in lines:
            pan_date_match = re.search(r'\b(\d{2}[/\\.\\-\\]\d{2}[/\\.\\-\\]\d{4})\b', line)
            if pan_date_match:
                extracted_fields['Date of Birth'] = pan_date_match.group(1)
                print(f"  [Date Success - PAN Fallback] Found general date: '{extracted_fields['Date of Birth']}' from line: '{line}'")
                date_found = True
                break

    # PAN
    print("\n--- Starting PAN Extraction ---")
    for line in lines:
        pan_match = re.search(r'\b([A-Z]{5}\d{4}[A-Z]{1})\b', line)
        if pan_match:
            extracted_fields['PAN'] = pan_match.group(1)
            print(f"  [PAN Success] Found: '{extracted_fields['PAN']}'")
            break

    # Aadhar Number
    print("\n--- Starting Aadhar Number Extraction ---")
    for line in lines:
        aadhar_match = re.search(r'\b(\d{4}\s\d{4}\s\d{4})\b', line)
        if aadhar_match:
            extracted_fields['Aadhar Number'] = aadhar_match.group(1)
            print(f"  [Aadhar Success] Found: '{extracted_fields['Aadhar Number']}'")
            break

    # Gender
    print("\n--- Starting Gender Extraction ---")
    for line in lines:
        # The Hindi terms "पुरुष|महिला" are kept here for filtering *avoid* patterns and keywords_that_are_never_names,
        # but the actual *extraction* of the gender for this field relies on the English regex pattern.
        gender_match = re.search(r'\b(MALE|FEMALE|TRANSGENDER|पुरुष|महिला)\b', line, re.IGNORECASE)
        if gender_match:
            extracted_fields['Gender'] = gender_match.group(1).capitalize()
            print(f"  [Gender Success] Found: '{gender_match.group(1)}'") # Log the actual extracted string
            break
            
    return extracted_fields # Return the dictionary

# --- FastAPI Endpoints ---

# Serve index.html at the root URL
@app.get("/")
async def read_root():
    """
    Serves the index.html file from the static directory.
    """
    return FileResponse("static/index.html")

@app.post("/parse-document-data/")
async def parse_document_data(request_data: DocumentParseRequest):
    extracted_fields = await _process_document_for_extraction(request_data.extracted_text, request_data.forms)
    print(f"Backend Final Extracted Fields for JSON response: {extracted_fields}")
    return JSONResponse(content=extracted_fields)

@app.post("/download-csv/")
async def download_csv(request_data: DocumentParseRequest):
    print("\n--- CSV Download Request Received ---")
    extracted_fields = await _process_document_for_extraction(request_data.extracted_text, request_data.forms)

    # Renamed 'Father\'s Name' to 'Fathers Name / Relation Name' in the columns list
    columns = ['Document type', 'Name', 'Fathers Name / Relation Name', 'DOB/ Date of Incorporation', 'PAN', 'Aadhar Number', 'Gender', 'Address']
    
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)

    writer.writeheader()
    
    # Map extracted_fields keys to CSV columns
    row_to_write = {}
    for col in columns:
        if col == 'DOB/ Date of Incorporation':
            row_to_write[col] = extracted_fields.get('Date of Birth', '') # Date of Birth field holds DOB/Incorporation date
        elif col == 'Document type':
            row_to_write[col] = extracted_fields.get('Document type', '')
        elif col == 'Fathers Name / Relation Name': # Mapping 'Father\'s Name' to 'Fathers Name / Relation Name'
            row_to_write[col] = extracted_fields.get('Father\'s Name', '')
        else:
            row_to_write[col] = extracted_fields.get(col, '')

    writer.writerow(row_to_write)
    
    output.seek(0)

    headers = {
        "Content-Disposition": "attachment; filename=extracted_document_data.csv",
        "Content-Type": "text/csv"
    }
    print("--- CSV Generation Complete. Returning StreamingResponse ---")
    return StreamingResponse(output, headers=headers)

# --- FastAPI Endpoint for Document Upload (Initial Textract Call) ---

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    """
    Receives an uploaded document, processes it with AWS Textract,
    and returns the raw extracted text, tables, and key-value pairs.
    """
    if S3_BUCKET_NAME == 'your-textract-input-bucket':
        raise HTTPException(
            status_code=500,
            detail="S3_BUCKET_NAME is not configured. Please update the backend script with your actual S3 bucket name."
        )

    if not file.content_type.startswith(('image/', 'application/pdf')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload an image (PNG, JPEG) or PDF."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"Uploaded file saved temporarily to: {temp_file_path}")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

        s3_object_location = upload_document_to_s3(temp_file_path, S3_BUCKET_NAME, f"uploads/{file.filename}")

        if not s3_object_location:
            raise HTTPException(status_code=500, detail="Failed to upload document to S3.")

        try:
            job_id = start_textract_job({'S3Object': s3_object_location}, job_type='ANALYZE_DOCUMENT')
            textract_results = get_textract_job_results(job_id)

            if not textract_results or not textract_results.get('Blocks'):
                raise HTTPException(status_code=500, detail="Textract processing failed or returned no blocks.")

            extracted_text = extract_text_from_blocks(textract_results['Blocks'])
            parsed_tables = parse_textract_tables(textract_results['Blocks'])
            parsed_forms = parse_textract_forms(textract_results['Blocks'])

            return JSONResponse(content={
                "status": "success",
                "message": "Document processed successfully",
                "filename": file.filename,
                "extracted_text": extracted_text,
                "tables": parsed_tables,
                "forms": parsed_forms
            })
        except HTTPException as e:
            raise e
        except Exception as e:
            traceback.print_exc()
            print(f"An unexpected error occurred during Textract processing: {e}")
            raise HTTPException(status_code=500, detail=f"An internal server error occurred. See server logs for details.")
