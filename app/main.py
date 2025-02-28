import os
import json
import re
import gradio as gr
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

# Hardcoded directories
KNOWLEDGE_DIR = "data/knowledge"
PROCESSED_DIR = "data/processed"

# Ensure directories exist
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Existing functions ---

def extract_pages_from_pdf(file_path):
    """Extract text from each page of the PDF file and return as a list."""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return pages

def is_valid_json(data):
    """Check if the given string is valid JSON."""
    try:
        json.loads(data)
        return True
    except Exception:
        return False

def generate_qa_pair(chunk_text, llm, prompt_template):
    """
    Uses the given LLM to generate a JSON-formatted question-answer pair
    from the provided text chunk.
    Returns a tuple (qa_pair, raw_response).
    """
    prompt = prompt_template.format(chunk=chunk_text.strip())
    raw_response = llm.invoke(prompt)
    
    # 1. Try if the raw response is directly valid JSON.
    if is_valid_json(raw_response):
        return json.loads(raw_response), raw_response

    # 2. Try to extract JSON from a triple-backtick code block.
    match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.S)
    if match and is_valid_json(match.group(1)):
        return json.loads(match.group(1)), raw_response

    # 3. As a fallback, try to extract JSON enclosed in single backticks.
    match = re.search(r'`(.*?)`', raw_response, re.S)
    if match and is_valid_json(match.group(1)):
        return json.loads(match.group(1)), raw_response

    print("Warning: Received invalid JSON for a chunk, skipping.")
    return None, raw_response

# --- PDF Processing Function ---

def process_pdf(pdf_path, output_dir, llm, prompt_template):
    """
    Process a single PDF file:
      - Extract pages, group them with a sliding window, split into sub-chunks,
        and use the LLM to generate Q/A pairs.
      - Append each valid response directly to a JSONL file named after the PDF.
    """
    pages = extract_pages_from_pdf(pdf_path)
    num_pages = len(pages)
    
    # Group pages using a sliding window centered on odd-numbered pages.
    window_chunks = []
    for i in range(num_pages):
        if (i + 1) % 2 == 1:
            start = max(0, i - 2)
            end = min(num_pages, i + 3)
            window_text = "\n\n".join(pages[start:end])
            window_chunks.append(window_text)
    
    sub_chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    
    # Prepare output file path.
    pdf_filename = os.path.basename(pdf_path)
    output_filename = os.path.splitext(pdf_filename)[0] + ".jsonl"
    output_filepath = os.path.join(output_dir, output_filename)
    
    valid_responses_count = 0
    invalid_responses_count = 0
    
    # Open the output file once in append mode.
    with open(output_filepath, "a") as f:
        for window_index, window in enumerate(window_chunks, start=1):
            sub_chunks = sub_chunk_splitter.split_text(window)
            print(f"Window {window_index} split into {len(sub_chunks)} sub-chunks.")
            for sub_index, sub_chunk in enumerate(sub_chunks, start=1):
                print(f"Processing sub-chunk {sub_index}/{len(sub_chunks)} from window {window_index} of {pdf_path}...")
                qa_pair, raw_response = generate_qa_pair(sub_chunk, llm, prompt_template)
                if qa_pair is not None:
                    qa_pair["source_file"] = pdf_path
                    qa_pair["window_index"] = window_index
                    qa_pair["subchunk_index"] = sub_index
                    f.write(json.dumps(qa_pair) + "\n")
                    f.flush()  
                    valid_responses_count += 1
                else:
                    invalid_responses_count += 1
    
    return (f"Processed '{pdf_filename}'. Valid responses: {valid_responses_count}, "
            f"Invalid responses: {invalid_responses_count}. Appended to '{output_filepath}'.")

def process_all_pdfs():

    llm = OllamaLLM(
        model="deepseek-r1:7b",
        temperature=0.3,
        num_ctx=4096,
        base_url="http://ollama:11434"
    )
    
    prompt_template = (
        "You are a science researcher. "
        "When given a body of text, focus on extracting and discussing the technical details and coding concepts. "
        "Your response should be precise, fact-based, and written in a scholarly tone. "
        "Generate a question and answer pair in JSON format that reflects this analytical approach. "
        "Each JSON must have exactly two keys: 'question' and 'answer'.\n\n"
        "Generate a question and answer pair based solely on the following text:\n\n"
        "{chunk}\n\n"
        "Only respond with the JSON and no additional text."
    )
    
    pdf_files = [os.path.join(KNOWLEDGE_DIR, f) for f in os.listdir(KNOWLEDGE_DIR) if f.lower().endswith(".pdf")]
    messages = []
    for pdf_path in pdf_files:
        result = process_pdf(pdf_path, PROCESSED_DIR, llm, prompt_template)
        messages.append(result)
    
    return "\n".join(messages)

# --- Gradio Interface Functions ---

def list_pdfs_fixed():
    if not os.path.exists(KNOWLEDGE_DIR):
        return []
    return [f for f in os.listdir(KNOWLEDGE_DIR) if f.lower().endswith(".pdf")]

def upload_pdf_fixed(file):
    if file is None:
        return "No file uploaded.", list_pdfs_fixed()
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    
    if hasattr(file, "read"):
        data = file.read()
        filename = os.path.basename(file.name)
    else:
        with open(file, "rb") as f:
            data = f.read()
        filename = os.path.basename(file)
    
    destination = os.path.join(KNOWLEDGE_DIR, filename)
    with open(destination, "wb") as f:
        f.write(data)
    
    return f"Uploaded '{filename}' to '{KNOWLEDGE_DIR}'.", list_pdfs_fixed()

def delete_pdf_fixed(pdf_filename):
    if isinstance(pdf_filename, list):
        pdf_filename = pdf_filename[0] if pdf_filename else ""
    file_path = os.path.join(KNOWLEDGE_DIR, pdf_filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        status = f"Deleted '{pdf_filename}'."
    else:
        status = f"'{pdf_filename}' does not exist."
    return status, list_pdfs_fixed()

# Gradio Interface

def main_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# PDF Management and Processing Interface")
        
        with gr.Tab("Upload PDF"):
            with gr.Row():
                upload_input = gr.File(label="Upload PDF", file_types=[".pdf"], file_count="single")
            upload_btn = gr.Button("Upload PDF")
            upload_status = gr.Textbox(label="Upload Status")
            upload_btn.click(fn=upload_pdf_fixed, inputs=upload_input, outputs=[upload_status])
        
        with gr.Tab("Delete PDF"):
            with gr.Row():
                refresh_btn_del = gr.Button("Refresh PDF List")
            current_files = list_pdfs_fixed()
            default_value = current_files[0] if current_files else None
            pdf_list_del = gr.Dropdown(choices=current_files, value=default_value, label="Select PDF to Delete")
            delete_status = gr.Textbox(label="Deletion Status")
            delete_btn = gr.Button("Delete Selected PDF")
            refresh_btn_del.click(fn=list_pdfs_fixed, inputs=[], outputs=pdf_list_del)
            delete_btn.click(fn=delete_pdf_fixed, inputs=pdf_list_del, outputs=[delete_status, pdf_list_del])
        
        with gr.Tab("Process PDFs"):
            gr.Markdown("Review the PDFs in your knowledge directory before processing. Click Refresh to see the updated list.")
            with gr.Row():
                refresh_btn_proc = gr.Button("Refresh PDF List")
            pdf_list_proc = gr.Textbox(label="PDFs Ready for Processing", interactive=False, lines=5)
            refresh_btn_proc.click(fn=lambda: "\n".join(list_pdfs_fixed()),
                                     inputs=[], outputs=pdf_list_proc)
            process_btn = gr.Button("Process All PDFs")
            process_output = gr.Textbox(label="Processing Output", lines=10)
            process_btn.click(fn=process_all_pdfs, inputs=[], outputs=process_output)
    
    return demo

if __name__ == "__main__":
    demo = main_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
