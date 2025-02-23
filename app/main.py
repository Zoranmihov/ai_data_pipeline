import os
import json
import re
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # kept for compatibility
from langchain_ollama import OllamaLLM

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
    from the provided text chunk. Returns a tuple (qa_pair, raw_response)
    where qa_pair is the parsed JSON (or None if invalid) and raw_response
    is the full text response from the model.
    """
    prompt = prompt_template.format(chunk=chunk_text.strip())
    raw_response = llm.invoke(prompt)  # Use invoke() to avoid deprecation warning.
    
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

def main():
    # --- Step 1: Find all PDF files in the data directory ---
    data_dir = "data"
    pdf_files = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.lower().endswith(".pdf")
    ]
    
    if not pdf_files:
        print("No PDF files found in the data directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) in '{data_dir}'.")

    # --- Step 2: Initialize the LLM ---
    print("Initializing LLM...")
    llm = OllamaLLM(
        model="deepseek-r1:7b",
        temperature=0.3,
        num_ctx=4096,
        base_url="http://ollama:11434"
    )
    
    # Define a prompt template for generating a JSON Q/A pair in a scholarly tone.
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

    
    # --- Step 3: Prepare output files ---
    valid_output_filename = "responses.jsonl"
    invalid_output_filename = "invalid_responses.jsonl"
    sample_invalid_filename = "sample_invalid_response.json"
    sample_valid_saved = False
    sample_invalid_saved = False

    print("Processing PDF files and appending responses to files...")

    # We'll also instantiate a text splitter for the sub-chunking.
    # Adjust chunk_size and overlap as needed.
    sub_chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )

    with open(valid_output_filename, "a") as valid_outfile, open(invalid_output_filename, "a") as invalid_outfile:
        for pdf_file in pdf_files:
            print(f"\nProcessing {pdf_file}...")
            pages = extract_pages_from_pdf(pdf_file)
            num_pages = len(pages)
            print(f"Extracted {num_pages} pages from {pdf_file}.")

            # --- Step 4: Group pages using a sliding window of 5 pages ---
            # We use odd-numbered pages as the window center.
            window_chunks = []
            for i in range(num_pages):
                if (i + 1) % 2 == 1:  # center on odd-numbered pages
                    start = max(0, i - 2)
                    end = min(num_pages, i + 3)  # covers pages i-2 to i+2
                    window_text = "\n\n".join(pages[start:end])
                    window_chunks.append(window_text)
            print(f"Created {len(window_chunks)} page-window chunks from {pdf_file}.")

            # --- Step 5: For each window, further split into sub-chunks for QNA generation ---
            for window_index, window in enumerate(window_chunks, start=1):
                # Split the window text into smaller sub-chunks.
                sub_chunks = sub_chunk_splitter.split_text(window)
                print(f"Window {window_index} split into {len(sub_chunks)} sub-chunks.")
                for sub_index, sub_chunk in enumerate(sub_chunks, start=1):
                    print(f"Processing sub-chunk {sub_index}/{len(sub_chunks)} from window {window_index} of {pdf_file}...")
                    qa_pair, raw_response = generate_qa_pair(sub_chunk, llm, prompt_template)
                    if qa_pair is not None:
                        # Record source and indices for reference.
                        qa_pair["source_file"] = pdf_file
                        qa_pair["window_index"] = window_index
                        qa_pair["subchunk_index"] = sub_index
                        valid_outfile.write(json.dumps(qa_pair) + "\n")
                        valid_outfile.flush()
                    else:
                        # Log invalid response details for analysis.
                        log_entry = {
                            "source_file": pdf_file,
                            "window_index": window_index,
                            "subchunk_index": sub_index,
                            "raw_response": raw_response,
                            # Optionally include a snippet of the sub-chunk text (first 500 characters)
                            "subchunk_text_snippet": sub_chunk[:500]
                        }
                        invalid_outfile.write(json.dumps(log_entry) + "\n")
                        invalid_outfile.flush()

                        if not sample_invalid_saved:
                            with open(sample_invalid_filename, "w") as sample_invalid_file:
                                sample_invalid_file.write(json.dumps(log_entry, indent=4))
                            sample_invalid_saved = True

    print(f"\nAll responses have been appended to {valid_output_filename}.")
    print(f"All invalid responses have been appended to {invalid_output_filename}.")
    if sample_invalid_saved:
        print(f"A sample invalid response has been saved to {sample_invalid_filename}.")

if __name__ == "__main__":
    main()
