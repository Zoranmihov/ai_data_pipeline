from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import gradio as gr

# Load and parse the JSONL data
file_path = "data/training.jsonl"

qa_data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        qa_data.append(json.loads(line))

# Extract only the questions and answers
questions = [item["question"] for item in qa_data]
answers = [item["answer"] for item in qa_data]

# Initialize an embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings for stored questions
question_embeddings = embedding_model.encode(questions)

# Index embeddings with FAISS
dimension = question_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(question_embeddings))

# Function to retrieve answer based on user input
def retrieve_answer(user_query):
    # Encode the query
    query_embedding = embedding_model.encode([user_query])
    
    # Perform the search, capturing both distances and indices
    distances, indices = faiss_index.search(np.array(query_embedding), 1)
    
    # Define a threshold for distance;
    threshold = 1.0  
    
    # Check if the best match is similar enough
    if distances[0][0] > threshold:
        return "I don't have information on that subject."
    
    # Otherwise, return the corresponding answer
    matched_index = indices[0][0]
    return answers[matched_index]


# Set up the Gradio interface
iface = gr.Interface(
    fn=retrieve_answer,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=gr.Textbox(label="Answer"),
    title="Product Q&A Chatbot",
    description="Enter a product-related question and get a precise answer from the dataset."
)

# Launch the Gradio app on localhost
iface.launch(server_name="0.0.0.0", server_port=7861)
