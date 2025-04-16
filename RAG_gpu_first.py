import tkinter as tk
from tkinter import scrolledtext
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import fitz  # PyMuPDF for PDF processing
import pandas as pd
import numpy as npA
from tqdm.auto import tqdm
import textwrap
import os

# Initialize GPU and models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)

# Load LLM model
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Function to process PDF and extract text
def process_pdf(pdf_path):
    """
    Extracts text from a PDF file and splits it into pages.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc), desc="Processing PDF"):
        text = page.get_text().replace("\n", " ").strip()
        pages_and_texts.append({"page_number": page_number, "text": text})
    return pages_and_texts

# Function to chunk text into smaller pieces
def chunk_text(text, max_tokens=128):
    """
    Splits a large text into smaller chunks of a specified maximum token length.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to embed text chunks
def embed_chunks(chunks):
    """
    Embeds text chunks using the embedding model.
    """
    return embedding_model.encode(chunks, batch_size=32, convert_to_tensor=True)

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, embeddings, chunks, top_k=5):
    """
    Retrieves the most relevant chunks for a query using cosine similarity.
    """
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(scores, k=top_k)
    return [(chunks[idx], scores[idx].item()) for idx in top_results.indices]

# Function to format the prompt
def format_prompt(query, context_items):
    """
    Formats the query and context into a specific and structured prompt for the LLM.
    """
    context = "\n".join([f"- {item}" for item in context_items])
    prompt = f"""
    You are an expert assistant. Use the provided context to answer the query accurately and concisely.
    Context:
    {context}
    Query: {query}
    Answer:
    """
    return prompt.strip()

# Function to verify facts
def verify_facts(answer, context_items):
    """
    Verifies if the generated answer aligns with the retrieved context.
    """
    combined_context = " ".join(context_items)
    answer_embedding = embedding_model.encode(answer, convert_to_tensor=True)
    context_embedding = embedding_model.encode(combined_context, convert_to_tensor=True)
    similarity_score = util.cos_sim(answer_embedding, context_embedding).item()
    threshold = 0.7
    if similarity_score >= threshold:
        return f"Fact Verified ✅ (Similarity: {similarity_score:.2f})"
    else:
        return f"Fact Not Verified ❌ (Similarity: {similarity_score:.2f})"

# Function to handle query and generate response
def generate_response():
    query = query_entry.get()
    if not query:
        output_text_widget.insert(tk.END, "Please enter a query.\n")
        return

    # Process PDF and chunk text (replace with your PDF path)
    pdf_path = "/home/arsenal/projects/ml/Local_RAG/human-nutrition-text.pdf"
    if not os.path.exists(pdf_path):
        output_text_widget.insert(tk.END, "PDF file not found. Please ensure the file exists.\n")
        return

    pages_and_texts = process_pdf(pdf_path)
    chunks = [chunk for page in pages_and_texts for chunk in chunk_text(page["text"], max_tokens=128)]

    # Embed chunks
    embeddings = embed_chunks(chunks)

    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, embeddings, chunks)
    context_items = [item[0] for item in relevant_chunks]

    # Format the prompt
    prompt = format_prompt(query, context_items)

    # Generate response
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(input_ids["input_ids"], max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Verify facts
    verification_result = verify_facts(answer, context_items)

    # Display the response
    output_text_widget.delete(1.0, tk.END)
    output_text_widget.insert(tk.END, f"Query: {query}\n\nGenerated Answer:\n{answer}\n\n{verification_result}")

# Create the GUI
root = tk.Tk()
root.title("Query and Answer Generator")

# Query input
query_label = tk.Label(root, text="Enter your query:")
query_label.pack(pady=5)
query_entry = tk.Entry(root, width=50)
query_entry.pack(pady=5)

# Generate button
generate_button = tk.Button(root, text="Generate Answer", command=generate_response)
generate_button.pack(pady=10)

# Output display
output_label = tk.Label(root, text="Generated Answer:")
output_label.pack(pady=5)
output_text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=15)
output_text_widget.pack(pady=5)

# Run the GUI
root.mainloop()