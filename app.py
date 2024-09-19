#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Upgrade pip to the latest version
get_ipython().system('pip install --upgrade pip')

# Install dependencies with specific versions to avoid conflicts
get_ipython().system('pip install torch==2.0.0 transformers==4.33.0 datasets==2.13.0 faiss-cpu==1.7.3 apache-tika==1.24 boto3==1.26.0 awscli==1.27.0 streamlit==1.18.0 pandas==2.0.0')


# In[4]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install torch transformers datasets faiss-cpu apache-tika boto3 awscli streamlit pandas')


# In[5]:


# Upgrade pip
get_ipython().system('pip install --upgrade pip')

# Install necessary packages with specific versions
get_ipython().system('pip install torch==2.0.0 transformers==4.33.0 datasets==2.13.0 faiss-cpu==1.7.3 tika boto3==1.26.0 awscli==1.27.0 streamlit==1.18.0 pandas==2.0.0')


# In[6]:


import pandas as pd
import tika
from tika import parser

# Initialize Tika JVM
tika.initVM()

# Function to load the dataset from JSON
def load_dataset(file_path):
    df = pd.read_json(file_path)
    return df

# Function to extract text from arXiv dataset
def extract_text_from_df(df):
    texts = []
    for index, row in df.iterrows():
        # Combine relevant fields into a single text
        text = f"Title: {row['title']}\nAbstract: {row['abstract']}\nContent: {row['content']}"
        texts.append(text)
    return texts


# In[7]:


from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

# Function to get embeddings from text
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.detach().numpy()


# In[8]:


import faiss

# Initialize FAISS index
dimension = 768  # BERT embedding size
index = faiss.IndexFlatL2(dimension)

# Function to add embeddings to the FAISS index
def add_to_index(embedding):
    index.add(np.array(embedding))

# Function to search for similar embeddings
def search_similar_embeddings(query_embedding, top_k=5):
    distances, indices = index.search(np.array(query_embedding), top_k)
    return indices, distances


# In[9]:


from transformers import RagTokenizer, RagTokenForGeneration

# Load RAG model and tokenizer
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
rag_model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

# Function to generate metadata using RAG
def generate_metadata(query, context):
    inputs = rag_tokenizer(query, context, return_tensors='pt')
    outputs = rag_model.generate(**inputs)
    generated_metadata = rag_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_metadata


# In[10]:


import streamlit as st

def run_streamlit_ui():
    st.title("Metadata Generator")

    uploaded_file = st.file_uploader("Upload arXiv JSON dataset", type='json')
    if uploaded_file is not None:
        # Load dataset from uploaded file
        df = pd.read_json(uploaded_file)
        texts = extract_text_from_df(df)

        # Generate embeddings and add to index
        for text in texts:
            embedding = get_embeddings(text)
            add_to_index(embedding)
        
        st.text("Dataset uploaded and processed successfully.")

        query = st.text_input("Enter your query")
        if query:
            # Generate embedding for the query
            query_embedding = get_embeddings(query)
            
            # Search for similar documents
            indices, distances = search_similar_embeddings(query_embedding)
            
            # Retrieve the top results
            top_results = [texts[i] for i in indices[0]]
            
            # Generate metadata
            context = " ".join(top_results)
            generated_metadata = generate_metadata(query, context)
            
            st.text_area("Generated Metadata", generated_metadata)

# Run the Streamlit UI
# Save this code to a file named `app.py` and run: streamlit run app.py


# In[12]:


# Install the necessary packages if not already installed
get_ipython().system('pip install pyngrok')

from pyngrok import ngrok
import subprocess
import time

# Define the path to your Streamlit app
streamlit_script = 'app.ipynb'  # Change to the path of your Streamlit script if needed

# Start Streamlit using subprocess
def start_streamlit():
    process = subprocess.Popen(['streamlit', 'run', streamlit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

# Set up ngrok to expose Streamlit
def setup_ngrok():
    # Open a new ngrok tunnel on port 8501 (default for Streamlit)
    public_url = ngrok.connect(port='8501')
    print(f"Streamlit app is live at {public_url}")

# Start Streamlit and ngrok
streamlit_process = start_streamlit()
time.sleep(5)  # Give Streamlit a few seconds to start
setup_ngrok()


# In[ ]:




