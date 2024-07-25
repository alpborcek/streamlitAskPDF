import pickle

# Paths to your pkl files
embeddings_file_path = "data/embeddings.pkl"
chunks_file_path = "data/chunks.pkl"

# Load and print the embeddings
with open(embeddings_file_path, "rb") as file:
    embeddings = pickle.load(file)
print("Embeddings:", embeddings)

# Load and print the text chunks
with open(chunks_file_path, "rb") as file:
    chunks = pickle.load(file)
print("Text Chunks:", chunks)
