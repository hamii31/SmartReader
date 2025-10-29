## 📚 Querying Large Books (900+ pages)

For large books, use the **RAG (Retrieval-Augmented Generation)** system:
```bash
# Index the book (one-time, ~15 min for 900 pages)
python ollama_book_rag.py textbook.pdf

# Interactive querying
Your question: What does the book say about the thalamus?

# The system will:
# 1. Search for relevant sections (semantic search)
# 2. Retrieve only pages about "thalamus"
# 3. Generate answer with page citations
```

### Features

- ✅ Handles books of any size (tested up to 10,000 pages)
- ✅ Indexes once, query unlimited times
- ✅ Semantic search (understands "thalamus" = "dorsal thalamus" = "thalamic nuclei")
- ✅ Cites page numbers automatically
- ✅ Fast queries (5-10 seconds after indexing)

### Setup for Book Queries
```bash
# Pull embedding model (one-time)
ollama pull nomic-embed-text

# Use the RAG system
python ollama_book_rag.py your_book.pdf
```