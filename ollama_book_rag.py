"""
ollama_book_rag.py - RAG system for querying large PDF books
Features permanent cache in user AppData directory
"""

import sys
import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import pickle

try:
    import PyPDF2
    import requests
except ImportError:
    import subprocess
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2", "requests", "numpy"])
    import PyPDF2
    import requests


class TextChunk:
    """Represents a chunk of text from the book"""
    
    def __init__(self, text: str, page_number: int, chunk_id: int, embedding=None, metadata=None):
        self.text = text
        self.page_number = page_number
        self.chunk_id = chunk_id
        self.embedding = embedding
        self.metadata = metadata if metadata is not None else {}


class BookRAGSystem:
    """
    RAG (Retrieval-Augmented Generation) system for large books
    
    Features:
    - Permanent cache in user AppData directory
    - Indexes entire book into searchable chunks
    - Uses embeddings for semantic search
    - Retrieves only relevant sections for a query
    - Generates answers based on relevant context
    """
    
    def __init__(
        self, 
        model_name: str = "llama3.2:1b",
        embedding_model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
        self.api_url = f"{ollama_host}/api"
        
        self.chunks = []
        self.index_built = False
        
        # Set permanent cache directory
        self.cache_dir = self._get_cache_directory()
        
        self.check_models()

        # Pre-load the generation model to avoid first-query timeouts
        self.preload_model()
    
    def _get_cache_directory(self) -> str:
        """
        Get permanent cache directory based on platform
        
        Returns:
            Path to cache directory
        """
        if sys.platform == 'win32':
            # Windows: AppData\Roaming\SmartReader\cache
            cache_dir = os.path.join(os.environ.get('APPDATA', ''), 'SmartReader', 'cache')
        elif sys.platform == 'darwin':
            # macOS: ~/Library/Application Support/SmartReader/cache
            cache_dir = os.path.expanduser('~/Library/Application Support/SmartReader/cache')
        else:
            # Linux: ~/.config/smartreader/cache
            cache_dir = os.path.expanduser('~/.config/smartreader/cache')
        
        # Create directory if it doesn't exist
        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"✓ Cache directory: {cache_dir}")
        except Exception as e:
            print(f"Warning: Could not create cache directory: {e}")
            # Fallback to temp directory
            import tempfile
            cache_dir = os.path.join(tempfile.gettempdir(), 'SmartReader', 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            print(f"✓ Using fallback cache: {cache_dir}")
        
        return cache_dir

    def preload_model(self):
        """
        Pre-load the generation model into memory
        This prevents first-query timeouts
        """
        print(f"🔄 Pre-loading model: {self.model_name}...")
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": "Ready",
                    "stream": False
                },
                timeout=120  # Give it 2 minutes to load
            )
            
            if response.status_code == 200:
                print(f"✓ Model {self.model_name} loaded and ready")
                return True
            else:
                print(f"⚠️ Model pre-load returned status {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"⚠️ Model took too long to pre-load (will load on first query)")
            return False
            
        except Exception as e:
            print(f"⚠️ Could not pre-load model: {e}")
            print("Model will load on first query instead")
            return False
    
    def get_cache_path(self, pdf_path: str) -> str:
        """
        Generate cache file path for a given PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Full path to cache file
        """
        # Create unique hash based on file path and modification time
        file_stat = os.stat(pdf_path)
        unique_string = f"{pdf_path}_{file_stat.st_size}_{file_stat.st_mtime}"
        file_hash = hashlib.md5(unique_string.encode()).hexdigest()
        
        # Use filename plus hash for cache name
        pdf_name = Path(pdf_path).stem
        # Clean filename (remove special characters)
        safe_name = "".join(c for c in pdf_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        cache_filename = f"{safe_name}_{file_hash[:8]}.pkl"
        
        return os.path.join(self.cache_dir, cache_filename)
    
    def is_cached(self, pdf_path: str) -> bool:
        """
        Check if a PDF has been cached
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if cached, False otherwise
        """
        cache_path = self.get_cache_path(pdf_path)
        return os.path.exists(cache_path)
    
    def save_cache(self, pdf_path: str):
        """
        Save current index to cache
        
        Args:
            pdf_path: Path to the PDF file
        """
        cache_path = self.get_cache_path(pdf_path)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            print(f"✓ Cache saved: {os.path.basename(cache_path)}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def load_cache(self, pdf_path: str) -> bool:
        """
        Load index from cache
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if successfully loaded, False otherwise
        """
        cache_path = self.get_cache_path(pdf_path)
        
        try:
            with open(cache_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            self.index_built = True  # CRITICAL: Mark index as built
            print(f"✓ Loaded {len(self.chunks)} chunks from cache")
            print(f"✓ Index marked as built: {self.index_built}")
            return True
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            return False
    
    def get_cache_stats(self) -> Tuple[str, int, float]:
        """
        Get cache statistics
        
        Returns:
            Tuple of (cache_dir, num_files, total_size_mb)
        """
        if not os.path.exists(self.cache_dir):
            return self.cache_dir, 0, 0.0
        
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f))
            for f in files
        )
        
        size_mb = total_size / (1024 * 1024)
        return self.cache_dir, len(files), size_mb
    
    def clear_cache(self):
        """Clear all cached files"""
        if os.path.exists(self.cache_dir):
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            print("✓ Cache cleared")
    
    def check_models(self):
        """Check if required models are available"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            models = [m['name'] for m in response.json().get('models', [])]
            
            if self.embedding_model not in models:
                print(f"📥 Pulling embedding model: {self.embedding_model}")
                print("This is needed for semantic search (one-time setup)...")
                os.system(f"ollama pull {self.embedding_model}")
            
            print(f"✓ Models ready: {self.model_name}, {self.embedding_model}")
        except Exception as e:
            print(f"Warning: Could not verify models: {e}")
    
    def extract_text_with_pages(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> List[Tuple[int, str]]:
        """
        Extract text from PDF, keeping track of page numbers
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            List of (page_number, text) tuples
        """
        print(f"📖 Extracting text from: {pdf_path}")
        pages_text = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                print(f"Total pages: {total_pages}")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    if progress_callback:
                        progress_callback(page_num, total_pages, f"Extracting page {page_num}/{total_pages}")
                    
                    text = page.extract_text()
                    if text.strip():
                        pages_text.append((page_num, text))
                    
                    if page_num % 50 == 0:
                        print(f"  Processed {page_num}/{total_pages} pages...")
                
                print(f"✓ Extracted text from {len(pages_text)} pages")
                return pages_text
        
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return []
    
    def create_chunks(
        self, 
        pages_text: List[Tuple[int, str]], 
        chunk_size: int = 1000,
        overlap: int = 200,
        progress_callback: Optional[Callable] = None
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks for better context
        
        Args:
            pages_text: List of (page_num, text) tuples
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
            progress_callback: Optional callback(current, total, message)
        """
        print(f"✂️  Creating chunks (size={chunk_size}, overlap={overlap})...")
        chunks = []
        chunk_id = 0
        
        total_pages = len(pages_text)
        
        for idx, (page_num, page_text) in enumerate(pages_text):
            if progress_callback:
                progress_callback(idx + 1, total_pages, f"Chunking page {idx + 1}/{total_pages}")
            
            # Split by paragraphs first
            paragraphs = page_text.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append(TextChunk(
                            text=current_chunk.strip(),
                            page_number=page_num,
                            chunk_id=chunk_id,
                            metadata={'length': len(current_chunk)}
                        ))
                        chunk_id += 1
                    
                    # Keep overlap from previous chunk
                    current_chunk = current_chunk[-overlap:] + para + "\n\n"
            
            # Add remaining text
            if current_chunk.strip():
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    page_number=page_num,
                    chunk_id=chunk_id,
                    metadata={'length': len(current_chunk)}
                ))
                chunk_id += 1
        
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text using Ollama
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['embedding']
            else:
                print(f"Error getting embedding: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def build_index(self, pdf_path: str, progress_callback: Optional[Callable] = None):
        """
        Build searchable index of the entire book with automatic caching
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback(progress, status_text)
        """
        # Check if cache exists
        if self.is_cached(pdf_path):
            if progress_callback:
                progress_callback(100, "Loading cached index...")
            
            print(f"📦 Loading cached index...")
            if self.load_cache(pdf_path):
                if progress_callback:
                    progress_callback(100, "Cache loaded successfully!")
                return
        
        # Extract and chunk text
        if progress_callback:
            progress_callback(10, "Extracting text from PDF...")
        
        pages_text = self.extract_text_with_pages(pdf_path, progress_callback)
        if not pages_text:
            print("Failed to extract text")
            return
        
        if progress_callback:
            progress_callback(25, "Creating text chunks...")
        
        self.chunks = self.create_chunks(pages_text, progress_callback=progress_callback)
        
        # Generate embeddings for each chunk
        print("🧮 Generating embeddings for semantic search...")
        print("This may take a while for large books (one-time process)...")
        
        total_chunks = len(self.chunks)
        
        for i, chunk in enumerate(self.chunks):
            if i % 20 == 0:
                print(f"  Processing chunk {i+1}/{total_chunks}...")
                
                # Update progress (25% to 90% range for embeddings)
                progress = 25 + int((i / total_chunks) * 65)
                if progress_callback:
                    progress_callback(progress, f"Generating embeddings... ({i+1}/{total_chunks})")
            
            # Truncate very long chunks for embedding
            text_for_embedding = chunk.text[:500]
            chunk.embedding = self.get_embedding(text_for_embedding)
        
        self.index_built = True
        print("✓ Index built successfully")
        
        # Save cache
        if progress_callback:
            progress_callback(95, "Saving cache...")
        
        self.save_cache(pdf_path)
        
        if progress_callback:
            progress_callback(100, "Indexing complete!")
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a_array = np.array(a)
        b_array = np.array(b)
        
        return np.dot(a_array, b_array) / (
            np.linalg.norm(a_array) * np.linalg.norm(b_array) + 1e-10
        )
    
    def search_relevant_chunks(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Tuple[TextChunk, float]]:
        """
        Search for chunks most relevant to the query
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self.index_built:
            print("Error: Index not built. Call build_index() first.")
            return []
        
        print(f"🔍 Searching for: '{query}'")
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        if not query_embedding:
            print("Failed to get query embedding")
            return []
        
        # Calculate similarity with all chunks
        similarities = []
        for chunk in self.chunks:
            if chunk.embedding:
                similarity = self.cosine_similarity(query_embedding, chunk.embedding)
                similarities.append((chunk, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        top_results = similarities[:top_k]
        
        print(f"✓ Found {len(top_results)} relevant sections")
        for i, (chunk, score) in enumerate(top_results[:3], 1):
            print(f"  {i}. Page {chunk.page_number} (similarity: {score:.3f})")
        
        return top_results
    
    def generate_answer(
        self, 
        query: str, 
        context_chunks: List[Tuple[TextChunk, float]],
        max_context_length: int = 4000
    ) -> str:
        """
        Generate answer using relevant context
        
        Args:
            query: User's question
            context_chunks: Relevant chunks with similarity scores
            max_context_length: Maximum characters to include in context
            
        Returns:
            Generated answer
        """
        # Build context from top chunks
        context_parts = []
        current_length = 0
        
        for chunk, score in context_chunks:
            if current_length + len(chunk.text) > max_context_length:
                break
            context_parts.append(
                f"[Page {chunk.page_number}]\n{chunk.text}\n"
            )
            current_length += len(chunk.text)
        
        context = "\n---\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful assistant analyzing a book. Based on the following excerpts from the book, answer the user's question comprehensively.

Question: {query}

Relevant excerpts from the book:

{context}

Instructions:
1. Provide a comprehensive answer based on the excerpts above
2. Cite page numbers when referencing specific information
3. If the information spans multiple pages, mention the page range
4. If the excerpts don't fully answer the question, acknowledge what's covered and what's not
5. Be specific and detailed in your response

Answer:"""
        
        print("💭 Generating answer...")
        
        # Call Ollama
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 800,
                        "num_ctx": 4096
                    }
                },
                timeout=180
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return f"Error: {response.status_code}"
        
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def query(self, question: str, top_k: int = 10) -> Dict:
        """
        Main query interface - search and generate answer
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.index_built:
            return {
                "error": "Index not built. Call build_index() first.",
                "answer": None
            }
        
        # Search for relevant chunks
        relevant_chunks = self.search_relevant_chunks(question, top_k=top_k)
        
        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the book.",
                "sources": [],
                "pages": []
            }
        
        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)
        
        # Extract source information
        pages = sorted(set([chunk.page_number for chunk, _ in relevant_chunks]))
        sources = [
            {
                "page": chunk.page_number,
                "similarity": float(score),
                "preview": chunk.text[:200] + "..."
            }
            for chunk, score in relevant_chunks[:5]
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "pages": pages,
            "total_chunks_used": len(relevant_chunks)
        }


def main():
    """Example usage"""
    import sys
    
    print("="*80)
    print("📚 Book RAG System - Query Large PDFs with Ollama")
    print("="*80)
    
    # Initialize system
    rag = BookRAGSystem(
        model_name="llama3.2:1b",
        embedding_model="nomic-embed-text"
    )
    
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("\nEnter path to book PDF: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    # Build index (with automatic caching)
    rag.build_index(pdf_path)
    
    print("\n" + "="*80)
    print("Index built! You can now query the book.")
    print("="*80)
    
    # Interactive query loop
    while True:
        print("\n" + "-"*80)
        query = input("\nYour question (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Query the book
        result = rag.query(query, top_k=10)
        
        # Display results
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result['answer'])
        
        print("\n" + "-"*80)
        print(f"📄 Sources: Pages {', '.join(map(str, result['pages']))}")
        print(f"📊 Used {result['total_chunks_used']} relevant sections")
        
        # Option to see sources
        show_sources = input("\nShow source excerpts? (y/n): ").lower()
        if show_sources == 'y':
            print("\n" + "="*80)
            print("SOURCE EXCERPTS:")
            print("="*80)
            for i, source in enumerate(result['sources'], 1):
                print(f"\n{i}. Page {source['page']} (relevance: {source['similarity']:.3f})")
                print(f"{source['preview']}")
                print("-" * 60)


if __name__ == "__main__":
    main()
