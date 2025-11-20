# SmartReader

> Your personal AI-powered book assistant that runs completely offline

SmartReader is a Windows desktop application that transforms how you interact with PDF documents. Upload any book or paper, ask questions in plain English, and get instant answers with exact page citations—all powered by AI that runs locally on your machine.


[![GitHub release](https://img.shields.io/github/v/release/hamii31/SmartReader?style=flat-square)](https://github.com/hamii31/SmartReader/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/github/downloads/hamii31/SmartReader/total?style=flat-square)](https://github.com/hamii31/SmartReader/releases)

---

## Features
- **100% Local** - All processing happens on your machine
- **No Cloud Dependencies** - Your documents never leave your computer
- **Completely Offline** - Works without internet after initial setup
- **Plain English Queries** - Ask questions like you're talking to a person
- **Context-Aware Answers** - Get relevant responses based on your document
- **Page Citations** - Every answer includes exact page references
- **Smart Caching** - Books are indexed once, then load instantly
- **Large Document Support** - Handle 900+ page books effortlessly
- **Fast Responses** - Get answers in 5-10 seconds
- **Setup Wizard** - Guided installation for first-time users
- **Clean Interface** - Intuitive design focused on productivity
- **No Configuration** - Works out of the box

## Automated Setup
- Setup wizard downloads models automatically
- No manual terminal commands needed
- One-click installation experience

## Data Storage & Cache Management

SmartReader stores cached book indexes in a permanent location:

**Windows:** `C:\Users\YourName\AppData\Roaming\SmartReader\cache`  
**macOS:** `~/Library/Application Support/SmartReader/cache`  
**Linux:** `~/.config/smartreader/cache`

### Managing Cache
- **View cache location:** Tools → Cache Location
- **Clear all cached books:** Tools → Clear Cache
- **Manual cleanup:** Simply delete the cache folder

Cache files are named uniquely based on PDF content, so:
- Same PDF = reuses cache 
- Modified PDF = creates new cache 
- Renamed PDF = still finds cache 
---

## Download

### Latest Release (v2.0.0)

**[⬇️ Download SmartReader.exe](https://github.com/hamii31/SmartReader/releases/download/v2.0.0/SmartReader.exe)**

*Size: ~25 MB | Platform: Windows 10/11 | License: MIT*

**Check regularly for new releases!**

## Quick Start

### Installation (3 steps, ~10 minutes)

1. **Download SmartReader** (25 MB)
   - [⬇️ Download SmartReader.exe](https://github.com/hamii31/SmartReader/releases/download/v1.1.3/SmartReader.exe)

2. **Install Ollama** (600 MB)
   - Setup wizard will guide you
   - Download from: https://ollama.com
   - Simple 1-click installer

3. **Download Models** (2.2 GB - automatic)
   - SmartReader downloads these for you
   - llama3.2:3b + nomic-embed-text
   - Takes 5-10 minutes

**Total Size:** ~2.5 GB  
**Total Time:** ~10-15 minutes  
**Then:** Use forever, completely offline!

### After Setup
- Works offline
- Cache books once, query forever

### First Use

1. **Launch SmartReader** (after setup)
2. **Click** "Load PDF Book"
3. **Select** your PDF file
4. **Wait** for indexing (first time only: 10-15 mins for large books)
5. **Ask questions** in the query box
6. **Get answers** with page citations!

### Example Queries
```
"What is the main argument of chapter 3?"
"Summarize the methodology section"
"What does the author say about climate change?"
"List all the conclusions mentioned in the book"
"What are the key findings on page 45-50?"
```

## Technology Stack

### Core Technologies
- **Python 3.14** - Application framework
- **Tkinter** - Cross-platform GUI
- **Ollama** - Local LLM inference engine

### AI/ML Components
- **llama3.2:3b** - Advanced Reasoning Language model (3B parameters)
- **nomic-embed-text** - Text embedding model
- **RAG Architecture** - Retrieval-Augmented Generation

### Libraries
- **PyPDF2** - PDF parsing and text extraction
- **NumPy** - Vector operations and similarity search
- **Requests** -  HTTP client (setup only, checks Ollama server status)
- **PyInstaller** - Executable packaging

---

## Architecture

SmartReader implements a **Retrieval-Augmented Generation (RAG)** pipeline:

1. **PDF Ingestion** 
   - Extract text while preserving page numbers
   - Handle complex layouts and multi-column formats

2. **Intelligent Chunking**
   - Split into overlapping segments (1000 chars, 200 overlap)
   - Maintain context across chunk boundaries

3. **Semantic Embedding**
   - Convert text to 768-dimensional vectors
   - Use nomic-embed-text for high-quality embeddings

4. **Vector Indexing**
   - Create searchable database with NumPy
   - Cache indexes for instant subsequent loads

5. **Query Processing**
   - Embed user query in same vector space
   - Find top-K relevant chunks via cosine similarity

6. **Answer Generation**
   - Feed context to llama3.2:3b
   - Generate coherent answers with citations
   - Return page references for verification

---

## Building from Source

### Prerequisites
```bash
# Python 3.10 or higher
python --version

# Ollama installed
ollama --version

# Git (optional, for cloning)
git --version
```

### Installation Steps
```bash
# Clone repository
git clone https://github.com/hamii31/SmartReader.git
cd SmartReader

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Run application
python main_launcher.py
```

### Building Executable
```bash
# Build with PyInstaller
python build_executable.py

# Output location
cd dist
SmartReader.exe
```
---

## Troubleshooting

### Windows Security Warning

**Issue:** Windows shows "Unknown publisher" warning

**Solution:** This is normal for unsigned apps. Click "More info" → "Run anyway"

### Antivirus False Positive

**Issue:** Antivirus blocks SmartReader.exe

**Solution:** 
1. Whitelist SmartReader.exe in your antivirus
2. The app is safe—false positives are common with PyInstaller executables

### Ollama Connection Error

**Issue:** "Could not connect to Ollama"

**Solution:**
1. Ensure Ollama is installed
2. Run `ollama serve` in Command Prompt
3. Keep that window open while using SmartReader

### Slow First Query

**Issue:** First query takes 2-3 minutes

**Solution:** This is normal - Ollama is loading the model into memory. Subsequent queries are fast.

### Out of Memory

**Issue:** Application crashes with large PDFs

**Solution:**
- Ensure you have 16GB RAM
- Close other applications
- Try smaller PDF files (under 500 pages)

---

## Reporting Bugs
- Use the [issue tracker](https://github.com/hamii31/SmartReader/issues)
- Include steps to reproduce
- Attach error logs if available

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
MIT License

Copyright (c) 2025 YOUR_NAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---