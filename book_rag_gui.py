"""
Professional Desktop GUI for SmartReader
Beautiful, user-friendly interface for querying large books locally with permanent cache
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
from pathlib import Path
from datetime import datetime
from ollama_book_rag import BookRAGSystem


class ModernButton(tk.Button):
    """Custom styled button with hover effects"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            relief=tk.FLAT,
            borderwidth=0,
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            **kwargs
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.default_bg = kwargs.get('bg', '#667eea')
        
    def on_enter(self, e):
        if self['state'] != 'disabled':
            self['background'] = '#5568d3'
    
    def on_leave(self, e):
        if self['state'] != 'disabled':
            self['background'] = self.default_bg


class BookRAGApp:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("SmartReader")
        self.root.geometry("1000x750")
        self.root.minsize(900, 600)
        
        # Configure colors
        self.colors = {
            'primary': '#667eea',
            'primary_dark': '#5568d3',
            'secondary': '#764ba2',
            'bg_light': '#f8f9fa',
            'bg_medium': '#e9ecef',
            'text_dark': '#212529',
            'text_medium': '#6c757d',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'white': '#ffffff'
        }
        
        # Application state
        self.rag = None
        self.current_book = None
        self.current_book_path = None
        self.is_indexing = False
        self.is_querying = False
        
        # Configure root window
        self.root.configure(bg=self.colors['bg_light'])
        
        # Setup menu bar first
        self.create_menu()
        
        # Setup UI
        self.setup_ui()
        
        # Initialize RAG system
        self.initialize_rag()
        
        # Check Ollama on startup
        self.root.after(100, self.check_ollama)
    
    def initialize_rag(self):
        """Initialize RAG system on startup"""
        try:
            self.rag = BookRAGSystem(model_name="llama3.2:1b")
            print(f"RAG system initialized")
            print(f"Cache directory: {self.rag.cache_dir}")
        except Exception as e:
            print(f"Warning: Could not initialize RAG: {e}")
            self.rag = None
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load PDF Book", command=self.upload_pdf, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Cache Location", command=self.show_cache_location)
        tools_menu.add_command(label="Clear Cache", command=self.clear_cache_dialog)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.upload_pdf())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
    
    def setup_ui(self):
        """Create the user interface"""
        
        # ==================== HEADER ====================
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=100)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="SmartReader",
            font=("Segoe UI", 28, "bold"),
            bg=self.colors['primary'],
            fg=self.colors['white']
        )
        title_label.pack(pady=(20, 5))
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="Query any book locally with AI - 100% private, no internet needed",
            font=("Segoe UI", 11),
            bg=self.colors['primary'],
            fg=self.colors['white']
        )
        subtitle_label.pack()
        
        # ==================== MAIN CONTENT ====================
        main_frame = tk.Frame(self.root, bg=self.colors['bg_light'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Book management
        left_panel = tk.Frame(main_frame, bg=self.colors['white'], relief=tk.RAISED, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10), pady=0)
        left_panel.config(width=280)
        left_panel.pack_propagate(False)
        
        # Book section title
        book_section_label = tk.Label(
            left_panel,
            text="Book Management",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['white'],
            fg=self.colors['text_dark']
        )
        book_section_label.pack(pady=(15, 10), padx=15, anchor='w')
        
        # Upload button
        self.upload_btn = ModernButton(
            left_panel,
            text="Load PDF Book",
            command=self.upload_pdf,
            bg=self.colors['primary'],
            fg=self.colors['white'],
            activebackground=self.colors['primary_dark'],
            activeforeground=self.colors['white'],
            padx=20,
            pady=12
        )
        self.upload_btn.pack(pady=10, padx=15, fill=tk.X)
        
        # Current book info
        book_info_frame = tk.Frame(left_panel, bg=self.colors['bg_light'])
        book_info_frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(
            book_info_frame,
            text="Current Book:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['bg_light'],
            fg=self.colors['text_medium']
        ).pack(anchor='w', pady=(5, 2))
        
        self.book_name_label = tk.Label(
            book_info_frame,
            text="No book loaded",
            font=("Segoe UI", 9),
            bg=self.colors['bg_light'],
            fg=self.colors['text_dark'],
            wraplength=240,
            justify=tk.LEFT
        )
        self.book_name_label.pack(anchor='w')
        
        # Status indicator
        status_frame = tk.Frame(left_panel, bg=self.colors['bg_light'])
        status_frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(
            status_frame,
            text="Status:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['bg_light'],
            fg=self.colors['text_medium']
        ).pack(anchor='w', pady=(5, 2))
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            font=("Segoe UI", 9),
            bg=self.colors['bg_light'],
            fg=self.colors['text_medium']
        )
        self.status_label.pack(anchor='w')
        
        # Progress bar
        self.progress = ttk.Progressbar(
            left_panel,
            mode='determinate',
            length=250
        )
        # Don't pack by default - only show when needed
        
        # Progress text label (create but don't pack)
        self.progress_text_label = tk.Label(
            left_panel,
            text="",
            font=("Segoe UI", 8),
            bg=self.colors['white'],
            fg=self.colors['text_medium']
        )
        
        # Info section
        info_frame = tk.Frame(left_panel, bg=self.colors['bg_light'])
        info_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        tk.Label(
            info_frame,
            text="How it works:",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_light'],
            fg=self.colors['text_dark']
        ).pack(anchor='w', pady=(0, 8))
        
        instructions = [
            "1. Load a PDF book",
            "2. Wait for indexing (first time)",
            "3. Ask questions!",
            "",
            "✓ 100% Local",
            "✓ Completely Private",
            "✓ Works Offline"
        ]
        
        for instruction in instructions:
            tk.Label(
                info_frame,
                text=instruction,
                font=("Segoe UI", 9),
                bg=self.colors['bg_light'],
                fg=self.colors['text_medium'],
                justify=tk.LEFT
            ).pack(anchor='w', pady=1)
        
        # Right panel - Chat interface
        right_panel = tk.Frame(main_frame, bg=self.colors['white'], relief=tk.RAISED, borderwidth=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Chat title
        chat_title_frame = tk.Frame(right_panel, bg=self.colors['white'])
        chat_title_frame.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        tk.Label(
            chat_title_frame,
            text="Ask Questions",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['white'],
            fg=self.colors['text_dark']
        ).pack(side=tk.LEFT)
        
        # Clear chat button
        self.clear_btn = ModernButton(
            chat_title_frame,
            text="Clear chat",
            command=self.clear_chat,
            bg=self.colors['bg_medium'],
            fg=self.colors['text_dark'],
            activebackground=self.colors['bg_medium'],
            activeforeground=self.colors['text_dark'],
            padx=15,
            pady=6
        )
        self.clear_btn.pack(side=tk.RIGHT)
        
        # Messages area
        messages_frame = tk.Frame(right_panel, bg=self.colors['bg_light'])
        messages_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        # Custom scrolled text with better styling
        self.messages = scrolledtext.ScrolledText(
            messages_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg=self.colors['bg_light'],
            relief=tk.FLAT,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.messages.pack(fill=tk.BOTH, expand=True)
        
        # Text tags styling
        self.messages.tag_config("user", foreground=self.colors['primary'], font=("Segoe UI", 10, "bold"))
        self.messages.tag_config("bot", foreground=self.colors['text_dark'], font=("Segoe UI", 10))
        self.messages.tag_config("system", foreground=self.colors['text_medium'], font=("Segoe UI", 9, "italic"))
        self.messages.tag_config("sources", foreground=self.colors['success'], font=("Segoe UI", 9))
        self.messages.tag_config("error", foreground=self.colors['danger'], font=("Segoe UI", 10))
        self.messages.tag_config("timestamp", foreground=self.colors['text_medium'], font=("Segoe UI", 8))
        
        # Example questions
        examples_frame = tk.Frame(right_panel, bg=self.colors['white'])
        examples_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        tk.Label(
            examples_frame,
            text="Try asking:",
            font=("Segoe UI", 9),
            bg=self.colors['white'],
            fg=self.colors['text_medium']
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        example_questions = [
            "What is the main topic?",
            "Summarize chapter 1",
            "What are the key findings?"
        ]
        
        for question in example_questions:
            btn = tk.Button(
                examples_frame,
                text=question,
                command=lambda q=question: self.use_example(q),
                font=("Segoe UI", 8),
                bg=self.colors['bg_light'],
                fg=self.colors['primary'],
                relief=tk.FLAT,
                padx=8,
                pady=4,
                cursor="hand2"
            )
            btn.pack(side=tk.LEFT, padx=2)
        
        # Input area
        input_frame = tk.Frame(right_panel, bg=self.colors['white'])
        input_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        # Question input
        self.question_input = tk.Entry(
            input_frame,
            font=("Segoe UI", 11),
            bg=self.colors['bg_light'],
            fg=self.colors['text_dark'],
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.question_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        self.question_input.bind('<Return>', lambda e: self.ask_question())
        
        # Ask button
        self.ask_btn = ModernButton(
            input_frame,
            text="Ask ➜",
            command=self.ask_question,
            bg=self.colors['primary'],
            fg=self.colors['white'],
            activebackground=self.colors['primary_dark'],
            activeforeground=self.colors['white'],
            state=tk.DISABLED,
            padx=25,
            pady=10
        )
        self.ask_btn.pack(side=tk.RIGHT)
        
        # ==================== STATUS BAR ====================
        status_bar = tk.Frame(self.root, bg=self.colors['bg_medium'], height=30)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_bar_label = tk.Label(
            status_bar,
            text="Ready",
            font=("Segoe UI", 9),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_medium'],
            anchor='w'
        )
        self.status_bar_label.pack(side=tk.LEFT, padx=10)
        
        # Version label
        tk.Label(
            status_bar,
            text="v1.1.3",
            font=("Segoe UI", 8),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_medium']
        ).pack(side=tk.RIGHT, padx=10)
    
    def check_ollama(self):
        """Check if Ollama is running on startup"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                self.add_system_message("Ollama is running and ready")
            else:
                self.show_ollama_error()
        except:
            self.show_ollama_error()
    
    def show_ollama_error(self):
        """Show error if Ollama is not running"""
        error_msg = """Ollama is not running!

        Please start Ollama first:
        1. Open a terminal/command prompt
        2. Run: ollama serve
        3. Keep that window open
        4. Then use this application

        Or make sure Ollama is installed:
        https://ollama.com/download"""
        
        messagebox.showerror("Ollama Not Running", error_msg)
        self.add_system_message("Ollama not detected. Please start Ollama first.")
        self.upload_btn.config(state=tk.DISABLED)
    
    def show_cache_location(self):
        """Show cache location and statistics"""
        if not self.rag:
            messagebox.showinfo("Cache Info", "RAG system not initialized yet.")
            return
        
        cache_dir, num_books, size_mb = self.rag.get_cache_stats()
        
        messagebox.showinfo(
            "Cache Information",
            f"Cache Directory:\n{cache_dir}\n\n"
            f"Cached Books: {num_books}\n"
            f"Total Size: {size_mb:.2f} MB\n\n"
            "Cache persists even if you move the application.\n"
            "You can manually delete this folder to free up space."
        )
    
    def clear_cache_dialog(self):
        """Clear all cached book data"""
        if not self.rag:
            messagebox.showinfo("Cache Info", "RAG system not initialized yet.")
            return
        
        cache_dir, num_books, size_mb = self.rag.get_cache_stats()
        
        if num_books == 0:
            messagebox.showinfo("Cache Empty", "No cached books to clear.")
            return
        
        response = messagebox.askyesno(
            "Clear Cache",
            f"This will delete {num_books} cached book(s) ({size_mb:.2f} MB).\n\n"
            "Books will need to be re-indexed when loaded again.\n\n"
            "Continue?"
        )
        
        if response:
            try:
                self.rag.clear_cache()
                messagebox.showinfo("Success", "Cache cleared successfully!")
                self.add_system_message("Cache cleared")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear cache:\n{e}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """SmartReader v1.1.3

        AI-Powered Book Assistant

        Features:
        - Query any PDF book locally
        - 100% private and offline
        - Permanent cache system
        - Powered by Ollama and RAG

        Built with Python, Tkinter, and Ollama

        © 2025"""
        
        messagebox.showinfo("About SmartReader", about_text)
    
    def upload_pdf(self):
        """Handle PDF upload"""
        filepath = filedialog.askopenfilename(
            title="Select PDF Book",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        # Check file size
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if file_size_mb > 100:
            response = messagebox.askyesno(
                "Large File",
                f"This file is {file_size_mb:.1f} MB. Indexing may take a long time.\n\nContinue?"
            )
            if not response:
                return
        
        # Store filepath
        self.current_book_path = filepath
        self.current_book = Path(filepath).name
        self.book_name_label.config(text=self.current_book)
        
        # Update UI for indexing
        self.is_indexing = True
        self.upload_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Indexing...", fg=self.colors['warning'])
        
        # Show progress bar
        self.progress.config(mode='determinate', value=0)
        self.progress.pack(padx=15, pady=5, fill=tk.X)
        self.progress_text_label.pack(padx=15, pady=(0, 10))
        
        self.status_bar_label.config(text="Indexing book... This may take 10-15 minutes for large books")
        
        self.add_system_message(f"Loading book: {self.current_book}")
        self.add_system_message("Indexing... (this only happens once per book)")
        
        # Progress callback - FLEXIBLE SIGNATURE
        def update_progress(*args):
            if len(args) == 2:
                # Called as: update_progress(percentage, message)
                progress_percent, message = args
            elif len(args) == 3:
                # Called as: update_progress(current, total, message)
                current, total, message = args
                progress_percent = int((current / total) * 100) if total > 0 else 0
            else:
                return  # Invalid call
            
            self.root.after(0, lambda: self.progress.config(value=progress_percent))
            self.root.after(0, lambda: self.progress_text_label.config(text=message))
            self.root.after(0, lambda: self.status_bar_label.config(text=message))
        
        # Index in background thread
        def index_book():
            try:
                pdf_path = self.current_book_path
                
                # Ensure RAG system exists
                if not self.rag:
                    print("RAG system was None, creating new instance...")
                    self.rag = BookRAGSystem(model_name="llama3.2")
                
                print("="*80)
                print(f"Cache directory: {self.rag.cache_dir}")
                print(f"PDF path: {pdf_path}")
                print(f"Is cached: {self.rag.is_cached(pdf_path)}")
                print(f"Index built before: {self.rag.index_built}")
                print("="*80)
                
                # Build index (handles caching automatically)
                self.rag.build_index(pdf_path, progress_callback=update_progress)
                
                print("="*80)
                print(f"Index built after: {self.rag.index_built}")
                print(f"Number of chunks: {len(self.rag.chunks)}")
                if len(self.rag.chunks) > 0:
                    print(f"First chunk has embedding: {self.rag.chunks[0].embedding is not None}")
                print("="*80)
                
                # Verify before calling complete
                if not self.rag.index_built:
                    raise Exception("Index failed to build - index_built flag is False")
                
                if len(self.rag.chunks) == 0:
                    raise Exception("Index failed to build - no chunks created")
                
                self.root.after(0, self.on_index_complete)
                
            except Exception as error:
                error_msg = str(error)
                print(f"ERROR during indexing: {error_msg}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.on_index_error(error_msg))
        
        thread = threading.Thread(target=index_book, daemon=True)
        thread.start()
    
    def on_index_complete(self):
        """Called when indexing is complete"""
        self.is_indexing = False
        self.progress.config(value=100)
        
        # Verify index is actually built
        if not self.rag or not self.rag.index_built:
            print("ERROR: Index should be built but index_built is False!")
            self.on_index_error("Index was not properly built")
            return
        
        # Hide progress after a moment
        self.root.after(2000, lambda: self.progress.pack_forget())
        self.root.after(2000, lambda: self.progress_text_label.pack_forget())
        
        self.upload_btn.config(state=tk.NORMAL)
        self.question_input.config(state=tk.NORMAL)
        self.ask_btn.config(state=tk.NORMAL)
        
        self.status_label.config(text="Ready", fg=self.colors['success'])
        self.status_bar_label.config(text="Ready to answer questions!")
        
        self.add_system_message(f"Book '{self.current_book}' loaded successfully!")
        self.add_system_message("You can now ask questions about the book.")
        
        self.question_input.focus()
    
    def on_index_error(self, error):
        """Called when indexing fails"""
        self.is_indexing = False
        self.progress.pack_forget()
        self.progress_text_label.pack_forget()
        
        self.upload_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Error", fg=self.colors['danger'])
        self.status_bar_label.config(text="Error indexing book")
        
        self.add_error_message(f"Failed to index book: {error}")
        
        messagebox.showerror(
            "Indexing Error",
            f"Failed to index the book:\n\n{error}\n\nPlease try again or choose a different file."
        )
    
    def ask_question(self):
        """Handle question submission"""
        question = self.question_input.get().strip()
        
        if not question:
            return
        
        # Debug check
        print(f"Asking question: {question}")
        print(f"RAG exists: {self.rag is not None}")
        print(f"Index built: {self.rag.index_built if self.rag else 'N/A'}")
        print(f"Chunks loaded: {len(self.rag.chunks) if self.rag else 0}")
        
        if not self.rag or not self.rag.index_built:
            print("ERROR: No book loaded or index not built!")
            messagebox.showwarning("No Book Loaded", "Please load a PDF book first.")
            return
        
        # Clear input
        self.question_input.delete(0, tk.END)
        
        # Add user message
        self.add_user_message(question)
        
        # Update UI for querying
        self.is_querying = True
        self.ask_btn.config(state=tk.DISABLED)
        self.question_input.config(state=tk.DISABLED)
        self.status_bar_label.config(text="Searching and generating answer...")
        
        # Query in background thread
        def query_book():
            try:
                result = self.rag.query(question, top_k=5)
                self.root.after(0, lambda: self.on_query_complete(result))
            except Exception as error:
                error_msg = str(error)
                print(f"ERROR during query: {error_msg}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.on_query_error(error_msg))
        
        thread = threading.Thread(target=query_book, daemon=True)
        thread.start()
    
    def on_query_complete(self, result):
        """Handle query completion"""
        print("="*80)
        print("ON_QUERY_COMPLETE CALLED")
        print(f"Result keys: {result.keys()}")
        print(f"Answer length: {len(result.get('answer', ''))}")
        print(f"Has error: {'error' in result}")
        print("="*80)
        
        self.is_querying = False
        self.ask_btn.config(state=tk.NORMAL)
        self.question_input.config(state=tk.NORMAL)
        self.status_bar_label.config(text="Ready")
        
        try:
            if "error" in result:
                error_msg = result.get('answer', 'Unknown error occurred')
                print(f"Displaying error: {error_msg}")
                self.add_error_message(error_msg)
            else:
                # Build response message
                response = ""
                
                # Show query type (optional)
                if result.get("query_type"):
                    query_type_emoji = {
                        'summary': '📋',
                        'specific_page': '📄',
                        'chapter': '📖',
                        'specific': '🔍',
                        'vague': '❓'
                    }
                    emoji = query_type_emoji.get(result["query_type"], '🔍')
                    query_type = result["query_type"].replace('_', ' ').title()
                    response += f"{emoji} Query Type: {query_type}\n"
                
                # Show pages consulted
                if result.get("pages"):
                    page_list = ", ".join(str(p) for p in result["pages"])
                    response += f"📄 Pages consulted: {page_list}\n\n"
                
                # Add the answer
                answer_text = result.get("answer", "")
                if answer_text:
                    response += answer_text
                    print(f"Adding bot message with {len(response)} characters")
                    self.add_bot_message(response, pages=result.get("pages"))
                else:
                    print("WARNING: Empty answer!")
                    self.add_error_message("Received empty answer from system")
                
                # Show sources in a separate system message
                if result.get("sources") and len(result["sources"]) > 0:
                    sources_text = "📚 Sources:\n"
                    for i, source in enumerate(result["sources"][:3], 1):
                        preview = source['preview'].replace('\n', ' ')[:150]
                        sources_text += f"{i}. Page {source['page']} (relevance: {source['similarity']:.2%})\n   {preview}...\n"
                    self.add_system_message(sources_text)
        
        except Exception as e:
            print(f"Error in on_query_complete: {e}")
            import traceback
            traceback.print_exc()
            self.add_error_message(f"Error displaying results: {e}")
    
    def on_query_error(self, error):
        """Called when query fails"""
        self.is_querying = False
        self.ask_btn.config(state=tk.NORMAL)
        self.question_input.config(state=tk.NORMAL)
        self.status_bar_label.config(text="Error during query")
        
        self.add_error_message(f"Error: {error}")
    
    def use_example(self, question):
        """Use an example question"""
        if self.rag and self.rag.index_built and not self.is_querying:
            self.question_input.delete(0, tk.END)
            self.question_input.insert(0, question)
            self.ask_question()
    
    def clear_chat(self):
        """Clear the chat messages"""
        response = messagebox.askyesno(
            "Clear Chat",
            "Are you sure you want to clear all messages?"
        )
        if response:
            self.messages.config(state=tk.NORMAL)
            self.messages.delete(1.0, tk.END)
            self.messages.config(state=tk.DISABLED)
            self.add_system_message("Chat cleared")
    
    def add_user_message(self, text):
        """Add user message to chat"""
        timestamp = datetime.now().strftime("%H:%M")
        self.messages.config(state=tk.NORMAL)
        self.messages.insert(tk.END, f"\n[{timestamp}] ", "timestamp")
        self.messages.insert(tk.END, "You: ", "user")
        self.messages.insert(tk.END, f"{text}\n", "bot")
        self.messages.see(tk.END)
        self.messages.config(state=tk.DISABLED)
    
    def add_bot_message(self, text, pages=None):
        """Add bot message to chat"""
        timestamp = datetime.now().strftime("%H:%M")
        self.messages.config(state=tk.NORMAL)
        self.messages.insert(tk.END, f"\n[{timestamp}] ", "timestamp")
        self.messages.insert(tk.END, "Assistant: ", "user")
        self.messages.insert(tk.END, f"\n{text}\n", "bot")
        
        if pages and len(pages) > 0:
            self.messages.insert(tk.END, f"Sources: Pages {', '.join(map(str, pages))}\n", "sources")
        
        self.messages.see(tk.END)
        self.messages.config(state=tk.DISABLED)
    
    def add_system_message(self, text):
        """Add system message to chat"""
        self.messages.config(state=tk.NORMAL)
        self.messages.insert(tk.END, f"\n{text}\n", "system")
        self.messages.see(tk.END)
        self.messages.config(state=tk.DISABLED)
    
    def add_error_message(self, text):
        """Add error message to chat"""
        timestamp = datetime.now().strftime("%H:%M")
        self.messages.config(state=tk.NORMAL)
        self.messages.insert(tk.END, f"\n[{timestamp}] ", "timestamp")
        self.messages.insert(tk.END, f"{text}\n", "error")
        self.messages.see(tk.END)
        self.messages.config(state=tk.DISABLED)


def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Set application icon (if you have one)
    # root.iconbitmap("icon.ico")
    
    app = BookRAGApp(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()