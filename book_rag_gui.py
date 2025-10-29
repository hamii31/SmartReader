"""
Professional Desktop GUI for Ollama Book RAG
A beautiful, user-friendly interface for querying large books locally
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
        
        # Setup UI
        self.setup_ui()
        
        # Check Ollama on startup
        self.root.after(100, self.check_ollama)
    
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
            text="📖 Book Management",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['white'],
            fg=self.colors['text_dark']
        )
        book_section_label.pack(pady=(15, 10), padx=15, anchor='w')
        
        # Upload button
        self.upload_btn = ModernButton(
            left_panel,
            text="📄 Load PDF Book",
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
            text="● Ready",
            font=("Segoe UI", 9),
            bg=self.colors['bg_light'],
            fg=self.colors['text_medium']
        )
        self.status_label.pack(anchor='w')
        
        # Progress bar
        self.progress = ttk.Progressbar(
            left_panel,
            mode='indeterminate',
            length=250
        )
        # Don't pack by default - only show when needed
        
        # Info section
        info_frame = tk.Frame(left_panel, bg=self.colors['bg_light'])
        info_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        tk.Label(
            info_frame,
            text="💡 How it works:",
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
            text="💬 Ask Questions",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['white'],
            fg=self.colors['text_dark']
        ).pack(side=tk.LEFT)
        
        # Clear chat button
        self.clear_btn = ModernButton(
            chat_title_frame,
            text="🗑️ Clear",
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
        
        # Configure text tags for styling
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
            text="v1.0.0",
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
                self.add_system_message("✓ Ollama is running and ready")
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
        self.add_system_message("⚠️ Ollama not detected. Please start Ollama first.")
        self.upload_btn.config(state=tk.DISABLED)
    
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
        
        # Store filepath as instance variable
        self.current_book_path = filepath
        self.current_book = Path(filepath).name
        self.book_name_label.config(text=self.current_book)
        
        # Update UI for indexing
        self.is_indexing = True
        self.upload_btn.config(state=tk.DISABLED)
        self.status_label.config(text="● Indexing...", fg=self.colors['warning'])
        
        # Show progress bar with determinate mode
        self.progress.config(mode='determinate')
        self.progress['value'] = 0
        self.progress.pack(padx=15, pady=10, fill=tk.X)
        
        # Add progress text label
        self.progress_text_label = tk.Label(
            self.progress.master,
            text="Starting indexing...",
            font=("Segoe UI", 8),
            bg=self.colors['bg_light'],
            fg=self.colors['text_medium']
        )
        self.progress_text_label.pack(padx=15, pady=(0, 5))
        
        self.status_bar_label.config(text="Indexing book... This may take 10-15 minutes for large books")
        
        self.add_system_message(f"Loading book: {self.current_book}")
        self.add_system_message("⏳ Indexing... (this only happens once)")
        
        # Progress callback
        def update_progress(progress, status):
            self.root.after(0, lambda: self.progress.config(value=progress))
            self.root.after(0, lambda: self.progress_text_label.config(text=status))
            self.root.after(0, lambda: self.status_bar_label.config(text=status))
        
        # Index in background thread
        def index_book():
            try:
                # Use self.current_book_path instead of filepath
                pdf_path = self.current_book_path
                
                self.rag = BookRAGSystem(model_name="llama3.2")
                
                # Use default cache path (in ./cache folder)
                cache_path = self.rag.get_default_cache_path(pdf_path)
                print(f"📦 Cache path: {cache_path}")
                print(f"📄 PDF path: {pdf_path}")
                
                self.rag.build_index(pdf_path, cache_path=cache_path, progress_callback=update_progress)
                
                self.root.after(0, self.on_index_complete)
            except Exception as error:
                error_msg = str(error)
                print(f"ERROR during indexing: {error_msg}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda msg=error_msg: self.on_index_error(msg))
        
        thread = threading.Thread(target=index_book, daemon=True)
        thread.start()
        
    def on_index_complete(self):
        """Called when indexing is complete"""
        self.is_indexing = False
        self.progress.config(value=100)
        
        # Hide progress after a moment
        self.root.after(2000, lambda: self.progress.pack_forget())
        if hasattr(self, 'progress_text_label'):
            self.root.after(2000, lambda: self.progress_text_label.pack_forget())
        
        self.upload_btn.config(state=tk.NORMAL)
        self.question_input.config(state=tk.NORMAL)
        self.ask_btn.config(state=tk.NORMAL)
        
        self.status_label.config(text="● Ready", fg=self.colors['success'])
        self.status_bar_label.config(text="Ready to answer questions!")
        
        self.add_system_message(f"✓ Book '{self.current_book}' loaded successfully!")
        self.add_system_message("You can now ask questions about the book.")
        
        self.question_input.focus()
    
    def on_index_error(self, error):
        """Called when indexing fails"""
        self.is_indexing = False
        self.progress.stop()
        self.progress.pack_forget()
        
        self.upload_btn.config(state=tk.NORMAL)
        self.status_label.config(text="● Error", fg=self.colors['danger'])
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
        
        if not self.rag:
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
        self.status_bar_label.config(text="🔍 Searching and generating answer...")
        
        # Query in background thread
        def query_book():
            try:
                result = self.rag.query(question, top_k=10)
                self.root.after(0, lambda res=result: self.on_query_complete(res))  # Fixed lambda
            except Exception as error:  # Changed 'e' to 'error'
                error_msg = str(error)
                self.root.after(0, lambda msg=error_msg: self.on_query_error(msg))  # Fixed lambda
        
        thread = threading.Thread(target=query_book, daemon=True)
        thread.start()
    
    def on_query_complete(self, result):
        """Called when query is complete"""
        self.is_querying = False
        self.ask_btn.config(state=tk.NORMAL)
        self.question_input.config(state=tk.NORMAL)
        self.status_bar_label.config(text="Ready to answer questions!")
        
        answer = result['answer']
        pages = result['pages']
        
        self.add_bot_message(answer, pages)
        
        self.question_input.focus()
    
    def on_query_error(self, error):
        """Called when query fails"""
        self.is_querying = False
        self.ask_btn.config(state=tk.NORMAL)
        self.question_input.config(state=tk.NORMAL)
        self.status_bar_label.config(text="Error during query")
        
        self.add_error_message(f"Error: {error}")
    
    def use_example(self, question):
        """Use an example question"""
        if self.rag and not self.is_querying:
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
            self.messages.insert(tk.END, f"📄 Sources: Pages {', '.join(map(str, pages))}\n", "sources")
        
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
        self.messages.insert(tk.END, f"❌ {text}\n", "error")
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