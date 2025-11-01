"""
Setup Wizard - First-time setup for SmartReader
Automated installation of Ollama and AI models
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import sys
import requests
import threading
import time


class SetupWizard:
    def __init__(self, root, on_complete=None):
        self.root = root
        self.root.title("SmartReader Setup")
        self.root.geometry("700x600")
        self.root.resizable(False, False)
        
        self.on_complete = on_complete
        self.current_step = 0
        self.ollama_installed = False
        self.models_downloaded = False
        
        # Colors
        self.colors = {
            'primary': '#4a90e2',
            'success': '#5cb85c',
            'warning': '#f0ad4e',
            'danger': '#d9534f',
            'dark': '#333333',
            'light': '#f5f5f5'
        }
        
        self.center_window()
        self.create_ui()
        self.show_step(0)
    
    def center_window(self):
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - 700) // 2
        y = (self.root.winfo_screenheight() - 600) // 2
        self.root.geometry(f'700x600+{x}+{y}')
    
    def create_ui(self):
        # Main container
        main = tk.Frame(self.root, bg='white')
        main.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = tk.Frame(main, bg=self.colors['primary'], height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="SmartReader Setup", font=('Arial', 24, 'bold'),
                 bg=self.colors['primary'], fg='white').pack(expand=True)
        
        # Content area (scrollable)
        content_frame = tk.Frame(main, bg='white')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        # Canvas for scrolling
        self.canvas = tk.Canvas(content_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient='vertical', command=self.canvas.yview)
        
        self.scrollable_frame = tk.Frame(self.canvas, bg='white')
        self.scrollable_frame.bind(
            '<Configure>',
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw', width=600)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mouse wheel scrolling
        def scroll(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.canvas.bind_all("<MouseWheel>", scroll)
        
        # Footer
        footer = tk.Frame(main, bg=self.colors['light'], height=100)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        
        # Progress bar
        progress_container = tk.Frame(footer, bg=self.colors['light'])
        progress_container.pack(fill=tk.X, padx=50, pady=(1, 10))
        
        self.step_label = tk.Label(progress_container, text="Step 1 of 5",
                                   font=('Arial', 9), bg=self.colors['light'])
        self.step_label.pack()
        
        self.progress = ttk.Progressbar(progress_container, length=600, mode='determinate')
        self.progress.pack(pady=5)
        
        # Buttons
        btn_container = tk.Frame(footer, bg=self.colors['light'])
        btn_container.pack(fill=tk.X, padx=50, pady=(1, 15))
        
        self.back_button = tk.Button(btn_container, text="← Back",
                                     command=self.go_back,
                                     font=('Arial', 11, 'bold'),
                                     bg='#d0d0d0', fg='black',
                                     width=12, height=2,
                                     state=tk.DISABLED, cursor='hand2')
        self.back_button.pack(side=tk.LEFT)
        
        self.next_button = tk.Button(btn_container, text="Next →",
                                     command=self.go_next,
                                     font=('Arial', 11, 'bold'),
                                     bg=self.colors['primary'], fg='white',
                                     width=12, height=2, cursor='hand2')
        self.next_button.pack(side=tk.RIGHT)
        
        self.cancel_button = tk.Button(btn_container, text="Cancel",
                                       command=self.cancel,
                                       font=('Arial', 10),
                                       bg='white', fg=self.colors['dark'],
                                       width=10, height=2, cursor='hand2')
        self.cancel_button.pack(side=tk.RIGHT, padx=10)
    
    def clear_content(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.canvas.yview_moveto(0)  # Scroll to top
    
    def show_step(self, step):
        self.current_step = step
        self.clear_content()
        
        # Update progress
        self.progress['value'] = (step / 4) * 100
        self.step_label.config(text=f"Step {step + 1} of 5")
        
        # Update back button
        self.back_button.config(state=tk.NORMAL if step > 0 else tk.DISABLED)
        
        # Show appropriate step
        if step == 0:
            self.step_welcome()
        elif step == 1:
            self.step_check_system()
        elif step == 2:
            self.step_install_ollama()
        elif step == 3:
            self.step_download_models()
        elif step == 4:
            self.step_complete()
    
    def go_next(self):
        print(f"Next: step {self.current_step} → {self.current_step + 1}")
        if self.current_step < 4:
            self.show_step(self.current_step + 1)
    
    def go_back(self):
        print(f"Back: step {self.current_step} → {self.current_step - 1}")
        if self.current_step > 0:
            self.show_step(self.current_step - 1)
    
    def cancel(self):
        if messagebox.askyesno("Cancel Setup", 
                              "Are you sure you want to cancel?\n\nSmartReader won't work without setup."):
            self.root.destroy()
            sys.exit(0)
    
    # ==================== STEP 1: WELCOME ====================
    def step_welcome(self):
        tk.Label(self.scrollable_frame, text="Welcome to SmartReader! 👋",
                font=('Arial', 22, 'bold'), bg='white').pack(pady=(20, 10))
        
        tk.Label(self.scrollable_frame, text="Your AI-powered book assistant",
                font=('Arial', 12), bg='white', fg='#666').pack(pady=(0, 30))
        
        # Info box
        info = tk.Frame(self.scrollable_frame, bg='#e8f4f8', relief=tk.SOLID, borderwidth=1)
        info.pack(fill=tk.X, pady=15)
        
        info_text = """This setup wizard will:

  ✓  Install Ollama (AI engine)
  ✓  Download language models (~1.5 GB)
  ✓  Configure everything for you

After setup, you'll be able to:

  •  Upload any PDF book or document
  •  Ask questions in plain English
  •  Get instant answers with page citations
  •  Work completely offline and privately"""
        
        tk.Label(info, text=info_text, font=('Arial', 10),
                bg='#e8f4f8', justify=tk.LEFT).pack(padx=25, pady=20)
        
        # Requirements
        tk.Label(self.scrollable_frame, text="⚠️  Requirements",
                font=('Arial', 11, 'bold'), bg='white', fg='#ff6600').pack(anchor='w', pady=(20, 10))
        
        requirements = [
            "• Windows 10 or Windows 11",
            "• 4GB RAM minimum (8GB recommended)",
            "• 5GB free disk space",
            "• Internet connection (for initial setup only)",
            "• Estimated setup time: 10-15 minutes"
        ]
        
        for req in requirements:
            tk.Label(self.scrollable_frame, text=req, font=('Arial', 10),
                    bg='white', fg='#333').pack(anchor='w', pady=2)
        
        tk.Label(self.scrollable_frame, text="Click 'Next' when you're ready to begin",
                font=('Arial', 10, 'italic'), bg='white',
                fg=self.colors['primary']).pack(pady=(30, 0))
    
    # ==================== STEP 2: CHECK SYSTEM ====================
    def step_check_system(self):
        tk.Label(self.scrollable_frame, text="Checking Your System 🔍",
                font=('Arial', 20, 'bold'), bg='white').pack(pady=(30, 15))
        
        tk.Label(self.scrollable_frame, text="Please wait while we check what's already installed...",
                font=('Arial', 11), bg='white', fg='#666').pack(pady=(0, 30))
        
        # Status box
        status_box = tk.Frame(self.scrollable_frame, bg='#f5f5f5',
                             relief=tk.SOLID, borderwidth=1)
        status_box.pack(fill=tk.X, pady=10)
        
        self.ollama_status = tk.Label(status_box, text="⏳  Checking for Ollama...",
                                     font=('Arial', 11), bg='#f5f5f5',
                                     anchor='w', fg='#666')
        self.ollama_status.pack(fill=tk.X, padx=25, pady=(20, 10))
        
        self.models_status = tk.Label(status_box, text="⏳  Checking for AI models...",
                                     font=('Arial', 11), bg='#f5f5f5',
                                     anchor='w', fg='#666')
        self.models_status.pack(fill=tk.X, padx=25, pady=10)
        
        self.overall_status = tk.Label(status_box, text="",
                                      font=('Arial', 12, 'bold'), bg='#f5f5f5',
                                      anchor='w')
        self.overall_status.pack(fill=tk.X, padx=25, pady=(15, 20))
        
        # Disable next button
        self.next_button.config(state=tk.DISABLED)
        
        # Start check in background
        threading.Thread(target=self.check_system, daemon=True).start()
    
    def check_system(self):
        # Check Ollama
        time.sleep(1)
        try:
            result = subprocess.run(['ollama', '--version'], capture_output=True, timeout=5)
            self.ollama_installed = (result.returncode == 0)
            
            if self.ollama_installed:
                version = result.stdout.decode().strip()
                self.root.after(0, lambda: self.ollama_status.config(
                    text=f"✓  Ollama is installed ({version})", fg=self.colors['success']))
            else:
                self.root.after(0, lambda: self.ollama_status.config(
                    text="✗  Ollama is not installed", fg=self.colors['danger']))
        except:
            self.ollama_installed = False
            self.root.after(0, lambda: self.ollama_status.config(
                text="✗  Ollama is not installed", fg=self.colors['danger']))
        
        # Check models (only if Ollama is installed)
        if self.ollama_installed:
            time.sleep(1)
            try:
                result = subprocess.run(['ollama', 'list'], capture_output=True, timeout=5)
                output = result.stdout.decode()
                
                # Check for llama3.2:1b specifically
                has_llama = 'llama3.2:1b' in output or 'llama3.2' in output
                has_embed = 'nomic-embed-text' in output
                self.models_downloaded = has_llama and has_embed
                
                if self.models_downloaded:
                    self.root.after(0, lambda: self.models_status.config(
                        text="✓  AI models are installed (llama3.2:1b, nomic-embed-text)", 
                        fg=self.colors['success']))
                else:
                    missing = []
                    if not has_llama: missing.append('llama3.2:1b')
                    if not has_embed: missing.append('nomic-embed-text')
                    self.root.after(0, lambda m=missing: self.models_status.config(
                        text=f"✗  Missing models: {', '.join(m)}", fg=self.colors['danger']))
            except:
                self.models_downloaded = False
                self.root.after(0, lambda: self.models_status.config(
                    text="✗  AI models are not installed", fg=self.colors['danger']))
        else:
            self.root.after(0, lambda: self.models_status.config(
                text="⊘  Skipped (Ollama not installed)", fg='#999'))
        
        # Show overall result
        self.root.after(0, self.finish_check)
    
    def finish_check(self):
        if self.ollama_installed and self.models_downloaded:
            self.overall_status.config(
                text="🎉  Everything is ready! Click Next to finish.",
                fg=self.colors['success'])
            self.current_step = 3  # Skip to completion
        elif self.ollama_installed and not self.models_downloaded:
            self.overall_status.config(
                text="📥  Need to download AI models. Click Next to continue.",
                fg=self.colors['warning'])
            self.current_step = 2  # Skip Ollama install
        else:
            self.overall_status.config(
                text="📦  Need to install Ollama and models. Click Next to continue.",
                fg=self.colors['warning'])
        
        self.next_button.config(state=tk.NORMAL)
    
    # ==================== STEP 3: INSTALL OLLAMA ====================
    def step_install_ollama(self):
        tk.Label(self.scrollable_frame, text="Install Ollama ⚙️",
                font=('Arial', 20, 'bold'), bg='white').pack(pady=(30, 15))
        
        tk.Label(self.scrollable_frame, text="Ollama is the AI engine that powers SmartReader",
                font=('Arial', 11), bg='white', fg='#666').pack()
        
        # Instructions
        info = tk.Frame(self.scrollable_frame, bg='#fff9e6', relief=tk.SOLID, borderwidth=1)
        info.pack(fill=tk.X, pady=30)
        
        instructions = """Please install Ollama manually:

1. Visit: https://ollama.com/download
2. Download OllamaSetup.exe
3. Run the installer
4. Wait for installation to complete
5. Come back here and click Next"""
        
        tk.Label(info, text=instructions, font=('Arial', 11),
                bg='#fff9e6', justify=tk.LEFT).pack(padx=25, pady=20)
        
        # Open website button
        def open_ollama_site():
            import webbrowser
            webbrowser.open('https://ollama.com/download')
        
        tk.Button(self.scrollable_frame, text="🌐  Open Ollama Website",
                 command=open_ollama_site,
                 font=('Arial', 11, 'bold'),
                 bg=self.colors['primary'], fg='white',
                 padx=20, pady=10, cursor='hand2').pack(pady=10)
        
        tk.Label(self.scrollable_frame, text="💡 After installing, click Next",
                font=('Arial', 10, 'italic'), bg='white', fg='#666').pack(pady=20)
        
        self.ollama_installed = True  # Assume user will install
    
    # ==================== STEP 4: DOWNLOAD MODELS (AUTOMATED) ====================
    def step_download_models(self):
        tk.Label(self.scrollable_frame, text="Download AI Models 🧠",
                font=('Arial', 20, 'bold'), bg='white').pack(pady=(30, 15))
        
        tk.Label(self.scrollable_frame, text="Downloading optimized models (~1.5 GB total)",
                font=('Arial', 11), bg='white', fg='#666').pack(pady=(0, 20))
        
        # Info box
        info = tk.Frame(self.scrollable_frame, bg='#e8f4f8', relief=tk.SOLID, borderwidth=1)
        info.pack(fill=tk.X, pady=15)
        
        info_text = """Models being downloaded:

  •  llama3.2:1b (1.3 GB) - Fast generation model
  •  nomic-embed-text (274 MB) - Semantic search

This may take 5-10 minutes depending on your internet speed."""
        
        tk.Label(info, text=info_text, font=('Arial', 10),
                bg='#e8f4f8', justify=tk.LEFT).pack(padx=25, pady=15)
        
        # Status area
        status_frame = tk.Frame(self.scrollable_frame, bg='#f5f5f5',
                               relief=tk.SOLID, borderwidth=1)
        status_frame.pack(fill=tk.X, pady=20)
        
        self.download_status = tk.Label(status_frame, text="Preparing to download...",
                                       font=('Arial', 11, 'bold'), bg='#f5f5f5',
                                       fg=self.colors['dark'])
        self.download_status.pack(padx=25, pady=15)
        
        self.download_progress = ttk.Progressbar(status_frame, length=500, mode='indeterminate')
        self.download_progress.pack(padx=25, pady=(0, 15))
        self.download_progress.start()
        
        # Disable buttons during download
        self.next_button.config(state=tk.DISABLED)
        self.back_button.config(state=tk.DISABLED)
        
        # Start download in background
        threading.Thread(target=self.download_models_thread, daemon=True).start()
    
    def download_models_thread(self):
        """Download models automatically"""
        models = [
            ("llama3.2:1b", "Generation model"),
            ("nomic-embed-text", "Embedding model")
        ]
        
        for i, (model, description) in enumerate(models, 1):
            try:
                # Update status
                self.root.after(0, lambda m=model, d=description: 
                    self.download_status.config(
                        text=f"Downloading {m}\n({d}) - {i}/{len(models)}"
                    ))
                
                # Download model
                print(f"📥 Pulling {model}...")
                result = subprocess.run(
                    ['ollama', 'pull', model],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode == 0:
                    print(f"✓ Successfully downloaded {model}")
                else:
                    print(f"❌ Failed to download {model}: {result.stderr}")
                    self.root.after(0, lambda m=model: self.download_failed(m))
                    return
                    
            except subprocess.TimeoutExpired:
                print(f"❌ Timeout downloading {model}")
                self.root.after(0, lambda m=model: self.download_failed(m, "timeout"))
                return
            except Exception as e:
                print(f"❌ Error downloading {model}: {e}")
                self.root.after(0, lambda m=model, err=str(e): self.download_failed(m, err))
                return
        
        # All downloads successful
        self.root.after(0, self.download_complete)
    
    def download_complete(self):
        """Called when all models are downloaded"""
        self.download_progress.stop()
        self.download_status.config(
            text="✓ All models downloaded successfully!",
            fg=self.colors['success']
        )
        
        self.models_downloaded = True
        
        # Enable next button
        self.next_button.config(state=tk.NORMAL)
        self.back_button.config(state=tk.NORMAL)
        
        # Auto-advance after 2 seconds
        self.root.after(2000, lambda: self.show_step(4))
    
    def download_failed(self, model, error="unknown"):
        """Called when download fails"""
        self.download_progress.stop()
        self.download_status.config(
            text=f"❌ Failed to download {model}",
            fg=self.colors['danger']
        )
        
        error_msg = f"""Failed to download model: {model}
Error: {error}

Please try:
1. Check your internet connection
2. Restart SmartReader setup
3. Or download manually:
   - Open Command Prompt
   - Run: ollama pull {model}"""
        
        messagebox.showerror("Download Failed", error_msg)
        
        # Enable back button
        self.back_button.config(state=tk.NORMAL)
    
    # ==================== STEP 5: COMPLETE ====================
    def step_complete(self):
        # Hide cancel button
        self.cancel_button.pack_forget()
        
        tk.Label(self.scrollable_frame, text="🎉",
                font=('Arial', 72), bg='white').pack(pady=(40, 20))
        
        tk.Label(self.scrollable_frame, text="Setup Complete!",
                font=('Arial', 24, 'bold'), bg='white',
                fg=self.colors['success']).pack(pady=(0, 10))
        
        tk.Label(self.scrollable_frame, text="SmartReader is ready to use!",
                font=('Arial', 13), bg='white', fg='#666').pack(pady=(0, 30))
        
        # Features box
        features_box = tk.Frame(self.scrollable_frame, bg='#e8f8e8',
                               relief=tk.SOLID, borderwidth=1)
        features_box.pack(fill=tk.X, pady=15)
        
        features = [
            "✓  Upload any PDF book or document",
            "✓  Ask questions in plain English",
            "✓  Get instant, cited answers (20-30 seconds)",
            "✓  Work completely offline",
            "✓  100% private - no data leaves your computer",
            "✓  Optimized for fast CPU performance"
        ]
        
        for feature in features:
            tk.Label(features_box, text=feature, font=('Arial', 11),
                    bg='#e8f8e8', anchor='w').pack(padx=30, pady=5, fill=tk.X)
        
        tk.Label(self.scrollable_frame, text="Click 'Launch' to start using SmartReader!",
                font=('Arial', 12, 'bold'), bg='white',
                fg=self.colors['primary']).pack(pady=(30, 0))
        
        # Change Next button to Launch
        self.next_button.config(text="Launch SmartReader",
                               command=self.launch_app,
                               bg=self.colors['success'])
        self.back_button.config(state=tk.DISABLED)
    
    def launch_app(self):
        self.root.destroy()
        if self.on_complete:
            self.on_complete()
        else:
            try:
                from book_rag_gui import BookRAGApp
                root = tk.Tk()
                BookRAGApp(root)
                root.mainloop()
            except:
                messagebox.showerror("Error", "Could not launch SmartReader")
                sys.exit(1)


def main():
    root = tk.Tk()
    SetupWizard(root)
    root.mainloop()


if __name__ == '__main__':
    main()