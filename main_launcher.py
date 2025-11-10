"""
Main Launcher for SmartReader
Checks if setup is needed and launches the appropriate interface
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os
from pathlib import Path

VERSION = "v2.0"


def get_config_dir():
    """
    Get application config directory
    
    Returns:
        Path to config directory (creates if doesn't exist)
    """
    if sys.platform == 'win32':
        config_dir = os.path.join(os.environ.get('APPDATA', ''), 'SmartReader')
    elif sys.platform == 'darwin':
        config_dir = os.path.expanduser('~/Library/Application Support/SmartReader')
    else:
        config_dir = os.path.expanduser('~/.config/smartreader')
    
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def is_setup_complete():
    """
    Check if setup has been completed before
    
    Returns:
        True if setup was previously completed, False otherwise
    """
    config_file = os.path.join(get_config_dir(), 'setup_complete.flag')
    exists = os.path.exists(config_file)
    print(f"Setup complete flag: {config_file} - Exists: {exists}")
    return exists


def mark_setup_complete():
    """Mark setup as completed by creating flag file"""
    config_file = os.path.join(get_config_dir(), 'setup_complete.flag')
    try:
        with open(config_file, 'w') as f:
            f.write(f'Setup completed successfully - {VERSION}')
        print(f"Setup marked complete: {config_file}")
    except Exception as e:
        print(f"Warning: Could not create setup flag: {e}")


def is_upgraded_from_v1():
    """
    Check if this is an upgrade from v1.x to v2.0
    
    Returns:
        True if upgrading from v1, False otherwise
    """
    config_file = os.path.join(get_config_dir(), 'setup_complete.flag')
    if not os.path.exists(config_file):
        return False
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
            # Check if file contains old version string
            if 'v1.' in content or 'Setup completed successfully' == content.strip():
                return True
    except:
        pass
    
    return False


def mark_migration_shown():
    """Mark that migration notice has been shown"""
    config_file = os.path.join(get_config_dir(), 'migration_v2_shown.flag')
    try:
        with open(config_file, 'w') as f:
            f.write('Migration notice shown')
        print("Migration notice marked as shown")
    except Exception as e:
        print(f"Warning: Could not create migration flag: {e}")


def should_show_migration_notice():
    """Check if we should show the migration notice"""
    if not is_upgraded_from_v1():
        return False
    
    # Check if we've already shown it
    config_file = os.path.join(get_config_dir(), 'migration_v2_shown.flag')
    return not os.path.exists(config_file)


def show_migration_notice():
    """Show migration notice for v1 -> v2 upgrade"""
    response = messagebox.askyesnocancel(
        "SmartReader v2.0",
        "Welcome to SmartReader v2.0!\n\n"
        "NEW FEATURES:\n"
        "🧠 Chain-of-Thought Reasoning\n"
        "📊 Confidence Scoring\n"
        "💬 Multi-Turn Conversations\n"
        "⚡ llama3.2:3b Model (smarter!)\n\n"
        "RECOMMENDED: Clear your old cache to take advantage\n"
        "of the new 3b model. Books will need to be re-indexed\n"
        "once (10-15 min for large books).\n\n"
        "Clear cache now and start fresh?\n\n"
        "[Yes] = Clear cache & proceed\n"
        "[No] = Keep old cache\n"
        "[Cancel] = Don't ask again"
    )
    
    mark_migration_shown()
    
    if response is True:
        # User wants to clear cache
        try:
            from ollama_book_rag import BookRAGSystem
            rag = BookRAGSystem(model_name="llama3.2:3b")
            rag.clear_cache()
            messagebox.showinfo(
                "Cache Cleared",
                "Old cache cleared successfully!\n\n"
                "Your books will be re-indexed with the\n"
                "enhanced 3b model when you load them."
            )
        except Exception as e:
            messagebox.showwarning(
                "Cache Clear Failed",
                f"Could not clear cache automatically:\n{e}\n\n"
                "You can clear it manually via Tools → Clear Cache"
            )
    elif response is False:
        # User wants to keep old cache
        messagebox.showinfo(
            "Using Existing Cache",
            "Your existing cache will be preserved.\n\n"
            "Note: Old caches use the 1b model.\n"
            "For best results, consider clearing the\n"
            "cache via Tools → Clear Cache."
        )
    # else: Cancel - don't ask again (already marked)


def check_ollama_installed():
    """
    Check if Ollama is installed and accessible
    
    Returns:
        True if Ollama is installed, False otherwise
    """
    try:
        if sys.platform == 'win32':
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
        
        if result.returncode == 0:
            print(f"Ollama detected: {result.stdout.strip()}")
            return True
        else:
            print("Ollama not found (non-zero return code)")
            return False
    
    except FileNotFoundError:
        print("Ollama not found (command not found)")
        return False
    except subprocess.TimeoutExpired:
        print("Ollama check timed out")
        return False
    except Exception as e:
        print(f"Ollama check failed: {e}")
        return False


def check_models_downloaded():
    """
    Check if required AI models are downloaded
    ENHANCED: Now checks for llama3.2:3b instead of 1b
    
    Returns:
        True if both required models exist, False otherwise
    """
    try:
        if sys.platform == 'win32':
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
        
        if result.returncode == 0:
            output = result.stdout
            # ENHANCED: Check for 3b model
            has_llama = "llama3.2:3b" in output or "llama3.2" in output
            has_embed = "nomic-embed-text" in output
            
            if has_llama and has_embed:
                print("All required models found")
                return True
            else:
                missing = []
                if not has_llama:
                    missing.append("llama3.2:3b")
                if not has_embed:
                    missing.append("nomic-embed-text")
                print(f"Missing models: {', '.join(missing)}")
                return False
        else:
            print("Could not list models")
            return False
    
    except Exception as e:
        print(f"Model check failed: {e}")
        return False


def ensure_ollama_running():
    """
    Make sure Ollama server is running
    
    Returns:
        True if Ollama is running, False otherwise
    """
    # Check if already running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("Ollama server is running")
            return True
    except:
        pass
    
    # Try to start Ollama
    print("Starting Ollama server...")
    try:
        if sys.platform == 'win32':
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        # Give it time to start
        import time
        time.sleep(3)
        
        # Check if it started
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                print("Ollama server started successfully")
                return True
        except:
            pass
        
        print("Could not verify Ollama server")
        return False
    
    except Exception as e:
        print(f"Could not start Ollama server: {e}")
        return False


def launch_setup_wizard():
    """Launch the setup wizard"""
    print("Launching setup wizard...")
    
    try:
        import setup_wizard
        
        def on_wizard_complete():
            """Called when wizard finishes successfully"""
            print("Setup wizard completed")
            mark_setup_complete()
            launch_main_app()
        
        root = tk.Tk()
        wizard = setup_wizard.SetupWizard(root, on_complete=on_wizard_complete)
        root.mainloop()
    
    except ImportError as e:
        messagebox.showerror(
            "File Not Found",
            f"Could not find setup_wizard.py\n\n{e}\n\n"
            "Make sure all files are in the same folder."
        )
        sys.exit(1)
    except Exception as e:
        messagebox.showerror(
            "Setup Error",
            f"An error occurred launching the setup wizard:\n\n{e}"
        )
        sys.exit(1)


def launch_main_app():
    """Launch the main application"""
    print("Launching main application...")
    
    # Show migration notice if upgrading from v1
    if should_show_migration_notice():
        show_migration_notice()
    
    # Ensure Ollama is running
    if not ensure_ollama_running():
        response = messagebox.askyesno(
            "Ollama Not Running",
            "Ollama server is not running.\n\n"
            "SmartReader needs Ollama to work.\n\n"
            "Do you want to try starting it automatically?"
        )
        if response:
            if not ensure_ollama_running():
                messagebox.showwarning(
                    "Manual Start Required",
                    "Could not start Ollama automatically.\n\n"
                    "Please start Ollama manually:\n"
                    "1. Open Command Prompt\n"
                    "2. Run: ollama serve\n"
                    "3. Keep that window open\n"
                    "4. Restart SmartReader"
                )
                sys.exit(0)
    
    # Launch GUI
    try:
        import book_rag_gui
        root = tk.Tk()
        app = book_rag_gui.BookRAGApp(root)
        root.mainloop()
    
    except ImportError as e:
        messagebox.showerror(
            "File Not Found",
            f"Could not find book_rag_gui.py\n\n{e}\n\n"
            "Make sure all files are in the same folder.\n\n"
            "Required files:\n"
            "- main_launcher.py\n"
            "- book_rag_gui_enhanced.py\n"
            "- ollama_book_rag.py"
        )
        sys.exit(1)
    except Exception as e:
        messagebox.showerror(
            "Launch Error",
            f"An error occurred launching SmartReader:\n\n{e}"
        )
        import traceback
        traceback.print_exc()
        sys.exit(1)


def show_splash_screen():
    """
    Show splash screen while checking setup
    
    Returns:
        Splash screen window
    """
    splash = tk.Tk()
    splash.title("SmartReader")
    splash.geometry("450x380")  # Slightly taller for enhanced features
    splash.resizable(False, False)
    splash.configure(bg='#667eea')  # Updated color scheme
    
    # Center window on screen
    splash.update_idletasks()
    x = (splash.winfo_screenwidth() - 450) // 2
    y = (splash.winfo_screenheight() - 380) // 2
    splash.geometry(f'450x380+{x}+{y}')
    
    # Remove window decorations for cleaner look
    splash.overrideredirect(True)
    
    # Add a border
    border = tk.Frame(splash, bg='#5568d3', bd=0)
    border.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    content = tk.Frame(border, bg='#667eea')
    content.pack(fill=tk.BOTH, expand=True)
    
    # Logo
    tk.Label(
        content,
        text="📚",
        font=('Arial', 70),
        bg='#667eea',
        fg='white'
    ).pack(pady=(50, 15))
    
    # App name
    tk.Label(
        content,
        text="SmartReader",
        font=('Arial', 28, 'bold'),
        bg='#667eea',
        fg='white'
    ).pack(pady=(0, 5))
    
    # Tagline
    tk.Label(
        content,
        text="AI-Powered Book Assistant",
        font=('Arial', 12),
        bg='#667eea',
        fg='white'
    ).pack(pady=(0, 20))
    
    # Enhanced features
    features_frame = tk.Frame(content, bg='#667eea')
    features_frame.pack(pady=(0, 20))
    
    features = ["🧠 Chain-of-Thought", "📊 Confidence Scores", "💬 Multi-Turn Context"]
    for feature in features:
        tk.Label(
            features_frame,
            text=feature,
            font=('Arial', 9),
            bg='#667eea',
            fg='white'
        ).pack()
    
    # Loading message
    tk.Label(
        content,
        text="Initializing...",
        font=('Arial', 11),
        bg='#667eea',
        fg='white'
    ).pack(pady=(10, 0))
    
    # Version
    tk.Label(
        content,
        text=VERSION,
        font=('Arial', 8),
        bg='#667eea',
        fg='#c4d7ff'
    ).pack(side=tk.BOTTOM, pady=15)
    
    splash.update()
    return splash


def main():
    """Main entry point"""
    print("=" * 70)
    print(f"SmartReader {VERSION}")
    print("=" * 70)
    print()
    
    # Show splash screen
    splash = show_splash_screen()
    
    try:
        import time
        time.sleep(1.5)  # Show splash a bit longer to show features
        
        # Check if setup has been completed before
        if is_setup_complete():
            print("Previous setup detected")
            
            # Quick verification that everything still works
            ollama_ok = check_ollama_installed()
            models_ok = check_models_downloaded() if ollama_ok else False
            
            if ollama_ok and models_ok:
                # Everything is good - launch main app
                print("All systems ready - launching main application")
                splash.destroy()
                launch_main_app()
            else:
                # Something is broken - offer to re-run setup
                splash.destroy()
                
                # Check what's missing
                if not ollama_ok:
                    missing_msg = "Ollama is not installed or not accessible."
                elif not models_ok:
                    missing_msg = "Required AI models (llama3.2:3b, nomic-embed-text) are not installed."
                else:
                    missing_msg = "Some required components are missing."
                
                response = messagebox.askyesno(
                    "Setup Issue Detected",
                    f"{missing_msg}\n\n"
                    "Would you like to run the setup wizard again to fix this?\n\n"
                    "Note: SmartReader requires llama3.2:3b model."
                )
                
                if response:
                    launch_setup_wizard()
                else:
                    # User declined - try to launch anyway
                    messagebox.showinfo(
                        "Launching Anyway",
                        "SmartReader will try to launch, but some features may not work.\n\n"
                        "If you have problems, please run the setup wizard or install:\n"
                        "ollama pull llama3.2:3b"
                    )
                    launch_main_app()
        else:
            # First run - launch setup wizard
            print("First run detected - launching setup wizard")
            splash.destroy()
            launch_setup_wizard()
    
    except Exception as e:
        print(f"Fatal error in main: {e}")
        
        import traceback
        traceback.print_exc()
        
        try:
            splash.destroy()
        except:
            pass
        
        messagebox.showerror(
            "Startup Error",
            f"SmartReader encountered an error during startup:\n\n{e}\n\n"
            "Please try restarting the application.\n\n"
            "If this problem persists, please check that:\n"
            "1. Ollama is installed (ollama.com)\n"
            "2. Required models are installed:\n"
            "   ollama pull llama3.2:3b\n"
            "   ollama pull nomic-embed-text"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
