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
            f.write('Setup completed successfully')
        print(f"✓ Setup marked complete: {config_file}")
    except Exception as e:
        print(f"Warning: Could not create setup flag: {e}")


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
            print(f"✓ Ollama detected: {result.stdout.strip()}")
            return True
        else:
            print("✗ Ollama not found (non-zero return code)")
            return False
    
    except FileNotFoundError:
        print("✗ Ollama not found (command not found)")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Ollama check timed out")
        return False
    except Exception as e:
        print(f"✗ Ollama check failed: {e}")
        return False


def check_models_downloaded():
    """
    Check if required AI models are downloaded
    
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
            has_llama = "llama3.2" in output
            has_embed = "nomic-embed-text" in output
            
            if has_llama and has_embed:
                print("✓ All required models found")
                return True
            else:
                missing = []
                if not has_llama:
                    missing.append("llama3.2")
                if not has_embed:
                    missing.append("nomic-embed-text")
                print(f"✗ Missing models: {', '.join(missing)}")
                return False
        else:
            print("✗ Could not list models")
            return False
    
    except Exception as e:
        print(f"✗ Model check failed: {e}")
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
            print("✓ Ollama server is running")
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
                print("✓ Ollama server started successfully")
                return True
        except:
            pass
        
        print("✗ Could not verify Ollama server")
        return False
    
    except Exception as e:
        print(f"✗ Could not start Ollama server: {e}")
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
    
    # Launch main GUI
    try:
        import book_rag_gui
        
        root = tk.Tk()
        app = book_rag_gui.BookRAGApp(root)
        root.mainloop()
    
    except ImportError as e:
        messagebox.showerror(
            "File Not Found",
            f"Could not find book_rag_gui.py\n\n{e}\n\n"
            "Make sure all files are in the same folder."
        )
        sys.exit(1)
    except Exception as e:
        messagebox.showerror(
            "Launch Error",
            f"An error occurred launching SmartReader:\n\n{e}"
        )
        sys.exit(1)


def show_splash_screen():
    """
    Show splash screen while checking setup
    
    Returns:
        Splash screen window
    """
    splash = tk.Tk()
    splash.title("SmartReader")
    splash.geometry("450x350")
    splash.resizable(False, False)
    splash.configure(bg='#4a90e2')
    
    # Center window on screen
    splash.update_idletasks()
    x = (splash.winfo_screenwidth() - 450) // 2
    y = (splash.winfo_screenheight() - 350) // 2
    splash.geometry(f'450x350+{x}+{y}')
    
    # Remove window decorations for cleaner look
    splash.overrideredirect(True)
    
    # Add a border
    border = tk.Frame(splash, bg='#2c5aa0', bd=0)
    border.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    content = tk.Frame(border, bg='#4a90e2')
    content.pack(fill=tk.BOTH, expand=True)
    
    # Logo
    tk.Label(
        content,
        text="📚",
        font=('Arial', 70),
        bg='#4a90e2',
        fg='white'
    ).pack(pady=(60, 15))
    
    # App name
    tk.Label(
        content,
        text="SmartReader",
        font=('Arial', 28, 'bold'),
        bg='#4a90e2',
        fg='white'
    ).pack(pady=(0, 5))
    
    # Tagline
    tk.Label(
        content,
        text="AI-Powered Book Assistant",
        font=('Arial', 12),
        bg='#4a90e2',
        fg='white'
    ).pack(pady=(0, 40))
    
    # Loading message
    tk.Label(
        content,
        text="Initializing...",
        font=('Arial', 11),
        bg='#4a90e2',
        fg='white'
    ).pack()
    
    # Version
    tk.Label(
        content,
        text="v1.0.0",
        font=('Arial', 8),
        bg='#4a90e2',
        fg='#a0c4ff'
    ).pack(side=tk.BOTTOM, pady=20)
    
    splash.update()
    return splash


def main():
    """Main entry point"""
    print("=" * 70)
    print("SmartReader v1.0.0")
    print("=" * 70)
    print()
    
    # Show splash screen
    splash = show_splash_screen()
    
    try:
        import time
        time.sleep(1)  # Show splash for at least 1 second
        
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
                
                response = messagebox.askyesno(
                    "Setup Issue Detected",
                    "Some required components are missing or not working properly.\n\n"
                    "Would you like to run the setup wizard again to fix this?"
                )
                
                if response:
                    launch_setup_wizard()
                else:
                    # User declined - try to launch anyway
                    messagebox.showinfo(
                        "Launching Anyway",
                        "SmartReader will try to launch, but some features may not work.\n\n"
                        "If you have problems, please run the setup wizard."
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
            "If this problem persists, please contact support."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()