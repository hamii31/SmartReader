"""
Build SmartReader as a standalone Windows executable
"""

import PyInstaller.__main__
import os
import sys

print("=" * 70)
print("🔨 Building SmartReader Executable")
print("=" * 70)

APP_NAME = "SmartReader"
VERSION = "1.0.0"
ICON = "icon.ico"  # Optional

# Check if icon exists
if not os.path.exists(ICON):
    print(f"⚠️  No icon found. Building without icon.")
    icon_param = []
else:
    icon_param = [f'--icon={ICON}']
    print(f"✓ Using icon: {ICON}")

print(f"\n📦 Building {APP_NAME} v{VERSION}...")
print("   This will take 3-5 minutes...\n")

# Build configuration
build_params = [
    'main_launcher.py',                    # Entry point
    f'--name={APP_NAME}',                  # Executable name
    '--onefile',                           # Single .exe file
    '--windowed',                          # No console window
    '--clean',                             # Clean build
    '--noconfirm',                         # Don't ask for confirmation
    
    # Add Python files as data
    '--add-data=setup_wizard.py;.',
    '--add-data=book_rag_gui.py;.',
    '--add-data=ollama_book_rag.py;.',
    
    # Hidden imports (modules not auto-detected)
    '--hidden-import=tkinter',
    '--hidden-import=tkinter.ttk',
    '--hidden-import=tkinter.filedialog',
    '--hidden-import=tkinter.scrolledtext',
    '--hidden-import=tkinter.messagebox',
    '--hidden-import=PIL',
    '--hidden-import=PIL.Image',
    '--hidden-import=numpy',
    '--hidden-import=PyPDF2',
    '--hidden-import=requests',
    '--hidden-import=pickle',
    '--hidden-import=threading',
    '--hidden-import=pathlib',
    '--hidden-import=datetime',
    
    # Collect all submodules
    '--collect-all=tkinter',
    '--collect-all=PyPDF2',
    
    # Don't use UPX compression (more compatible)
    '--noupx',
] + icon_param

# Run PyInstaller
try:
    PyInstaller.__main__.run(build_params)
    
    print("\n" + "=" * 70)
    print("✅ BUILD COMPLETE!")
    print("=" * 70)
    
    exe_path = f"dist/{APP_NAME}.exe"
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\n📂 Location: {exe_path}")
        print(f"📦 Size: {size_mb:.1f} MB")
    
    print("\n✨ Features:")
    print("   • Single executable file")
    print("   • No Python installation needed")
    print("   • Double-click to run")
    print("   • Works on any Windows 10/11 PC")
    
    print("\n🚀 To test:")
    print(f"   cd dist")
    print(f"   {APP_NAME}.exe")
    
    print("\n📝 To distribute:")
    print(f"   Just share: dist/{APP_NAME}.exe")
    print("   Users double-click and it works!")
    print("=" * 70 + "\n")

except Exception as e:
    print(f"\n❌ BUILD FAILED!")
    print(f"Error: {e}")
    sys.exit(1)