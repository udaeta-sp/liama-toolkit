"""LIAMA Toolkit launcher."""
import sys
import os
import traceback

# Ensure src is in path regardless of working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "src"))

try:
    from liama.main import main
    main()
except Exception:
    # Show error in a message box if GUI fails to start
    error = traceback.format_exc()
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "LIAMA - Error", error)
    except Exception:
        print(error, file=sys.stderr)
    sys.exit(1)
