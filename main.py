import sys
import os
import warnings
from PyQt5.QtWidgets import QApplication, QMessageBox

# Try to import MainWindow, but provide fallback if it fails
try:
    from gui.main_window import MainWindow
    MAIN_WINDOW_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Error importing MainWindow: {str(e)}")
    MAIN_WINDOW_AVAILABLE = False

def main():
    # Ensure required directories exist
    os.makedirs('data/known_faces', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/models/pose', exist_ok=True)
    os.makedirs('screenshots', exist_ok=True)
    
    app = QApplication(sys.argv)
    
    if not MAIN_WINDOW_AVAILABLE:
        # Show error message if MainWindow couldn't be imported
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setWindowTitle("Import Error")
        error_msg.setText("Could not import required modules.")
        error_msg.setInformativeText("There was an error importing TensorFlow. "
                                    "Some features may be disabled.")
        error_msg.setDetailedText("The application will run with limited functionality. "
                                 "Action recognition features will be disabled.")
        error_msg.exec_()
        return 1
    
    window = MainWindow()
    window.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
