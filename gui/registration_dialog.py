from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from models.face_detector import FaceDetector


class RegistrationDialog(QDialog):
    """Dialog for student registration."""

    def __init__(self, parent=None, student_id=None):
        """
        Initialize registration dialog.
        
        Args:
            parent: Parent widget
            student_id: Student ID for editing existing student
        """
        super().__init__(parent)
        self.setWindowTitle("Student Registration")
        self.setMinimumSize(800, 600)
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.cap = None
        self.face_image = None
        self.face_encoding = None
        self.student_id = student_id
        
        # Setup UI
        self.setup_ui()
        self.setup_camera()
        
        # Load student data if editing
        if student_id:
            self.load_student_data()
    
    def setup_ui(self):
        """Setup the registration dialog UI."""
        layout = QVBoxLayout(self)
        
        # Student info section
        info_layout = QHBoxLayout()
        
        # Left side - form
        form_layout = QVBoxLayout()
        
        # ID input
        id_layout = QHBoxLayout()
        id_label = QLabel("Student ID:")
        self.id_input = QLineEdit()
        if self.student_id:
            self.id_input.setText(self.student_id)
            self.id_input.setReadOnly(True)
        id_layout.addWidget(id_label)
        id_layout.addWidget(self.id_input)
        form_layout.addLayout(id_layout)
        
        # Name input
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        self.name_input = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        form_layout.addLayout(name_layout)
        
        # Email input
        email_layout = QHBoxLayout()
        email_label = QLabel("Class Name:")
        self.email_input = QLineEdit()
        email_layout.addWidget(email_label)
        email_layout.addWidget(self.email_input)
        form_layout.addLayout(email_layout)
        
        info_layout.addLayout(form_layout)
        
        # Right side - face capture
        face_layout = QVBoxLayout()
        self.face_label = QLabel()
        self.face_label.setFixedSize(320, 240)
        face_layout.addWidget(self.face_label)
        
        # Camera controls
        camera_btn_layout = QHBoxLayout()
        self.capture_btn = QPushButton("Capture Face")
        self.capture_btn.clicked.connect(self.capture_face)
        camera_btn_layout.addWidget(self.capture_btn)
        
        self.retake_btn = QPushButton("Retake")
        self.retake_btn.clicked.connect(self.retake)
        self.retake_btn.setEnabled(False)
        camera_btn_layout.addWidget(self.retake_btn)
        
        face_layout.addLayout(camera_btn_layout)
        info_layout.addLayout(face_layout)
        
        layout.addLayout(info_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_student)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def setup_camera(self):
        """Initialize the camera capture."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            msg = ("Could not open camera. "
                  "Please check connection.")
            QMessageBox.warning(self, "Camera Error", msg)
            return
        
        # Start timer for frame updates
        self.timer = self.parent().timer
        self.timer.timeout.connect(self.update_frame)
    
    def update_frame(self):
        """Update the camera frame display."""
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # Convert to QImage and display
            qt_image = QImage(
                rgb_frame.data, w, h, bytes_per_line,
                QImage.Format_RGB888
            )
            self.face_label.setPixmap(
                QPixmap.fromImage(qt_image).scaled(
                    320, 240, Qt.KeepAspectRatio
                )
            )
    
    def capture_face(self):
        """Capture the current frame for face detection."""
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Detect face and get encoding
            face_locations = self.face_detector.detect_faces(frame)
            if not face_locations:
                msg = "No face detected. Please try again."
                QMessageBox.warning(
                    self, "Face Detection Error", msg
                )
                return
                
            self.face_image = frame
            self.face_encoding = (
                self.face_detector.compute_face_encoding(
                    frame, face_locations[0]
                )
            )
            
            # Update UI
            self.capture_btn.setEnabled(False)
            self.retake_btn.setEnabled(True)
            
            # Show captured image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_frame.data, w, h, bytes_per_line,
                QImage.Format_RGB888
            )
            self.face_label.setPixmap(
                QPixmap.fromImage(qt_image).scaled(
                    320, 240, Qt.KeepAspectRatio
                )
            )
    
    def retake(self):
        """Reset face capture."""
        self.face_image = None
        self.face_encoding = None
        self.capture_btn.setEnabled(True)
        self.retake_btn.setEnabled(False)
    
    def load_student_data(self):
        """Load existing student data for editing."""
        student = self.parent().database.get_student(self.student_id)
        if student:
            self.name_input.setText(student[1])
            self.email_input.setText(student[2])
    
    def save_student(self):
        """Save student information."""
        # Validate inputs
        student_id = self.id_input.text().strip()
        name = self.name_input.text().strip()
        class_name = self.email_input.text().strip()  # Using email field for class name temporarily
        
        if not student_id or not name or not class_name:
            QMessageBox.warning(
                self, "Validation Error",
                "Please fill in all required fields."
            )
            return
        
        if self.face_image is None or self.face_encoding is None:
            QMessageBox.warning(
                self, "Validation Error",
                "Please capture a face image."
            )
            return
        
        try:
            # Save face image
            image_path = f"faces/{student_id}.jpg"
            cv2.imwrite(image_path, self.face_image)
            
            # Add student to database
            success = self.parent().database.add_student(
                student_id,
                name,
                class_name,
                self.face_encoding.tobytes(),
                image_path
            )
            
            if success:
                QMessageBox.information(
                    self, "Success",
                    f"Student '{name}' has been registered successfully."
                )
                self.accept()
            else:
                QMessageBox.warning(
                    self, "Error",
                    "Failed to register student. Please try again."
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"An error occurred: {str(e)}"
            )
            
    def closeEvent(self, event):
        """Handle dialog close event."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()
