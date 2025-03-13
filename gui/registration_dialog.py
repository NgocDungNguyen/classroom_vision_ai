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
        email_label = QLabel("Email:")
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
        """Save student data to database."""
        student_id = self.id_input.text().strip()
        name = self.name_input.text().strip()
        email = self.email_input.text().strip()
        
        # Validate inputs
        if not student_id or not name or not email:
            msg = "Please fill in all fields."
            QMessageBox.warning(self, "Validation Error", msg)
            return
        
        if not self.student_id and not self.face_encoding:
            msg = "Please capture a face photo."
            QMessageBox.warning(self, "Face Required", msg)
            return
        
        try:
            # Save face image if captured
            if self.face_image is not None:
                face_path = f"faces/{student_id}.jpg"
                cv2.imwrite(face_path, self.face_image)
            
            # Add or update student
            if self.student_id:
                self.parent().database.update_student(
                    student_id, name, email
                )
            else:
                self.parent().database.add_student(
                    student_id, name, email, self.face_encoding
                )
            
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to save student: {str(e)}"
            )
    
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGridLayout, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import os


class RegistrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Register New Student")
        self.setMinimumSize(1000, 600)
        self.setup_ui()
        self.setup_camera()
        self.captured_images = []

    def setup_ui(self):
        layout = QGridLayout()
        self.setLayout(layout)
        
        # Style the dialog
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f2f5;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
            }
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
        """)

        # Left panel - Camera feed
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setStyleSheet(
            "border: 2px solid #ddd; border-radius: 8px;"
        )
        layout.addWidget(self.camera_label, 0, 0, 4, 1)

        # Right panel - Student info
        info_layout = QVBoxLayout()
        
        # Student ID
        id_layout = QHBoxLayout()
        id_label = QLabel("Student ID:")
        self.id_input = QLineEdit()
        id_layout.addWidget(id_label)
        id_layout.addWidget(self.id_input)
        info_layout.addLayout(id_layout)

        # Student Name
        name_layout = QHBoxLayout()
        name_label = QLabel("Full Name:")
        self.name_input = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        info_layout.addLayout(name_layout)

        # Capture status
        self.status_label = QLabel("Look at different angles for capture")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        info_layout.addWidget(self.status_label)

        # Capture button
        self.capture_btn = QPushButton("Capture Face (0/3)")
        self.capture_btn.clicked.connect(self.capture_face)
        info_layout.addWidget(self.capture_btn)

        # Save button
        self.save_btn = QPushButton("Save Student")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_student)
        info_layout.addWidget(self.save_btn)

        layout.addLayout(info_layout, 0, 1)

    def setup_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            image = QImage(
                rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            self.camera_label.setPixmap(
                QPixmap.fromImage(image).scaled(640, 480, Qt.KeepAspectRatio)
            )

    def capture_face(self):
        if len(self.captured_images) < 3:
            ret, frame = self.capture.read()
            if ret:
                self.captured_images.append(frame)
                self.capture_btn.setText(
                    f"Capture Face ({len(self.captured_images)}/3)"
                )
                if len(self.captured_images) == 3:
                    self.save_btn.setEnabled(True)
                    self.status_label.setText("Ready to save!")
                    self.status_label.setStyleSheet(
                        "color: #4CAF50; font-weight: bold;"
                    )

    def save_student(self):
        student_id = self.id_input.text().strip()
        name = self.name_input.text().strip()
        
        if not student_id or not name:
            QMessageBox.warning(self, "Error", "Please fill in all fields")
            return

        # Create directory for student faces
        student_dir = os.path.join("faces", student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        # Save captured images
        for i, image in enumerate(self.captured_images):
            path = os.path.join(student_dir, f"face_{i}.jpg")
            cv2.imwrite(path, image)

        self.accept()

    def closeEvent(self, event):
        self.capture.release()
        self.timer.stop()
        event.accept()
