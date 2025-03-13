from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from models.face_detector import FaceDetector
from models.database import Database
from gui.registration_dialog import RegistrationDialog


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        """Initialize main window."""
        super().__init__()
        self.setWindowTitle("Classroom Vision AI")
        self.setMinimumSize(1024, 768)

        # Initialize components
        self.database = Database()
        self.face_detector = FaceDetector()
        self.cap = None
        self.monitoring = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Setup UI
        self.setup_ui()
        self.setup_camera()

    def setup_ui(self):
        """Setup the main window UI."""
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # Monitoring tab
        monitoring_tab = QWidget()
        monitoring_layout = QVBoxLayout(monitoring_tab)

        # Camera feed
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        monitoring_layout.addWidget(self.camera_label)

        # Controls
        controls_layout = QHBoxLayout()
        self.toggle_btn = QPushButton("Start Monitoring")
        self.toggle_btn.clicked.connect(self.toggle_monitoring)
        controls_layout.addWidget(self.toggle_btn)

        self.register_btn = QPushButton("Register Student")
        self.register_btn.clicked.connect(self.show_registration)
        controls_layout.addWidget(self.register_btn)

        monitoring_layout.addLayout(controls_layout)
        tab_widget.addTab(monitoring_tab, "Monitoring")

        # Analytics tab
        analytics_tab = QWidget()
        analytics_layout = QVBoxLayout(analytics_tab)
        analytics_label = QLabel("Analytics Coming Soon!")
        analytics_layout.addWidget(analytics_label)
        tab_widget.addTab(analytics_tab, "Analytics")

    def setup_camera(self):
        """Initialize the camera capture."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.camera_label.setText(
                "Could not open camera. Please check connection."
            )
            return

        self.timer.start(30)  # 30ms = ~33 fps

    def update_frame(self):
        """Update the camera frame display."""
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            if self.monitoring:
                # Detect faces
                face_locations = self.face_detector.detect_faces(frame)
                for face_loc in face_locations:
                    # Draw rectangle around face
                    top, right, bottom, left = face_loc
                    cv2.rectangle(
                        frame, (left, top), (right, bottom),
                        (0, 255, 0), 2
                    )

            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            # Convert to QImage and display
            qt_image = QImage(
                rgb_frame.data, w, h, bytes_per_line,
                QImage.Format_RGB888
            )
            self.camera_label.setPixmap(
                QPixmap.fromImage(qt_image).scaled(
                    640, 480, Qt.KeepAspectRatio
                )
            )

    def toggle_monitoring(self):
        """Toggle face detection monitoring."""
        self.monitoring = not self.monitoring
        self.toggle_btn.setText(
            "Stop Monitoring" if self.monitoring else "Start Monitoring"
        )

    def show_registration(self):
        """Show the student registration dialog."""
        dialog = RegistrationDialog(self)
        dialog.exec_()

    def closeEvent(self, event):
        """Clean up resources when window is closed."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                             QPushButton, QLabel, QProgressBar, QLineEdit,
                             QComboBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QFrame, QScrollArea, QHBoxLayout,
                             QGridLayout, QMessageBox, QFileDialog, QDateEdit)
from PyQt5.QtCore import Qt, QTimer, QDate
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
import cv2
import numpy as np
from datetime import datetime
from models.face_detector import FaceDetector
from models.behavior_monitor import BehaviorMonitor
from models.behavior_trainer import BehaviorTrainer
from models.database import Database
from gui.registration_dialog import RegistrationDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classroom Vision AI")
        self.setMinimumSize(1200, 800)
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.behavior_monitor = BehaviorMonitor()
        self.behavior_trainer = BehaviorTrainer()
        self.database = Database()
        self.setup_ui()
        self.setup_camera()
        
        # Initialize monitoring state
        self.monitoring = False
        self.training = False
        self.current_behavior = None
        self.class_start_time = None

    def setup_ui(self):
        """Setup the main UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Monitoring tab
        monitoring_tab = QWidget()
        monitoring_layout = QVBoxLayout(monitoring_tab)
        
        # Camera feed
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(960, 720)
        self.camera_label.setStyleSheet(
            "border: 2px solid #ddd; border-radius: 8px;"
        )
        monitoring_layout.addWidget(self.camera_label)
        
        # Controls
        controls_layout = QVBoxLayout()
        
        # Time display
        self.time_label = QLabel("Class Duration: 00:00:00")
        self.time_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #333;"
        )
        controls_layout.addWidget(self.time_label)
        
        # Attendance progress
        self.attendance_bar = QProgressBar()
        self.attendance_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        controls_layout.addWidget(self.attendance_bar)
        
        # Monitoring toggle
        self.monitor_btn = QPushButton("Start Monitoring")
        self.monitor_btn.clicked.connect(self.toggle_monitoring)
        self.monitor_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        controls_layout.addWidget(self.monitor_btn)
        
        monitoring_layout.addLayout(controls_layout)
        tabs.addTab(monitoring_tab, "Monitoring")
        
        # Training tab
        training_tab = self.create_training_tab()
        tabs.addTab(training_tab, "Training")
        
        # Analytics tab
        analytics_tab = self.create_analytics_tab()
        tabs.addTab(analytics_tab, "Analytics")
        
        # Students tab
        students_tab = QWidget()
        students_layout = QVBoxLayout(students_tab)
        
        # Search and filter section
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search students...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
        search_layout.addWidget(self.search_input)
        
        register_btn = QPushButton("Register New Student")
        register_btn.clicked.connect(self.show_registration)
        register_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
        """)
        search_layout.addWidget(register_btn)
        
        students_layout.addLayout(search_layout)

        # Students table
        self.students_table = QTableWidget()
        self.students_table.setColumnCount(5)
        self.students_table.setHorizontalHeaderLabels([
            "ID", "Name", "Class", "Attendance Rate", "Recent Behavior"
        ])
        self.students_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        students_layout.addWidget(self.students_table)

        tabs.addTab(students_tab, "Student Management")

        # Register button
        register_btn = QPushButton("Register New Student")
        register_btn.clicked.connect(self.show_registration)
        register_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
        """)
        layout.addWidget(register_btn)

    def create_training_tab(self):
        """Create the training tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Training controls
        controls_layout = QHBoxLayout()
        
        # Video selection
        video_btn = QPushButton("Load Training Video")
        video_btn.clicked.connect(self.load_training_video)
        controls_layout.addWidget(video_btn)
        
        # Behavior selection
        self.behavior_combo = QComboBox()
        self.behavior_combo.addItems([
            'attentive', 'inattentive', 'hand_raised', 
            'sleeping', 'using_phone'
        ])
        controls_layout.addWidget(self.behavior_combo)
        
        # Training controls
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.toggle_training)
        controls_layout.addWidget(self.train_btn)
        
        layout.addLayout(controls_layout)
        
        # Training preview
        self.training_label = QLabel()
        self.training_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.training_label)
        
        # Training progress
        self.training_progress = QProgressBar()
        layout.addWidget(self.training_progress)
        
        return tab
        
    def create_analytics_tab(self):
        """Create the analytics tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Date range selection
        date_layout = QHBoxLayout()
        
        date_layout.addWidget(QLabel("Start Date:"))
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate())
        date_layout.addWidget(self.start_date)
        
        date_layout.addWidget(QLabel("End Date:"))
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        date_layout.addWidget(self.end_date)
        
        layout.addLayout(date_layout)
        
        # Analytics controls
        controls_layout = QHBoxLayout()
        
        generate_btn = QPushButton("Generate Analytics")
        generate_btn.clicked.connect(self.generate_analytics)
        controls_layout.addWidget(generate_btn)
        
        export_combo = QComboBox()
        export_combo.addItems(['CSV', 'JSON'])
        controls_layout.addWidget(export_combo)
        
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(
            lambda: self.export_data(export_combo.currentText().lower())
        )
        controls_layout.addWidget(export_btn)
        
        layout.addLayout(controls_layout)
        
        # Analytics display
        self.analytics_label = QLabel()
        self.analytics_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.analytics_label)
        
        return tab
        
    def setup_camera(self):
        """Initialize the camera with proper error handling."""
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Try different camera indices
        for i in range(2):  # Try first two camera indices
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow
            if cap.isOpened():
                self.capture = cap
                self.timer.start(30)
                break
        
        if self.capture is None:
            QMessageBox.warning(
                self,
                "Camera Error",
                "No camera found. Please connect a camera and restart."
            )

    def update_frame(self):
        """Update the camera frame with error handling."""
        if self.capture is None or not self.capture.isOpened():
            return
            
        ret, frame = self.capture.read()
        if not ret:
            return
            
        # Process frame
        faces = self.face_detector.detect_faces(frame)
        
        if self.monitoring:
            behaviors = self.behavior_monitor.analyze_frame(frame)
            self.update_analytics(behaviors)
            
            # Draw behavior indicators
            for behavior in behaviors:
                if 'face_location' in behavior:
                    x1, y1, x2, y2 = behavior['face_location']
                    color = (0, 255, 0)  # Green for attentive
                    if behavior['type'] != 'attentive':
                        color = (0, 0, 255)  # Red for other behaviors
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add behavior label
                    label = f"{behavior['type']} ({behavior['confidence']:.2f})"
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
        
        # Convert frame for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        self.camera_label.setPixmap(
            QPixmap.fromImage(image).scaled(960, 720, Qt.KeepAspectRatio)
        )

    def toggle_monitoring(self):
        self.monitoring = not self.monitoring
        if self.monitoring:
            self.monitor_btn.setText("Stop Monitoring")
            self.monitor_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
            """)
            self.class_start_time = datetime.now()
        else:
            self.monitor_btn.setText("Start Monitoring")
            self.monitor_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.class_start_time = None

    def update_analytics(self, behaviors):
        if not self.class_start_time:
            return
            
        # Update class duration
        duration = datetime.now() - self.class_start_time
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        self.time_label.setText(
            f"Class Duration: {hours:02d}:{minutes:02d}:{seconds:02d}"
        )
        
        # Update attendance progress
        total_students = 30  # Example total
        present_students = len(set(b['student'] for b in behaviors))
        attendance_percent = (present_students / total_students) * 100
        self.attendance_bar.setValue(int(attendance_percent))
        self.attendance_bar.setFormat(
            f"Attendance: {present_students}/{total_students}"
        )
        
        # Update statistics
        stats = []
        behavior_counts = {}
        for b in behaviors:
            behavior_counts[b['type']] = behavior_counts.get(
                b['type'], 0
            ) + 1
        
        stats.append(f"Total Students Present: {present_students}")
        for behavior, count in behavior_counts.items():
            stats.append(f"{behavior.title()}: {count}")
        
        self.stats_label = QLabel("\n".join(stats))
        self.stats_label.setStyleSheet(
            "font-size: 14px; color: #333;"
        )

    def load_training_video(self):
        """Load a video file for training."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Video",
            "",
            "Video Files (*.mp4 *.avi)"
        )
        
        if file_path:
            self.training_video, self.total_frames = (
                self.behavior_trainer.load_training_data(file_path)
            )
            self.training_progress.setMaximum(self.total_frames)
            self.train_btn.setEnabled(True)
            
    def toggle_training(self):
        """Toggle training mode."""
        self.training = not self.training
        
        if self.training:
            self.train_btn.setText("Stop Training")
            self.current_behavior = self.behavior_combo.currentText()
            self.training_timer = QTimer()
            self.training_timer.timeout.connect(self.update_training)
            self.training_timer.start(30)
        else:
            self.train_btn.setText("Start Training")
            self.training_timer.stop()
            
    def update_training(self):
        """Update training frame and collect annotations."""
        if not self.training_video.isOpened():
            self.toggle_training()
            return
            
        ret, frame = self.training_video.read()
        if not ret:
            self.toggle_training()
            return
            
        # Detect faces and landmarks
        faces = self.face_detector.detect_faces(frame)
        
        if faces:
            # Save annotation
            self.behavior_trainer.save_annotation(
                frame,
                faces[0],  # Use first detected face
                self.current_behavior,
                datetime.now()
            )
        
        # Update progress
        current_frame = self.training_video.get(cv2.CAP_PROP_POS_FRAMES)
        self.training_progress.setValue(int(current_frame))
        
        # Display frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        image = QImage(
            rgb_frame.data, w, h, ch * w, QImage.Format_RGB888
        )
        self.training_label.setPixmap(
            QPixmap.fromImage(image).scaled(960, 720, Qt.KeepAspectRatio)
        )
        
    def generate_analytics(self):
        """Generate analytics visualizations."""
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()
        
        self.behavior_trainer.generate_analytics(start_date, end_date)
        
        # Display analytics images
        analytics_path = (
            self.behavior_trainer.analytics_dir / 'behavior_distribution.png'
        )
        if analytics_path.exists():
            pixmap = QPixmap(str(analytics_path))
            self.analytics_label.setPixmap(
                pixmap.scaled(800, 600, Qt.KeepAspectRatio)
            )
            
    def export_data(self, format):
        """Export analytics data."""
        self.behavior_trainer.export_data(format)
        QMessageBox.information(
            self,
            "Export Complete",
            f"Data exported to {self.behavior_trainer.analytics_dir}"
        )

    def show_registration(self):
        dialog = RegistrationDialog(self)
        dialog.exec_()

    def closeEvent(self, event):
        self.capture.release()
        self.timer.stop()
        event.accept()
