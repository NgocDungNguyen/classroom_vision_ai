from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QPushButton, QLabel, QProgressBar, QLineEdit,
    QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QScrollArea, QHBoxLayout,
    QGridLayout, QMessageBox, QFileDialog, QDateEdit,
    QDialog, QTimeEdit, QCalendarWidget
)
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
import time


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classroom Vision AI")
        self.setMinimumSize(1200, 800)
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.behavior_monitor = BehaviorMonitor()
        self.database = Database()
        self.setup_ui()
        self.setup_camera()
        
        # Initialize monitoring state
        self.monitoring = False
        self.training = False
        self.current_behavior = None
        self.class_start_time = None
        self.current_class = None
        self.check_in_window_active = False
        self.current_attendance = []
        self.check_in_times = {}
        
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
        
        # Class selection
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Current Class:"))
        self.class_combo = QComboBox()
        self.update_class_list()
        class_layout.addWidget(self.class_combo)
        
        # Add class button
        add_class_btn = QPushButton("Add Class")
        add_class_btn.clicked.connect(self.show_add_class_dialog)
        class_layout.addWidget(add_class_btn)
        monitoring_layout.addLayout(class_layout)
        
        # Start class button
        start_class_btn = QPushButton("Start Class")
        start_class_btn.clicked.connect(self.start_class)
        monitoring_layout.addWidget(start_class_btn)
        
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
            # Convert face detections to recognized students format
            recognized_students = [
                {
                    'id': f'student_{i}',  # Placeholder ID until face recognition is implemented
                    'face_location': face
                }
                for i, face in enumerate(faces)
            ]
            
            # Get behaviors and annotated frame
            behaviors, annotated_frame = self.behavior_monitor.analyze_frame(
                frame, recognized_students
            )
            
            # Update analytics with detected behaviors
            self.update_analytics(behaviors)
            
            # Display annotated frame
            frame = annotated_frame
        else:
            # Just draw face rectangles when not monitoring
            for face in faces:
                top, right, bottom, left = face
                cv2.rectangle(
                    frame, (left, top), (right, bottom),
                    (0, 255, 0), 2
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

    def show_add_class_dialog(self):
        """Show dialog for adding a new class."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Class")
        layout = QVBoxLayout(dialog)
        
        # Class details
        form_layout = QGridLayout()
        form_layout.addWidget(QLabel("Class Name:"), 0, 0)
        name_input = QLineEdit()
        form_layout.addWidget(name_input, 0, 1)
        
        form_layout.addWidget(QLabel("Subject:"), 1, 0)
        subject_input = QLineEdit()
        form_layout.addWidget(subject_input, 1, 1)
        
        form_layout.addWidget(QLabel("Room:"), 2, 0)
        room_input = QLineEdit()
        form_layout.addWidget(room_input, 2, 1)
        
        # Schedule
        form_layout.addWidget(QLabel("Schedule:"), 3, 0)
        schedule_layout = QHBoxLayout()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        day_checks = []
        for day in days:
            check = QPushButton(day)
            check.setCheckable(True)
            schedule_layout.addWidget(check)
            day_checks.append(check)
        form_layout.addLayout(schedule_layout, 3, 1)
        
        # Time
        time_layout = QHBoxLayout()
        start_time = QTimeEdit()
        end_time = QTimeEdit()
        time_layout.addWidget(QLabel("Time:"))
        time_layout.addWidget(start_time)
        time_layout.addWidget(QLabel("to"))
        time_layout.addWidget(end_time)
        form_layout.addLayout(time_layout, 4, 1)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        save_btn.clicked.connect(lambda: self.save_class(
            name_input.text(),
            subject_input.text(),
            room_input.text(),
            [day_checks[i].isChecked() for i in range(len(days))],
            start_time.time(),
            end_time.time(),
            dialog
        ))
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec_()
        
    def save_class(self, name, subject, room, days, start_time, end_time, dialog):
        """Save a new class to the database."""
        if not name or not subject or not room:
            QMessageBox.warning(self, "Error", "Please fill in all required fields")
            return
            
        # Generate a unique class ID based on subject and name
        class_id = f"{subject[:3]}{len(name)}_{int(time.time())}"
        class_id = class_id.upper()
        
        schedule = {
            'days': days,
            'start_time': start_time.toString('HH:mm'),
            'end_time': end_time.toString('HH:mm')
        }
        
        try:
            success = self.database.add_class(class_id, name, subject, room, schedule)
            if success:
                self.update_class_list()
                dialog.accept()
                QMessageBox.information(self, "Success", f"Class '{name}' has been created successfully.")
            else:
                QMessageBox.warning(self, "Error", "Failed to create class. Please try again.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
        
    def update_class_list(self):
        """Update the class selection combo box."""
        self.class_combo.clear()
        classes = self.database.get_classes()
        for class_info in classes:
            self.class_combo.addItem(
                f"{class_info['name']} - {class_info['subject']}",
                class_info['id']
            )
            
    def start_class(self):
        """Start monitoring a class session."""
        if not self.current_class:
            QMessageBox.warning(
                self, "Error", "Please select a class first"
            )
            return
            
        self.class_start_time = datetime.now()
        self.check_in_window_active = True
        self.monitoring = True
        
        # Start check-in window timer (15 minutes)
        QTimer.singleShot(900000, self.close_check_in_window)
        
    def close_check_in_window(self):
        """Close the check-in window and finalize initial attendance."""
        self.check_in_window_active = False
        if self.monitoring:
            # Generate initial attendance report
            self.generate_attendance_report()
            
    def generate_attendance_report(self):
        """Generate attendance report for current session."""
        if not self.current_class:
            return
            
        attendance_data = {
            'class_id': self.current_class,
            'date': datetime.now().date(),
            'students': self.current_attendance,
            'check_in_times': self.check_in_times
        }
        
        self.database.save_attendance(attendance_data)
        
    def update_analytics(self, behaviors):
        """Update analytics with behavior and attendance data."""
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
        
        # Get class roster
        if self.current_class:
            total_students = len(
                self.database.get_class_students(self.current_class)
            )
        else:
            total_students = 0
            
        # Update attendance tracking
        present_students = len(set(b['student_id'] for b in behaviors))
        attendance_percent = (
            (present_students / total_students) * 100 if total_students > 0 else 0
        )
        self.attendance_bar.setValue(int(attendance_percent))
        self.attendance_bar.setFormat(
            f"Attendance: {present_students}/{total_students}"
        )
        
        # Track check-ins during window
        if self.check_in_window_active:
            for behavior in behaviors:
                student_id = behavior['student_id']
                if student_id not in self.check_in_times:
                    self.check_in_times[student_id] = datetime.now()
                    
        # Update behavior statistics
        behavior_counts = {}
        student_behaviors = {}
        
        for b in behaviors:
            # Count behaviors
            behavior_counts[b['type']] = behavior_counts.get(
                b['type'], 0
            ) + 1
            
            # Track student-specific behaviors
            student_id = b['student_id']
            if student_id not in student_behaviors:
                student_behaviors[student_id] = {}
            student_behaviors[student_id][b['type']] = (
                student_behaviors[student_id].get(b['type'], 0) + 1
            )
            
        # Update statistics display
        stats = []
        stats.append(f"Total Students Present: {present_students}")
        for behavior, count in behavior_counts.items():
            stats.append(f"{behavior.title()}: {count}")
            
        # Add student-specific statistics
        stats.append("\nStudent Behaviors:")
        for student_id, behaviors in student_behaviors.items():
            student_name = self.database.get_student_name(student_id)
            behavior_list = [
                f"{b}: {c}" for b, c in behaviors.items()
            ]
            stats.append(
                f"{student_name}: {', '.join(behavior_list)}"
            )
            
        self.stats_label.setText("\n".join(stats))
        
    def generate_analytics(self):
        """Generate analytics visualizations."""
        if not self.start_date.date() or not self.end_date.date():
            QMessageBox.warning(
                self, "Error", "Please select date range"
            )
            return
            
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()
        
        # Get analytics data
        attendance_data = self.database.get_attendance_data(
            start_date, end_date
        )
        behavior_data = self.database.get_behavior_data(
            start_date, end_date
        )
        
        # Generate visualizations
        self.plot_attendance_trends(attendance_data)
        self.plot_behavior_distribution(behavior_data)
        self.plot_student_engagement(behavior_data)
        
    def plot_attendance_trends(self, data):
        """Plot attendance trends over time."""
        # Implementation for attendance visualization
        pass
        
    def plot_behavior_distribution(self, data):
        """Plot distribution of different behaviors."""
        # Implementation for behavior distribution visualization
        pass
        
    def plot_student_engagement(self, data):
        """Plot student engagement metrics."""
        # Implementation for engagement visualization
        pass
        
    def export_data(self, format):
        """Export analytics data in specified format."""
        if not self.start_date.date() or not self.end_date.date():
            QMessageBox.warning(
                self, "Error", "Please select date range"
            )
            return
            
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()
        
        # Get data
        attendance_data = self.database.get_attendance_data(
            start_date, end_date
        )
        behavior_data = self.database.get_behavior_data(
            start_date, end_date
        )
        
        # Export path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report",
            "",
            f"{format.upper()} Files (*.{format.lower()})"
        )
        
        if file_path:
            if format.lower() == 'csv':
                self.export_csv(file_path, attendance_data, behavior_data)
            else:
                self.export_json(file_path, attendance_data, behavior_data)
                
    def export_csv(self, path, attendance_data, behavior_data):
        """Export data in CSV format."""
        # Implementation for CSV export
        pass
        
    def export_json(self, path, attendance_data, behavior_data):
        """Export data in JSON format."""
        # Implementation for JSON export
        pass

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
        
    def show_registration(self):
        dialog = RegistrationDialog(self)
        dialog.exec_()

    def closeEvent(self, event):
        self.capture.release()
        self.timer.stop()
        event.accept()
