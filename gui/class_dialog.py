from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QLineEdit, QMessageBox, QComboBox,
                             QTimeEdit, QListWidget, QGroupBox)
from PyQt5.QtCore import Qt, QTime
import uuid

class ClassManagementDialog(QDialog):
    def __init__(self, student_manager, parent=None):
        super().__init__(parent)
        self.student_manager = student_manager
        self.setup_ui()

    def setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Class Management")
        self.setModal(True)
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)

        # Class Information
        class_group = QGroupBox("Class Information")
        class_layout = QVBoxLayout()

        # Class ID and Name
        id_name_layout = QHBoxLayout()
        
        # Class ID
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Class ID:"))
        self.id_input = QLineEdit()
        id_layout.addWidget(self.id_input)
        id_name_layout.addLayout(id_layout)

        # Class Name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_input = QLineEdit()
        name_layout.addWidget(self.name_input)
        id_name_layout.addLayout(name_layout)

        class_layout.addLayout(id_name_layout)

        # Subject and Room
        subject_room_layout = QHBoxLayout()
        
        # Subject
        subject_layout = QHBoxLayout()
        subject_layout.addWidget(QLabel("Subject:"))
        self.subject_input = QLineEdit()
        subject_layout.addWidget(self.subject_input)
        subject_room_layout.addLayout(subject_layout)

        # Room
        room_layout = QHBoxLayout()
        room_layout.addWidget(QLabel("Room:"))
        self.room_input = QLineEdit()
        room_layout.addWidget(self.room_input)
        subject_room_layout.addLayout(room_layout)

        class_layout.addLayout(subject_room_layout)

        # Schedule
        schedule_layout = QHBoxLayout()
        
        # Day selection
        schedule_layout.addWidget(QLabel("Day:"))
        self.day_combo = QComboBox()
        self.day_combo.addItems(["Monday", "Tuesday", "Wednesday", 
                                "Thursday", "Friday", "Saturday", "Sunday"])
        schedule_layout.addWidget(self.day_combo)

        # Time selection
        schedule_layout.addWidget(QLabel("Start Time:"))
        self.start_time = QTimeEdit()
        self.start_time.setDisplayFormat("HH:mm")
        schedule_layout.addWidget(self.start_time)

        schedule_layout.addWidget(QLabel("End Time:"))
        self.end_time = QTimeEdit()
        self.end_time.setDisplayFormat("HH:mm")
        schedule_layout.addWidget(self.end_time)

        class_layout.addLayout(schedule_layout)

        class_group.setLayout(class_layout)
        layout.addWidget(class_group)

        # Student Management
        student_group = QGroupBox("Student Management")
        student_layout = QHBoxLayout()

        # Available students
        available_layout = QVBoxLayout()
        available_layout.addWidget(QLabel("Available Students:"))
        self.available_list = QListWidget()
        available_layout.addWidget(self.available_list)
        student_layout.addLayout(available_layout)

        # Add/Remove buttons
        button_layout = QVBoxLayout()
        self.add_button = QPushButton("Add >")
        self.add_button.clicked.connect(self.add_students)
        self.remove_button = QPushButton("< Remove")
        self.remove_button.clicked.connect(self.remove_students)
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        student_layout.addLayout(button_layout)

        # Enrolled students
        enrolled_layout = QVBoxLayout()
        enrolled_layout.addWidget(QLabel("Enrolled Students:"))
        self.enrolled_list = QListWidget()
        enrolled_layout.addWidget(self.enrolled_list)
        student_layout.addLayout(enrolled_layout)

        student_group.setLayout(student_layout)
        layout.addWidget(student_group)

        # Dialog buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_class)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Load existing students
        self.load_students()

    def load_students(self):
        """Load existing students into the available list."""
        self.available_list.clear()
        for student_id, student_data in self.student_manager.students.items():
            self.available_list.addItem(f"{student_id} - {student_data['name']}")

    def add_students(self):
        """Add selected students to the enrolled list."""
        for item in self.available_list.selectedItems():
            self.enrolled_list.addItem(item.text())
            self.available_list.takeItem(self.available_list.row(item))

    def remove_students(self):
        """Remove selected students from the enrolled list."""
        for item in self.enrolled_list.selectedItems():
            self.available_list.addItem(item.text())
            self.enrolled_list.takeItem(self.enrolled_list.row(item))

    def save_class(self):
        """Save the class information."""
        class_id = self.id_input.text().strip()
        name = self.name_input.text().strip()
        subject = self.subject_input.text().strip()
        room = self.room_input.text().strip()

        if not all([class_id, name, subject, room]):
            QMessageBox.warning(self, "Error", "Please fill all required fields.")
            return

        # Create schedule
        schedule = {
            'day': self.day_combo.currentText(),
            'start_time': self.start_time.time().toString("HH:mm"),
            'end_time': self.end_time.time().toString("HH:mm")
        }

        # Get enrolled students
        enrolled_students = []
        for i in range(self.enrolled_list.count()):
            student_id = self.enrolled_list.item(i).text().split(' - ')[0]
            enrolled_students.append(student_id)

        try:
            # Add class to database
            if self.student_manager.add_class(class_id, name, subject, schedule, room):
                # Add students to class
                for student_id in enrolled_students:
                    self.student_manager.add_student_to_class(student_id, class_id)
                
                QMessageBox.information(self, "Success", "Class created successfully!")
                self.accept()
            else:
                QMessageBox.warning(self, "Error", "Class ID already exists.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create class: {str(e)}")
