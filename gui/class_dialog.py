from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QLineEdit, QPushButton, QMessageBox, QComboBox,
                           QFormLayout, QGroupBox, QDialogButtonBox, QListWidget,
                           QListWidgetItem, QSplitter, QWidget, QTableWidget,
                           QTableWidgetItem, QHeaderView, QAbstractItemView,
                           QCheckBox, QDateTimeEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QDateTime
from PyQt5.QtGui import QIcon, QFont
import os
from datetime import datetime

class ClassDialog(QDialog):
    """Dialog for managing class information and student enrollment."""
    
    class_updated = pyqtSignal(dict)
    
    def __init__(self, database, parent=None, class_id=None):
        super().__init__(parent)
        self.database = database
        self.class_id = class_id
        self.class_data = None
        self.enrolled_students = []
        self.available_students = []
        
        # Set window properties
        self.setWindowTitle("Class Management")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        
        # Load class data if editing existing class
        if class_id:
            self.class_data = self.database.get_class(class_id)
            if self.class_data:
                self.setWindowTitle(f"Edit Class: {self.class_data['name']}")
        
        self.setup_ui()
        self.load_students()
        
        # If editing, populate form with class data
        if self.class_data:
            self.populate_form()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Class information section
        class_info_group = QGroupBox("Class Information")
        form_layout = QFormLayout()
        
        # Class ID
        self.id_input = QLineEdit()
        if self.class_id:
            self.id_input.setText(str(self.class_id))
            self.id_input.setReadOnly(True)
        else:
            self.id_input.setPlaceholderText("e.g., CS101")
        form_layout.addRow("Class ID:", self.id_input)
        
        # Class Name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Introduction to Computer Science")
        form_layout.addRow("Class Name:", self.name_input)
        
        # Subject
        self.subject_input = QLineEdit()
        self.subject_input.setPlaceholderText("e.g., Computer Science")
        form_layout.addRow("Subject:", self.subject_input)
        
        # Room
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText("e.g., Room 101")
        form_layout.addRow("Room:", self.location_input)
        
        # Schedule
        schedule_group = QGroupBox("Class Schedule")
        schedule_layout = QVBoxLayout()
        
        # Start time
        start_time_layout = QHBoxLayout()
        start_time_layout.addWidget(QLabel("Start Time:"))
        self.start_time_edit = QDateTimeEdit()
        self.start_time_edit.setDisplayFormat("yyyy-MM-dd hh:mm AP")
        self.start_time_edit.setDateTime(QDateTime.currentDateTime())
        start_time_layout.addWidget(self.start_time_edit)
        schedule_layout.addLayout(start_time_layout)
        
        # End time
        end_time_layout = QHBoxLayout()
        end_time_layout.addWidget(QLabel("End Time:"))
        self.end_time_edit = QDateTimeEdit()
        self.end_time_edit.setDisplayFormat("yyyy-MM-dd hh:mm AP")
        self.end_time_edit.setDateTime(
            QDateTime.currentDateTime().addSecs(3600)  # Add 1 hour
        )
        end_time_layout.addWidget(self.end_time_edit)
        schedule_layout.addLayout(end_time_layout)
        
        schedule_group.setLayout(schedule_layout)
        form_layout.addRow(schedule_group)
        
        class_info_group.setLayout(form_layout)
        main_layout.addWidget(class_info_group)
        
        # Student enrollment section
        enrollment_group = QGroupBox("Student Enrollment")
        enrollment_layout = QHBoxLayout()
        
        # Available students list
        available_group = QGroupBox("Available Students")
        available_layout = QVBoxLayout()
        
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        available_layout.addWidget(self.available_list)
        
        # Add button
        add_btn = QPushButton("Add >>")
        add_btn.clicked.connect(self.add_students)
        available_layout.addWidget(add_btn)
        
        available_group.setLayout(available_layout)
        enrollment_layout.addWidget(available_group)
        
        # Enrolled students list
        enrolled_group = QGroupBox("Enrolled Students")
        enrolled_layout = QVBoxLayout()
        
        self.enrolled_list = QListWidget()
        self.enrolled_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        enrolled_layout.addWidget(self.enrolled_list)
        
        # Remove button
        remove_btn = QPushButton("<< Remove")
        remove_btn.clicked.connect(self.remove_students)
        enrolled_layout.addWidget(remove_btn)
        
        enrolled_group.setLayout(enrolled_layout)
        enrollment_layout.addWidget(enrolled_group)
        
        enrollment_group.setLayout(enrollment_layout)
        main_layout.addWidget(enrollment_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        self.setLayout(main_layout)
    
    def load_students(self):
        """Load available and enrolled students."""
        # Get all students
        all_students = self.database.get_all_students()
        
        # If editing existing class, get enrolled students
        if self.class_id:
            enrolled_students = self.database.get_enrolled_students(self.class_id)
            self.enrolled_students = [s for s in all_students if s['id'] in [es['student_id'] for es in enrolled_students]]
            self.available_students = [s for s in all_students if s['id'] not in [es['student_id'] for es in enrolled_students]]
        else:
            self.available_students = all_students
            self.enrolled_students = []
        
        # Populate lists
        self.update_student_lists()
    
    def update_student_lists(self):
        """Update the available and enrolled student lists."""
        # Clear lists
        self.available_list.clear()
        self.enrolled_list.clear()
        
        # Add available students
        for student in self.available_students:
            item = QListWidgetItem(f"{student['id']} - {student['name']}")
            item.setData(Qt.UserRole, student['id'])
            self.available_list.addItem(item)
        
        # Add enrolled students
        for student in self.enrolled_students:
            item = QListWidgetItem(f"{student['id']} - {student['name']}")
            item.setData(Qt.UserRole, student['id'])
            self.enrolled_list.addItem(item)
    
    def add_students(self):
        """Add selected students to the class."""
        selected_items = self.available_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            student_id = item.data(Qt.UserRole)
            student = next((s for s in self.available_students if s['id'] == student_id), None)
            if student:
                self.enrolled_students.append(student)
                self.available_students.remove(student)
        
        self.update_student_lists()
    
    def remove_students(self):
        """Remove selected students from the class."""
        selected_items = self.enrolled_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            student_id = item.data(Qt.UserRole)
            student = next((s for s in self.enrolled_students if s['id'] == student_id), None)
            if student:
                self.available_students.append(student)
                self.enrolled_students.remove(student)
        
        self.update_student_lists()
    
    def populate_form(self):
        """Populate form with existing class data."""
        if not self.class_data:
            return
        
        self.id_input.setText(str(self.class_data['id']))
        self.name_input.setText(self.class_data['name'])
        self.subject_input.setText(self.class_data['subject'])
        self.location_input.setText(self.class_data['location'])
        
        # Load schedule
        schedule = eval(self.class_data['schedule'])
        if 'start_time' in schedule:
            try:
                start_time = QDateTime.fromString(
                    schedule['start_time'], 
                    "yyyy-MM-dd hh:mm:ss"
                )
                self.start_time_edit.setDateTime(start_time)
            except:
                pass
        
        if 'end_time' in schedule:
            try:
                end_time = QDateTime.fromString(
                    schedule['end_time'], 
                    "yyyy-MM-dd hh:mm:ss"
                )
                self.end_time_edit.setDateTime(end_time)
            except:
                pass
    
    def validate_inputs(self):
        """Validate form inputs."""
        class_id = self.id_input.text().strip()
        name = self.name_input.text().strip()
        
        if not class_id:
            QMessageBox.warning(self, "Validation Error", "Class ID is required.")
            return False
        
        if not name:
            QMessageBox.warning(self, "Validation Error", "Class name is required.")
            return False
        
        # Check if class ID already exists (for new classes)
        if not self.class_id:
            existing_class = self.database.get_class(class_id)
            if existing_class:
                QMessageBox.warning(
                    self, "Validation Error", 
                    f"Class ID '{class_id}' already exists. Please use a different ID."
                )
                return False
        
        # Validate schedule
        start_time = self.start_time_edit.dateTime()
        end_time = self.end_time_edit.dateTime()
        
        if start_time >= end_time:
            QMessageBox.warning(
                self, "Validation Error", 
                "End time must be after start time."
            )
            return False
        
        return True
    
    def accept(self):
        """Handle dialog acceptance."""
        if not self.validate_inputs():
            return
        
        class_id = self.id_input.text().strip()
        name = self.name_input.text().strip()
        subject = self.subject_input.text().strip()
        room = self.location_input.text().strip()
        
        # Create schedule dictionary
        schedule = {
            'start_time': self.start_time_edit.dateTime().toString("yyyy-MM-dd hh:mm:ss"),
            'end_time': self.end_time_edit.dateTime().toString("yyyy-MM-dd hh:mm:ss")
        }
        
        try:
            # Add or update class
            if self.class_id:
                # Update existing class
                success = self.database.update_class(
                    class_id, name, subject, room, str(schedule)
                )
                if not success:
                    QMessageBox.warning(
                        self, "Update Error", 
                        "Failed to update class. Please try again."
                    )
                    return
            else:
                # Add new class
                success = self.database.add_class(
                    class_id, name, subject, room, schedule
                )
                if not success:
                    QMessageBox.warning(
                        self, "Creation Error", 
                        "Failed to create class. Please try again."
                    )
                    return
            
            # Update enrollments
            # First, get current enrollments
            current_enrollments = []
            if self.class_id:
                enrolled = self.database.get_enrolled_students(class_id)
                current_enrollments = [e['student_id'] for e in enrolled]
            
            # Determine students to add and remove
            new_enrollments = [s['id'] for s in self.enrolled_students]
            
            students_to_add = [sid for sid in new_enrollments if sid not in current_enrollments]
            students_to_remove = [sid for sid in current_enrollments if sid not in new_enrollments]
            
            # Add new enrollments
            for student_id in students_to_add:
                self.database.enroll_student(class_id, student_id)
            
            # Remove old enrollments
            for student_id in students_to_remove:
                self.database.unenroll_student(class_id, student_id)
            
            # Get updated class data
            class_data = self.database.get_class(class_id)
            
            # Emit signal with class data
            self.class_updated.emit(class_data)
            
            QMessageBox.information(
                self, "Success", 
                f"Class '{name}' has been {'updated' if self.class_id else 'created'} successfully."
            )
            
            super().accept()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"An error occurred: {str(e)}"
            )
    
    def reject(self):
        """Handle dialog rejection."""
        super().reject()
