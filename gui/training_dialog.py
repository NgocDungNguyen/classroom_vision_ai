import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                            QComboBox, QLineEdit, QFileDialog, QProgressBar, QMessageBox,
                            QTabWidget, QWidget, QListWidget, QListWidgetItem, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from models.action_recognition import ActionRecognizer

class TrainingThread(QThread):
    progress_updated = pyqtSignal(int)
    training_complete = pyqtSignal(bool, str)
    
    def __init__(self, action_recognizer, sequences, labels):
        super().__init__()
        self.action_recognizer = action_recognizer
        self.sequences = sequences
        self.labels = labels
        
    def run(self):
        try:
            # Train the model
            history = self.action_recognizer.train_model(self.sequences, self.labels)
            
            # Emit completion signal
            self.training_complete.emit(True, "Training completed successfully!")
        except Exception as e:
            self.training_complete.emit(False, f"Error during training: {str(e)}")
            
class ActionTrainingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Action Recognition Training")
        self.setMinimumSize(800, 600)
        
        # Initialize action recognizer
        self.action_recognizer = ActionRecognizer()
        self.training_data = {}  # {action_name: {behavior_type: [video_paths]}}
        
        # Create UI
        self.init_ui()
        
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        
        # Create tabs
        self.tab_widget = QTabWidget()
        self.tab_actions = QWidget()
        self.tab_training = QWidget()
        self.tab_testing = QWidget()
        
        self.tab_widget.addTab(self.tab_actions, "Actions")
        self.tab_widget.addTab(self.tab_training, "Training")
        self.tab_widget.addTab(self.tab_testing, "Testing")
        
        # Setup each tab
        self.setup_actions_tab()
        self.setup_training_tab()
        self.setup_testing_tab()
        
        main_layout.addWidget(self.tab_widget)
        
        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        main_layout.addWidget(close_button)
        
        self.setLayout(main_layout)
        
    def setup_actions_tab(self):
        layout = QVBoxLayout()
        
        # Action management section
        action_section = QGridLayout()
        
        # Action name input
        action_section.addWidget(QLabel("Action Name:"), 0, 0)
        self.action_name_input = QLineEdit()
        action_section.addWidget(self.action_name_input, 0, 1)
        
        # Behavior type selection
        action_section.addWidget(QLabel("Behavior Type:"), 1, 0)
        self.behavior_type_combo = QComboBox()
        self.behavior_type_combo.addItems(["good", "bad"])
        action_section.addWidget(self.behavior_type_combo, 1, 1)
        
        # Add action button
        add_action_btn = QPushButton("Add Action")
        add_action_btn.clicked.connect(self.add_action)
        action_section.addWidget(add_action_btn, 2, 0, 1, 2)
        
        layout.addLayout(action_section)
        
        # Action list
        layout.addWidget(QLabel("Defined Actions:"))
        self.action_list = QListWidget()
        layout.addWidget(self.action_list)
        
        # Remove action button
        remove_action_btn = QPushButton("Remove Selected Action")
        remove_action_btn.clicked.connect(self.remove_action)
        layout.addWidget(remove_action_btn)
        
        self.tab_actions.setLayout(layout)
        
    def setup_training_tab(self):
        layout = QVBoxLayout()
        
        # Action selection for training
        training_section = QGridLayout()
        
        training_section.addWidget(QLabel("Select Action:"), 0, 0)
        self.training_action_combo = QComboBox()
        training_section.addWidget(self.training_action_combo, 0, 1)
        
        training_section.addWidget(QLabel("Behavior Type:"), 1, 0)
        self.training_behavior_combo = QComboBox()
        self.training_behavior_combo.addItems(["good", "bad"])
        training_section.addWidget(self.training_behavior_combo, 1, 1)
        
        # Add training data button
        add_training_btn = QPushButton("Add Training Video")
        add_training_btn.clicked.connect(self.add_training_data)
        training_section.addWidget(add_training_btn, 2, 0, 1, 2)
        
        layout.addLayout(training_section)
        
        # Training data list
        layout.addWidget(QLabel("Training Data:"))
        self.training_data_list = QListWidget()
        layout.addWidget(self.training_data_list)
        
        # Remove training data button
        remove_training_btn = QPushButton("Remove Selected Training Data")
        remove_training_btn.clicked.connect(self.remove_training_data)
        layout.addWidget(remove_training_btn)
        
        # Training progress
        layout.addWidget(QLabel("Training Progress:"))
        self.training_progress = QProgressBar()
        layout.addWidget(self.training_progress)
        
        # Start training button
        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.clicked.connect(self.start_training)
        layout.addWidget(self.start_training_btn)
        
        self.tab_training.setLayout(layout)
        
    def setup_testing_tab(self):
        layout = QVBoxLayout()
        
        # Video preview section
        preview_layout = QHBoxLayout()
        
        # Video display
        self.video_preview = QLabel("No video selected")
        self.video_preview.setAlignment(Qt.AlignCenter)
        self.video_preview.setMinimumSize(400, 300)
        self.video_preview.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        preview_layout.addWidget(self.video_preview)
        
        # Controls
        controls_layout = QVBoxLayout()
        
        # Select test video button
        select_test_btn = QPushButton("Select Test Video")
        select_test_btn.clicked.connect(self.select_test_video)
        controls_layout.addWidget(select_test_btn)
        
        # Test video button
        self.test_video_btn = QPushButton("Test Video")
        self.test_video_btn.clicked.connect(self.test_video)
        self.test_video_btn.setEnabled(False)
        controls_layout.addWidget(self.test_video_btn)
        
        # Results label
        self.results_label = QLabel("Results will appear here")
        controls_layout.addWidget(self.results_label)
        
        preview_layout.addLayout(controls_layout)
        layout.addLayout(preview_layout)
        
        self.tab_testing.setLayout(layout)
        
    def add_action(self):
        action_name = self.action_name_input.text().strip()
        if not action_name:
            QMessageBox.warning(self, "Input Error", "Please enter an action name")
            return
            
        behavior_type = self.behavior_type_combo.currentText()
        action_id = f"{action_name}_{behavior_type}"
        
        # Check if action already exists
        for i in range(self.action_list.count()):
            if self.action_list.item(i).text() == action_id:
                QMessageBox.warning(self, "Duplicate Action", "This action already exists")
                return
                
        # Add to list widget
        self.action_list.addItem(action_id)
        
        # Add to action recognizer
        if action_id not in self.action_recognizer.actions:
            self.action_recognizer.actions.append(action_id)
            
        # Add to training action combo
        if action_name not in [self.training_action_combo.itemText(i) for i in range(self.training_action_combo.count())]:
            self.training_action_combo.addItem(action_name)
            
        # Initialize training data structure
        if action_name not in self.training_data:
            self.training_data[action_name] = {"good": [], "bad": []}
            
        # Clear input
        self.action_name_input.clear()
        
    def remove_action(self):
        selected_items = self.action_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "Please select an action to remove")
            return
            
        for item in selected_items:
            action_id = item.text()
            action_name = action_id.split("_")[0]
            
            # Remove from list widget
            self.action_list.takeItem(self.action_list.row(item))
            
            # Remove from action recognizer
            if action_id in self.action_recognizer.actions:
                self.action_recognizer.actions.remove(action_id)
                
            # Check if all behaviors of this action are removed
            remaining_behaviors = False
            for i in range(self.action_list.count()):
                if self.action_list.item(i).text().startswith(f"{action_name}_"):
                    remaining_behaviors = True
                    break
                    
            # If no behaviors left, remove from training combo and data
            if not remaining_behaviors:
                for i in range(self.training_action_combo.count()):
                    if self.training_action_combo.itemText(i) == action_name:
                        self.training_action_combo.removeItem(i)
                        break
                        
                if action_name in self.training_data:
                    del self.training_data[action_name]
                    
    def add_training_data(self):
        if self.training_action_combo.count() == 0:
            QMessageBox.warning(self, "No Actions", "Please add actions first")
            return
            
        action_name = self.training_action_combo.currentText()
        behavior_type = self.training_behavior_combo.currentText()
        
        # Open file dialog to select video
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Training Video", "", "Video Files (*.mp4 *.avi *.mov *.wmv)"
        )
        
        if file_path:
            # Add to training data
            self.training_data[action_name][behavior_type].append(file_path)
            
            # Add to list widget
            item_text = f"{action_name} ({behavior_type}): {os.path.basename(file_path)}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, file_path)
            self.training_data_list.addItem(item)
            
    def remove_training_data(self):
        selected_items = self.training_data_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "Please select training data to remove")
            return
            
        for item in selected_items:
            file_path = item.data(Qt.UserRole)
            item_text = item.text()
            
            # Parse action and behavior from item text
            parts = item_text.split("(")
            action_name = parts[0].strip()
            behavior_type = parts[1].split(")")[0].strip()
            
            # Remove from training data
            if action_name in self.training_data and behavior_type in self.training_data[action_name]:
                if file_path in self.training_data[action_name][behavior_type]:
                    self.training_data[action_name][behavior_type].remove(file_path)
                    
            # Remove from list widget
            self.training_data_list.takeItem(self.training_data_list.row(item))
            
    def start_training(self):
        # Check if we have training data
        has_data = False
        for action in self.training_data:
            for behavior in self.training_data[action]:
                if self.training_data[action][behavior]:
                    has_data = True
                    break
            if has_data:
                break
                
        if not has_data:
            QMessageBox.warning(self, "No Training Data", "Please add training data first")
            return
            
        # Prepare sequences and labels
        all_sequences = []
        all_labels = []
        
        # Disable UI during training
        self.start_training_btn.setEnabled(False)
        self.training_progress.setValue(0)
        
        try:
            # Process each video and extract sequences
            for action in self.training_data:
                for behavior in self.training_data[action]:
                    for video_path in self.training_data[action][behavior]:
                        sequences, labels = self.action_recognizer.collect_training_data(
                            video_path, action, behavior
                        )
                        all_sequences.extend(sequences)
                        all_labels.extend(labels)
                        
            if not all_sequences:
                QMessageBox.warning(self, "Training Error", "Could not extract valid sequences from videos")
                self.start_training_btn.setEnabled(True)
                return
                
            # Start training thread
            self.training_thread = TrainingThread(self.action_recognizer, all_sequences, all_labels)
            self.training_thread.progress_updated.connect(self.update_training_progress)
            self.training_thread.training_complete.connect(self.training_completed)
            self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"Error preparing training data: {str(e)}")
            self.start_training_btn.setEnabled(True)
            
    def update_training_progress(self, value):
        self.training_progress.setValue(value)
        
    def training_completed(self, success, message):
        self.start_training_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Training Complete", message)
            self.training_progress.setValue(100)
        else:
            QMessageBox.critical(self, "Training Error", message)
            self.training_progress.setValue(0)
            
    def select_test_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Test Video", "", "Video Files (*.mp4 *.avi *.mov *.wmv)"
        )
        
        if file_path:
            self.test_video_path = file_path
            self.test_video_btn.setEnabled(True)
            
            # Display first frame
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_preview.setPixmap(QPixmap.fromImage(q_img).scaled(
                    self.video_preview.width(), self.video_preview.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
            cap.release()
            
    def test_video(self):
        if not hasattr(self, 'test_video_path'):
            QMessageBox.warning(self, "No Video", "Please select a test video first")
            return
            
        if not os.path.exists(self.action_recognizer.model_path):
            QMessageBox.warning(self, "No Model", "Please train a model first")
            return
            
        try:
            # Load the model if not already loaded
            if not self.action_recognizer.model:
                self.action_recognizer.model = load_model(self.action_recognizer.model_path)
                
            # Process the video
            cap = cv2.VideoCapture(self.test_video_path)
            sequences = []
            current_sequence = []
            
            while cap.isOpened() and len(current_sequence) < self.action_recognizer.sequence_length:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                keypoints = self.action_recognizer.extract_keypoints(frame)
                current_sequence.append(keypoints)
                
            cap.release()
            
            if len(current_sequence) == self.action_recognizer.sequence_length:
                action, confidence = self.action_recognizer.predict_action(np.array(current_sequence))
                self.results_label.setText(f"Predicted: {action}\nConfidence: {confidence:.2f}")
            else:
                self.results_label.setText("Could not extract enough frames for prediction")
                
        except Exception as e:
            QMessageBox.critical(self, "Testing Error", f"Error testing video: {str(e)}")
            self.results_label.setText(f"Error: {str(e)}")
