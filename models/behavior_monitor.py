import cv2
import numpy as np
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple


class BehaviorType(Enum):
    ATTENTIVE = "attentive"
    INATTENTIVE = "inattentive"
    HAND_RAISED = "hand_raised"
    SLEEPING = "sleeping"
    USING_PHONE = "using_phone"


class BehaviorMonitor:
    """Monitor and analyze student behaviors in real-time using OpenCV."""
    
    def __init__(self):
        """Initialize behavior monitoring."""
        self.current_behaviors = {}  # student_id -> (behavior_type, start_time, confidence)
        self.behavior_history = {}   # student_id -> list of behaviors
        self.active_class_id = None
        
        # Load Haar cascades for feature detection
        cascade_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            f"{cascade_dir}/haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            f"{cascade_dir}/haarcascade_eye.xml"
        )
        
        # Initialize behavior tracking
        self.prev_head_pos = {}  # student_id -> (x, y)
        self.head_movement_threshold = 30
        self.eye_aspect_ratio_threshold = 0.2
        self.behavior_duration_threshold = 3.0  # seconds
        
    def set_active_class(self, class_id: str):
        """Set the active class for behavior monitoring."""
        self.active_class_id = class_id
        
    def analyze_frame(self, frame, recognized_students: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
        """
        Analyze a frame to detect student behaviors.
        
        Args:
            frame: Video frame to analyze
            recognized_students: List of recognized student dicts with 'id' and 'face_location'
            
        Returns:
            Tuple of (behavior_list, annotated_frame)
        """
        if not self.active_class_id:
            return [], frame
            
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annotated_frame = frame.copy()
        behaviors = []
        current_time = datetime.now()
        
        for student in recognized_students:
            student_id = student['id']
            face_loc = student['face_location']
            
            if not face_loc:
                continue
                
            # Extract face region
            top, right, bottom, left = face_loc
            face_roi = gray[top:bottom, left:right]
            
            # Detect eyes in face region
            eyes = self.eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            
            # Analyze behavior based on features
            behavior = self._analyze_behavior(
                student_id, face_loc, eyes, current_time
            )
            
            if behavior:
                behaviors.append(behavior)
                self._draw_behavior_indicator(
                    annotated_frame,
                    face_loc,
                    behavior['type'],
                    behavior['confidence']
                )
                
        return behaviors, annotated_frame
        
    def _analyze_behavior(
        self,
        student_id: str,
        face_loc: Tuple[int, int, int, int],
        eyes: np.ndarray,
        current_time: datetime
    ) -> Dict:
        """Analyze student behavior based on detected features."""
        top, right, bottom, left = face_loc
        face_center = ((left + right) // 2, (top + bottom) // 2)
        
        # Check head movement
        if student_id in self.prev_head_pos:
            prev_x, prev_y = self.prev_head_pos[student_id]
            movement = np.sqrt(
                (face_center[0] - prev_x)**2 + 
                (face_center[1] - prev_y)**2
            )
            
            if movement > self.head_movement_threshold:
                behavior_type = BehaviorType.INATTENTIVE.value
                confidence = min(movement / 100, 0.9)
            else:
                behavior_type = BehaviorType.ATTENTIVE.value
                confidence = 0.8
        else:
            behavior_type = BehaviorType.ATTENTIVE.value
            confidence = 0.7
            
        # Update head position
        self.prev_head_pos[student_id] = face_center
        
        # Check for sleeping based on eye detection
        if len(eyes) == 0:
            behavior_type = BehaviorType.SLEEPING.value
            confidence = 0.85
            
        # Create behavior record
        behavior = {
            'student_id': student_id,
            'type': behavior_type,
            'confidence': confidence,
            'timestamp': current_time,
            'face_location': face_loc
        }
        
        return behavior
        
    def _draw_behavior_indicator(
        self,
        frame: np.ndarray,
        face_loc: Tuple[int, int, int, int],
        behavior_type: str,
        confidence: float
    ):
        """Draw behavior indicator on frame."""
        top, right, bottom, left = face_loc
        
        # Choose color based on behavior
        if behavior_type == BehaviorType.ATTENTIVE.value:
            color = (0, 255, 0)  # Green
        elif behavior_type == BehaviorType.SLEEPING.value:
            color = (255, 0, 0)  # Red
        else:
            color = (0, 0, 255)  # Blue
            
        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Add behavior label
        label = f"{behavior_type} ({confidence:.2f})"
        cv2.putText(
            frame,
            label,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
