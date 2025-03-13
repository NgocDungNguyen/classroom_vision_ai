import face_recognition
import numpy as np
from typing import List, Tuple, Optional
import cv2


class FaceDetector:
    """Face detection and recognition using face_recognition library."""

    def __init__(self):
        """Initialize face detector."""
        self.known_face_encodings = []
        self.known_face_ids = []

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find faces
        face_locations = face_recognition.face_locations(rgb_frame)
        return face_locations

    def compute_face_encoding(
        self,
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Compute face encoding for recognition.

        Args:
            frame: Input frame
            face_location: Face location tuple

        Returns:
            Face encoding as numpy array, or None if failed
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get face encoding
            face_encodings = face_recognition.face_encodings(
                rgb_frame,
                [face_location]
            )

            if face_encodings:
                return face_encodings[0]
            return None
        except Exception:
            return None

    def add_known_face(
        self,
        face_encoding: np.ndarray,
        student_id: str
    ) -> None:
        """
        Add a known face for recognition.

        Args:
            face_encoding: Face encoding to add
            student_id: Student ID to associate with face
        """
        self.known_face_encodings.append(face_encoding)
        self.known_face_ids.append(student_id)

    def recognize_face(
        self,
        face_encoding: np.ndarray
    ) -> Optional[str]:
        """
        Recognize a face from known faces.

        Args:
            face_encoding: Face encoding to recognize

        Returns:
            Student ID if recognized, None otherwise
        """
        if not self.known_face_encodings:
            return None

        # Compare with known faces
        matches = face_recognition.compare_faces(
            self.known_face_encodings,
            face_encoding
        )

        if True in matches:
            match_index = matches.index(True)
            return self.known_face_ids[match_index]
        return None
import cv2
import dlib
import numpy as np
from pathlib import Path
from typing import List, Tuple


class FaceDetector:
    """Face detection and recognition using dlib."""

    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(
            "data/models/shape_predictor_68_face_landmarks.dat"
        )
        self.face_encoder = dlib.face_recognition_model_v1(
            "data/models/dlib_face_recognition_resnet_model_v1.dat"
        )
        self.known_faces = {}

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame.
        
        Args:
            frame: RGB frame from camera
            
        Returns:
            List of face rectangles (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        face_rects = []
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            face_rects.append((x, y, w, h))
        
        return face_rects

    def encode_face(self, frame: np.ndarray, face_rect: tuple) -> np.ndarray:
        """Generate encoding for a detected face.
        
        Args:
            frame: RGB frame from camera
            face_rect: Face rectangle (x, y, w, h)
            
        Returns:
            128-dimensional face encoding
        """
        x, y, w, h = face_rect
        face = dlib.rectangle(x, y, x + w, y + h)
        
        # Get facial landmarks
        shape = self.shape_predictor(frame, face)
        
        # Generate face encoding
        face_encoding = np.array(
            self.face_encoder.compute_face_descriptor(frame, shape)
        )
        
        return face_encoding

    def compare_faces(
        self, face_encoding: np.ndarray, tolerance: float = 0.6
    ) -> str:
        """Compare a face encoding with known faces.
        
        Args:
            face_encoding: Face encoding to compare
            tolerance: Maximum distance for a match
            
        Returns:
            Student ID if match found, else None
        """
        if not self.known_faces:
            return None
            
        for student_id, encodings in self.known_faces.items():
            for known_encoding in encodings:
                distance = np.linalg.norm(face_encoding - known_encoding)
                if distance < tolerance:
                    return student_id
        
        return None

    def add_known_face(
        self, student_id: str, face_encoding: np.ndarray
    ) -> None:
        """Add a known face encoding for a student.
        
        Args:
            student_id: Student identifier
            face_encoding: Face encoding to add
        """
        if student_id in self.known_faces:
            self.known_faces[student_id].append(face_encoding)
        else:
            self.known_faces[student_id] = [face_encoding]

    def load_known_faces(self, faces_dir: str) -> None:
        """Load known faces from directory.
        
        Args:
            faces_dir: Directory containing face images
        """
        faces_path = Path(faces_dir)
        if not faces_path.exists():
            return
            
        for student_dir in faces_path.iterdir():
            if student_dir.is_dir():
                student_id = student_dir.name
                for face_file in student_dir.glob("*.jpg"):
                    frame = cv2.imread(str(face_file))
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        faces = self.detect_faces(frame)
                        if faces:
                            encoding = self.encode_face(frame, faces[0])
                            self.add_known_face(student_id, encoding)
