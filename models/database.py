import sqlite3
import numpy as np
from datetime import datetime


class Database:
    def __init__(self, db_path="classroom.db"):
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create students table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                class_name TEXT NOT NULL,
                face_encoding BLOB,
                face_image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)

        conn.commit()
        conn.close()

    def add_student(self, student_id, name, face_encoding, 
                   face_image_path, class_name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO students 
                (id, name, class_name, face_encoding, face_image_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, name, class_name, face_encoding, face_image_path)
            )
            conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            conn.close()

    def get_student(self, student_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM students WHERE id = ?",
            (student_id,)
        )
        student = cursor.fetchone()
        conn.close()

        if student:
            return {
                'id': student[0],
                'name': student[1],
                'class_name': student[2],
                'face_encoding': student[3],
                'face_image_path': student[4],
                'created_at': student[5]
            }
        return None

    def get_all_students(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM students")
        students = cursor.fetchall()
        conn.close()

        return [{
            'id': s[0],
            'name': s[1],
            'class_name': s[2],
            'face_encoding': s[3],
            'face_image_path': s[4],
            'created_at': s[5]
        } for s in students]

    def identify_student(self, face_encoding):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, face_encoding FROM students")
        students = cursor.fetchall()
        conn.close()

        if not students:
            return None

        min_distance = float('inf')
        matched_student = None

        test_encoding = np.frombuffer(face_encoding, dtype=np.float64)
        for student_id, stored_encoding in students:
            if stored_encoding is None:
                continue
            
            stored = np.frombuffer(stored_encoding, dtype=np.float64)
            distance = np.linalg.norm(test_encoding - stored)
            
            if distance < min_distance and distance < 0.6:
                min_distance = distance
                matched_student = self.get_student(student_id)

        return matched_student

    def record_attendance(self, student_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Check if attendance already recorded in last 5 minutes
            cursor.execute(
                """
                SELECT COUNT(*) FROM attendance 
                WHERE student_id = ? 
                AND timestamp > datetime('now', '-5 minutes')
                """,
                (student_id,)
            )
            recent_count = cursor.fetchone()[0]

            if recent_count == 0:
                cursor.execute(
                    """
                    INSERT INTO attendance (student_id, timestamp)
                    VALUES (?, datetime('now'))
                    """,
                    (student_id,)
                )
                conn.commit()
                return True
            return False
        finally:
            conn.close()

    def get_attendance_records(self, date=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if date:
            cursor.execute(
                """
                SELECT s.id, s.name, s.class_name, a.timestamp
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                WHERE date(a.timestamp) = date(?)
                ORDER BY a.timestamp DESC
                """,
                (date,)
            )
        else:
            cursor.execute(
                """
                SELECT s.id, s.name, s.class_name, a.timestamp
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                ORDER BY a.timestamp DESC
                """
            )

        records = cursor.fetchall()
        conn.close()

        return [{
            'student_id': r[0],
            'name': r[1],
            'class_name': r[2],
            'timestamp': r[3]
        } for r in records]
