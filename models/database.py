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

        # Drop existing tables to ensure schema consistency
        cursor.execute("DROP TABLE IF EXISTS behaviors")
        cursor.execute("DROP TABLE IF EXISTS class_students")
        cursor.execute("DROP TABLE IF EXISTS attendance")
        cursor.execute("DROP TABLE IF EXISTS classes")
        cursor.execute("DROP TABLE IF EXISTS students")

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

        # Create classes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS classes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                subject TEXT NOT NULL,
                room TEXT NOT NULL,
                schedule TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create class_students table (many-to-many relationship)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS class_students (
                class_id TEXT,
                student_id TEXT,
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (class_id, student_id),
                FOREIGN KEY (class_id) REFERENCES classes(id),
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)

        # Create attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                class_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id),
                FOREIGN KEY (class_id) REFERENCES classes(id)
            )
        """)
        
        # Create behaviors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS behaviors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                class_id TEXT,
                behavior_type TEXT NOT NULL,
                confidence REAL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration REAL,
                FOREIGN KEY (student_id) REFERENCES students(id),
                FOREIGN KEY (class_id) REFERENCES classes(id)
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

    def record_attendance(self, student_id, class_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Check if attendance already recorded in last 5 minutes
            cursor.execute(
                """
                SELECT COUNT(*) FROM attendance 
                WHERE student_id = ? AND class_id = ? 
                AND timestamp > datetime('now', '-5 minutes')
                """,
                (student_id, class_id)
            )
            recent_count = cursor.fetchone()[0]

            if recent_count == 0:
                cursor.execute(
                    """
                    INSERT INTO attendance (student_id, class_id, timestamp)
                    VALUES (?, ?, datetime('now'))
                    """,
                    (student_id, class_id)
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

    def add_class(self, class_id, name, subject, room, schedule):
        """Add a new class to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                INSERT INTO classes (id, name, subject, room, schedule)
                VALUES (?, ?, ?, ?, ?)
                """,
                (class_id, name, subject, room, str(schedule))
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False
        finally:
            conn.close()

    def get_class(self, class_id):
        """Get class information by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT id, name, subject, room, schedule, created_at
                FROM classes
                WHERE id = ?
                """,
                (class_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'subject': row[2],
                    'room': row[3],
                    'schedule': row[4],
                    'created_at': row[5]
                }
            return None
        finally:
            conn.close()

    def get_classes(self):
        """Get all classes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT id, name, subject, room, schedule, created_at
                FROM classes
                ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()
            
            return [{
                'id': row[0],
                'name': row[1],
                'subject': row[2],
                'room': row[3],
                'schedule': row[4],
                'created_at': row[5]
            } for row in rows]
        finally:
            conn.close()

    def update_class(self, class_id, name, subject, room, schedule):
        """Update an existing class."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                UPDATE classes
                SET name = ?, subject = ?, room = ?, schedule = ?
                WHERE id = ?
                """,
                (name, subject, room, schedule, class_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False
        finally:
            conn.close()

    def delete_class(self, class_id):
        """Delete a class and its enrollments."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete enrollments first
            cursor.execute(
                """
                DELETE FROM class_students
                WHERE class_id = ?
                """,
                (class_id,)
            )
            
            # Delete class
            cursor.execute(
                """
                DELETE FROM classes
                WHERE id = ?
                """,
                (class_id,)
            )
            
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False
        finally:
            conn.close()

    def enroll_student(self, class_id, student_id):
        """Enroll a student in a class."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                INSERT INTO class_students (class_id, student_id)
                VALUES (?, ?)
                """,
                (class_id, student_id)
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False
        finally:
            conn.close()

    def unenroll_student(self, class_id, student_id):
        """Remove a student from a class."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                DELETE FROM class_students
                WHERE class_id = ? AND student_id = ?
                """,
                (class_id, student_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False
        finally:
            conn.close()

    def get_enrolled_students(self, class_id):
        """Get all students enrolled in a class."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT s.id, s.name, s.class_name, cs.joined_at
                FROM students s
                JOIN class_students cs ON s.id = cs.student_id
                WHERE cs.class_id = ?
                ORDER BY cs.joined_at DESC
                """,
                (class_id,)
            )
            rows = cursor.fetchall()
            
            return [{
                'id': row[0],
                'name': row[1],
                'class_name': row[2],
                'joined_at': row[3]
            } for row in rows]
        finally:
            conn.close()

    def get_student_classes(self, student_id):
        """Get all classes a student is enrolled in."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT c.id, c.name, c.subject, c.room, c.schedule, cs.joined_at
                FROM classes c
                JOIN class_students cs ON c.id = cs.class_id
                WHERE cs.student_id = ?
                ORDER BY cs.joined_at DESC
                """,
                (student_id,)
            )
            rows = cursor.fetchall()
            
            return [{
                'id': row[0],
                'name': row[1],
                'subject': row[2],
                'room': row[3],
                'schedule': row[4],
                'joined_at': row[5]
            } for row in rows]
        finally:
            conn.close()

    def save_attendance(self, attendance_data):
        """Save attendance data for a class session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for student_id in attendance_data['students']:
                check_in_time = attendance_data['check_in_times'].get(
                    student_id, attendance_data['date']
                )
                cursor.execute(
                    """
                    INSERT INTO attendance 
                    (student_id, class_id, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (student_id, attendance_data['class_id'], check_in_time)
                )
            conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            conn.close()
            
    def get_attendance_data(self, start_date, end_date):
        """Get attendance data for a date range."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT s.name, a.timestamp, c.name as class_name
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            JOIN class_students cs ON s.id = cs.student_id
            JOIN classes c ON cs.class_id = c.id
            WHERE date(a.timestamp) BETWEEN date(?) AND date(?)
            ORDER BY a.timestamp
            """,
            (start_date, end_date)
        )
        records = cursor.fetchall()
        conn.close()
        
        return [{
            'student_name': r[0],
            'timestamp': r[1],
            'class_name': r[2]
        } for r in records]
        
    def record_behavior(self, student_id, class_id, behavior_type, 
                       confidence, start_time, end_time, duration):
        """Record a student behavior."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                INSERT INTO behaviors 
                (student_id, class_id, behavior_type, confidence,
                 start_time, end_time, duration)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (student_id, class_id, behavior_type, confidence,
                 start_time, end_time, duration)
            )
            conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            conn.close()
            
    def get_behavior_data(self, start_date, end_date):
        """Get behavior data for a date range."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT s.name, b.behavior_type, b.confidence,
                   b.start_time, b.end_time, b.duration,
                   c.name as class_name
            FROM behaviors b
            JOIN students s ON b.student_id = s.id
            JOIN classes c ON b.class_id = c.id
            WHERE date(b.start_time) BETWEEN date(?) AND date(?)
            ORDER BY b.start_time
            """,
            (start_date, end_date)
        )
        records = cursor.fetchall()
        conn.close()
        
        return [{
            'student_name': r[0],
            'behavior_type': r[1],
            'confidence': r[2],
            'start_time': r[3],
            'end_time': r[4],
            'duration': r[5],
            'class_name': r[6]
        } for r in records]
        
    def get_student_name(self, student_id):
        """Get a student's name by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name FROM students WHERE id = ?",
            (student_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else "Unknown Student"

    def get_enrolled_students(self, class_id):
        """Get all students enrolled in a class."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT s.id as student_id, s.name, cs.joined_at
            FROM students s
            JOIN class_students cs ON s.id = cs.student_id
            WHERE cs.class_id = ?
            """,
            (class_id,)
        )
        students = cursor.fetchall()
        conn.close()
        
        return [{
            'student_id': s[0],
            'name': s[1],
            'joined_at': s[2]
        } for s in students]
        
    def enroll_student(self, class_id, student_id):
        """Enroll a student in a class."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                INSERT INTO class_students (class_id, student_id)
                VALUES (?, ?)
                """,
                (class_id, student_id)
            )
            conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            conn.close()
            
    def unenroll_student(self, class_id, student_id):
        """Remove a student from a class."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                DELETE FROM class_students 
                WHERE class_id = ? AND student_id = ?
                """,
                (class_id, student_id)
            )
            conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            conn.close()
