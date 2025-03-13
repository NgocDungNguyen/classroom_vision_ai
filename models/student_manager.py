import os
import json
import csv
from typing import List, Dict, Optional, Tuple
import pandas as pd
from datetime import datetime

class StudentManager:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)

        self.data_dir = data_dir
        self.students_file = os.path.join(data_dir, 'students.json')
        self.classes_file = os.path.join(data_dir, 'classes.json')
        self.attendance_file = os.path.join(data_dir, 'attendance.json')
        self.students = self._load_data(self.students_file, {})
        self.classes = self._load_data(self.classes_file, {})
        self.attendance = self._load_data(self.attendance_file, {})

        self.attendance_log_file = os.path.join(data_dir, 'attendance_log.csv')
        self._ensure_attendance_log_exists()

    def _load_data(self, file_path: str, default: dict) -> dict:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {file_path}, creating new file")
                return default
        return default

    def _save_data(self, data: dict, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def _ensure_attendance_log_exists(self):
        if not os.path.exists(self.attendance_log_file):
            os.makedirs(os.path.dirname(self.attendance_log_file), exist_ok=True)
            with open(self.attendance_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'Time', 'Student', 'Status'])

    def add_student(self, student_id: str, name: str, class_ids: List[str] = None) -> bool:
        if student_id in self.students:
            return False

        self.students[student_id] = {
            'name': name,
            'class_ids': class_ids or [],
            'registration_date': datetime.now().isoformat()
        }
        self._save_data(self.students, self.students_file)
        return True

    def get_student_by_name(self, name: str) -> Optional[str]:
        for student_id, data in self.students.items():
            if data['name'].lower() == name.lower():
                return student_id
        return None

    def add_class(self, class_id: str, name: str, subject: str, schedule: Dict[str, str], room: str) -> bool:
        if class_id in self.classes:
            return False

        self.classes[class_id] = {
            'name': name,
            'subject': subject,
            'schedule': schedule,
            'room': room,
            'students': [],
            'creation_date': datetime.now().isoformat()
        }
        self._save_data(self.classes, self.classes_file)
        return True

    def add_student_to_class(self, student_id: str, class_id: str) -> bool:
        if student_id not in self.students or class_id not in self.classes or student_id in self.classes[class_id]['students']:
            return False

        self.classes[class_id]['students'].append(student_id)
        if class_id not in self.students[student_id]['class_ids']:
            self.students[student_id]['class_ids'].append(class_id)

        self._save_data(self.classes, self.classes_file)
        self._save_data(self.students, self.students_file)
        return True

    def record_attendance(self, student_name: str, status: str = "Present") -> bool:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        with open(self.attendance_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([date_str, time_str, student_name, status])

        student_id = self.get_student_by_name(student_name)
        if student_id and self.classes:
            class_id = next(iter(self.classes.keys()))

            if class_id not in self.attendance:
                self.attendance[class_id] = {}

            if date_str not in self.attendance[class_id]:
                self.attendance[class_id][date_str] = {}

            self.attendance[class_id][date_str][student_id] = {
                'status': status,
                'check_in_time': time_str
            }

            self._save_data(self.attendance, self.attendance_file)

        return True

    def get_attendance_records(self) -> List[Tuple[str, str, str]]:
        records = []

        if os.path.exists(self.attendance_log_file):
            with open(self.attendance_log_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 4:
                        date, time, student, status = row
                        records.append((student, time, status))

        return records

    def get_student_attendance(self, student_id: str, class_id: Optional[str] = None) -> Dict:
        attendance_records = {}

        if class_id:
            if class_id in self.attendance and student_id in self.students:
                attendance_records[class_id] = {
                    date: data[student_id]
                    for date, data in self.attendance[class_id].items()
                    if student_id in data
                }
        else:
            for class_id, class_data in self.attendance.items():
                attendance_records[class_id] = {
                    date: data[student_id]
                    for date, data in class_data.items()
                    if student_id in data
                }

        return attendance_records

    def get_class_attendance(self, class_id: str, date: Optional[str] = None) -> Dict:
        if class_id not in self.attendance:
            return {}

        if date:
            return self.attendance[class_id].get(date, {})
        return self.attendance[class_id]

    def generate_attendance_report(self, class_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        if class_id not in self.attendance:
            return pd.DataFrame()

        data = []
        for date, attendance_data in self.attendance[class_id].items():
            if start_date <= date <= end_date:
                for student_id, status in attendance_data.items():
                    student_name = self.students[student_id]['name']
                    data.append({
                        'Date': date,
                        'Student ID': student_id,
                        'Student Name': student_name,
                        'Status': status['status'],
                        'Check-in Time': status.get('check_in_time', 'N/A')
                    })

        return pd.DataFrame(data)

    def get_all_students(self) -> Dict[str, Dict]:
        return self.students

    def get_all_classes(self) -> Dict[str, Dict]:
        return self.classes

    def get_student(self, student_id: str) -> Optional[Dict]:
        return self.students.get(student_id)

    def get_class(self, class_id: str) -> Optional[Dict]:
        return self.classes.get(class_id)

    def update_student(self, student_id: str, name: str = None, class_ids: List[str] = None) -> bool:
        if student_id not in self.students:
            return False

        if name:
            self.students[student_id]['name'] = name
        if class_ids is not None:
            self.students[student_id]['class_ids'] = class_ids

        self._save_data(self.students, self.students_file)
        return True

    def update_class(self, class_id: str, name: str = None, subject: str = None, schedule: Dict[str, str] = None, room: str = None) -> bool:
        if class_id not in self.classes:
            return False

        if name:
            self.classes[class_id]['name'] = name
        if subject:
            self.classes[class_id]['subject'] = subject
        if schedule:
            self.classes[class_id]['schedule'] = schedule
        if room:
            self.classes[class_id]['room'] = room

        self._save_data(self.classes, self.classes_file)
        return True

    def remove_student(self, student_id: str) -> bool:
        if student_id not in self.students:
            return False

        for class_id in self.students[student_id]['class_ids']:
            if class_id in self.classes:
                if student_id in self.classes[class_id]['students']:
                    self.classes[class_id]['students'].remove(student_id)

        del self.students[student_id]

        self._save_data(self.students, self.students_file)
        self._save_data(self.classes, self.classes_file)
        return True

    def remove_class(self, class_id: str) -> bool:
        if class_id not in self.classes:
            return False

        for student_id in self.classes[class_id]['students']:
            if student_id in self.students:
                if class_id in self.students[student_id]['class_ids']:
                    self.students[student_id]['class_ids'].remove(class_id)

        del self.classes[class_id]

        self._save_data(self.students, self.students_file)
        self._save_data(self.classes, self.classes_file)
        return True
