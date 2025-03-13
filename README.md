# Classroom Vision AI System

A real-time student attendance tracking system using computer vision and facial recognition technology.

## Features

- **Student Registration**
  - Register students with facial recognition
  - Store student information and facial data
  - Multiple angle face capture for better accuracy

- **Attendance Tracking**
  - Automated face recognition-based attendance
  - Real-time attendance monitoring
  - Efficient check-in tracking with 5-minute cooldown

- **User Interface**
  - Modern PyQt5-based interface
  - Live camera feed display
  - Multiple tabs for different functions
  - Student and attendance record viewing

## Requirements

- Python 3.8+
- Webcam or USB camera
- Required Python packages:
  - opencv-python==4.8.0.76
  - dlib==19.24.2
  - numpy==1.26.0
  - PyQt5==5.15.9
  - pathlib==1.0.1
  - sqlite3==3.42.0

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd classroom_vision_ai
```
1. Create a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate
```
1. Install required packages:
```bash
pip install -r requirements.txt
```
1. Download required model files:
- Download the shape predictor from:
  [Shape Predictor Model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Extract and place it in `data/models/shape_predictor_68_face_landmarks.dat`
- Download face recognition model from:
  [Face Recognition Model](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)
- Place it in `data/models/dlib_face_recognition_resnet_model_v1.dat`

## Usage

1. Start the application:
```bash
python main.py
```
1. Register Students:
   - Click "Register Student" button
   - Enter student ID, name, and class
   - Follow the face capture process
   - Click "Register" to save
1. Track Attendance:
   - Click "Take Attendance" to begin tracking
   - The system will automatically recognize students
   - Attendance is recorded with a 5-minute cooldown
   - Click "Stop Attendance" when done

## Project Structure
```plaintext
classroom_vision_ai/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── gui/                   # GUI components
│   ├── main_window.py     # Main application window
│   └── registration_dialog.py  # Student registration dialog
├── models/                # Core functionality
│   ├── face_detector.py   # Face detection and recognition
│   └── database.py        # Database management
├── data/                  # Data storage
│   ├── models/           # AI model files
│   └── classroom.db      # SQLite database
└── faces/                # Student face images storage

## Notes
- The system runs locally and does not require internet connectivity
- All data is stored in a local SQLite database
- Face images are stored in the faces/ directory
- Face recognition uses dlib's state-of-the-art models

## Privacy and Security
- All data is stored locally on your machine
- No data is transmitted to external servers
- Face images are stored securely in a local directory
- The system does not store raw video footage
