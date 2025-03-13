import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class BehaviorTrainer:
    """Train behavior recognition models using labeled data."""

    def __init__(self):
        self.behaviors = {
            'attentive': 0,
            'inattentive': 1,
            'hand_raised': 2,
            'sleeping': 3,
            'using_phone': 4
        }
        self.label_encoder = LabelEncoder()
        self.training_data = []
        self.labels = []
        self.landmarks = []
        
        # Create directories
        self.data_dir = Path('data')
        self.models_dir = self.data_dir / 'models'
        self.training_dir = self.data_dir / 'training'
        self.analytics_dir = self.data_dir / 'analytics'
        
        for dir_path in [self.data_dir, self.models_dir, 
                        self.training_dir, self.analytics_dir]:
            dir_path.mkdir(exist_ok=True)

    def load_training_data(self, video_path):
        """Load a video for training and annotation.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoCapture object and total frames
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap, total_frames

    def save_annotation(self, frame, landmarks, behavior, timestamp):
        """Save annotated frame with landmarks and behavior label.
        
        Args:
            frame: Video frame
            landmarks: List of landmark points
            behavior: Behavior label
            timestamp: Frame timestamp
        """
        # Create unique filename
        filename = f"annotation_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.training_dir / filename
        
        # Save frame as image
        img_path = str(filepath).replace('.json', '.jpg')
        cv2.imwrite(img_path, frame)
        
        # Save annotation data
        data = {
            'behavior': behavior,
            'landmarks': landmarks.tolist(),
            'timestamp': timestamp.isoformat(),
            'image_path': img_path
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.training_data.append(frame)
        self.labels.append(behavior)
        self.landmarks.append(landmarks)

    def train_model(self):
        """Train behavior recognition model using collected data."""
        if not self.training_data:
            raise ValueError("No training data available")
            
        # Convert data to numpy arrays
        X = np.array(self.landmarks)
        y = self.label_encoder.fit_transform(self.labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model (placeholder for actual model training)
        print("Training model...")
        # Add your preferred model training code here
        
        # Save model
        model_path = self.models_dir / 'behavior_model.pkl'
        # Add model saving code here
        
        return {
            'accuracy': 0.85,  # Replace with actual metrics
            'samples': len(self.training_data),
            'classes': self.behaviors.keys()
        }

    def generate_analytics(self, start_date=None, end_date=None):
        """Generate analytics visualizations and reports.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
        """
        # Load behavior data
        behavior_data = self._load_behavior_data()
        
        if not behavior_data:
            return
            
        # Filter by date range
        if start_date and end_date:
            behavior_data = behavior_data[
                (behavior_data['timestamp'] >= start_date) &
                (behavior_data['timestamp'] <= end_date)
            ]
        
        # Generate visualizations
        self._plot_behavior_distribution(behavior_data)
        self._plot_attendance_trends(behavior_data)
        self._plot_behavior_timeline(behavior_data)
        self._generate_summary_report(behavior_data)

    def _load_behavior_data(self):
        """Load behavior data from database or files."""
        data_files = list(self.training_dir.glob('*.json'))
        if not data_files:
            return None
            
        data_list = []
        for file in data_files:
            with open(file) as f:
                data = json.load(f)
                data_list.append({
                    'timestamp': datetime.fromisoformat(data['timestamp']),
                    'behavior': data['behavior']
                })
        
        return pd.DataFrame(data_list)

    def _plot_behavior_distribution(self, data):
        """Plot distribution of behaviors."""
        plt.figure(figsize=(10, 6))
        data['behavior'].value_counts().plot(kind='bar')
        plt.title('Behavior Distribution')
        plt.xlabel('Behavior Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(self.analytics_dir / 'behavior_distribution.png')
        plt.close()

    def _plot_attendance_trends(self, data):
        """Plot attendance trends over time."""
        plt.figure(figsize=(12, 6))
        daily_attendance = data.groupby(
            data['timestamp'].dt.date
        ).size()
        daily_attendance.plot(kind='line', marker='o')
        plt.title('Daily Attendance Trends')
        plt.xlabel('Date')
        plt.ylabel('Number of Students')
        plt.tight_layout()
        plt.savefig(self.analytics_dir / 'attendance_trends.png')
        plt.close()

    def _plot_behavior_timeline(self, data):
        """Plot behavior changes over time."""
        plt.figure(figsize=(12, 6))
        behavior_timeline = data.pivot_table(
            index=data['timestamp'].dt.date,
            columns='behavior',
            aggfunc='size',
            fill_value=0
        )
        behavior_timeline.plot(kind='area', stacked=True)
        plt.title('Behavior Timeline')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend(title='Behaviors', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(self.analytics_dir / 'behavior_timeline.png')
        plt.close()

    def _generate_summary_report(self, data):
        """Generate summary report of behavior and attendance."""
        report = {
            'total_sessions': len(data['timestamp'].dt.date.unique()),
            'total_students': len(data),
            'behavior_counts': data['behavior'].value_counts().to_dict(),
            'date_range': {
                'start': data['timestamp'].min().strftime('%Y-%m-%d'),
                'end': data['timestamp'].max().strftime('%Y-%m-%d')
            }
        }
        
        report_path = self.analytics_dir / 'summary_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

    def export_data(self, format='csv'):
        """Export behavior and attendance data.
        
        Args:
            format: Export format ('csv' or 'json')
        """
        data = self._load_behavior_data()
        if data is None:
            return
            
        if format == 'csv':
            output_path = self.analytics_dir / 'behavior_data.csv'
            data.to_csv(output_path, index=False)
        else:
            output_path = self.analytics_dir / 'behavior_data.json'
            data.to_json(output_path, orient='records')
