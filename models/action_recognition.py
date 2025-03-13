import os
import cv2
import numpy as np
import warnings

# Force CPU mode and disable DirectML
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['DISABLE_TFDML'] = '1'

try:
    # Configure TensorFlow before importing
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    TF_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"TensorFlow import failed: {str(e)}")
    TF_AVAILABLE = False


class ActionRecognizer:
    def __init__(self, model_path='data/models/action_model.h5'):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Action recognition is disabled.")

        self.model_path = model_path
        self.actions = []
        self.sequence_length = 30
        self.model = None
        self.tf_available = TF_AVAILABLE

        # Initialize OpenCV's DNN models for pose estimation
        try:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'data', 'models', 'pose')
            os.makedirs(model_dir, exist_ok=True)

            pose_model_path = os.path.join(model_dir, 'graph_opt.pb')
            if os.path.exists(pose_model_path):
                self.pose_net = cv2.dnn.readNetFromTensorflow(pose_model_path)
            else:
                print(f"Pose model not found at {pose_model_path}")
                self.pose_net = None
        except Exception as e:
            print(f"Error initializing pose model: {str(e)}")
            self.pose_net = None

        self.threshold = 0.2
        self.BODY_PARTS = {
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17
        }


    def extract_keypoints(self, frame):
        """Extract keypoints using OpenCV's DNN pose estimation."""
        if self.pose_net is None:
            # Return empty keypoints if pose model is not available
            return np.zeros(len(self.BODY_PARTS) * 3)

        frame_height, frame_width = frame.shape[:2]

        try:
            # Prepare input blob and perform inference
            blob = cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
            self.pose_net.setInput(blob)
            output = self.pose_net.forward()

            # Extract keypoints
            keypoints = []
            for i in range(len(self.BODY_PARTS)):
                # Get confidence map
                prob_map = output[0, i, :, :]
                prob_map = cv2.resize(prob_map, (frame_width, frame_height))

                # Find global maxima of the probmap
                minVal, prob, minLoc, point = cv2.minMaxLoc(prob_map)

                if prob > self.threshold:
                    x = (frame_width * point[0]) / output.shape[3]
                    y = (frame_height * point[1]) / output.shape[2]
                    keypoints.extend([x/frame_width, y/frame_height, prob])
                else:
                    keypoints.extend([0, 0, 0])

            return np.array(keypoints)
        except Exception as e:
            print(f"Error extracting keypoints: {str(e)}")
            return np.zeros(len(self.BODY_PARTS) * 3)


    def create_model(self):
        """Create LSTM model for action recognition."""
        if not self.tf_available:
            print("TensorFlow not available, cannot create model")
            return False

        input_shape = (self.sequence_length, len(self.BODY_PARTS) * 3)  # 3 values per keypoint (x, y, conf)

        self.model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
            Dropout(0.2),
            LSTM(128, return_sequences=True, activation='relu'),
            Dropout(0.2),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(len(self.actions), activation='softmax')
        ])
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return True


    def collect_training_data(self, video_path, action_label, behavior_type):
        """Extract frames and keypoints from training video."""
        if not self.tf_available:
            print("TensorFlow not available, cannot collect training data")
            return [], []

        sequences = []
        labels = []

        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            current_sequence = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                keypoints = self.extract_keypoints(frame)
                current_sequence.append(keypoints)

                if len(current_sequence) == self.sequence_length:
                    sequences.append(current_sequence)
                    labels.append(self.actions.index(f"{action_label}_{behavior_type}"))
                    current_sequence = []

                frame_count += 1

            cap.release()
            return sequences, labels
        except Exception as e:
            print(f"Error collecting training data: {str(e)}")
            return [], []


    def train_model(self, sequences, labels):
        """Train the action recognition model."""
        if not self.tf_available:
            print("TensorFlow not available, cannot train model")
            return None

        try:
            X = np.array(sequences)
            y = to_categorical(labels).astype('float32')

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            if not self.model:
                if not self.create_model():
                    return None

            history = self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            return history
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None


    def predict_action(self, sequence):
        """Predict action from a sequence of frames."""
        if not self.tf_available:
            print("TensorFlow not available, cannot predict action")
            return "Unknown", 0.0

        try:
            if not self.model:
                if os.path.exists(self.model_path):
                    self.model = load_model(self.model_path)
                else:
                    raise ValueError("No trained model found")

            sequence = np.expand_dims(sequence, axis=0)
            prediction = self.model.predict(sequence)
            action_idx = np.argmax(prediction)
            confidence = prediction[0][action_idx]

            return self.actions[action_idx], confidence
        except Exception as e:
            print(f"Error predicting action: {str(e)}")
            return "Unknown", 0.0


    def draw_skeleton(self, frame, keypoints):
        """Draw the skeleton on the frame for visualization."""
        try:
            pairs = [
                (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
                (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
                (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
            ]

            for pair in pairs:
                partFrom = pair[0]
                partTo = pair[1]

                idFrom = partFrom * 3
                idTo = partTo * 3

                if (idFrom + 2 < len(keypoints) and idTo + 2 < len(keypoints) and
                    keypoints[idFrom + 2] > self.threshold and 
                    keypoints[idTo + 2] > self.threshold):

                    x1 = int(keypoints[idFrom] * frame.shape[1])
                    y1 = int(keypoints[idFrom + 1] * frame.shape[0])
                    x2 = int(keypoints[idTo] * frame.shape[1])
                    y2 = int(keypoints[idTo + 1] * frame.shape[0])

                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (x1, y1), 4, (0, 0, 255), -1)
                    cv2.circle(frame, (x2, y2), 4, (0, 0, 255), -1)

            return frame
        except Exception as e:
            print(f"Error drawing skeleton: {str(e)}")
            return frame
