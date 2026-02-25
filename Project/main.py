"""
LSTM do klasyfikacji stanów twarzy BEZ użycia gotowych modeli
Klasy: sleep (śpię), tired (zmęczona), normal (stan normalny)
Z głosowym odczytem wyników
"""
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pyttsx3  # Для голосового вывода
try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except (ImportError, AttributeError):
    try:
        import mediapipe.solutions as solutions

        mp_face_mesh = solutions.face_mesh
        mp_drawing = solutions.drawing_utils
    except:
        print("ERROR: Could not import MediaPipe")
        print("Run: pip uninstall mediapipe && pip install mediapipe==0.10.9")
        raise


# ******************** CZĘŚĆ 1: EKSTRAKCJA CECH

class FaceFeatureExtractor:
    """EAR and MAR from video """

    def __init__(self):
        try:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"MediaPipe initialization error: {e}")
            print("Try: pip install mediapipe==0.10.9")
            raise


        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]

    def calculate_ear(self, landmarks, eye_indices):
        """
        Eye Aspect Ratio
        Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])

        # Vertical distance
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])

        # Horizontal distance
        horizontal = np.linalg.norm(points[0] - points[3])

        # Counting EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def calculate_mar(self, landmarks, mouth_indices):
        """
        Own realization of Mouth Aspect Ratio
        Formula: MAR = (||p3-p11|| + ||p5-p9||) / (2 * ||p1-p7||)
        """
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in mouth_indices])

        # Vertical odlegosci
        vertical_1 = np.linalg.norm(points[2] - points[10])
        vertical_2 = np.linalg.norm(points[4] - points[8])

        # Horizontal odleglosci
        horizontal = np.linalg.norm(points[0] - points[6])

        # Count MAR
        mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return mar

    def extract_features_from_video(self, video_path):
        """Ekstrakcja cech z wideo klatka po klatce"""

        cap = cv2.VideoCapture(str(video_path))
        features = []

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # BGR -> RGB для MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Count EAR for both eye
                left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
                right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0

                # Count MAR
                mar = self.calculate_mar(landmarks, self.MOUTH)

                features.append([avg_ear, mar])

        cap.release()
        print(f"  Frames processed: {frame_count}, features extracted: {len(features)}")
        return np.array(features)

    def __del__(self):
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
        except:
            pass


def process_dataset(data_dir, output_dir):
    """
   Przetwarzanie zbioru danych i zapisywanie cech
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n Contents of '{data_dir}':")
    for item in data_dir.iterdir():
        print(f"  - {item.name} ({'folder' if item.is_dir() else 'file'})")

    extractor = FaceFeatureExtractor()

    # Map
    # eyes -> sleep (0), yawn -> tired (1), normal -> normal (2)
    class_mapping = {'eyes': 0, 'yawn': 1, 'normal': 2}
    class_names_display = {0: 'sleep', 1: 'tired', 2: 'normal'}

    print("\nStarting video processing...")

    for class_name, class_idx in class_mapping.items():
        class_path = data_dir / class_name
        if not class_path.exists():
            print(f" Warning: folder {class_name} not found")
            continue

        video_files = list(class_path.glob('*.mp4')) + \
                      list(class_path.glob('*.avi')) + \
                      list(class_path.glob('*.mov'))

        print(f"\n Class '{class_name}': found {len(video_files)} videos")

        for idx, video_file in enumerate(video_files, 1):
            print(f"  [{idx}/{len(video_files)}] {video_file.name}")

            try:
                features = extractor.extract_features_from_video(video_file)

                if len(features) > 0:
                    # Сохраняем с новым именем класса для дальнейшего использования
                    display_name = class_names_display[class_idx]
                    output_file = output_dir / f"{display_name}_{video_file.stem}.npy"
                    np.save(output_file, features)
                else:
                    print(f"    ⚠️ Face not detected!")
            except Exception as e:
                print(f"     Error: {e}")

    print("\n Processing completed!")


# ******************* CZĘŚĆ 2: PRZYGOTOWANIE DANNYCH

def create_sequences(features, sequence_length=50, step=15):
    sequences = []

    for i in range(0, len(features) - sequence_length + 1, step):
        sequence = features[i:i + sequence_length]
        sequences.append(sequence)

    return sequences


def load_and_prepare_data(features_dir, sequence_length=50, step=15):
    features_dir = Path(features_dir)

    X = []
    y = []

    class_mapping = {'sleep': 0, 'tired': 1, 'normal': 2}

    print("\nLoading features...")

    for class_name, class_idx in class_mapping.items():
        class_files = list(features_dir.glob(f'{class_name}_*.npy'))
        print(f"  {class_name}: {len(class_files)} files")

        for npy_file in class_files:
            features = np.load(npy_file)
            sequences = create_sequences(features, sequence_length, step)

            for seq in sequences:
                X.append(seq)
                y.append(class_idx)

    X = np.array(X)
    y = np.array(y)

    print(f"\nTotal samples: {len(X)}")
    print(f"  sleep: {np.sum(y == 0)}")
    print(f"  tired: {np.sum(y == 1)}")
    print(f"  normal: {np.sum(y == 2)}")

    # One-hot encoding
    y_categorical = to_categorical(y, num_classes=3)

    return X, y_categorical


# *****************3: LSTM MODEL

def create_custom_lstm_model(sequence_length=50, num_features=2, num_classes=3):

    model = Sequential(name='Custom_LSTM_Classifier')

    # Layer  1: LSTM слой (рекуррентный)
    # 64 neurona LSTM - pamiec kolejnosc
    model.add(LSTM(
        units=64,
        input_shape=(sequence_length, num_features),
        return_sequences=False,
        name='LSTM_Layer'
    ))

    # Dropout fight with overfitting
    model.add(Dropout(0.3, name='Dropout_Regularization'))

    # Layer 2: Dense
    # 32 neurona for uploading from LSTM
    model.add(Dense(
        units=32,
        activation='relu',  # ReLU so not linear
        name='Dense_Hidden'
    ))

    # Layer 3: Выходной слой
    # 3 neurona (sleep, tired, normal)
    model.add(Dense(
        units=num_classes,
        activation='softmax',  # Softmax for probabilities
        name='Output_Layer'
    ))

    # Compilacja
    model.compile(
        optimizer='adam',  #Adam
        loss='categorical_crossentropy',  # Funkcja straty
        metrics=['accuracy']  # Мetryka dokładności
    )

    return model


# **************  4: Learning with custom model

def train_custom_model(X, y, epochs=50, batch_size=16):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )

    print(f"\n Data sizes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")


    print("\n Creating custom LSTM model...")
    model = create_custom_lstm_model(
        sequence_length=X_train.shape[1],
        num_features=X_train.shape[2],
        num_classes=y_train.shape[1]
    )

    print("\n Model architecture:")
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]

    # Learn
    print("\n Training started...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Final rate
    print("\n Model evaluation:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    return model, history


# ************************5: REAL-TIME with voice output

class VoiceRealtimeClassifier:

    def __init__(self, model_path, sequence_length=50):
        self.model = tf.keras.models.load_model(model_path)
        self.sequence_length = sequence_length
        self.extractor = FaceFeatureExtractor()
        self.buffer = []

        self.class_names = ['Sleep', 'Tired', 'Normal']
        self.class_colors = [(255, 0, 0), (0, 165, 255), (0, 255, 0)]  # Blue, Orange, Green

        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Скорость речи
            self.engine.setProperty('volume', 0.9)  # Громкость

            # Установка английского голоса
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'english' in voice.name.lower() or 'en' in voice.languages:
                    self.engine.setProperty('voice', voice.id)
                    break

            self.voice_enabled = True
            print(" Voice output enabled")
        except:
            self.voice_enabled = False
            print(" Voice output unavailable")

        self.last_prediction = None
        self.prediction_stable_count = 0
        self.stability_threshold = 5  # Сколько кадров подряд нужно для озвучки

    def speak(self, text):
        """Озвучить текст"""
        if self.voice_enabled:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                pass

    def run(self):
        """Запуск классификации с голосом"""
        cap = cv2.VideoCapture(0)

        print("\n" + "=" * 50)
        print(" CAMERA STARTED WITH VOICE OUTPUT")
        print("=" * 50)
        print("Press 'q' to exit")
        print("Press 's' to speak current state")
        print("=" * 50 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Obtaining
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.extractor.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_ear = self.extractor.calculate_ear(landmarks, self.extractor.LEFT_EYE)
                right_ear = self.extractor.calculate_ear(landmarks, self.extractor.RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0
                mar = self.extractor.calculate_mar(landmarks, self.extractor.MOUTH)

                self.buffer.append([ear, mar])

                if len(self.buffer) > self.sequence_length:
                    self.buffer.pop(0)

                # Prediction
                if len(self.buffer) == self.sequence_length:
                    sequence = np.array([self.buffer])
                    prediction = self.model.predict(sequence, verbose=0)[0]
                    class_idx = np.argmax(prediction)
                    confidence = prediction[class_idx]

                    if class_idx == self.last_prediction:
                        self.prediction_stable_count += 1
                    else:
                        self.prediction_stable_count = 0
                        self.last_prediction = class_idx

                    if self.prediction_stable_count == self.stability_threshold:
                        self.speak(self.class_names[class_idx])
                        print(f" Statusq: {self.class_names[class_idx]}")

                    # SShowing results
                    color = self.class_colors[class_idx]
                    text = f"{self.class_names[class_idx]}: {confidence:.2%}"

                    # Background for text
                    cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)
                    cv2.putText(frame, text, (20, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                # Showing EAR и MAR
                cv2.putText(frame, f"Eyes: {ear:.3f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Mouth: {mar:.3f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Face not detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Face State Classifier (q - exit, s - speak)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and self.last_prediction is not None:
                self.speak(self.class_names[self.last_prediction])

        cap.release()
        cv2.destroyAllWindows()


# ==================== Main ====================
#
# if __name__ == "__main__":
#     # congig
#     DATA_DIR = "dataset"  # Folder with folders: eyes, yawn, normal
#     FEATURES_DIR = "features"
#     MODEL_PATH = "my_custom_lstm_model.h5"
#     SEQUENCE_LENGTH = 50
#
#     print("=" * 60)
#     print("PROJECT: CUSTOM LSTM MODEL FOR FACE STATE CLASSIFICATION")
#     print("=" * 60)
#     print("\n Folder mapping:")
#     print("  eyes/ -> Sleep")
#     print("  yawn/ -> Tired")
#     print("  normal/ -> Normal")
#     print(f"\n Looking for data folder: {Path(DATA_DIR).absolute()}")
#
#     print("\n" + "=" * 60)
#     print("STEP 1: Feature extraction from videos (custom implementation)")
#     print("=" * 60)
#     process_dataset(DATA_DIR, FEATURES_DIR)
#
#     features_dir = Path(FEATURES_DIR)
#     if not features_dir.exists() or len(list(features_dir.glob('*.npy'))) == 0:
#         print("\n ERROR: No processed data!")
#         print(f"Check that folder '{DATA_DIR}' has subfolders:")
#         print("  - eyes (with videos) -> will be 'sleep'")
#         print("  - yawn (with videos) -> will be 'tired'")
#         print("  - normal (with videos) -> stays 'normal'")
#         exit(1)
#
#     print("\n" + "=" * 60)
#     print("STEP 2: Data preparation (custom implementation)")
#     print("=" * 60)
#     X, y = load_and_prepare_data(FEATURES_DIR, SEQUENCE_LENGTH)
#
#     print("\n" + "=" * 60)
#     print("STEP 3: Training custom LSTM model")
#     print("=" * 60)
#     model, history = train_custom_model(X, y, epochs=50, batch_size=16)
#
#     model.save(MODEL_PATH)
#     print(f"\n Custom model saved: {MODEL_PATH}")
#
#     print("\n" + "=" * 60)
#     print("STEP 5: Camera launch with voice output")
#     print("=" * 60)
#     print("\n Install pyttsx3 for voice: pip install pyttsx3")
#
#     try:
#         classifier = VoiceRealtimeClassifier(MODEL_PATH, SEQUENCE_LENGTH)
#         classifier.run()
#     except Exception as e:
#         print(f"\n Camera error: {e}")
#         print("You may need to install: pip install pyttsx3")

# ... (wyżej są klasy FaceFeatureExtractor i VoiceRealtimeClassifier - zostaw je!)

# ==================== GŁÓWNA FUNKCJA (TYLKO KAMERA) ====================

if __name__ == "__main__":
    import os

    # KONFIGURACJA
    MODEL_PATH = "my_custom_lstm_model.h5"  # Upewnij się, że ten plik istnieje!
    SEQUENCE_LENGTH = 50

    print("*" * 60)
    print("TRYB: REAL-TIME INFERENCE (TYLKO KAMERA)")
    print("*" * 60)

    # Sprawdzenie, czy model istnieje
    if not Path(MODEL_PATH).exists():
        print(f"\n BŁĄD: Nie znaleziono pliku modelu '{MODEL_PATH}'")
        print("Musisz najpierw uruchomić pełny skrypt treningowy, aby stworzyć model.")
        print("Upewnij się, że plik .h5 znajduje się w tym samym folderze co skrypt.")
    else:
        print(f"\n Znaleziono model: {MODEL_PATH}")
        print(" Uruchamianie kamery...")

        try:
            # Inicjalizacja i uruchomienie klasyfikatora
            classifier = VoiceRealtimeClassifier(MODEL_PATH, SEQUENCE_LENGTH)
            classifier.run()
        except Exception as e:
            print(f"\n Wystąpił błąd: {e}")
            print("Sprawdź, czy masz zainstalowane biblioteki: opencv-python, tensorflow, mediapipe, pyttsx3")