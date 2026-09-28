from collections import Counter, deque
from pathlib import Path
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
EMOTIONS = ("Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised")
EMOJI_FILES = {
    "Angry": ("angry.png", "angry.jpg", "angry.jpeg"),
    "Disgusted": ("disgusted.png",),
    "Fearful": ("fearful.png",),
    "Happy": ("happy.png", "happy.jpg", "happy.jpeg"),
    "Neutral": ("neutral.png",),
    "Sad": ("sad.png", "sad.jpg", "sad.jpeg"),
    "Surprised": ("surprised.png", "surpriced.png"),
}
EMOJI_TEXT = {
    "Angry": "😠",
    "Disgusted": "🤢",
    "Fearful": "😨",
    "Happy": "😄",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprised": "😮",
}
FACE_LANDMARK_MODEL = ROOT / "face_landmarker.task"
FACE_SIZE = (48, 48)
FRAME_SIZE = (800, 600)


def build_emotion_model():
    from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from keras.models import Sequential

    model = Sequential([
        Conv2D(32, 3, activation="relu", input_shape=(48, 48, 1)),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D(2),
        Dropout(0.25),
        Conv2D(128, 3, activation="relu"),
        MaxPooling2D(2),
        Conv2D(128, 3, activation="relu"),
        MaxPooling2D(2),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation="relu"),
        Dropout(0.5),
        Dense(len(EMOTIONS), activation="softmax"),
    ])
    return model


def load_emotion_model():
    model_path = ROOT / "model.h5"
    if not model_path.is_file():
        return None

    try:
        from keras.models import load_model

        try:
            return load_model(str(model_path), compile=False)
        except (ValueError, TypeError):
            model = build_emotion_model()
            model.load_weights(str(model_path))
            return model
    except Exception as error:
        print(f"Could not load model.h5: {error}")
        return None


class EmojiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Emojify")
        self.root.geometry("1200x760")
        self.root.minsize(950, 650)
        self.root.configure(bg="#151515")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.model = load_emotion_model()
        self.landmarker = None
        self.landmarker_result = None
        self.face_detector = self.load_face_detector()
        self.camera = None
        self.camera_starting = False
        self.emoji_images = self.load_emoji_images()
        self.prediction_history = {}
        self.current_frame = None
        self.running = True
        self.last_frame_time = time.monotonic()

        self.build_gui()
        self.status_label.configure(text=self.get_status())
        self.root.after(100, self.start_camera)
        self.root.after(100, self.start_landmarker)

    def load_landmarker(self):
        if not FACE_LANDMARK_MODEL.is_file():
            return None
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(FACE_LANDMARK_MODEL)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=5,
                output_face_blendshapes=True,
            )
            return vision.FaceLandmarker.create_from_options(options)
        except Exception as error:
            print(f"Could not load face_landmarker.task: {error}")
            return None

    def start_landmarker(self):
        threading.Thread(target=self._load_landmarker_worker, daemon=True).start()

    def _load_landmarker_worker(self):
        landmarker = self.load_landmarker()
        self.root.after(0, lambda: self._landmarker_ready(landmarker))

    def _landmarker_ready(self, landmarker):
        self.landmarker = landmarker
        self.status_label.configure(text=self.get_status())

    @staticmethod
    def load_face_detector():
        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade.is_file():
            return None
        detector = cv2.CascadeClassifier(str(cascade))
        return None if detector.empty() else detector

    @staticmethod
    def open_camera():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera.release()
            return None
        return camera

    def start_camera(self):
        if self.camera_starting or self.camera is not None:
            return
        self.camera_starting = True
        self.status_label.configure(text="Connecting to camera...")
        threading.Thread(target=self._open_camera_worker, daemon=True).start()

    def _open_camera_worker(self):
        camera = self.open_camera()
        self.root.after(0, lambda: self._camera_ready(camera))

    def _camera_ready(self, camera):
        self.camera_starting = False
        self.camera = camera
        self.status_label.configure(text=self.get_status())
        if self.camera is not None:
            self.update_frame()
        elif self.model is None and self.landmarker is not None:
            self.status_label.configure(
                text="Camera unavailable. Basic expression fallback is ready for uploaded photos."
            )

    def load_emoji_images(self):
        images = {}
        for emotion, filenames in EMOJI_FILES.items():
            path = next(
                (ROOT / "emojis" / filename for filename in filenames
                 if (ROOT / "emojis" / filename).is_file()),
                None,
            )
            if path is None:
                continue
            try:
                image = Image.open(path).convert("RGBA")
                image.thumbnail((230, 230))
                images[emotion] = ImageTk.PhotoImage(image)
            except Exception as error:
                print(f"Could not load {path.name}: {error}")
        return images

    def build_gui(self):
        tk.Label(
            self.root,
            text="AI EMOJIFY",
            bg="#151515",
            fg="#f4f0e8",
            font=("Segoe UI", 26, "bold"),
        ).pack(pady=(18, 2))
        tk.Label(
            self.root,
            text="Real-time facial emotion recognition",
            bg="#151515",
            fg="#a9b0b7",
            font=("Segoe UI", 11),
        ).pack(pady=(0, 5))
        self.status_label = tk.Label(
            self.root,
            text="Starting...",
            bg="#151515",
            fg="#e0ad62",
            font=("Segoe UI", 10),
            wraplength=1000,
        )
        self.status_label.pack(pady=(0, 10))

        content = tk.Frame(self.root, bg="#151515")
        content.pack(fill=tk.BOTH, expand=True, padx=25, pady=10)
        self.video_label = tk.Label(content, bg="#242424", text="Camera or uploaded photo")
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = tk.Frame(content, bg="#151515", width=330)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        side.pack_propagate(False)
        tk.Label(
            side, text="Detected Emotion", bg="#151515", fg="#a9b0b7",
            font=("Segoe UI", 11),
        ).pack(pady=(15, 3))
        self.emotion_label = tk.Label(
            side, text="Waiting...", bg="#151515", fg="#f4f0e8",
            font=("Segoe UI", 24, "bold"),
        )
        self.emotion_label.pack(pady=(0, 5))
        self.emoji_label = tk.Label(side, bg="#151515")
        self.emoji_label.pack(pady=5)
        self.confidence_label = tk.Label(
            side, text="Confidence: 0.0%", bg="#151515", fg="#72d6a0",
            font=("Segoe UI", 13, "bold"),
        )
        self.confidence_label.pack(pady=4)
        self.face_count_label = tk.Label(
            side, text="Faces: 0", bg="#151515", fg="#a9b0b7",
            font=("Segoe UI", 10),
        )
        self.face_count_label.pack(pady=2)
        self.fps_label = tk.Label(
            side, text="FPS: 0.0", bg="#151515", fg="#a9b0b7",
            font=("Segoe UI", 10),
        )
        self.fps_label.pack(pady=2)
        tk.Frame(side, bg="#303030", height=1).pack(fill=tk.X, padx=20, pady=15)
        tk.Label(
            side, text="Emotion Probabilities", bg="#151515", fg="#f4f0e8",
            font=("Segoe UI", 11, "bold"),
        ).pack(pady=(0, 8))
        self.probability_labels = {}
        for emotion in EMOTIONS:
            label = tk.Label(
                side, text=f"{emotion}: 0.0%", bg="#151515", fg="#a9b0b7",
                anchor="w", font=("Segoe UI", 9),
            )
            label.pack(fill=tk.X, padx=25, pady=1)
            self.probability_labels[emotion] = label

        buttons = tk.Frame(self.root, bg="#151515")
        buttons.pack(pady=(5, 18))
        for text, command in (
            ("Capture", self.capture_image),
            ("Upload Photo", self.upload_photo),
            ("Restart", self.restart_camera),
            ("Quit", self.close),
        ):
            tk.Button(
                buttons, text=text, command=command,
                bg="#29313a", fg="white", activebackground="#414c58",
                activeforeground="white", relief=tk.FLAT, padx=17, pady=8,
                font=("Segoe UI", 10, "bold"), cursor="hand2",
            ).pack(side=tk.LEFT, padx=5)

    def get_status(self):
        if self.model is None and self.landmarker is not None:
            return "Basic expression estimates enabled; add model.h5 for trained seven-class recognition."
        if self.model is None:
            return "model.h5 is missing or could not be loaded."
        if not self.emoji_images:
            return "Model loaded, but emoji images are missing from emojis/."
        if self.camera is None:
            return "Camera unavailable. Use Upload Photo to analyze an image."
        return "Emotion model ready."

    def detect_faces(self, frame):
        self.landmarker_result = None
        height, width = frame.shape[:2]
        faces = []

        if self.face_detector is not None:
            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

        if self.landmarker is not None:
            import mediapipe as mp

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.landmarker_result = self.landmarker.detect_for_video(
                image, int(time.monotonic() * 1000)
            )
            if len(faces) == 0:
                for landmarks in self.landmarker_result.face_landmarks:
                    x_values = [point.x for point in landmarks]
                    y_values = [point.y for point in landmarks]
                    pad_x = int((max(x_values) - min(x_values)) * width * 0.15)
                    pad_y = int((max(y_values) - min(y_values)) * height * 0.15)
                    x1 = max(0, int(min(x_values) * width) - pad_x)
                    y1 = max(0, int(min(y_values) * height) - pad_y)
                    x2 = min(width, int(max(x_values) * width) + pad_x)
                    y2 = min(height, int(max(y_values) * height) + pad_y)
                    if x2 > x1 and y2 > y1:
                        faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    @staticmethod
    def prepare_face(face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, FACE_SIZE).astype(np.float32) / 255.0
        return gray[None, :, :, None]

    def predict_emotion(self, face, face_index=0):
        if self.model is not None:
            try:
                probabilities = self.model.predict(self.prepare_face(face), verbose=0)[0]
                index = int(np.argmax(probabilities))
                return EMOTIONS[index], float(probabilities[index]), probabilities
            except Exception as error:
                print(f"Prediction error: {error}")
                return "Neutral", 0.0, None

        if (
            self.landmarker_result is None
            or face_index >= len(self.landmarker_result.face_blendshapes)
        ):
            return "Neutral", 0.0, None

        blendshapes = {
            item.category_name: item.score
            for item in self.landmarker_result.face_blendshapes[face_index]
        }

        def avg(*names):
            return sum(blendshapes.get(name, 0.0) for name in names) / len(names)

        scores = np.array([
            avg("browDownLeft", "browDownRight", "mouthPressLeft", "mouthPressRight"),
            avg("noseSneerLeft", "noseSneerRight", "mouthUpperUpLeft", "mouthUpperUpRight"),
            avg("eyeWideLeft", "eyeWideRight", "browInnerUp", "jawOpen"),
            avg("mouthSmileLeft", "mouthSmileRight"),
            max(0.05, blendshapes.get("neutral", 0.0)),
            avg("mouthFrownLeft", "mouthFrownRight", "browInnerUp"),
            avg("jawOpen", "eyeWideLeft", "eyeWideRight", "browInnerUp"),
        ], dtype=np.float32)
        total = float(scores.sum())
        probabilities = scores / total if total else np.full(7, 1 / 7, dtype=np.float32)
        index = int(np.argmax(probabilities))
        return EMOTIONS[index], float(probabilities[index]), probabilities

    def smooth_emotion(self, face_id, emotion):
        history = self.prediction_history.setdefault(face_id, deque(maxlen=7))
        history.append(emotion)
        return Counter(history).most_common(1)[0][0]

    def process_frame(self, frame):
        faces = self.detect_faces(frame)
        best_result = None
        for face_id, (x, y, width, height) in enumerate(faces):
            face = frame[y:y + height, x:x + width]
            if face.size == 0:
                continue
            emotion, confidence, probabilities = self.predict_emotion(face, face_id)
            emotion = self.smooth_emotion(face_id, emotion)
            color = (72, 190, 163) if confidence >= 0.2 else (80, 80, 200)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(
                frame, f"{emotion} {confidence * 100:.1f}%", (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
            )
            result = {"emotion": emotion, "confidence": confidence, "probabilities": probabilities}
            if best_result is None or confidence > best_result["confidence"]:
                best_result = result
        self.update_panel(best_result, len(faces))
        return frame

    def update_panel(self, result, face_count):
        self.face_count_label.configure(text=f"Faces: {face_count}")
        if result is None:
            self.emotion_label.configure(text="No Face")
            self.confidence_label.configure(text="Confidence: 0.0%")
            self.emoji_label.configure(image="", text="")
            for emotion, label in self.probability_labels.items():
                label.configure(text=f"{emotion}: 0.0%")
            return

        emotion = result["emotion"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        self.emotion_label.configure(text=emotion)
        self.confidence_label.configure(text=f"Confidence: {confidence * 100:.1f}%")
        emoji = self.emoji_images.get(emotion)
        if emoji is not None:
            self.emoji_label.configure(image=emoji, text="")
            self.emoji_label.image = emoji
        else:
            self.emoji_label.configure(
                image="", text=EMOJI_TEXT.get(emotion, ""),
                font=("Segoe UI Emoji", 65), fg="white",
            )
        if probabilities is not None:
            for emotion, value in zip(EMOTIONS, probabilities):
                self.probability_labels[emotion].configure(
                    text=f"{emotion}: {float(value) * 100:.1f}%"
                )

    def update_frame(self):
        if not self.running or self.camera is None:
            return
        success, frame = self.camera.read()
        if not success:
            self.status_label.configure(text="Could not read from camera.")
            self.root.after(200, self.update_frame)
            return
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = self.process_frame(frame)
        now = time.monotonic()
        elapsed = now - self.last_frame_time
        self.last_frame_time = now
        self.fps_label.configure(text=f"FPS: {1 / elapsed:.1f}" if elapsed else "FPS: 0.0")
        self.show_frame(frame)
        self.root.after(30, self.update_frame)

    def show_frame(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo
        self.current_frame = frame.copy()

    def capture_image(self):
        if self.current_frame is None:
            messagebox.showwarning("Capture", "No camera frame available.")
            return
        capture_directory = ROOT / "captures"
        capture_directory.mkdir(parents=True, exist_ok=True)
        output = capture_directory / f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        if cv2.imwrite(str(output), self.current_frame):
            self.status_label.configure(text=f"Captured: {output.name}")

    def upload_photo(self):
        path = filedialog.askopenfilename(
            title="Select a photo",
            filetypes=(("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")),
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Image error", "Could not open this image.")
            return
        image = cv2.resize(image, FRAME_SIZE)
        self.show_frame(self.process_frame(image))
        self.status_label.configure(text="Photo analyzed.")

    def restart_camera(self):
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        self.prediction_history.clear()
        self.start_camera()

    def close(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
        if self.landmarker is not None:
            self.landmarker.close()
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    app = EmojiApp(window)
    window.mainloop()
