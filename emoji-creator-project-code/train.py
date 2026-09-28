from pathlib import Path
from collections import deque, Counter
from datetime import datetime
import csv
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent

EMOTION_NAMES = (
    "Angry",
    "Disgusted",
    "Fearful",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised",
)

EMOTION_FILES = {
    "Angry": ("angry.png",),
    "Disgusted": ("disgusted.png",),
    "Fearful": ("fearful.png",),
    "Happy": ("happy.png",),
    "Neutral": ("neutral.png",),
    "Sad": ("sad.png",),
    "Surprised": (
        "surprised.png",
        "surpriced.png",
    ),
}

ROOT_EMOJI = ROOT / "emojis"

CAPTURE_DIR = ROOT / "captures"
RESULT_DIR = ROOT / "results"

CAPTURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_emotion_model():
    model_path = ROOT / "model.h5"

    if not model_path.is_file():
        return None

    try:
        from keras.models import load_model

        model = load_model(
            str(model_path),
            compile=False,
        )

        return model

    except Exception as first_error:

        try:
            from keras import Sequential
            from keras.layers import (
                Conv2D,
                Dense,
                Dropout,
                Flatten,
                MaxPooling2D,
                BatchNormalization,
                Input,
            )

            model = Sequential(
                [
                    Input(shape=(48, 48, 1)),

                    Conv2D(
                        32,
                        3,
                        activation="relu",
                        padding="same",
                    ),
                    BatchNormalization(),

                    Conv2D(
                        32,
                        3,
                        activation="relu",
                        padding="same",
                    ),
                    MaxPooling2D(2),
                    Dropout(0.25),

                    Conv2D(
                        64,
                        3,
                        activation="relu",
                        padding="same",
                    ),
                    BatchNormalization(),

                    Conv2D(
                        64,
                        3,
                        activation="relu",
                        padding="same",
                    ),
                    MaxPooling2D(2),
                    Dropout(0.25),

                    Conv2D(
                        128,
                        3,
                        activation="relu",
                        padding="same",
                    ),
                    BatchNormalization(),

                    Conv2D(
                        128,
                        3,
                        activation="relu",
                        padding="same",
                    ),
                    MaxPooling2D(2),
                    Dropout(0.30),

                    Flatten(),

                    Dense(
                        512,
                        activation="relu",
                    ),
                    BatchNormalization(),
                    Dropout(0.50),

                    Dense(
                        7,
                        activation="softmax",
                    ),
                ]
            )

            model.load_weights(
                str(model_path)
            )

            return model

        except Exception as second_error:
            print(
                "Could not load emotion model:"
            )
            print(first_error)
            print(second_error)

            return None


def load_object_detector():
    try:
        from ultralytics import YOLO

        model_path = ROOT / "yolo11n.pt"

        if model_path.is_file():
            model = YOLO(
                str(model_path)
            )
        else:
            model = YOLO(
                "yolo11n.pt"
            )

        return model

    except Exception as error:
        print(
            "Object detector unavailable:"
        )
        print(error)
        return None


class VisionApplication:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "AI Vision and Emotion Analyzer"
        )

        self.root.geometry(
            "1280x760"
        )

        self.root.minsize(
            1050,
            680,
        )

        self.root.configure(
            bg="#151515"
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close,
        )

        self.emotion_model = (
            load_emotion_model()
        )

        self.object_model = (
            load_object_detector()
        )

        self.face_detector = cv2.CascadeClassifier(
            str(
                Path(cv2.data.haarcascades)
                / "haarcascade_frontalface_default.xml"
            )
        )

        if self.face_detector.empty():
            self.face_detector = None

        self.face_landmarker = self.load_face_landmarker()

        self.camera = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW,
        )

        if not self.camera.isOpened():
            self.camera.release()

            self.camera = cv2.VideoCapture(
                0
            )

        if not self.camera.isOpened():
            self.camera = None

        self.running = True

        self.last_time = time.time()

        self.fps = 0.0

        self.emotion_history = deque(
            maxlen=15
        )

        self.current_emotion = "Neutral"

        self.current_confidence = 0.0

        self.current_people = 0

        self.current_objects = 0

        self.total_detections = 0

        self.emoji_images = {}

        self.load_emojis()

        self.build_interface()

        self.prepare_csv()

        if self.emotion_model is None:
            self.status_var.set(
                "model.h5 not found"
            )

        elif self.object_model is None:
            self.status_var.set(
                "Emotion model ready. "
                "Object detector unavailable."
            )

        else:
            self.status_var.set(
                "AI vision system ready"
            )

        if self.camera is not None:
            self.update_camera()

    @staticmethod
    def load_face_landmarker():
        model_path = ROOT / "face_landmarker.task"

        if not model_path.is_file():
            return None

        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(model_path)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=10,
            )

            return vision.FaceLandmarker.create_from_options(
                options
            )
        except Exception as error:
            print("Face landmarker unavailable:")
            print(error)
            return None

    def load_emojis(self):

        for emotion, files in EMOTION_FILES.items():

            for filename in files:

                path = ROOT_EMOJI / filename

                if path.is_file():

                    try:
                        image = Image.open(
                            path
                        ).convert("RGBA")

                        image.thumbnail(
                            (190, 190)
                        )

                        self.emoji_images[
                            emotion
                        ] = ImageTk.PhotoImage(
                            image
                        )

                        break

                    except Exception:
                        pass

    def build_interface(self):

        title = tk.Label(
            self.root,
            text="AI VISION AND EMOTION ANALYZER",
            bg="#151515",
            fg="#f4f0e8",
            font=(
                "Segoe UI",
                22,
                "bold",
            ),
        )

        title.pack(
            pady=(15, 4)
        )

        self.status_var = tk.StringVar()

        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#151515",
            fg="#e0ad62",
            font=(
                "Segoe UI",
                10,
            ),
        )

        status.pack(
            pady=(0, 10)
        )

        content = tk.Frame(
            self.root,
            bg="#151515",
        )

        content.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=10,
        )

        self.video_label = tk.Label(
            content,
            bg="#242424",
        )

        self.video_label.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
        )

        panel = tk.Frame(
            content,
            bg="#202020",
            width=300,
        )

        panel.pack(
            side=tk.RIGHT,
            fill=tk.Y,
            padx=(18, 0),
        )

        panel.pack_propagate(False)

        self.emotion_var = tk.StringVar(
            value="Neutral"
        )

        self.confidence_var = tk.StringVar(
            value="Confidence: 0.0%"
        )

        self.people_var = tk.StringVar(
            value="People: 0"
        )

        self.object_var = tk.StringVar(
            value="Objects: 0"
        )

        self.fps_var = tk.StringVar(
            value="FPS: 0.0"
        )

        tk.Label(
            panel,
            text="CURRENT EMOTION",
            bg="#202020",
            fg="#aaaaaa",
            font=(
                "Segoe UI",
                10,
                "bold",
            ),
        ).pack(
            pady=(25, 4)
        )

        tk.Label(
            panel,
            textvariable=self.emotion_var,
            bg="#202020",
            fg="#ffffff",
            font=(
                "Segoe UI",
                24,
                "bold",
            ),
            wraplength=260,
        ).pack(
            pady=5
        )

        tk.Label(
            panel,
            textvariable=self.confidence_var,
            bg="#202020",
            fg="#dddddd",
            font=(
                "Segoe UI",
                11,
            ),
        ).pack(
            pady=5
        )

        self.emoji_label = tk.Label(
            panel,
            bg="#202020",
        )

        self.emoji_label.pack(
            pady=15
        )

        statistics = tk.Frame(
            panel,
            bg="#292929",
        )

        statistics.pack(
            fill=tk.X,
            padx=18,
            pady=10,
        )

        tk.Label(
            statistics,
            textvariable=self.people_var,
            bg="#292929",
            fg="#ffffff",
            font=("Segoe UI", 11),
        ).pack(
            pady=6
        )

        tk.Label(
            statistics,
            textvariable=self.object_var,
            bg="#292929",
            fg="#ffffff",
            font=("Segoe UI", 11),
        ).pack(
            pady=6
        )

        tk.Label(
            statistics,
            textvariable=self.fps_var,
            bg="#292929",
            fg="#ffffff",
            font=("Segoe UI", 11),
        ).pack(
            pady=6
        )

        button_frame = tk.Frame(
            self.root,
            bg="#151515",
        )

        button_frame.pack(
            pady=12
        )

        self.make_button(
            button_frame,
            "Capture",
            self.capture_image,
        ).pack(
            side=tk.LEFT,
            padx=5,
        )

        self.make_button(
            button_frame,
            "Upload Photo",
            self.upload_photo,
        ).pack(
            side=tk.LEFT,
            padx=5,
        )

        self.make_button(
            button_frame,
            "Clear History",
            self.clear_history,
        ).pack(
            side=tk.LEFT,
            padx=5,
        )

        self.make_button(
            button_frame,
            "Quit",
            self.close,
        ).pack(
            side=tk.LEFT,
            padx=5,
        )

    def make_button(
        self,
        parent,
        text,
        command,
    ):

        return tk.Button(
            parent,
            text=text,
            command=command,
            bg="#333333",
            fg="#ffffff",
            activebackground="#505050",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=18,
            pady=8,
            font=(
                "Segoe UI",
                10,
                "bold",
            ),
        )

    def prepare_csv(self):

        self.csv_path = (
            RESULT_DIR
            / "detection_results.csv"
        )

        if not self.csv_path.is_file():

            with open(
                self.csv_path,
                "w",
                newline="",
                encoding="utf8",
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    [
                        "timestamp",
                        "emotion",
                        "confidence",
                        "people",
                        "objects",
                    ]
                )

    def log_result(
        self,
        emotion,
        confidence,
        people,
        objects,
    ):

        with open(
            self.csv_path,
            "a",
            newline="",
            encoding="utf8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    emotion,
                    round(
                        confidence * 100,
                        2,
                    ),
                    people,
                    objects,
                ]
            )

    def detect_faces(
        self,
        frame,
    ):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        faces = []

        if self.face_detector is not None:
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

        if not len(faces) and self.face_landmarker is not None:
            import mediapipe as mp

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )
            result = self.face_landmarker.detect_for_video(
                image,
                int(time.monotonic() * 1000),
            )

            height, width = gray.shape

            for landmarks in result.face_landmarks:
                x_values = [point.x for point in landmarks]
                y_values = [point.y for point in landmarks]
                x1 = max(0, int(min(x_values) * width))
                y1 = max(0, int(min(y_values) * height))
                x2 = min(width, int(max(x_values) * width))
                y2 = min(height, int(max(y_values) * height))

                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2 - x1, y2 - y1))

        return gray, faces

    def predict_emotion(
        self,
        gray,
        face,
    ):

        if self.emotion_model is None:
            return (
                "Neutral",
                0.0,
                None,
            )

        x, y, w, h = face

        padding_x = int(w * 0.12)
        padding_y = int(h * 0.12)

        x1 = max(
            0,
            x - padding_x,
        )

        y1 = max(
            0,
            y - padding_y,
        )

        x2 = min(
            gray.shape[1],
            x + w + padding_x,
        )

        y2 = min(
            gray.shape[0],
            y + h + padding_y,
        )

        crop = gray[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return (
                "Neutral",
                0.0,
                None,
            )

        crop = cv2.resize(
            crop,
            (48, 48),
        )

        crop = crop.astype(
            np.float32
        )

        crop = crop / 255.0

        input_data = crop[
            None,
            :,
            :,
            None,
        ]

        prediction = (
            self.emotion_model.predict(
                input_data,
                verbose=0,
            )[0]
        )

        index = int(
            np.argmax(prediction)
        )

        emotion = (
            EMOTION_NAMES[index]
        )

        confidence = float(
            prediction[index]
        )

        return (
            emotion,
            confidence,
            prediction,
        )

    def smooth_emotion(
        self,
        emotion,
    ):

        self.emotion_history.append(
            emotion
        )

        counts = Counter(
            self.emotion_history
        )

        return counts.most_common(1)[0][0]

    def detect_objects(
        self,
        frame,
    ):

        if self.object_model is None:
            return []

        try:

            results = self.object_model.predict(
                source=frame,
                conf=0.35,
                verbose=False,
            )

            detections = []

            if not results:
                return detections

            result = results[0]

            if result.boxes is None:
                return detections

            names = result.names

            for box in result.boxes:

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidence = float(
                    box.conf[0]
                    .cpu()
                    .numpy()
                )

                class_id = int(
                    box.cls[0]
                    .cpu()
                    .numpy()
                )

                label = str(
                    names[class_id]
                )

                x1, y1, x2, y2 = coordinates

                detections.append(
                    (
                        label,
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                )

            return detections

        except Exception as error:

            print(
                "Object detection error:"
            )

            print(error)

            return []

    def draw_object_detections(
        self,
        frame,
        detections,
    ):

        for (
            label,
            confidence,
            x1,
            y1,
            x2,
            y2,
        ) in detections:

            if label == "person":
                continue

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 180, 60),
                2,
            )

            text = (
                f"{label} "
                f"{confidence * 100:.0f}%"
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    max(
                        0,
                        y1 - 28,
                    ),
                ),
                (
                    x1 + 150,
                    y1,
                ),
                (255, 180, 60),
                cv2.FILLED,
            )

            cv2.putText(
                frame,
                text,
                (
                    x1 + 4,
                    max(
                        18,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    def draw_people(
        self,
        frame,
        detections,
    ):

        people = [
            item
            for item in detections
            if item[0] == "person"
        ]

        for index, item in enumerate(
            people,
            start=1,
        ):

            (
                label,
                confidence,
                x1,
                y1,
                x2,
                y2,
            ) = item

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (80, 210, 160),
                2,
            )

            text = (
                f"Person {index} "
                f"{confidence * 100:.0f}%"
            )

            cv2.putText(
                frame,
                text,
                (
                    x1,
                    max(
                        20,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (80, 210, 160),
                2,
                cv2.LINE_AA,
            )

    def process_frame(
        self,
        frame,
    ):

        display = frame.copy()

        gray, faces = (
            self.detect_faces(
                frame
            )
        )

        object_detections = (
            self.detect_objects(
                frame
            )
        )

        self.draw_object_detections(
            display,
            object_detections,
        )

        self.draw_people(
            display,
            object_detections,
        )

        self.current_people = sum(
            1
            for item in object_detections
            if item[0] == "person"
        )

        self.current_objects = sum(
            1
            for item in object_detections
            if item[0] != "person"
        )

        best_emotion = "Neutral"
        best_confidence = 0.0

        for face in faces:

            (
                emotion,
                confidence,
                probabilities,
            ) = self.predict_emotion(
                gray,
                face,
            )

            if confidence > best_confidence:
                best_confidence = confidence
                best_emotion = emotion

            x, y, w, h = face

            smooth = self.smooth_emotion(
                emotion
            )

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (80, 190, 220),
                2,
            )

            text = (
                f"{smooth} "
                f"{confidence * 100:.1f}%"
            )

            cv2.rectangle(
                display,
                (
                    x,
                    max(
                        0,
                        y - 30,
                    ),
                ),
                (
                    x + 190,
                    y,
                ),
                (80, 190, 220),
                cv2.FILLED,
            )

            cv2.putText(
                display,
                text,
                (
                    x + 5,
                    max(
                        20,
                        y - 9,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        if len(faces) > 0:

            best_emotion = self.smooth_emotion(
                best_emotion
            )

        self.current_emotion = (
            best_emotion
        )

        self.current_confidence = (
            best_confidence
        )

        self.total_detections += (
            len(faces)
            + len(object_detections)
        )

        return display

    def update_statistics(self):

        self.emotion_var.set(
            self.current_emotion
        )

        self.confidence_var.set(
            "Confidence: "
            + f"{self.current_confidence * 100:.1f}%"
        )

        self.people_var.set(
            f"People: {self.current_people}"
        )

        self.object_var.set(
            f"Objects: {self.current_objects}"
        )

        self.fps_var.set(
            f"FPS: {self.fps:.1f}"
        )

        emoji = self.emoji_images.get(
            self.current_emotion
        )

        if emoji is not None:

            self.emoji_label.configure(
                image=emoji
            )

        else:

            self.emoji_label.configure(
                image=""
            )

    def update_camera(self):

        if not self.running:
            return

        if self.camera is None:
            self.status_var.set(
                "Camera unavailable"
            )

            return

        success, frame = (
            self.camera.read()
        )

        if not success:

            self.status_var.set(
                "Could not read camera"
            )

            self.root.after(
                200,
                self.update_camera,
            )

            return

        frame = cv2.resize(
            frame,
            (760, 570),
        )

        display = self.process_frame(
            frame
        )

        current_time = time.time()

        elapsed = (
            current_time
            - self.last_time
        )

        if elapsed > 0:

            instant_fps = (
                1.0 / elapsed
            )

            self.fps = (
                self.fps * 0.85
                + instant_fps * 0.15
            )

        self.last_time = current_time

        self.update_statistics()

        self.log_result(
            self.current_emotion,
            self.current_confidence,
            self.current_people,
            self.current_objects,
        )

        rgb = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            rgb
        )

        photo = ImageTk.PhotoImage(
            image
        )

        self.video_label.configure(
            image=photo
        )

        self.video_label.image = photo

        self.root.after(
            30,
            self.update_camera,
        )

    def capture_image(self):

        if self.camera is None:
            messagebox.showerror(
                "Camera",
                "Camera is unavailable.",
            )

            return

        success, frame = (
            self.camera.read()
        )

        if not success:
            messagebox.showerror(
                "Camera",
                "Could not capture image.",
            )

            return

        frame = cv2.resize(
            frame,
            (760, 570),
        )

        processed = self.process_frame(
            frame
        )

        filename = (
            datetime.now().strftime(
                "capture_%Y%m%d_%H%M%S.jpg"
            )
        )

        path = (
            CAPTURE_DIR
            / filename
        )

        cv2.imwrite(
            str(path),
            processed,
        )

        messagebox.showinfo(
            "Capture Saved",
            f"Image saved:\n{path}",
        )

    def upload_photo(self):

        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        image = cv2.imread(
            path
        )

        if image is None:

            messagebox.showerror(
                "Image",
                "Could not open the selected image.",
            )

            return

        image = cv2.resize(
            image,
            (760, 570),
        )

        processed = self.process_frame(
            image
        )

        rgb = cv2.cvtColor(
            processed,
            cv2.COLOR_BGR2RGB,
        )

        photo = ImageTk.PhotoImage(
            Image.fromarray(rgb)
        )

        window = tk.Toplevel(
            self.root
        )

        window.title(
            "Image Analysis"
        )

        window.geometry(
            "800x650"
        )

        label = tk.Label(
            window,
            image=photo,
            bg="#151515",
        )

        label.image = photo

        label.pack(
            fill=tk.BOTH,
            expand=True,
        )

        info = tk.Label(
            window,
            text=(
                f"Emotion: "
                f"{self.current_emotion}    "
                f"Confidence: "
                f"{self.current_confidence * 100:.1f}%    "
                f"People: "
                f"{self.current_people}    "
                f"Objects: "
                f"{self.current_objects}"
            ),
            font=(
                "Segoe UI",
                11,
                "bold",
            ),
            bg="#151515",
            fg="#ffffff",
        )

        info.pack(
            pady=10
        )

    def clear_history(self):

        self.emotion_history.clear()

        self.current_emotion = (
            "Neutral"
        )

        self.current_confidence = 0.0

        self.status_var.set(
            "Emotion history cleared"
        )

        self.update_statistics()

    def close(self):

        self.running = False

        if self.camera is not None:

            self.camera.release()

            self.camera = None

        if self.face_landmarker is not None:
            self.face_landmarker.close()
            self.face_landmarker = None

        self.root.destroy()


def main():

    root = tk.Tk()

    VisionApplication(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()