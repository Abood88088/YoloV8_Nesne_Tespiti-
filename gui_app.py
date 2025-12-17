import sys
import os
import cv2
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.25


class YOLOv8App(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLOv8 Detector")
        self.setGeometry(200, 100, 900, 900)

        if not os.path.exists(MODEL_PATH):
            QMessageBox.critical(self, "Error", "best.pt bulunamadı")
            sys.exit(1)

        self.model = YOLO(MODEL_PATH)
        self.image_path = None

        self.image_label = QLabel("Bir görüntü seçiniz")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border:1px solid gray;")
        self.image_label.setMinimumSize(800, 700)

        # ✅ EKLENEN KISIM (tespit edilen sınıflar ve sayıları)
        self.result_label = QLabel("Tespit sonuçları burada gösterilecek")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size:14px; padding:6px;")

        self.btn_select = QPushButton("Resim Seç")
        self.btn_select.clicked.connect(self.select_image)

        self.btn_detect = QPushButton("Tespiti Çalıştır")
        self.btn_detect.clicked.connect(self.detect_objects)
        self.btn_detect.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)   # ✅ layout’a eklendi
        layout.addWidget(self.btn_select)
        layout.addWidget(self.btn_detect)

        self.setLayout(layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Resim Seç", "", "Images (*.jpg *.png *.jpeg)"
        )
        if file_path:
            self.image_path = file_path
            self.show_image(file_path)
            self.btn_detect.setEnabled(True)

    def detect_objects(self):
        if self.image_path is None:
            return

        results = self.model.predict(
            source=self.image_path,
            conf=CONF_THRESHOLD,
            device="cpu",
            save=False
        )

        # Bounding box çizimi
        annotated = results[0].plot()
        self.show_cv_image(annotated)

        # ✅ EKLENEN KISIM (sınıf ve adet hesaplama)
        boxes = results[0].boxes
        names = self.model.names

        counts = {}
        if boxes is not None:
            for cls_id in boxes.cls.tolist():
                cls_name = names[int(cls_id)]
                counts[cls_name] = counts.get(cls_name, 0) + 1

        if counts:
            text = "Tespit Edilen Nesneler:\n"
            for k, v in counts.items():
                text += f"{k}: {v}\n"
        else:
            text = "Hiç nesne tespit edilmedi"

        self.result_label.setText(text)

    def show_image(self, path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def show_cv_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w

        qt_img = QImage(
            img_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qt_img)
        pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOv8App()
    window.show()
    sys.exit(app.exec_())
