import os
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, 
    QHBoxLayout, QWidget, QFileDialog, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Nesne Tespiti (Çatal & Kaşık) - CPU Modu")
        self.setGeometry(100, 100, 1200, 700)
        
        try:
            self.model = YOLO(MODEL_PATH)
            print(f"Model başarıyla yüklendi: {MODEL_PATH}")
        except Exception as e:
            print(f"HATA: Model yüklenemedi. 'best.pt' dosyasının bu klasörde olduğundan emin olun.")
            print(f"Hata detayı: {e}")
            sys.exit()

        self.current_image_path = None
        self.tagged_pixmap = None

        main_layout = QVBoxLayout()

        image_layout = QHBoxLayout()
        
        self.original_panel = QLabel("Lütfen bir resim seçin...")
        self.original_panel.setAlignment(Qt.AlignCenter)
        self.original_panel.setFrameShape(QFrame.Box)
        self.original_panel.setMinimumSize(500, 500)
        image_layout.addWidget(self.original_panel)

        self.tagged_panel = QLabel("Tahmin sonucu burada görünecek...")
        self.tagged_panel.setAlignment(Qt.AlignCenter)
        self.tagged_panel.setFrameShape(QFrame.Box)
        self.tagged_panel.setMinimumSize(500, 500)
        image_layout.addWidget(self.tagged_panel)
        
        main_layout.addLayout(image_layout)

        self.results_label = QLabel("Tespit Edilen Nesneler: ")
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        main_layout.addWidget(self.results_label)

        button_layout = QHBoxLayout()
        
        self.btn_select = QPushButton("1. Resim Seç (Select Image)")
        self.btn_select.clicked.connect(self.select_image)
        self.btn_select.setStyleSheet("font-size: 14px; padding: 10px;")
        button_layout.addWidget(self.btn_select)

        self.btn_test = QPushButton("2. Nesneleri Tespit Et (Test Image)")
        self.btn_test.clicked.connect(self.test_image)
        self.btn_test.setEnabled(False) 
        self.btn_test.setStyleSheet("font-size: 14px; padding: 10px;")
        button_layout.addWidget(self.btn_test)

        self.btn_save = QPushButton("3. Sonucu Kaydet (Save Image)")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("font-size: 14px; padding: 10px;")
        button_layout.addWidget(self.btn_save)
        
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Bir Resim Seç", "", "Resim Dosyaları (*.png *.jpg *.jpeg)")
        
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(self.original_panel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_panel.setPixmap(scaled_pixmap)
            
            self.tagged_panel.setText("Tahmin sonucu burada görünecek...")
            self.results_label.setText("Tespit Edilen Nesneler: ")
            self.btn_test.setEnabled(True)
            self.btn_save.setEnabled(False)
            self.tagged_pixmap = None

    def test_image(self):
        if self.current_image_path:
            try:
                results = self.model(self.current_image_path, device='cpu') 
            except Exception as e:
                self.results_label.setText(f"Tahmin hatası: {e}")
                return

            image_with_boxes = results[0].plot()
            
            rgb_image = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.tagged_panel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.tagged_panel.setPixmap(scaled_pixmap)
            self.tagged_pixmap = pixmap
            
            names = self.model.names
            detected_classes = [names[int(c)] for c in results[0].boxes.cls]
            counts = Counter(detected_classes)
            
            if not counts:
                result_text = "Tespit Edilen Nesneler: Hiçbir şey bulunamadı."
            else:
                result_text = "Tespit Edilen Nesneler: " + ", ".join([f"{count} {name}" for name, count in counts.items()])
            
            self.results_label.setText(result_text)
            self.btn_save.setEnabled(True)

    def save_image(self):
        if self.tagged_pixmap:
            file_path, _ = QFileDialog.getSaveFileName(self, "Resmi Kaydet", "", "PNG Dosyası (*.png);;JPG Dosyası (*.jpg)")
            
            if file_path:
                self.tagged_pixmap.save(file_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())