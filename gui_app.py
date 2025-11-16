import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, 
    QHBoxLayout, QWidget, QFileDialog, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import cv2
from collections import Counter

try:
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path and sys.platform == "win32":
        torch_lib_path = os.path.join(venv_path, "Lib", "site-packages", "torch", "lib")
        if os.path.exists(torch_lib_path):
            os.add_dll_directory(torch_lib_path)
            print(f"DLL Yolu Eklendi: {torch_lib_path}")
except Exception as e:
    print(f"DLL Düzeltmesi sırasında bir hata oluştu: {e}")

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
            sys.exit()

        self.current_image_path = None
        self.tagged_pixmap = None

        self.camera = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_frame)
        self.is_camera_running = False

        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        
        self.original_panel = QLabel("Lütfen bir resim seçin veya kamerayı başlatın...")
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
        
        self.btn_select = QPushButton("1. Resim Seç")
        self.btn_select.clicked.connect(self.select_image)
        self.btn_select.setStyleSheet("font-size: 14px; padding: 10px;")
        button_layout.addWidget(self.btn_select)

        self.btn_test = QPushButton("2. Resimden Tespit Et")
        self.btn_test.clicked.connect(self.test_image)
        self.btn_test.setEnabled(False) 
        self.btn_test.setStyleSheet("font-size: 14px; padding: 10px;")
        button_layout.addWidget(self.btn_test)

        self.btn_save = QPushButton("3. Sonucu Kaydet")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("font-size: 14px; padding: 10px;")
        button_layout.addWidget(self.btn_save)
        
        self.btn_camera = QPushButton("Ekstra: Kamerayı Başlat")
        self.btn_camera.clicked.connect(self.toggle_camera)
        self.btn_camera.setStyleSheet("font-size: 14px; padding: 10px; background-color: #DAF7A6;")
        button_layout.addWidget(self.btn_camera)
        
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def select_image(self):
        if self.is_camera_running:
            self.toggle_camera()
            
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
            results = self.model(self.current_image_path, device='cpu') 
            
            image_with_boxes, result_text = self.process_results(results)
            
            self.display_image(image_with_boxes, self.tagged_panel)
            self.results_label.setText(result_text)
            
            self.tagged_pixmap = self.convert_cv_to_pixmap(image_with_boxes)
            self.btn_save.setEnabled(True)

    def save_image(self):
        if self.tagged_pixmap:
            file_path, _ = QFileDialog.getSaveFileName(self, "Resmi Kaydet", "", "PNG Dosyası (*.png);;JPG Dosyası (*.jpg)")
            
            if file_path:
                self.tagged_pixmap.save(file_path)

    def toggle_camera(self):
        if self.is_camera_running:
            self.is_camera_running = False
            self.timer.stop()
            if self.camera:
                self.camera.release()
                self.camera = None
            
            self.btn_camera.setText("Ekstra: Kamerayı Başlat")
            self.btn_camera.setStyleSheet("font-size: 14px; padding: 10px; background-color: #DAF7A6;")
            self.btn_select.setEnabled(True)
            self.btn_test.setEnabled(False)
        
        else:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.results_label.setText("Hata: Kamera açılamadı.")
                self.camera = None
                return
            
            self.is_camera_running = True
            self.timer.start(33) 
            
            self.btn_camera.setText("Kamerayı Durdur")
            self.btn_camera.setStyleSheet("font-size: 14px; padding: 10px; background-color: #FFC300;")
            self.btn_select.setEnabled(False)
            self.btn_test.setEnabled(False)
            self.btn_save.setEnabled(False)

    def update_camera_frame(self):
        if not self.is_camera_running or not self.camera:
            return

        ret, frame = self.camera.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_image(rgb_frame, self.original_panel)

            results = self.model(frame, device='cpu', verbose=False)
            
            image_with_boxes, result_text = self.process_results(results)
            
            self.display_image(image_with_boxes, self.tagged_panel)
            self.results_label.setText(result_text)

    def process_results(self, results):
        image_with_boxes = results[0].plot()
        
        names = self.model.names
        detected_classes = [names[int(c)] for c in results[0].boxes.cls]
        counts = Counter(detected_classes)
        
        if not counts:
            result_text = "Tespit Edilen Nesneler: Hiçbir şey bulunamadı."
        else:
            result_text = "Tespit Edilen Nesneler: " + ", ".join([f"{count} {name}" for name, count in counts.items()])
            
        return image_with_boxes, result_text

    def convert_cv_to_pixmap(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)

    def display_image(self, img, label_panel):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(label_panel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label_panel.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        if self.is_camera_running:
            self.timer.stop()
            self.camera.release()
        event.accept()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
