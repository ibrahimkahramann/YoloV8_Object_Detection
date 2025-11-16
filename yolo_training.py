from ultralytics import YOLO
import os

def main():
    # 1. Modeli yükle
    model = YOLO('yolov8n.pt') 

    # 2. data.yaml dosyasının yolunu belirt
    data_yaml_path = os.path.join(os.getcwd(), "data.yaml")

    if not os.path.exists(data_yaml_path):
        print(f"HATA: 'data.yaml' dosyası bulunamadı. Roboflow'dan indirdiğiniz dosyanın burada olduğundan emin olun.")
        return

    print(f"data.yaml yolu: {data_yaml_path}")
    print("--- YOLOv8 Eğitimi (CPU) Başlıyor ---")

    # 3. Modeli CPU kullanarak eğit
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        device='cpu', # CPU'yu kullanmaya zorla
        project='YoloV8_Object_Detection_Runs', 
        name='fork_spoon_run_cpu'
    )

    print("--- Eğitim Tamamlandı ---")

    print(f"\nEğitim tamamlandı. Modeliniz 'YoloV8_Object_Detection_Runs/fork_spoon_run_cpu/weights/best.pt' yoluna kaydedildi.")

if __name__ == '__main__':
    main()