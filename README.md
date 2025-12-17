# YOLOv8 ile Elma ve Adaptör Nesnelerinin Algılanması

Bu çalışma, BLG407 Makine Öğrenmesi dersi kapsamında gerçekleştirilmiş bir
nesne tespiti uygulamasıdır.
Projede amaç, YOLOv8 mimarisi kullanılarak **elma** ve **şarj adaptörü**
nesnelerinin görüntüler üzerinde otomatik olarak tanınmasıdır.

Çalışmada kullanılan tüm görseller, farklı ortam ve koşullarda
**tarafımdan çekilmiş** olup herhangi bir hazır veri seti kullanılmamıştır.

---

## 1️⃣ Veri Seti Hazırlığı

Veri seti oluşturulurken modelin genelleme yeteneğini artırmak için
çeşitli sahne koşulları tercih edilmiştir:

- Farklı arka planlar
- Değişken ışık seviyeleri
- Nesnelerin tekli ve birlikte bulunduğu sahneler

| Nesne Türü | Görüntü Adedi |
|-----------|---------------|
| Elma | 100 |
| Adaptör | 100 |
| **Toplam** | **200** |

Tüm görüntüler **LabelImg** aracı kullanılarak elle etiketlenmiş
ve YOLOv8 formatına uygun şekilde kaydedilmiştir.

---

## 2️⃣ Etiketleme Formatı

Bu projede YOLO’nun standart etiketleme yapısı kullanılmıştır:

```text
class_id  x_center  y_center  width  height
```


## 3️⃣ Model Eğitimi

Model eğitimi Google Colab ortamında gerçekleştirilmiştir.
YOLOv8n mimarisi seçilmiş ve aşağıdaki eğitim ayarları kullanılmıştır:

• Görüntü boyutu: 640 × 640
• Epoch: 60
• Batch size: 8
• Varsayılan veri artırma teknikleri

```text
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="/content/dataset/data.yaml",
    epochs=60,
    imgsz=640,
    batch=8
)

```
