# Baki Metrosu : Suni Intellekt Monitorinq Sistemleri

Bu repozitoriya Baki Metrosunda sernisin axinini ve vaqon daxilindeki sixligi real-vaxt rejiminde teyin etmek ucun hazirlanmis suni intellekt modullarini ehtiva edir.

## Esas Modullar

### 1: Agilli Cixis Saygaci (people_number_out.py)
Metrodan cixan sernisinlerin sayini deqiq mueyyenlesdirmek ve sehv sayimlarin (False Positive) qarsisini almaq ucun istiqamet ve blacklist alqoritmi tetbiq edilir.

* Istiqamet Teyini: Saquli LineZone vasitesile kecid istiqametinin izlenmesi.
* Blacklist Mentiqi: Odenis aparatlarina teref daxil olan sernisin ID-leri qara siyahiyab alinir ve cixis sayina elave edilmir.
* Etiketleme: [CX] (Cixan sernisin) ve [OD] (Odenis ucun daxil olan sernisin).

### 2: Vaqon Sixligi Monitorinqi (metro_dashboard.py)
Qatar vaqonlarindaki sernisin sixligini real-vaxt rejiminde Streamlit interfeysi uzerinde vizuallasdirir.

* Coxsayli Kamera Desteyi: Eyni anda 5 vaqon kamerasinin analizi.
* Polygon Zone (ROI): Vaqon daxilindeki perspektiv zonasinin ayrilmasi.
* Ağıllı Kalibrasiya ve Hamarlama: Ilk kadrlar uzre sabit parametrlerin oyrenilmesi ve ani sicrayislarin filtrlenmesi.
* Sixliq Kateqoriyalari:
  * 0:12 sernisin : NORMAL
  * 13:20 sernisin : SIX
  * 20+ sernisin : COX SIX

## Qurasdirma

Teleb olunan kitabxanalari qurasdirmaq ucun:

```bash
pip install ultralytics supervision streamlit opencv-python numpy
```

ve ya `uv` paket meneceri ile:

```bash
uv sync
```

## Ishe Salma Qaydalari

### Agilli Cixis Saygacini Baslatmaq:
```bash
python people_number_out.py --video path/to/exit_video.mp4
```

### Vaqon Sixligi Monitorinqini (Dashboard) Baslatmaq:
```bash
streamlit run metro_dashboard.py
```

### Tek Vaqon Analizini Baslatmaq:
```bash
python frame.py --video path/to/vagon_video.avi
```

### Yeni Kamera Bucagi Ucun Koordinat Secmek:
```bash
python kordinat.py --video path/to/video.mp4
```

## Texnoloji Arxitektura

* Model: YOLOv8 (Nano ve Medium konfiqurasiyalari)
* Obiyekt Izleme: ByteTrack
* Geometrik Analiz: Supervision (LineZone ve PolygonZone)
* Istifadeci Interfeysi: Streamlit ve OpenCV

## Qeyd ve Mexfilik
Proqram tehlukesizlik ve video axinlarinin optimalligi meqsedile CPU ve GPU uzerinde minimal resurs serfiyyati ile calisacaq sekilde qurulmusdur.
