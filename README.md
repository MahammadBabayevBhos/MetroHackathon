Bakı Metrosu - Süni İntellekt Monitorinq Sistemləri
Esas Module lar: "people_number_out" and "metro_dashboard"
[!IMPORTANT]
Məxfilik Bildirişi: Bu layihədə istifadə olunan sintetik datalar (video görüntülər, kadrlar və nəticələr) həssas və strateji əhəmiyyətli olduğu üçün görüntülərin və ya deployment nəticələrinin ictimai paylaşılması qadağandır. Kodlar yalnız Bakı Metrosu sisteminin spesifik kamera bucaqlarına və koordinatlarına (LineZone, Polygons) uyğun optimallaşdırılmışdır.

Bu repozitoriya Bakı Metrosunda sərnişin axınını və vaqon sıxlığını idarə etmək üçün hazırlanmış iki əsas modulu ehtiva edir.

Modul 1: Ağıllı Çıxış Sayğacı (Blacklist Məntiqi)
Bu sistem metrodan çıxan sərnişinlərin sayını dəqiq müəyyənləşdirmək üçün nəzərdə tutulub. Sistemin əsas üstünlüyü "False Positive" (səhv sayım) hallarını aradan qaldıran xüsusi alqoritmdir.

İşləmə Prinsipi:
İstiqamət Təyini: Şaquli bir LineZone (Keçid Xətti) çəkilmişdir.

Blacklist (Qara Siyahı): Əgər bir şəxs çöldən içəri (ödəniş aparatlarına tərəf) daxil olursa, onun ID-si avtomatik olaraq qara siyahıya salınır.

Filtrləmə: * Əgər xətti keçən şəxs blacklist-də yoxdursa -> Metrodan çıxan sərnişin kimi sayılır (+1).

Əgər xətti keçən şəxs blacklist-dədirsə -> Ödəniş edib geri qayıdan şəxs hesab olunur və sayılmır.

Vizual Etiket: Kadrdakı sərnişinlər [CX] (Çıxan) və ya [OD] (Ödəniş üçün daxil olan) etiketləri ilə markalanır.

Modul 2: Vaqon Sıxlığı Monitorinqi (Streamlit Dashboard)
Bu modul qatar vaqonlarının daxilindəki sıxlığı real-vaxt rejimində analiz edir və dispetçer üçün vizuallaşdırır.

Texniki Özəlliklər:
Çoxsaylı Kamera Dəstəyi: Eyni anda 5 fərqli vaqonun (kamerasının) analizi.

Polygon Zone: Hər vaqon üçün xüsusi perspektiv zonası (ROI) təyin edilmişdir ki, kənarda qalan obyektlər sayıma təsir etməsin.

Ağıllı Kalibrasiya: İlk 10 saniyə ərzində sistem vaqonun strukturunu və stabil vəziyyətini öyrənir (Peak-Hold məntiqi).

Sıxlıq Səviyyələri:

🟢 NORMAL: 0 – 12 sərnişin.

🟠 SIX: 13 – 20 sərnişin.

🔴 ÇOX SIX: 20+ sərnişin.

Dashboard İnterfeysi:
Orbitron Futuristik Dizayn: Metro infrastrukturuna uyğun qaranlıq və neon mövzulu interfeys.

Dinamik İkonlar: Sərnişin sayına uyğun vizual insan ikonlarının dəyişməsi.

Stabilizasiya: SMOOTH_BUF_SIZE və CHANGE_THRESHOLD parametrləri sayəsində rəqəmlərin ani sıçrayışları (titrəmələri) hamarlanır.

Tələblər və Quraşdırma
Sistemi işə salmaq üçün aşağıdakı kitabxanaların quraşdırılması mütləqdir:

Bash
pip install ultralytics supervision streamlit opencv-python numpy
Proqramı Başlatmaq:
Çıxış Sayğacı üçün: python cixis_saygaci.py

Dashboard üçün: streamlit run vaqon_monitorinq.py

Memarlıq (Architecture)
Detector: YOLOv8 (Nano və Medium modelləri).

Tracker: ByteTrack (İnsanların kadrda itməməsi üçün).

Analiz: Supervision (Zonaların və xəttlərin idarəsi).

UI: Streamlit & OpenCV.

Qeyd: Modellər CPU üzərində işləmək üçün optimallaşdırılmışdır (imgsz=640).

