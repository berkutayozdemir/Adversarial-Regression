# Adversarial Regression (Conditional GAN)

## 🇬🇧 Project Overview
This project implements a **Conditional Generative Adversarial Network (cGAN)** for a regression task. Instead of a standard predictive model, we use a **Generator** that learns to produce realistic target values by competing against a **Discriminator**.

### Key Features
*   **Architecture**:
    *   **Generator**: `(Inputs + Noise) -> Hidden Layers -> Prediction`
    *   **Discriminator**: `(Inputs + Target) -> Hidden Layers -> Real/Fake Probability`
*   **Dataset**: Diabetes Dataset (Regression)
*   **Optimization**:
    *   Fixed **Mode Collapse** by scaling target variables and optimizing batch size (64).
    *   Achieved stable convergence where the Generator successfully learns the data distribution.
*   **Analysis**:
    *   Comparison of Probability Density Functions (PDF) between Real and Generated data.

### How to Run
```bash
# 1. Activate Environment
source venv/bin/activate

# 2. Train the GAN
python run_pipeline.py

# 3. Generate Reports
python visualize_density.py
python generate_report.py
```

---

## 🇹🇷 Proje Özeti
Bu proje, regresyon görevi için bir **Koşullu Üretken Çekişmeli Ağ (cGAN)** uygulamaktadır. Standart bir tahmin modeli yerine, bir **Ayırt Edici (Discriminator)** ile rekabet ederek gerçekçi hedef değerler üretmeyi öğrenen bir **Üretici (Generator)** kullanıyoruz.

### Temel Özellikler
*   **Mimari**:
    *   **Üretici**: `(Girdiler + Gürültü) -> Gizli Katmanlar -> Tahmin`
    *   **Ayırt Edici**: `(Girdiler + Hedef) -> Gizli Katmanlar -> Gerçek/Sahte Olasılığı`
*   **Veri Seti**: Diyabet Veri Seti (Regresyon)
*   **Optimizasyon**:
    *   Hedef değişkenlerin ölçeklendirilmesi ve toplu iş boyutunun (batch size) küçültülmesi (64) ile **Mod Çökmesi (Mode Collapse)** sorunu çözüldü.
    *   Üreticinin veri dağılımını başarıyla öğrendiği kararlı bir yakınsama sağlandı.
*   **Analiz**:
    *   Gerçek ve Üretilen veriler arasındaki Olasılık Yoğunluk Fonksiyonlarının (PDF) karşılaştırılması.

### Nasıl Çalıştırılır
```bash
# 1. Ortamı Aktifleştir
source venv/bin/activate

# 2. GAN'ı Eğit
python run_pipeline.py

# 3. Raporları Oluştur
python visualize_density.py
python generate_density_report.py
```
