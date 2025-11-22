# VisionTransformer-Comparison
Tugas Eksplorasi VT

# Eksperimen Vision Transformer untuk Klasifikasi Makanan Indonesia

Repositori ini berisi kode untuk melatih dan membandingkan dua model Vision Transformer ringan:

- `vit_tiny_patch16_224` (ViT Tiny)
- `deit_tiny_patch16_224` (DeiT Tiny)

pada tugas klasifikasi 5 kelas makanan Indonesia:

- `bakso`
- `gado_gado`
- `nasi_goreng`
- `rendang`
- `soto_ayam`

---

## 1. Struktur Folder

Struktur minimal proyek:

```text
EksplorasiVT/
│
├─ dataset/
│   ├─ 0001.jpg
│   ├─ 0002.jpg
│   └─ ...
├─ jawaban_klasifikasi.csv
├─ train.csv
├─ val.csv
├─ test.csv
├─ uji_coba.csv
│
├─ review_history.ipynb
├─ inference_klasifikasi.ipynb
├─ train_model.ipynb
│
│  (file hasil training)
├─ checkpoints/
│   ├─ vit_tiny_best.pth
│   └─ deit_tiny_best.pth
├─ deit_tiny_summary.csv
├─ deit_tiny_history.npy
├─ deit_tiny_classification_report.txt
├─ deit_tiny_confusion_matrix.png
├─ vit_tiny_history.npy
├─ vit_tiny_classification_report.txt
├─ vit_tiny_confusion_matrix.png
└─ vit_tiny_summary.csv
```

Format file CSV:

```csv
filename,label
0001.jpg,rendang
0002.jpg,bakso
...
```

> Untuk `test.csv` saat inference, kolom `label` boleh dikosongkan atau dihapus.

---

## 2. Persiapan Lingkungan

### 2.1 Versi Python
Disarankan menggunakan:
- Python 3.10 – 3.12

### 2.2 Instalasi Dependency

Dari folder proyek:

```bash
pip install -r requirements.txt
```

Contoh isi `requirements.txt`:

```text
torch
torchvision
timm
pandas
numpy
matplotlib
scikit-learn
tqdm
Pillow
```

---

## 3. Training Model

Script utama training:
- `train_vit_food.py`

Fungsi script:
- Membaca `train.csv` dan `val.csv`
- Melatih model ViT/DeiT Tiny
- Menyimpan:
  - checkpoint terbaik ke folder `checkpoints/`
  - history ke `.npy`
  - confusion matrix
  - classification report

### 3.1 Konfigurasi Model

Di dalam `train_vit_food.py`:

```python
MODEL_NAME = "vit_tiny_patch16_224"
MODEL_TAG  = "vit_tiny"
```

Untuk melatih ViT Tiny:
```bash
python train_vit_food.py
```

Untuk DeiT Tiny, ubah menjadi:
```python
MODEL_NAME = "deit_tiny_patch16_224"
MODEL_TAG  = "deit_tiny"
```

---

## 4. Melihat Kurva Training dari `.npy`

Script:
- `lihat_history.py`

Konfigurasi:
```python
HISTORY_PATH = "vit_tiny_history.npy"
```

Jalankan:
```bash
python lihat_history.py
```

Akan menampilkan grafik:
- Train vs Val Loss
- Train vs Val Accuracy

---

## 5. Inference dan Pembuatan CSV Hasil

Script:
- `inference_klasifikasi.py`

### 5.1 Konfigurasi Utama

```python
MODEL_NAME = "vit_tiny_patch16_224"
CHECKPOINT_PATH = "checkpoints/vit_tiny_best.pth"

DATA_DIR = "dataset"
TEST_CSV = "test.csv"
OUTPUT_CSV = "jawaban_klasifikasi.csv"
KUNCI_JAWABAN_CSV = "kunci_jawaban.csv"
```

Format `kunci_jawaban.csv`:
```csv
filename,label
0001.jpg,rendang
0002.jpg,bakso
```

### 5.2 Menjalankan Inference

```bash
python inference_klasifikasi.py
```

Output:
- `jawaban_klasifikasi.csv`
- Akurasi ditampilkan di terminal

Contoh output:
```text
===== HASIL EVALUASI =====
Total sampel : 279
Benar        : 275
Salah        : 4
Akurasi      : 98.57%
```

---

## 6. Catatan Penting

### 6.1 Urutan Kelas Harus Konsisten
```python
CLASSES = ["bakso", "gado_gado", "nasi_goreng", "rendang", "soto_ayam"]
```

### 6.2 Windows Compatibility
Gunakan:
```python
NUM_WORKERS = 0
```
untuk menghindari error DataLoader.

### 6.3 Mode CPU / GPU

Default:
```python
DEVICE = torch.device("cpu")
```

Untuk GPU:
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 7. Alur Singkat Penggunaan

1. Siapkan dataset + CSV  
2. Install dependency  
3. Training model  
4. Lihat hasil training  
5. Jalankan inference  
6. Cek hasil di `jawaban_klasifikasi.csv`

---

## 8. Tujuan Proyek

Mengevaluasi dan membandingkan:
- ViT Tiny
- DeiT Tiny

pada klasifikasi makanan Indonesia, serta menganalisis trade-off antara:
- Akurasi
- Stabilitas
- Kompleksitas model
- Kecepatan inferensi CPU
