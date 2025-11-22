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

## Cara Menggunakan Model (Inference Langsung)

### Langkah 0 – Download Dataset
Karena ukuran dataset cukup besar, dataset tidak disertakan langsung dalam repository.

Silakan download dataset melalui link berikut:  
👉 https://drive.google.com/file/d/1pR7ul4iOmhXq06Fhwj4TsSibc49KxS5U/view?usp=drive_link

File yang akan terunduh:
```
dataset.rar
```

Setelah selesai di-download:
1. Pindahkan file `dataset.rar` ke dalam folder proyek:
   ```
   EksplorasiVT/dataset/
   ```
2. Klik kanan → **Extract Here**

Pastikan setelah diekstrak, struktur folder menjadi seperti ini:

```text
EksplorasiVT/
│
├─ dataset/
│   ├─ 0001.jpg
│   ├─ 0002.jpg
│   ├─ 0003.jpg
│   └─ ...
└─ ...
```

⚠️ Jangan sampai gambarnya berada di dalam subfolder tambahan seperti:
```
dataset/dataset/0001.jpg  ❌ (SALAH)
```
Yang benar:
```
dataset/0001.jpg ✅
```

---

## Cara Menggunakan Model (Inference Langsung)

### Langkah 1 – Persiapan file
1. Download dan ekstrak seluruh file proyek ke dalam satu folder, misalnya:
   ```
   EksplorasiVT/
   ```

2. Pastikan struktur folder akhir seperti berikut:

```text
EksplorasiVT/
│
├─ dataset/
│   ├─ 0001.jpg
│   ├─ 0002.jpg
│   ├─ 0003.jpg
│   └─ ...
│
├─ inference_klasifikasi.ipynb
├─ uji_coba.csv
├─ test.csv
└─ checkpoints/
    ├─ vit_tiny_best.pth
    └─ deit_tiny_best.pth
```

---

### Langkah 2 – Menjalankan model
1. Buka file:
   ```
   inference_klasifikasi.ipynb
   ```

2. Jalankan seluruh cell dari atas ke bawah.

3. Model akan:
   - Membaca daftar gambar dari `test.csv`
   - Mengklasifikasikan gambar menggunakan model terpilih
   - Menghasilkan file:
     ```
     jawaban_klasifikasi.csv
     ```
     yang berisi nama file dan label hasil prediksi.

---

## Mengubah Gambar yang Ingin Diklasifikasikan

Untuk mengganti gambar uji:

1. Edit file berikut:
   - `uji_coba.csv`
   - atau `test.csv`

2. Pastikan format CSV seperti ini:
```csv
filename
0001.jpg
0002.jpg
0003.jpg
```

3. Simpan, lalu jalankan kembali notebook `inference_klasifikasi.ipynb`.

---

## Mengganti Model yang Digunakan

Secara default sistem menggunakan model ViT Tiny.

Jika ingin menggunakan model DeiT Tiny, ubah bagian berikut di dalam notebook:

```python
MODEL_NAME = "vit_tiny_patch16_224"
CHECKPOINT_PATH = "checkpoints/vit_tiny_best.pth"
```

menjadi:

```python
MODEL_NAME = "deit_tiny_patch16_224"
CHECKPOINT_PATH = "checkpoints/deit_tiny_best.pth"
```

Kemudian jalankan ulang seluruh cell notebook.

---

## Catatan Penting

- Pastikan nama file gambar di CSV **persis sama** dengan yang ada di folder `dataset/`.
- Jangan mengubah struktur folder kecuali benar-benar paham path yang digunakan.
- Jika model tidak bisa dimuat, pastikan file `.pth` ada di folder `checkpoints/`.

---

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
# MODEL_NAME = "deit_tiny_patch16_224"   # hapus komentar untuk ganti model

CHECKPOINT_PATH = "checkpoints/vit_tiny_best.pth"
# CHECKPOINT_PATH = "checkpoints/deit_tiny_best.pth" # hapus komentar untuk ganti model

DATA_DIR = "dataset"
TEST_CSV = "uji_coba.csv"
OUTPUT_CSV = "jawaban_klasifikasi.csv"
KUNCI_JAWABAN_CSV = "test.csv"
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
