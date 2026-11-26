# Klasifikasi Jenis Retakan Jalan Menggunakan SVM dan XGBoost (GBDT)

Repository ini berisi proyek akhir **Mata Kuliah Pengolahan Citra Digital (PCD)** oleh **Kelompok 3**. Proyek ini bertujuan untuk mengklasifikasikan tiga jenis retakan jalan, yaitu *Alligator Cracking*, *Longitudinal Cracking*, dan *Transverse Cracking*, menggunakan metode pengolahan citra digital dan machine learning.

Penelitian ini menggunakan kombinasi fitur tekstur, orientasi, serta morfologi dan topologi yang kemudian diklasifikasikan menggunakan **Support Vector Machine (SVM)** dan **Extreme Gradient Boosting (XGBoost)**.

## 👥 Anggota Kelompok 3

- **Ahmad Farhan Hidayat** — 25/559740/PA/23558
- **Muhammad Gauza Faliha** — 25/555851/PA/23315
- **Chelsea Christofera Antonioli Purnomo** — 25/558145/PA/23462
- **Muhammad Fahri Nur Sya'bani** — 25/561997/PA/23695
- **Saktyaveshavatar Dharmesthabuddhi** — 25/566653/PA/23920

## 📂 Dataset

Penelitian ini menggunakan dua sumber dataset:

1. **[PaveBench](https://doi.org/10.48550/arXiv.2604.02804)** — 8.987 citra.
2. **[Pavement Cracks - Roboflow](https://universe.roboflow.com/wzhen-vt-edu/pavement-cracks-2wi3m)** — 2.329 citra.

Dari kedua dataset tersebut, digunakan tiga kelas retakan yang sama, yaitu *Alligator Cracking*, *Longitudinal Cracking*, dan *Transverse Cracking*.

## ⚙️ Metodologi

Tahapan utama yang dilakukan dalam proyek ini meliputi:

1. **Preprocessing:** Grayscale conversion, Gaussian Filtering, CLAHE, dan Black Bottom-Hat Transform.
2. **Segmentation:** Otsu Thresholding, Canny Edge Detection, dan Connected Component Analysis.
3. **Feature Extraction:** GLCM, Block-LBP, Probabilistic Hough Transform, serta fitur morfologi dan topologi.
4. **Classification:** Menggunakan SVM dan XGBoost.
5. **Evaluation:** Membandingkan performa model menggunakan Macro F1-Score, F1-Score per kelas, dan Confusion Matrix.

## 🧪 Eksperimen

Eksperimen dilakukan untuk:
- Membandingkan performa SVM dan XGBoost.
- Menganalisis kontribusi setiap kelompok fitur terhadap hasil klasifikasi.
- Membandingkan hasil pelatihan menggunakan masing-masing dataset dan gabungan keduanya.

## 🛠️ Tools

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost

## 📌 Status Proyek

**In Development** — Repository ini dikembangkan sebagai bagian dari Proyek Akhir Mata Kuliah Pengolahan Citra Digital (PCD).
