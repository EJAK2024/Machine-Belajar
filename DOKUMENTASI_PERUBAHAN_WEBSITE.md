# Dokumentasi Perubahan Website Klasifikasi

File: `website_klasifikasi.html` (kanonik, di `tgs 2/`)
Salinan submission: `tgs 2/240180157_M. Erza Putra Yadi/WEB_Multi-Class.html`
Versi lama yang disimpan sebagai acuan: `website_backup.html` (202.974 byte)
Versi sekarang: **211.584 byte**

Isi dokumen ini: apa saja yang berubah, apa bedanya dengan versi lama yang
**belum punya evaluasi confusion matrix**, dan di mana letak kodenya.

---

## Ringkasan cepat

| Aspek | Sebelum | Sesudah |
|---|---|---|
| Evaluasi Naive Bayes | angka akurasi saja (99,55%) | **22×22 confusion matrix + precision/recall/F1 per kelas** |
| Protokol evaluasi NB | 80/20 split, sama seperti KNN/CART | **5-fold stratified cross-validation, out-of-fold**, hanya di 660 baris berlabel |
| Data NB | `multi.csv` (2.200 baris, semua berlabel) | **`multi_semi_label.csv`** (660 berlabel + 1.540 tanpa label) |
| Cara NB belajar | supervised biasa | **semi-supervised (self-training)**, 3 putaran, 1.503 pseudo-label |
| Data KNN & CART | `multi.csv`, fitur float | `multi.csv` **dibulatkan ke int** |
| Nilai input di browser | float apa adanya | **dibulatkan dulu** (`rint`) sebelum diskoring |
| Hasil prediksi | metode terpilih saja | **ketiga metode dihitung & dibandingkan** |
| Alasan kenapa metode menang | tidak ada | **daftar alasan berperingkat** |
| Tabel Contoh Data | tidak ada | **1.540 baris tanpa label, 50 baris per klik** |
| Ukuran pohon CART | 73 node / 37 daun / kedalaman 18 | **95 node / 48 daun / kedalaman 20** |
| Angka akurasi KNN | 99,09% | **97,73%** |
| Angka akurasi CART | 97,95% | **98,18%** |

---

## 1. Evaluasi Naive Bayes + Matriks Kekeliruan (perubahan utama)

### Dulu (sebelumnya)

Naive Bayes cuma ditampilkan sebagai **satu angka akurasi 99,55%**, dihitung
dengan protokol 80/20 yang sama seperti KNN dan CART. Tidak ada cara melihat
kelas mana yang sering tertukar dengan kelas mana. Di seluruh file versi lama
kata `confusion` dan `matriks` **tidak muncul sama sekali** (0 kemunculan).

Artinya, angka 99,55% itu terlihat sangat bagus, tapi Anda tidak tahu apakah
kebanyakan benar karena kelasnya mudah, atau karena memang modelnya bagus.

### Sekarang

Saat **Naive Bayes dipilih**, di bawah hasil prediksi muncul blok baru
**"Evaluasi Naive Bayes"** yang berisi:

1. **Matriks kekeliruan 22 × 22** — baris = kelas sebenarnya, kolom = kelas
   hasil prediksi. Angka di diagonal = benar, angka di luar diagonal = salah
   klasifikasi, dan ke mana kelas itu tersesat. Jumlah seluruh sel = 660.
2. **Metrik per kelas** — precision, recall, F1, dan support untuk tiap dari
   22 jenis tanaman.
3. **Rata-rata makro & tertimbang** — ringkasan di atas seluruh kelas.

**Protokol evaluasinya juga berubah** (ini yang membuat angkanya jujur):

| | Sebelum | Sesudah |
|---|---|---|
| Protokol | 80/20 split | **5-fold stratified cross-validation** |
| Data dievaluasi | data uji hasil split | **out-of-fold prediction** di 660 baris berlabel |
| `random_state` | 42 | 42 |
| Jumlah baris dinilai | ~440 | **660** (semua baris berlabel, tiap baris dinilai oleh model yang tidak pernah melihatnya) |

Kenapa bukan 80/20? Karena setelah semi-supervised, model NB dilatih pada
**2.163 baris** (660 berlabel + 1.503 pseudo-label). Data 80/20 biasa akan
bocor: baris yang dianggap "uji" justru ikut dipakai melabeli dirinya sendiri.
Cross-validation out-of-fold di 660 baris berlabel menghindari itu.

### Angka evaluasi NB sekarang

- Akurasi cross-validation: **98,64%** (651 benar dari 660)
- Precision / recall / F1 makro dan tertimbang: **0,9878 / 0,9864 / 0,9863**
- Seluruh **9 kesalahan** terkonsentrasi di 4 kasus:
  - pomegranate → coconut (5×)
  - rice → jute (3×)
  - jute → papaya (1×)

> Perhatikan: angka NB **turun** dari 99,55% (semu) ke 98,64% (jujur).
> Ini justru tanda metriknya benar — angka sebelumnya terlalu optimis karena
> model menilai dirinya sendiri pada data yang pernah ia lihat.

### Di mana kodenya

- `PARAMS.meta.nbEval` — matriks, metrik per kelas, makro/tertimbang, protokol
- `renderNbEval(det)` — merender tabel evaluasi, hanya dipanggil bila metode NB
- `PARAMS.meta.nbData` — riwayat self-training (lihat bagian 2)

---

## 2. Data Naive Bayes: semi-supervised (self-training)

**Sebelum:** semua metode memakai `multi.csv`, 2.200 baris, semuanya berlabel.

**Sekarang:**

| Metode | Sumber data |
|---|---|
| KNN, CART | `multi.csv` (dibulatkan) |
| **Naive Bayes** | **`multi_semi_label.csv`** — 660 baris berlabel + 1.540 baris tanpa label |

Cara NB belajar sekarang bersifat **semi-supervised**:

1. Latih Gaussian NB pada **660 baris berlabel**.
2. Tebak 1.540 baris sisanya; baris dengan keyakinan ≥ **90%** diberi
   *pseudo-label* (label hasil tebakan sendiri).
3. Latih ulang pada gabungan berlabel + pseudo-label. Ulangi sampai tidak ada
   baris baru yang lolos ambang.

Riwayatnya tersimpan di `PARAMS.meta.nbData`:

| Putaran | Pseudo-label baru | Sisa tanpa label |
|---|---|---|
| 1 | 1.489 | 51 |
| 2 | 13 | 38 |
| 3 | 1 | 37 |

Hasil akhir: **660 + 1.503 = 2.163 baris** untuk latih, **37 baris** tidak
dipakai (tidak berlabel dan keyakinannya di bawah 90%).

Selisih σ² memakai **ddof=1** (rumus kuliah). Jangan bandingkan dengan
`GaussianNB.var_` bawaan sklearn, yang memakai ddof=0.

---

## 3. Pembulatan fitur (rounding)

**Sebelum:** fitur disimpan dalam float penuh, dan nilai yang Anda ketik di
browser dipakai apa adanya.

**Sekarang:**

1. Kolom **K, temperature, humidity, ph, rainfall** pada `multi.csv`
   dibulatkan ke bilangan bulat sebelum di-embed (N dan P memang sudah int).
   Jadi fitur KNN/CART kini berada di grid yang sama persis dengan
   `multi_semi_label.csv`.
2. **Semua nilai input dibulatkan dulu** lewat fungsi `rint` (banker's
   rounding, sama seperti `np.rint`) sebelum dipakai menghitung — berlaku
   untuk ketiga metode, bukan hanya KNN.

Kenapa ini penting: `ph` memiliki **variance nol di 6 kelas NB** (apple,
banana, coconut, grapes, kidneybeans, papaya). Kalau input dibiarkan float,
σ² kelas-kelas itu jatuh ke nilai jepit `1e-9` dan kelas-kelas tersebut
"runtuh". Akibatnya akurasi NB di seluruh 2.200 baris jatuh ke **72,2%**.
Setelah dibulatkan: **98,68%**.

Konsekuensi yang terlihat: preset "Contoh 1" kini menampilkan nilai bulat,
sedangkan di dokumen `KNN, Naive Bayes, CART.docx` nilainya masih float.

---

## 4. Panel perbandingan tiga metode

**Sebelum:** hanya metode yang Anda pilih yang dihitung.

**Sekarang:** `hitungMetode(m, vals)` menghitung **ketiga metode** untuk data
yang sama, lalu `renderPerbandingan(det, semua, res)` menampilkan:

- **Tabel perbandingan** 3 baris: metode, prediksi, keyakinan, akurasi uji,
  protokol. Metode yang Anda pilih disorot; metode dengan akurasi uji
  tertinggi diberi tanda ★.
- **Banner sepakat / berselisih** — bila ketiga metode sepakat, hijau. Bila
  ada yang berbeda, ditampilkan pembagian suara, mis. `2× jute, 1× rice`.
- **Kotak alasan berperingkat** — kenapa metode tertentu dianggap paling
  akurat, urutannya mengikuti angka akurasi saat ini, bukan hardcode.

Peringkat akurasi berubah setelah pembulatan:

| Peringkat | Sebelum | Sesudah |
|---|---|---|
| 1 | Naive Bayes 99,55% | **Naive Bayes 98,64%** (5-fold CV) |
| 2 | KNN 99,09% | **CART 98,18%** (80/20) |
| 3 | CART 97,95% | **KNN 97,73%** (80/20) |

Angka-angka yang dipakai di teks alasan sudah diverifikasi terhadap data:

- Korelasi K–P = **0,74** → asumsi independensi naif terlanggar (baik untuk NB
  maupun intuisi jarak KNN).
- Rasio antar-varians kelas / within-varians: K ≈ **263×**, humidity ≈ **30×**.
- Akurasi resubstitusi di 2.200 baris membulat: CART **100%** (rakit berlebih),
  NB **2.171**, KNN **2.169**.

---

## 5. Tabel Contoh Data

**Sebelum:** Anda harus mengetik sendiri tujuh nilai fitur.

**Sekarang**, di kartu *Input Data Uji* (Langkah 2), di bawah tombol Proses:

- **1.540 baris tanpa label** dari `multi_semi_label.csv` ditampilkan dalam
  tabel, **50 baris per klik** tombol *Muat 50 baris lagi*.
- Kolom: `#`, N, P, K, Suhu, Kelembapan, pH, Curah Hujan, Label (tampil `?`),
  dan tombol `Uji`.
- **Klik baris mana pun** (atau tombol `Uji`-nya) untuk mengisikan form uji
  dengan tujuh nilai baris itu, lalu tinggal pilih metode dan tekan Proses.
- Header tabel menempel (sticky), kotak tabel tinggi maksimal 360px, dan baris
  bisa diakses lewat keyboard (Enter / Space).

---

## 6. Yang TIDAK berubah

- Trek biner / notebook `Regresi_Logistik_Biner.ipynb`, file `tes 1.xlsx`,
  `biner.csv`, dan `model_logreg_tumor.pkl` — tidak tersentuh.
- KNN tetap default **k = 5** (pilihan 1/3/5/7/9), tie-break tetap
  suara → total jarak → label.
- Tiga preset ("Contoh 1/2/3") tetap, hanya nilainya kini membulat.
- Protokol KNN/CART tetap 80/20 stratified, `random_state = 42`.
- Seluruh konten berbahasa Indonesia.
- Website tetap **offline satu file** — tanpa server, tanpa `fetch`.
- Konvensi DOM v1: `insertAdjacentHTML` untuk HTML statis, `.innerHTML =`
  hanya pada elemen yang baru dibuat. Jangan perkenalkan `Element.setHTML()`.

---

## 7. Cara membaca tampilan baru

1. Pilih metode di Langkah 1.
2. Isi data lewat preset, **Contoh Data**, atau ketik manual. Nilai otomatis
   dibulatkan ke bilangan bulat.
3. Tekan *Proses & Prediksi*.
4. Lihat **hasil utama** (prediksi metode terpilih).
5. Lihat **tabel perbandingan** — ketiga metode, siapa yang menang, dan kenapa.
6. Bila metode terpilih **Naive Bayes**, gulir sedikit untuk melihat
   **matriks kekeliruan 22×22** dan metrik per kelasnya.

---

## 8. Peta kode

| Bagian | Fungsi / konstanta |
|---|---|
| Data KNN & CART | `MULTI_CSV` (2.200 baris, fitur int) |
| Data NB | `SEMI_CSV` (`multi_semi_label.csv`) |
| Parameter model | `PARAMS.tree`, `PARAMS.nb`, `PARAMS.meta` |
| Pembulatan input | `rint(v)` |
| Hitung satu metode | `hitungMetode(m, vals)` — sekarang membulatkan input |
| Hitung semua metode | `hitung(vals)` → `hitungMetode(method, vals)` |
| Panel perbandingan | `renderPerbandingan(det, semua, res)` |
| Matriks kekeliruan NB | `renderNbEval(det)` |
| Ambil data NB | `getSemi()` → `{rows, labels, labeled, unlabeled}` |
| Tabel Contoh Data | `contohBaris()`, `muatContoh()`, `isiDariContoh(i)` |

---

## Catatan verifikasi

Seluruh klaim di atas sudah diperiksa:

- Akurasi CART di JS **sama persis** dengan sklearn di 2.200 baris membulat
  (0 selisih). KNN berbeda di **1 baris** (row 408) akibat tie-break.
- `node --check` lolos, penyeimbangan HTML 0 kesalahan.
- Tabel Contoh Data: 50 → 100 → 1.540 baris, tidak ada baris berlabel yang
  bocor, pengisian form cocok dengan sumber.
- `website_klasifikasi.html` dan `WEB_Multi-Class.html` identik byte-per-byte
  (MD5 `CCBD70B38E5A5DB5E6CA35335B7C01BF`).
- `AGENTS.md` sudah diperbarui mengikuti semua perubahan di atas.
