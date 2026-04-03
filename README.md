# NeuroScan AI — Brain Tumor MRI Segmentation & Staging

> Research prototype · Not for clinical use

---

## Project Structure

```
brain_tumor_app/
├── app.py                  ← Flask app (auth, routes, prediction)
├── training_for_grading.py
├── collab_download_masking_training.py     ← Multi-GPU distributed training script
├── requirements.txt
├── models/
│   └── best_brats_model.h5
│   └── best_grading_model.h5 ← Place your trained model here
├── static/
│   └── uploads/            ← Uploaded MRI files (auto-created)
├── flask_session/          ← Server-side session files (auto-created)
└── templates/
    ├── base.html
    ├── index.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    └── no_cookies.html
```

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install tensorflow             # GPU: pip install tensorflow[and-cuda]

# 3. Place your trained model
cp /path/to/best_brats_model.h5 models/

# 4. Run the app
python app.py
# → http://127.0.0.1:5000
```

---

## Security Architecture

| Feature | Implementation |
|---|---|
| Password hashing | `bcrypt` with cost factor 12 (auto-salted) |
| Session storage | Server-side filesystem (Flask-Session) — not JWT |
| Cookie flags | `HTTPONLY=True`, `SAMESITE=Lax` |
| Cookie probe | Login checks `session["cookie_probe"]` set on `/`; redirects to warning if missing |
| Session token | Random hex token stored per-session (CSRF mitigation) |
| File validation | Extension whitelist + `secure_filename()` + UUID rename |
| Input sanitization | All form inputs stripped/lowercased before use |

### Production Hardening Checklist
- [ ] Replace `app.secret_key` with env var: `SECRET_KEY=... python app.py`
- [ ] Set `SESSION_COOKIE_SECURE = True` (requires HTTPS)
- [ ] Add rate limiting (Flask-Limiter) on `/login` and `/predict`
- [ ] Set `MAX_CONTENT_LENGTH` based on your data size


---

## Distributed Training (`parallel_train.py`)

### Single Machine — All GPUs (MirroredStrategy)
```bash
python training.py(codes)
# TensorFlow auto-detects and uses all GPUs - (Trained on Google Colab using T4 GPU Runtime)
# Effective batch size = BATCH_SIZE × num_GPUs
```


```

### Big-Data Parallel Preprocessing
The `make_dataset()` function uses:
```python
ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)  # parallel CPU decode
ds.shuffle(buffer_size=512)                             # shuffled across files
ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)        # async GPU prefetch
```
This saturates GPU compute while CPUs prepare the next batch in parallel.

---

## WHO Glioma Grading — Medical Insights

### Grade I — Pilocytic Astrocytoma
- **Incidence**: ~6% of all gliomas; mainly children/young adults
- **Characteristics**: Well-circumscribed, cystic with mural nodule on MRI
- **Treatment**: Surgical resection (often curative)
- **Prognosis**: 5-year OS >95%
- **Key markers**: BRAF fusion (most common molecular alteration)

### Grade II — Diffuse Low-Grade Glioma
- **Incidence**: ~15% of gliomas; adults 30–45 years
- **Characteristics**: Infiltrative, no enhancement, T2/FLAIR hyperintense
- **Treatment**: Watch-and-wait OR radiation + PCV/TMZ chemo
- **Prognosis**: Median OS 5–15 years (IDH-mutant far better)
- **Key markers**: IDH1/2 mutation (better prognosis), 1p/19q codeletion (oligodendroglioma)

### Grade III — Anaplastic Glioma
- **Incidence**: ~20% of gliomas; adults 40–55 years
- **Characteristics**: Active mitosis, may enhance, necrosis rare
- **Treatment**: Surgery + radiation + TMZ (Stupp protocol)
- **Prognosis**: Median OS 2–5 years
- **Key markers**: IDH mutation, ATRX loss, TERT promoter mutation

### Grade IV — Glioblastoma (GBM)
- **Incidence**: ~47% of all malignant brain tumors; median age 64
- **Characteristics**: Ring-enhancing lesion, central necrosis, surrounding FLAIR edema
- **Treatment**: Surgery → Stupp (RT 60Gy + concurrent/adjuvant TMZ)
- **Prognosis**: Median OS ~15 months; 5-year OS ~5%
- **Key markers**: MGMT methylation (predicts TMZ response), EGFR amplification, PTEN loss

---

## Staging Heuristic (Current Implementation)
The current classifier uses tumor area fraction as a proxy. For production:

1. **Extract radiomics features** (PyRadiomics) from predicted masks: shape, texture, intensity stats
2. **Train a dedicated XGBoost or MLP** on radiomics + clinical data (age, KPS, IDH status)
3. **Alternatively**: fine-tune the grade head in `parallel_train.py` on labeled BraTS data

---

## Dataset
- **BraTS 2020**: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data
- Format: `.h5` slices with `image` (240×240×4) and `mask` (240×240×3) keys


<img width="1899" height="904" alt="image" src="https://github.com/user-attachments/assets/bfa546ea-4fda-4e76-81ff-a805c4dc84e9" />


<img width="1896" height="910" alt="image" src="https://github.com/user-attachments/assets/4c5e1b66-0c8d-4e1e-92ae-4809cbf117ba" />


<img width="1899" height="903" alt="image" src="https://github.com/user-attachments/assets/6d67f8bb-27d9-49a1-baaf-a0c99582a1a9" />


<img width="1890" height="902" alt="image" src="https://github.com/user-attachments/assets/68a8871a-7726-4641-a77f-4476051e7e47" />


<img width="1895" height="902" alt="image" src="https://github.com/user-attachments/assets/e67ccc2a-2f07-4e28-b1e1-83c288f20d0c" />


<img width="1893" height="901" alt="image" src="https://github.com/user-attachments/assets/f2c8238d-c015-4c39-a1cc-31a7d88165d3" />





