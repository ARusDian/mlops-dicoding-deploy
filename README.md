# Submission 2 — Heart Disease Risk Classification (TFX + TF-Serving + Monitoring)

<p><strong>Nama:</strong> Ahmad Rusdianto Andarina Syabani<br>
<strong>Username dicoding:</strong> arusdian</p>

<table>
<thead><tr><th>Bagian</th><th>Deskripsi</th></tr></thead>
<tbody>
    <tr>
      <td>Dataset</td>
      <td>
        <strong>Heart Disease Dataset</strong> (Public Health Dataset, UCI / Kaggle Mirror).<br>
        Data klinis pasien dari Cleveland, Hungary, Switzerland, dan Long Beach V.<br>
        Label biner <code>HeartDisease</code>: 0 = tidak, 1 = ya.<br>
        <em>Referensi:</em>
        <a href="https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction" target="_blank" rel="noopener">Kaggle – Heart Failure Prediction</a>.
      </td>
    </tr>
    <tr>
      <td>Masalah</td>
      <td>
        Memprediksi kemungkinan penyakit jantung berdasarkan fitur klinis tabular (usia, tekanan darah, kolesterol, dsb).<br>
        Kategori masalah: <strong>binary classification</strong>.
      </td>
    </tr>
    <tr>
      <td>Solusi machine learning</td>
      <td>
        Membangun <strong>pipeline TFX end-to-end</strong> pada <em>InteractiveContext</em>:<br>
        ExampleGen → StatisticsGen → SchemaGen → ExampleValidator → Transform → <u>Tuner</u> → Trainer → (Resolver) → Evaluator → Pusher.<br>
        Model diekspor sebagai <em>SavedModel</em> dengan signature <code>serving_default</code> yang menerima input <code>tf.Example</code> mentah.
      </td>
    </tr>
    <tr>
      <td>Metode pengolahan</td>
      <td>
        <ul>
          <li><strong>Numerik:</strong> Normalisasi z-score dengan <code>tft.scale_to_z_score</code> untuk <em>Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak</em>.</li>
          <li><strong>Kategorikal:</strong> <code>tft.compute_and_apply_vocabulary</code> (OOV=1) untuk <em>Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope</em>, lalu one-hot (drop-first) di Trainer.</li>
          <li><strong>Label:</strong> <code>HeartDisease</code> → <code>int64</code>.</li>
          <li>Anti-overfit & imbalance: L2, Dropout, BatchNorm, EarlyStopping, ReduceLROnPlateau.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Arsitektur model</td>
      <td>
        Concatenate(numerik + one-hot kategori) → Dense(96, ReLU, L2) → BN → Dropout(0.3) →<br>
        Dense(16, ReLU, L2) → BN → Dropout(0.3) → Dense(1, Sigmoid).<br>
        Optimizer Adam (lr 3e-4), Loss Binary Crossentropy (label smoothing 0.03).
      </td>
    </tr>
    <tr>
      <td>Metrik evaluasi</td>
      <td>
        TFMA: Binary Accuracy, AUC, Precision, Recall. Tambahan: Confusion Matrix & threshold (uji manual).
      </td>
    </tr>
    <tr>
      <td>Performa model</td>
      <td>
        <ul>
          <li><strong>Train (best epoch):</strong> Accuracy 0.9306 • AUC 0.9797</li>
          <li><strong>Validation (TFMA Overall):</strong> Accuracy 0.848 • AUC 0.929</li>
          <li><strong>Confusion Matrix (eval, 210 contoh):</strong> TN=76, FP=14, FN=18, TP=102</li>
        </ul>
        Catatan: Tuning meningkatkan stabilitas tanpa overfit berlebih.
      </td>
    </tr>
    <tr>
      <td>Best Hyperparameters (Tuner – Bayesian)</td>
      <td>
        <ul>
          <li><strong>units1</strong>: 96</li>
          <li><strong>units2</strong>: 16</li>
          <li><strong>dropout</strong>: 0.3</li>
          <li><strong>l2</strong>: 0.000031</li>
          <li><strong>lr</strong>: 0.0003</li>
          <li><strong>label_smoothing</strong>: 0.03</li>
        </ul>
      </td>
    </tr>
<tr>
<td>Opsi Deployment</td><td><strong>TensorFlow Serving 2.15</strong> melalui <strong>Dockerfile custom</strong> yang otomatis memilih versi model terbaru di direktori <code>serving_model/</code>. Deployment dilakukan di <strong>Railway.app</strong> dengan konfigurasi menggunakan <strong>railway.toml</strong>.</td></tr>
<tr><td>Web App (Serving)</td><td>
<b>Predict (REST):</b> <a href="https://mlops-dicoding-deploy-production.up.railway.app/v1/models/heart:predict">https://mlops-dicoding-deploy-production.up.railway.app/v1/models/heart:predict</a><br>
<b>Metadata:</b> <a href="https://mlops-dicoding-deploy-production.up.railway.app/v1/models/heart/metadata">https://mlops-dicoding-deploy-production.up.railway.app/v1/models/heart/metadata</a>
</td></tr>
<tr><td>Monitoring</td><td><strong>Prometheus</strong> mengumpulkan metrik dari endpoint <code>/monitoring/prometheus/metrics</code> TensorFlow Serving, dan <strong>Grafana</strong> menampilkan dashboard interaktif untuk latency, throughput, dan request metrics.</td></tr>
</tbody>
</table>



## 1. Pipeline TFX (Ringkasan)

Pipeline terdiri dari komponen berikut:

* **ExampleGen:** membaca dataset CSV
* **StatisticsGen:** menghasilkan statistik dataset
* **SchemaGen:** menurunkan skema fitur
* **ExampleValidator:** mendeteksi anomali/missing
* **Transform:** scaling & encoding fitur
* **Tuner:** hyperparameter search (BayesianOptimization)
* **Trainer:** model MLP dengan regularisasi
* **Resolver:** memilih model terbaik sebelumnya
* **Evaluator:** validasi performa via TFMA
* **Pusher:** ekspor SavedModel ke direktori `serving_model/arusdian/<timestamp>`

---

## 2. Dockerfile Deployment

```dockerfile
FROM tensorflow/serving:2.15.1

ARG MODEL_NAME=heart
ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_DIR=/models/${MODEL_NAME}
ENV GRPC_PORT=8500
ENV MONITORING_CONFIG=/monitoring.config
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV ENABLE_ONEDNN_OPTS=0

COPY monitoring/prometheus.config ${MONITORING_CONFIG}
COPY serving_model/arusdian/ /tmp/arusdian/

RUN set -e; \
    chosen="$(ls -1 /tmp/arusdian | grep -E '^[0-9]+$' | sort -n | tail -n 1 || true)"; \
    if [ -z "$chosen" ]; then chosen="$(ls -1t /tmp/arusdian | head -n 1 || true)"; fi; \
    echo "Chosen model version: $chosen"; \
    mkdir -p "${MODEL_DIR}/1"; \
    cp -R "/tmp/arusdian/${chosen}/"* "${MODEL_DIR}/1/"; \
    test -f "${MODEL_DIR}/1/saved_model.pb" || (echo "saved_model.pb not found" && exit 1)

EXPOSE 8501 8500

CMD /bin/sh -c '/usr/bin/tf_serving_entrypoint.sh \
  --rest_api_port=${PORT:-8501} \
  --port=${GRPC_PORT:-8500} \
  --model_name=${MODEL_NAME} \
  --model_base_path=${MODEL_DIR} \
  --monitoring_config_file=${MONITORING_CONFIG}'
```

---

## 3. Deploy via Railway

**railway.toml**

```toml
[build]
dockerfilePath = "Dockerfile"
context = "."

[variables]
MODEL_NAME = "heart"
GRPC_PORT  = "8500"
MONITORING_CONFIG = "/monitoring.config"
```

**Langkah Deploy:**

1. Hubungkan repo ke Railway.
2. Pilih root folder `submission-2`.
3. Railway akan mendeteksi Dockerfile otomatis.
4. Setelah build, Railway akan menjalankan perintah `CMD` dari Dockerfile.
5. Cek log:

   * “Chosen model version”
   * “Running ModelServer at 0.0.0.0:${PORT}”
6. Endpoint aktif di:

   * `/v1/models/heart` → metadata
   * `/v1/models/heart:predict` → prediksi

<div align="center">
  <img src="arusdian-deployment.png" width="600"/>
  <p><em>Model berhasil dideploy di Railway (TensorFlow Serving 2.15)</em></p>
</div>

---

## 4. Monitoring

**monitoring/prometheus.config**

```textproto
monitoring_config {
  prometheus_config {
    enable: true
    path: "/monitoring/prometheus/metrics"
  }
}
```

**monitoring/prometheus.yml**

```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'tfserving-railway'
    scheme: https
    metrics_path: /monitoring/prometheus/metrics
    static_configs:
      - targets: [ "mlops-dicoding-deploy-production.up.railway.app:443" ]
        labels:
          app: tfserving
          model_name: heart
    tls_config:
      insecure_skip_verify: true
```

---

## 5. Dashboard Monitoring (Prometheus + Grafana)

<table>
<tr>
<td><b>Prometheus Endpoint</b><br><img src="arusdian-monitoring.png" width="480"/></td>
<td><b>Grafana Dashboard</b><br><img src="arusdian-grafana-dashboard.png" width="480"/></td>
</tr>
</table>

**Metrik utama yang dimonitor:**

* `tensorflow:serving:request_count` → jumlah request total
* `tensorflow:serving:request_latency_bucket` → distribusi latency
* `histogram_quantile(0.5, …)` → latency p50
* `histogram_quantile(0.95, …)` → latency p95
* `tensorflow:cc:saved_model:load_attempt_count` → percobaan load model
* `tensorflow:serving:runtime_latency_bucket` → runtime eksekusi model

**Dashboard JSON:** [arusdian-dashboard.json](arusdian-dashboard.json)

---

## 6. Linting

Pengecekan kode menggunakan **Pylint**:

```
pylint modules/*.py --score=y
```

<div align="center">
  <img src="arusdian-pylint.png" width="480"/>
</div>

> Rata-rata skor Pylint: **10/10**, semua modul mengikuti konvensi PEP8.

---

## 7. Reproduksibilitas & Versi

| Komponen                  | Versi  |
| ------------------------- | ------ |
| Python                    | 3.9    |
| TensorFlow                | 2.15.* |
| TFX                       | 1.15.* |
| TensorFlow Transform      | 1.15.* |
| TensorFlow Model Analysis | 0.45.* |
| Keras Tuner               | 1.4.6  |
| Apache Beam               | 2.54.0 |

---

## 8. Checklist Submission

<table>
<thead><tr><th>Item</th><th>Status</th></tr></thead>
<tbody>
<tr><td>Pipeline TFX Lengkap</td><td>✅</td></tr>
<tr><td>Notebook Dokumentatif</td><td>✅</td></tr>
<tr><td>Modul Transform, Tuner, Trainer</td><td>✅</td></tr>
<tr><td>Dockerfile + Railway.toml</td><td>✅</td></tr>
<tr><td>Model Siap Serving (serving_model/)</td><td>✅</td></tr>
<tr><td>Monitoring (Prometheus + Grafana)</td><td>✅</td></tr>
<tr><td>Linting (Pylint)</td><td>✅</td></tr>
<tr><td>Endpoint Railway Berfungsi</td><td>✅</td></tr>
<tr><td>README Lengkap</td><td>✅</td></tr>
</tbody>
</table>

---

**Catatan:**
Untuk submission Dicoding, direktori `serving_model/` wajib disertakan agar model dapat diverifikasi.
Untuk produksi, model dapat disimpan di cloud storage (misal GCS/S3) dan diunduh saat build.
