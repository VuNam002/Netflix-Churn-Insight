# Netflix – Dự đoán khách hàng rời bỏ (churn)

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Dự án Kedro (`kedro 1.7.0`) xây dựng mô hình dự đoán khách hàng Netflix có rời dịch vụ hay không, từ bộ dữ liệu `netflix_customer_churn.csv` (5.000 khách hàng, cột mục tiêu `churned`: 1 = rời, 0 = ở lại). Dự án đi qua đủ 7 bước của một quy trình Machine Learning, từ thu thập dữ liệu đến triển khai API.

## Đọc dự án từ đâu

Nếu bạn mới mở repo này, đọc theo thứ tự sau (khoảng 10 phút):

1. **README này**: bảng 7 bước và [bản đồ cấu trúc](#cấu-trúc-dự-án).
2. [`conf/base/catalog.yml`](conf/base/catalog.yml): danh sách mọi dữ liệu đi vào và đi ra, nằm ở đâu, định dạng gì.
3. [`src/netflix/pipeline_registry.py`](src/netflix/pipeline_registry.py): dự án có những pipeline nào và ghép với nhau ra sao.
4. **Một pipeline hoàn chỉnh**, bắt đầu với [`data_processing`](src/netflix/pipelines/data_processing/): đọc `pipeline.py` trước (nối các bước), rồi `nodes.py` (phép xử lý thật).
5. [`conf/base/parameters_*.yml`](conf/base/): các con số có thể chỉnh (ngưỡng, tỉ lệ chia, siêu tham số).
6. Các notebook `notebooks/02` đến `06`: chạy từng ô để thấy dữ liệu thay đổi qua từng bước.

## Quy trình 7 bước ứng với dự án

| Bước | Việc | Chạy thật (lưu file) | Khám phá (notebook) |
|---|---|---|---|
| 1. Data collection | Biết dữ liệu từ đâu, mỗi cột nghĩa gì | `conf/base/catalog.yml` khai báo dữ liệu | xem [Bước 1](#bước-1-dữ-liệu) |
| 2. Statistics | Hiểu dữ liệu qua thống kê | `kedro run --pipeline eda` | `notebooks/02_statistics.ipynb` |
| 3. Data preprocessing | Làm sạch, tạo feature, chia train/test | `kedro run` (`data_processing`, `feature_engineering`) | `notebooks/03_preprocessing.ipynb` |
| 4. Data visualization | Biểu đồ khám phá dữ liệu | `kedro run --pipeline eda` | `notebooks/04_visualization.ipynb` |
| 5. Model building | So sánh mô hình, huấn luyện cái tốt nhất | `kedro run` (`training`) | `notebooks/05_modeling.ipynb` |
| 6. Model evaluation | Chấm trên tập test, phân tích lỗi | `kedro run` (`evaluation`) | `notebooks/06_evaluation.ipynb` |
| 7. Model deployment | Chấm điểm hàng loạt và API | `kedro run --pipeline inference`, `uvicorn`, Docker | |

**Cách học đề xuất:** với mỗi bước, mở notebook tương ứng và chạy từng ô để hiểu mình đang làm gì. Khi hiểu rồi thì chạy lệnh Kedro để tạo kết quả thật. Thứ tự notebook: 02 → 03 → 04 → 05, rồi `kedro run`, rồi mới 06 (notebook 06 đọc kết quả do `kedro run` tạo ra).

## Cài đặt và chạy nhanh

```powershell
cd C:\ML\netflix
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e ".[dev]"            # cài gói netflix + pytest, ruff

kedro run                          # bước 3, 5, 6: làm sạch, feature, huấn luyện, đánh giá
kedro run --pipeline eda           # bước 2, 4: thống kê và biểu đồ
kedro run --pipeline inference     # bước 7: chấm điểm khách mới (cần new_customers.csv)
pytest                             # chạy test
```

> `pytest` bao gồm `tests/test_run.py`, bài test này chạy cả pipeline thật nên ghi đè các file trong `data/` và thêm một phiên bản mô hình mới vào `data/06_models/`. Muốn bỏ qua: `pytest --deselect tests/test_run.py`.

Chạy từng pipeline riêng bằng `kedro run --pipeline <tên>`, với tên là một trong:
`data_processing`, `feature_engineering`, `training`, `evaluation`, `statistics`, `visualization`, `inference`, `eda`.

**Xem sơ đồ pipeline dạng hình (tuỳ chọn):** `pip install kedro-viz` rồi `kedro viz run`. Đây là cách nhanh nhất để thấy dữ liệu đi qua các node nào.

## Cấu trúc dự án

```
netflix/
├── conf/                                    <- cấu hình, KHÔNG chứa code
│   ├── base/
│   │   ├── catalog.yml                      <- KHAI BÁO dữ liệu: file nào, định dạng gì, ở đâu
│   │   ├── parameters.yml                   <- tham số dùng chung (test_size, random_state, ngưỡng quyết định)
│   │   └── parameters_<pipeline>.yml        <- tham số riêng của từng pipeline
│   └── local/
│       └── credentials.yml                  <- mật khẩu, API key. KHÔNG commit
├── data/                                    <- dữ liệu, KHÔNG commit (trừ thư mục rỗng)
│   ├── 01_raw/               <- dữ liệu gốc, chỉ đọc, không bao giờ sửa
│   ├── 02_intermediate/      <- đã đổi kiểu, làm sạch sơ bộ (hiện chưa dùng)
│   ├── 03_primary/           <- bảng chuẩn, sạch (netflix_clean)
│   ├── 04_feature/           <- bảng feature (model_features)
│   ├── 05_model_input/       <- X_train, X_test, y_train, y_test
│   ├── 06_models/            <- model đã train (classifier.pkl, có phiên bản theo thời gian)
│   ├── 07_model_output/      <- kết quả dự đoán (predictions.csv)
│   └── 08_reporting/         <- metrics, bảng so sánh, biểu đồ
├── notebooks/                <- 02 đến 06: khám phá và học từng bước
├── src/netflix/              <- toàn bộ mã nguồn
│   ├── pipeline_registry.py  <- đăng ký pipeline (thêm pipeline mới thì thêm vào đây)
│   ├── settings.py           <- cấu hình framework Kedro (ít khi sửa)
│   ├── pipelines/            <- mỗi pipeline là một thư mục con (nodes.py + pipeline.py)
│   │   ├── data_processing/      làm sạch                          (bước 3)
│   │   ├── feature_engineering/  tạo feature, chia train/test      (bước 3)
│   │   ├── statistics/           thống kê                          (bước 2)
│   │   ├── visualization/        biểu đồ                           (bước 4)
│   │   ├── training/             so sánh và huấn luyện mô hình     (bước 5)
│   │   ├── evaluation/           đánh giá trên tập test            (bước 6)
│   │   └── inference/            chấm điểm khách mới hàng loạt     (bước 7)
│   └── serving/app.py        <- API FastAPI                        (bước 7)
├── tests/
│   ├── conftest.py           <- dữ liệu giả dùng chung cho test
│   ├── pipelines/<tên>/      <- test cho từng pipeline
│   ├── serving/              <- test cho API
│   └── test_run.py           <- chạy thử cả pipeline trên dữ liệu thật
├── Dockerfile, .dockerignore, requirements-serving.txt   <- đóng gói API
├── pyproject.toml            <- thông tin gói, cấu hình pytest và ruff
└── requirements.txt          <- thư viện cần cài
```

**Quy ước của mỗi pipeline:** `nodes.py` chứa các hàm Python thuần (nhận dữ liệu, trả dữ liệu, không đọc/ghi file); `pipeline.py` nối các hàm đó bằng *tên dataset*; đường dẫn file nằm ở `catalog.yml`; con số có thể chỉnh nằm ở `parameters_*.yml`. Nhờ vậy muốn đổi ngưỡng hay đổi nơi lưu file thì không phải sửa code.

**Thư mục và file tự sinh, không sửa tay:** `data/` (trừ `01_raw`, tạo ra bởi `kedro run`), `.viz/`, `*.egg-info/`, `.coverage` (đã nằm trong `.gitignore`), cùng `.ruff_cache/` và `__pycache__/` (bộ nhớ đệm).

### Bốn khái niệm Kedro cần biết

| Khái niệm | Là gì | Nằm ở đâu |
|---|---|---|
| **Node** | Một hàm Python xử lý dữ liệu | `nodes.py` |
| **Pipeline** | Danh sách node, nối nhau bằng tên dữ liệu vào/ra | `pipeline.py` |
| **Catalog** | Bảng tra: tên dữ liệu → file, định dạng | `conf/base/catalog.yml` |
| **Parameters** | Các con số có thể chỉnh, truyền vào node bằng `params:<nhóm>` | `conf/base/parameters*.yml` |

Node A xuất ra dữ liệu tên `X`, node B nhận `X` thì Kedro tự biết B chạy sau A. Dữ liệu không có trong catalog chỉ nằm trong RAM và mất khi chạy xong.

### Từng pipeline làm gì

| Pipeline | Hàm trong `nodes.py` | Đọc | Ghi |
|---|---|---|---|
| `data_processing` | `clean_data` | `netflix_raw` | `netflix_clean` |
| `feature_engineering` | `create_features`, `split_data` | `netflix_clean` | `model_features`, `X_train`, `X_test`, `y_train`, `y_test` |
| `statistics` | `data_quality`, `target_distribution`, `describe_numeric`, `describe_categorical`, `churn_rate_by_feature`, `target_association` | `netflix_raw` | 6 file `stats_*` |
| `visualization` | `plot_eda` (gọi 5 hàm `plot_*`) | `netflix_clean` | `eda_figures` |
| `training` | `compare_models`, `train_best_model` | `X_train`, `y_train` | `model_comparison`, `classifier` |
| `evaluation` | `evaluate_model`, `evaluate_all_models`, `evaluate_by_segment`, `compute_feature_importance`, `plot_evaluation` | `classifier`, `X_test`, `y_test`, `model_comparison` | `metrics`, `model_comparison_test`, `metrics_by_segment`, `feature_importance`, `evaluation_figures` |
| `inference` | `predict_churn` (dùng lại `clean_data`, `create_features`) | `new_customers`, `classifier` | `predictions` |

Hai pipeline ghép sẵn trong `pipeline_registry.py`:
- **`kedro run`** (mặc định) = `data_processing` + `feature_engineering` + `training` + `evaluation`.
- **`eda`** = `data_processing` + `statistics` + `visualization`.

### Dữ liệu trong catalog

| Tên | File | Tạo bởi |
|---|---|---|
| `netflix_raw` | `data/01_raw/netflix_customer_churn.csv` | dữ liệu gốc |
| `netflix_clean` | `data/03_primary/netflix_clean.parquet` | `clean_data` |
| `model_features` | `data/04_feature/model_features.parquet` | `create_features` |
| `X_train`, `X_test`, `y_train`, `y_test` | `data/05_model_input/*.parquet` | `split_data` |
| `classifier` | `data/06_models/classifier.pkl/<thời gian>/` (mỗi lần chạy một phiên bản) | `train_best_model` |
| `model_comparison` | `data/08_reporting/training/model_comparison.csv` | `compare_models` |
| `metrics` | `data/08_reporting/metrics.json` | `evaluate_model` |
| `model_comparison_test` | `data/08_reporting/evaluation/model_comparison_test.csv` | `evaluate_all_models` |
| `metrics_by_segment`, `feature_importance` | `data/08_reporting/evaluation/*.csv` | `evaluate_by_segment`, `compute_feature_importance` |
| `evaluation_figures` | `data/08_reporting/evaluation/figures/` | `plot_evaluation` |
| `eda_figures` | `data/08_reporting/figures/eda/` | `plot_eda` |
| `stats_*` (6 file) | `data/08_reporting/statistics/` | các node của `statistics` |
| `new_customers` | `data/01_raw/new_customers.csv` | bạn tự cung cấp, xem [Bước 7](#bước-7-triển-khai) |
| `predictions` | `data/07_model_output/predictions.csv` | `predict_churn` |

### Tham số

| File | Chứa |
|---|---|
| `parameters.yml` | `model_options` (tỉ lệ chia train/test, `random_state`) và `decision.threshold` (ngưỡng quyết định) |
| `parameters_data_processing.yml` | tên cột id và cột mục tiêu |
| `parameters_feature_engineering.yml` | cột bị bỏ, cột lấy log, ngưỡng 30 ngày và 4 hồ sơ |
| `parameters_training.yml` | số fold, chỉ số xếp hạng, danh sách mô hình ứng viên và siêu tham số |
| `parameters_evaluation.yml` | cột dùng để phân tích theo nhóm, số lần xáo trộn khi tính độ quan trọng |
| `parameters_statistics.yml`, `parameters_visualization.yml` | tên cột, số nhóm chia, mức ý nghĩa thống kê |
| `parameters_inference.yml` | tên cột id trong kết quả dự đoán |

### Xử lý dữ liệu cho mô hình nằm ở đâu

Ba chỗ, theo đúng thứ tự dữ liệu đi qua:

1. **Làm sạch:** `clean_data` trong [`data_processing/nodes.py`](src/netflix/pipelines/data_processing/nodes.py).
2. **Tạo feature và chia dữ liệu:** `create_features` và `split_data` trong [`feature_engineering/nodes.py`](src/netflix/pipelines/feature_engineering/nodes.py).
3. **Mã hoá chữ thành số và chuẩn hoá:** nằm **trong mô hình**, ở hàm `_make_pipeline` của [`training/nodes.py`](src/netflix/pipelines/training/nodes.py). Cố ý đặt ở đây để các con số chuẩn hoá chỉ học từ phần train, tránh rò rỉ thông tin của tập test.

Khi chấm khách mới (pipeline `inference` và API), hệ thống dùng lại đúng `clean_data` và `create_features`, nên khách mới được xử lý giống hệt dữ liệu huấn luyện.

### Muốn làm X thì sửa ở đâu

| Muốn | Sửa |
|---|---|
| Đổi tỉ lệ train/test | `model_options.test_size` trong `parameters.yml` |
| Bỏ hoặc giữ một cột, đổi ngưỡng 30 ngày hoặc 4 hồ sơ | `parameters_feature_engineering.yml` |
| Thêm một feature mới | hàm `create_features` trong `feature_engineering/nodes.py` |
| Thêm một mô hình | `ESTIMATORS` trong `training/nodes.py` **và** một mục trong `parameters_training.yml` |
| Đổi số fold hoặc chỉ số xếp hạng mô hình | `cv_folds`, `scoring` trong `parameters_training.yml` |
| Đổi ngưỡng quyết định (mặc định 0,5) | `decision.threshold` trong `parameters.yml` |
| Phân tích theo nhóm khác (thay `subscription_type`) | `segment_column` trong `parameters_evaluation.yml` |
| Đổi nơi lưu file hoặc thêm dữ liệu lưu ra đĩa | `conf/base/catalog.yml` |
| Thêm một pipeline mới | `kedro pipeline create <tên>`, rồi thêm vào `pipeline_registry.py` |
| Đổi các trường API nhận vào | lớp `Customer` trong `serving/app.py` |
| Thêm test | `tests/pipelines/<tên pipeline>/test_pipeline.py` |

## Luồng dữ liệu

```
netflix_raw (CSV)
   │ clean_data                                   [data_processing]
   ▼
netflix_clean ─────────────► statistics, visualization   (kedro run --pipeline eda)
   │ create_features                              [feature_engineering]
   ▼
model_features
   │ split_data
   ▼
X_train, X_test, y_train, y_test
   │ compare_models → train_best_model            [training]
   ▼
model_comparison, classifier
   │ evaluate_model, evaluate_all_models,         [evaluation]
   │ evaluate_by_segment, compute_feature_importance, plot_evaluation
   ▼
metrics, model_comparison_test, metrics_by_segment, feature_importance, evaluation_figures

new_customers ─► clean_data ─► create_features ─► predict_churn ─► predictions   [inference]
```

## Chi tiết từng bước

### Bước 1: Dữ liệu
`data/01_raw/netflix_customer_churn.csv`: 5.000 dòng, 14 cột, mỗi khách hàng một dòng.

| Nhóm | Cột |
|---|---|
| Định danh | `customer_id` |
| Nhân khẩu | `age`, `gender`, `region` |
| Gói dịch vụ | `subscription_type`, `monthly_fee`, `number_of_profiles`, `payment_method` |
| Hành vi | `watch_hours`, `avg_watch_time_per_day`, `last_login_days`, `device`, `favorite_genre` |
| Mục tiêu | `churned` |

Lưu ý về nguồn dữ liệu (suy ra từ chính dữ liệu, chưa có tài liệu gốc xác nhận):
- `avg_watch_time_per_day` luôn bằng `watch_hours / (last_login_days + 1)`, nên tên cột dễ gây hiểu nhầm.
- Phí hàng tháng gắn cứng với gói (Basic 8,99, Standard 13,99, Premium 17,99), và các cột phân bố rất đều. **Nhiều khả năng đây là dữ liệu tổng hợp**, không phải dữ liệu thật của Netflix, nên điểm số rất cao của mô hình không thể coi là chất lượng khi dùng cho khách hàng thật.
- Không có cột thời gian, nên không thể chia train/test theo thời gian.

### Bước 2: Statistics (`statistics`)
Sáu node, mỗi node trả lời một câu hỏi và ghi file vào `data/08_reporting/statistics/`: chất lượng dữ liệu, cân bằng lớp, mô tả cột số, mô tả cột chữ, tỉ lệ churn theo nhóm, mối liên hệ với churn (Spearman, chi-square).

Phát hiện chính: không thiếu giá trị, không trùng; hai lớp cân bằng (50,3% rời); `watch_hours`, `last_login_days`, `avg_watch_time_per_day` liên hệ mạnh với churn; `age`, `gender`, `region`, `device`, `favorite_genre` **không** liên quan (p-value > 0,05).

### Bước 3: Tiền xử lý (`data_processing`, `feature_engineering`)
- `clean_data`: bỏ khách trùng, dòng thiếu, giá trị vô lý; ép kiểu. Với dữ liệu này không loại dòng nào.
- `create_features` (tham số trong `parameters_feature_engineering.yml`):
  - bỏ `customer_id` và 5 cột không liên quan ở bước 2;
  - `log1p` cho `watch_hours` và `avg_watch_time_per_day` (rất lệch phải);
  - `is_inactive` (không đăng nhập quá 30 ngày) và `has_many_profiles` (từ 4 hồ sơ), vì churn đổi mạnh quanh hai điểm đó.
- `split_data`: chia 80/20, có `stratify` theo `churned`.
- **Mã hoá chữ và chuẩn hoá số không làm ở đây**, mà nằm trong mô hình (bước 5), để chỉ học từ phần train và tránh rò rỉ dữ liệu test (*data leakage*).

### Bước 4: Visualization (`visualization`)
Năm biểu đồ lưu vào `data/08_reporting/figures/eda/`: cân bằng lớp, phân phối cột số theo lớp, tỉ lệ churn theo nhóm (cột số và cột chữ), ma trận tương quan. Biểu đồ vẽ bằng `matplotlib.figure.Figure` nên không cần giao diện đồ hoạ.

### Bước 5: Model building (`training`)
- `compare_models`: kiểm tra chéo (5 phần, theo ROC-AUC) cho từng ứng viên trong `parameters_training.yml`: hồi quy logistic, rừng ngẫu nhiên, gradient boosting, LightGBM và SVM. Ghi bảng xếp hạng vào `data/08_reporting/training/model_comparison.csv`, kèm `overfit_gap` để phát hiện học thuộc lòng.
- `train_best_model`: lấy ứng viên điểm cao nhất, huấn luyện lại trên toàn bộ tập train, lưu `classifier.pkl` (mỗi lần chạy một phiên bản).
- Muốn thêm mô hình: thêm vào `ESTIMATORS` trong `training/nodes.py` và thêm một mục trong `parameters_training.yml`. Mô hình phải có `predict_proba`, và nếu thư viện nằm ngoài scikit-learn (như `lightgbm`) thì thêm vào cả `requirements.txt` và `requirements-serving.txt`.

### Bước 6: Evaluation (`evaluation`)
Tập test chỉ dùng ở bước này, một lần. Kết quả vào `data/08_reporting/`:
- `metrics.json`: điểm của **mô hình đã chọn** (trường `model` cho biết là mô hình nào): accuracy, precision, recall, F1, ROC-AUC, kèm `baseline_accuracy` (độ chính xác nếu luôn đoán lớp phổ biến);
- `evaluation/model_comparison_test.csv`: **bảng tham khảo** chấm cả các mô hình ứng viên trên tập test. Mô hình đã chọn là dòng `selected = True`, chọn ở bước 5 bằng kiểm tra chéo. **Không dùng bảng này để chọn lại mô hình**: chọn theo điểm test thì tập test bị dùng để quyết định và điểm báo cáo sẽ lạc quan hơn thực tế. Vì vậy các dòng xếp theo thứ hạng kiểm tra chéo (`cv_rank`), không xếp theo điểm test;
- `evaluation/metrics_by_segment.csv`: điểm theo từng gói đăng ký;
- `evaluation/feature_importance.csv`: permutation importance;
- `evaluation/figures/`: ma trận nhầm lẫn, đường ROC, độ quan trọng feature.

Ngưỡng quyết định (`decision.threshold` trong `parameters.yml`, mặc định 0,5) dùng chung cho đánh giá, chấm điểm hàng loạt và API. Chọn ngưỡng là quyết định kinh doanh, xem mục 6.6 trong notebook 06.

### Bước 7: Triển khai
**Chấm điểm hàng loạt** (`inference`): đọc `data/01_raw/new_customers.csv` (cùng cột với file gốc, có hoặc không có `churned`), đi qua đúng `clean_data` và `create_features` như lúc huấn luyện, ghi `data/07_model_output/predictions.csv`. Tạo file mẫu 20 khách để thử:

```powershell
python -c "import pandas as pd; df = pd.read_csv('data/01_raw/netflix_customer_churn.csv').sample(20, random_state=1); df.drop(columns='churned').to_csv('data/01_raw/new_customers.csv', index=False)"
kedro run --pipeline inference
```

**API** (`src/netflix/serving/app.py`, FastAPI): tự nạp phiên bản mô hình mới nhất trong `data/06_models/`, hoặc file chỉ định bởi biến môi trường `CHURN_MODEL_PATH`.

```powershell
uvicorn netflix.serving.app:app --reload      # rồi mở http://127.0.0.1:8000/docs
```

| Đường dẫn | Việc |
|---|---|
| `GET /health` | kiểm tra dịch vụ và xem mô hình đang dùng |
| `POST /predict` | một khách hàng → xác suất rời và quyết định |
| `POST /predict/batch` | danh sách khách hàng |

Dữ liệu gửi lên được kiểm tra bằng Pydantic (ví dụ `subscription_type` chỉ nhận Basic/Standard/Premium); dữ liệu sai trả mã 422. Trang `/docs` cho phép thử trực tiếp trên trình duyệt.

**Docker** (chạy sau `kedro run`, vì ảnh chứa mô hình mới nhất):

```powershell
docker build -t netflix-churn .
docker run -p 8000:8000 netflix-churn
```

## Quy tắc làm việc

* Không xoá dòng nào trong `.gitignore`.
* Không commit dữ liệu (`data/`) lên repository.
* Không commit credentials hay cấu hình cục bộ; để tất cả trong `conf/local/`.
* Kết quả phải tái lập được: mọi tham số nằm trong `conf/`, có `random_state` cố định.
* Node là hàm thuần, chỉ làm việc với dữ liệu truyền vào; mọi việc đọc/ghi file do catalog xử lý.
* Mã hoá/chuẩn hoá nằm trong mô hình, không làm trước khi chia train/test.
* Đổi tham số trong `conf/` rồi chạy lại; không sửa số trực tiếp trong code.

## Kiểm thử và chất lượng mã

```powershell
pytest                                  # toàn bộ test
pytest --deselect tests/test_run.py     # bỏ test chạy cả pipeline (không ghi vào data/)
ruff check src tests                    # kiểm tra lỗi và phong cách
```

Test dùng dữ liệu giả có sẵn quy luật (`tests/conftest.py`) và đọc đúng các file tham số trong `conf/base`, nên đổi cấu hình mà làm hỏng code thì test báo ngay.

## Làm việc với notebook

Có hai cách mở notebook:
- **VS Code:** mở file `.ipynb`, bấm **Select Kernel** (góc trên bên phải) và chọn `C:\ML\.venv\Scripts\python.exe`.
- **Terminal:** `kedro jupyter lab`, tự dùng đúng môi trường.

Ô code đầu tiên của mỗi notebook tự kiểm tra kernel đã đủ thư viện chưa và tự tìm thư mục dự án; nếu thiếu gì nó in ra lệnh cài. Để xoá kết quả (output) khỏi notebook trước khi commit, dùng [`nbstripout`](https://github.com/kynan/nbstripout): `pip install nbstripout` rồi `nbstripout --install`. Kết quả vẫn giữ nguyên trên máy bạn.

## Đóng gói dự án Kedro

[Hướng dẫn đóng gói dự án Kedro](https://docs.kedro.org/en/stable/deploy/package_a_project/#package-an-entire-kedro-project)
