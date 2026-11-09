# Netflix – Dự đoán khách hàng rời bỏ (churn)

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Tổng quan

Dự án Kedro (`kedro 1.7.0`) dùng bộ dữ liệu `netflix_customer_churn.csv` (5.000 khách hàng) để xây dựng pipeline dự đoán khách hàng có rời dịch vụ hay không (cột mục tiêu `churned`: 1 = rời, 0 = ở lại).

Các cột trong dữ liệu gốc:

| Nhóm | Cột |
|---|---|
| Định danh | `customer_id` |
| Nhân khẩu | `age`, `gender`, `region` |
| Gói dịch vụ | `subscription_type`, `monthly_fee`, `number_of_profiles`, `payment_method` |
| Hành vi | `watch_hours`, `avg_watch_time_per_day`, `last_login_days`, `device`, `favorite_genre` |
| Mục tiêu | `churned` |

Tài liệu Kedro: https://docs.kedro.org

## Cài đặt và chạy nhanh

```bash
pip install -r requirements.txt   # cài thư viện
kedro run                         # chạy toàn bộ pipeline
kedro run --pipeline data_processing   # chạy riêng một pipeline
pytest                            # chạy test
```

## Cấu trúc dự án

```
netflix/
├── conf/
│   ├── base/
│   │   ├── catalog.yml                     <- KHAI BÁO dữ liệu: file nào, định dạng gì, ở đâu
│   │   ├── parameters.yml                  <- tham số dùng chung (test_size, random_state...)
│   │   └── parameters_data_processing.yml  <- tham số riêng của pipeline data_processing
│   ├── local/
│   │   └── credentials.yml                 <- mật khẩu, API key. KHÔNG commit
│   └── README.md
├── data/
│   ├── 01_raw/               <- dữ liệu gốc, chỉ đọc, không bao giờ sửa
│   ├── 02_intermediate/      <- đã đổi kiểu, làm sạch sơ bộ
│   ├── 03_primary/           <- bảng chuẩn, sạch
│   ├── 04_feature/           <- bảng feature
│   ├── 05_model_input/       <- dữ liệu đưa vào model
│   ├── 06_models/            <- model đã train (.pkl)
│   ├── 07_model_output/      <- kết quả dự đoán
│   └── 08_reporting/         <- metrics, biểu đồ
├── notebooks/                <- notebook khám phá dữ liệu (EDA), thử nghiệm
├── src/netflix/
│   ├── __init__.py
│   ├── __main__.py           <- cho phép chạy `python -m netflix`
│   ├── pipeline_registry.py  <- tự tìm và đăng ký pipeline
│   ├── settings.py           <- cấu hình framework Kedro
│   └── pipelines/            <- mỗi pipeline là một thư mục con
│       └── data_processing/
│           ├── __init__.py
│           ├── nodes.py      <- các hàm xử lý
│           └── pipeline.py   <- nối các hàm thành pipeline
├── tests/
│   ├── test_run.py
│   └── pipelines/data_processing/test_pipeline.py
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Luồng dữ liệu

Kedro xây đồ thị từ tên dataset: node nào xuất ra `X` thì node nào nhận `X` sẽ chạy sau nó.

```
netflix_raw (CSV, 01_raw)
   │  [nodes.py] làm sạch
   ▼
netflix_clean (Parquet, 03_primary)
   │  [nodes.py] tách train/test + huấn luyện
   ▼
classifier (Pickle, 06_models, có version)  ──►  metrics (JSON, 08_reporting)
```

Các dataset trên đã được khai báo trong `conf/base/catalog.yml`. Phần node và pipeline xử lý chúng **chưa được viết** (xem mục "Việc cần làm tiếp").

---

## Chi tiết từng thư mục và file

### `conf/` – Cấu hình (không chứa code)

Kedro gộp `conf/base` với `conf/local`; giá trị trong `local` ghi đè `base`. Thiết lập này nằm ở `CONFIG_LOADER_ARGS` trong `settings.py`.

#### `conf/base/catalog.yml`
- **Tác dụng:** danh sách mọi dataset của dự án: tên, loại, đường dẫn. Node chỉ gọi dataset theo tên, không biết file nằm ở đâu.
- **Nên viết:** mỗi dữ liệu đi vào hoặc ra khỏi pipeline cần một mục. Dataset không khai báo sẽ chỉ nằm trong RAM và mất khi chạy xong.
- **Hiện có:**

| Tên | Loại | Đường dẫn | Ghi chú |
|---|---|---|---|
| `netflix_raw` | `pandas.CSVDataset` | `data/01_raw/netflix_customer_churn.csv` | dữ liệu gốc |
| `netflix_clean` | `pandas.ParquetDataset` | `data/03_primary/netflix_clean.parquet` | bảng sạch |
| `classifier` | `pickle.PickleDataset` | `data/06_models/classifier.pkl` | `versioned: true`, mỗi lần chạy lưu một bản theo timestamp |
| `metrics` | `json.JSONDataset` | `data/08_reporting/metrics.json` | kết quả đánh giá |

- **Gợi ý thêm:** `X_train`, `X_test`, `y_train`, `y_test` (`05_model_input`), `predictions` (`07_model_output`), và các bảng trung gian ở `02_intermediate`/`04_feature` nếu muốn lưu để debug.

#### `conf/base/parameters.yml`
- **Tác dụng:** tham số dùng chung toàn dự án. Trong code truy cập bằng `params:model_options` hoặc `params:model_options.test_size`.
- **Nên viết:** mọi con số có thể muốn chỉnh (tỉ lệ chia, seed, siêu tham số, ngưỡng lọc) thay vì gõ cứng trong code.
- **Hiện có:** nhóm `model_options` gồm `test_size: 0.2`, `random_state: 42`, `max_features: 5000`, `C: 1.0`.
- **Lưu ý:** `max_features` chỉ có nghĩa nếu dùng bộ vector hoá văn bản (TF-IDF…). Dữ liệu hiện tại không có cột văn bản, nên có thể bỏ nếu không dùng.

#### `conf/base/parameters_data_processing.yml`
- **Tác dụng:** tham số riêng của pipeline `data_processing`, do Kedro sinh ra khi tạo pipeline.
- **Nên viết:** các tham số chỉ pipeline này dùng, ví dụ danh sách cột cần bỏ, ngưỡng loại outlier cho `age`. File hiện chỉ có chú thích mẫu.

#### `conf/local/credentials.yml`
- **Tác dụng:** chứa thông tin nhạy cảm (mật khẩu DB, khoá S3, API key). Được `.gitignore` bỏ qua.
- **Nên viết:** chỉ khi dữ liệu nằm ở nguồn cần xác thực. Catalog tham chiếu bằng `credentials: tên_khoá`. Dữ liệu CSV cục bộ hiện tại **không cần** file này.
- **Lưu ý:** tuyệt đối không commit. Có thể ghi ví dụ khoá cần thiết (không kèm giá trị thật) vào `conf/README.md` để người khác tự tạo.

#### `conf/README.md`
Ghi chú mặc định của Kedro. Mục "Instructions" còn trống, có thể viết hướng dẫn tạo `credentials.yml` tại đây.

---

### `data/` – Dữ liệu (không commit)

Các thư mục được đánh số theo [quy ước tầng dữ liệu của Kedro](https://docs.kedro.org/en/stable/getting_started/kedro_concepts/#data-layers). Dữ liệu chỉ đi từ số nhỏ lên số lớn. Mỗi thư mục có `.gitkeep` để Git giữ thư mục rỗng.

| Thư mục | Nội dung | Ví dụ trong dự án này |
|---|---|---|
| `01_raw` | Dữ liệu gốc, chỉ đọc, không sửa | `netflix_customer_churn.csv` |
| `02_intermediate` | Đổi kiểu dữ liệu, đổi tên cột, làm sạch sơ bộ | ép `churned` về `int`, chuẩn hoá chuỗi |
| `03_primary` | Bảng chuẩn, sạch, một dòng một khách hàng | `netflix_clean.parquet` |
| `04_feature` | Bảng feature | `watch_hours` theo nhóm, mã hoá `gender`, `device`… |
| `05_model_input` | Dữ liệu đưa thẳng vào model | `X_train`, `X_test`, `y_train`, `y_test` |
| `06_models` | Model đã huấn luyện | `classifier.pkl` |
| `07_model_output` | Kết quả dự đoán | xác suất churn của từng khách hàng |
| `08_reporting` | Metrics, biểu đồ | `metrics.json` (accuracy, F1, ROC-AUC…) |

---

### `notebooks/`
- **Tác dụng:** nơi khám phá dữ liệu và thử ý tưởng nhanh. Chạy bằng `kedro jupyter lab`, có sẵn biến `context`, `catalog`, `pipelines`, `session`.
- **Nên viết:** EDA (phân phối `churned`, tương quan, giá trị thiếu). Khi ý tưởng ổn, chuyển code sang `nodes.py`; không để logic chính chỉ nằm trong notebook.

---

### `src/netflix/` – Mã nguồn

#### `__init__.py`
Đánh dấu thư mục là package Python và khai báo `__version__` (pyproject đọc phiên bản từ đây).

#### `__main__.py`
Điểm vào để chạy `python -m netflix` hoặc lệnh `netflix` sau khi đóng gói. **Không cần sửa.**

#### `pipeline_registry.py`
- **Tác dụng:** trả về dict `{tên: Pipeline}` cho `kedro run`.
- **Cách hoạt động:** `find_pipelines()` tự quét `pipelines/*` và gọi `create_pipeline()` của từng thư mục. `__default__` là tổng của mọi pipeline, chính là thứ chạy khi gõ `kedro run`.
- **Nên viết:** thường giữ nguyên. Chỉ sửa khi muốn pipeline chạy mặc định khác, ví dụ `__default__ = data_processing + training`.

#### `settings.py`
- **Tác dụng:** cấu hình framework: hooks, config loader, session store…
- **Hiện có:** chỉ `CONFIG_LOADER_ARGS` (`base_env="base"`, `default_run_env="local"`); các mục còn lại đang bị comment.
- **Nên viết:** đăng ký `HOOKS` nếu cần (log, kiểm tra dữ liệu). Nếu không cần thì giữ nguyên.

#### `pipelines/` – mỗi pipeline một thư mục con

Tạo pipeline mới bằng `kedro pipeline create <tên>`. Kedro sinh sẵn thư mục, file test và file tham số tương ứng.

##### `pipelines/data_processing/nodes.py`
- **Tác dụng:** các **node** – hàm Python thuần, nhận đầu vào, trả đầu ra, không đọc hay ghi file.
- **Nên viết:** mỗi hàm làm một việc. Hiện file mới chỉ có docstring. Ví dụ:
  - `clean_data(netflix_raw) -> DataFrame`: bỏ trùng `customer_id`, xử lý giá trị thiếu, ép kiểu, bỏ giá trị bất thường (tuổi âm, `monthly_fee` ≤ 0).
  - `split_data(netflix_clean, params) -> X_train, X_test, y_train, y_test`: chia theo `test_size`, `random_state`.
  - `train_model(X_train, y_train, params) -> classifier`: huấn luyện với `C`.
  - `evaluate_model(classifier, X_test, y_test) -> dict`: trả về metrics.
- **Quy tắc:** không đọc đường dẫn trong node (việc của catalog), không gõ cứng tham số (việc của `parameters.yml`), không dùng `print` (ruff đang chặn `T201`), hãy dùng `logging`.

##### `pipelines/data_processing/pipeline.py`
- **Tác dụng:** nối các node thành pipeline bằng cách khai báo tên dataset vào/ra.
- **Hiện có:** `create_pipeline()` trả về `Pipeline([])`, nên **hiện pipeline rỗng**, chưa chạy gì.
- **Nên viết:**

```python
from kedro.pipeline import Node, Pipeline

from .nodes import clean_data, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(clean_data, inputs="netflix_raw", outputs="netflix_clean",
                 name="clean_data"),
            Node(split_data, inputs=["netflix_clean", "params:model_options"],
                 outputs=["X_train", "X_test", "y_train", "y_test"],
                 name="split_data"),
        ]
    )
```

##### `pipelines/data_processing/__init__.py`
Để trống là đủ. Đây là file giúp `find_pipelines()` nhận ra thư mục là một pipeline.

##### `pipelines/__init__.py`
Để trống.

---

### `tests/`

Cấu trúc thư mục test phản chiếu cấu trúc `src/`. Tên file phải bắt đầu bằng `test_`.

#### `tests/test_run.py`
- **Tác dụng:** test mẫu kiểm tra `kedro run` báo lỗi "Pipeline contains no nodes".
- **Lưu ý:** test này **sẽ fail** ngay khi pipeline có node, vì nó mong đợi pipeline rỗng. Hãy sửa lại thành test chạy pipeline thành công hoặc xoá đi khi viết xong pipeline đầu tiên.

#### `tests/pipelines/data_processing/test_pipeline.py`
- **Tác dụng:** test cho pipeline `data_processing`. Hiện chỉ có docstring.
- **Nên viết:** test riêng từng node với DataFrame nhỏ tự tạo, ví dụ `clean_data` có bỏ trùng `customer_id` không, `split_data` có đúng tỉ lệ 80/20 không. Chạy bằng `pytest`; ngưỡng coverage chỉnh ở `[tool.coverage.report]` (hiện `fail_under = 0`).

---

### File gốc

#### `pyproject.toml`
Cấu hình trung tâm: tên và phiên bản package, dependency, thông tin `[tool.kedro]`, cấu hình `pytest`/coverage và `ruff` (độ dài dòng 88, bật các rule `F, W, E, I, UP, PL, T201`). Cũng khai báo lệnh `netflix` và nhóm `dev` (pytest, ruff).
- **Nên sửa:** khi thêm dependency, đổi cấu hình lint/test.

#### `requirements.txt`
Danh sách thư viện cài bằng `pip install -r requirements.txt`. Hiện chỉ có `kedro`, `ipython`, `jupyterlab`, `notebook`.
- **Cần bổ sung:** catalog đang dùng CSV/Parquet/Pickle/JSON, nên cần thêm thư viện sau để `kedro run` chạy được:

```
kedro-datasets[pandas-csvdataset,pandas-parquetdataset]
pandas
pyarrow
scikit-learn
```

  Nếu dùng `ruff` hoặc `pytest`, cài thêm bằng `pip install -e ".[dev]"`.

#### `.gitignore`
Chặn `data/`, `conf/local/`, `__pycache__`, v.v. Không xoá dòng nào (xem quy tắc dưới).

---

## Quy tắc làm việc

* Không xoá dòng nào trong `.gitignore`.
* Không commit dữ liệu (`data/`) lên repository.
* Không commit credentials hay cấu hình cục bộ; để tất cả trong `conf/local/`.
* Kết quả phải tái lập được: mọi tham số nằm trong `conf/`, có `random_state` cố định.
* Node là hàm thuần, chỉ làm việc với dữ liệu truyền vào, mọi I/O do catalog xử lý.

## Việc cần làm tiếp

- [ ] Viết `clean_data` trong `nodes.py` và nối vào `pipeline.py` (`netflix_raw` → `netflix_clean`).
- [ ] Thêm các dataset mới vào `catalog.yml` (`X_train`, `X_test`, …).
- [ ] Tạo pipeline `training` (`kedro pipeline create training`): huấn luyện và đánh giá, xuất `classifier` và `metrics`.
- [ ] Bổ sung thư viện vào `requirements.txt` (`pandas`, `pyarrow`, `scikit-learn`, `kedro-datasets`).
- [ ] Viết test cho từng node; sửa hoặc xoá `tests/test_run.py`.
- [ ] Dọn `max_features` trong `parameters.yml` nếu không dùng.

## Làm việc với notebook

> Dùng `kedro jupyter notebook`, `kedro jupyter lab` hoặc `kedro ipython` để có sẵn các biến `context`, `session`, `catalog`, `pipelines`.

```bash
kedro jupyter notebook   # Jupyter Notebook
kedro jupyter lab        # JupyterLab
kedro ipython            # IPython
```

Jupyter, JupyterLab và IPython đã nằm trong `requirements.txt`. Để xoá output notebook trước khi commit, dùng [`nbstripout`](https://github.com/kynan/nbstripout) (`nbstripout --install`); output vẫn giữ nguyên ở máy bạn.

## Đóng gói dự án

[Hướng dẫn đóng gói dự án Kedro](https://docs.kedro.org/en/stable/deploy/package_a_project/#package-an-entire-kedro-project)
