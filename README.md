# CV 競賽平台 (CV Benchmark Platform)

內部 CV 競賽平台，支援 Classification、Detection、Segmentation 三種任務類型。

## 🚀 快速開始

### 開發環境

```bash
# 1. 安裝依賴
uv sync

# 2. 執行資料庫遷移
uv run python manage.py migrate

# 3. 建立管理員帳號
uv run python manage.py createsuperuser

# 4. 啟動開發伺服器
uv run python manage.py runserver

# 5. (另一個終端) 啟動 Q Worker
uv run python manage.py qcluster
```

訪問 http://127.0.0.1:8000/admin/ 進入管理後台。

### 生產環境 (Docker)

```bash
# 1. 複製環境變數範本
cp .env.example .env

# 2. 編輯 .env 填入真實設定
vim .env

# 3. 啟動所有服務
docker-compose up -d

# 4. 建立管理員帳號
docker-compose exec web python manage.py createsuperuser
```

## 📁 專案結構

```
benchmark_platform/
├── config/                 # Django 設定
├── competitions/           # 競賽核心 App
│   ├── models.py          # 資料模型
│   ├── admin.py           # Admin 後台
│   ├── views.py           # 參賽者介面
│   └── urls.py
├── scoring/               # 算分引擎
│   ├── engines/
│   │   ├── base.py        # 抽象基類
│   │   ├── classification.py
│   │   ├── detection.py
│   │   └── segmentation.py
│   └── tasks.py           # Django-Q2 非同步任務
├── templates/             # HTML 模板
├── tests/                 # 測試
├── Dockerfile
└── docker-compose.yml
```

## 🏆 功能特色

### 出題者 (Admin)
- 在 Django Admin 建立競賽
- 設定任務類型 (Classification/Detection/Segmentation)
- 上傳 Ground Truth
- 管理參賽白名單與專屬時間區間

### 參賽者
- 即時倒數計時顯示剩餘時間
- 上傳預測檔後自動評分
- 查看提交歷史與 Public Score
- 選擇最終提交版本

### 驗證人員 (Validator)
- 在 Admin 後台直接填入 Private Score
- 競賽結束後公布最終排名

## 📊 支援的評分指標

| 任務類型 | 評分指標 |
|----------|----------|
| Classification | Accuracy, F1-Score |
| Detection | mAP@0.5, mAP@[0.5:0.95] |
| Segmentation | mIoU |

## 📝 CSV 格式

### Classification
```csv
image_id,label
img_001,cat
img_002,dog
```

### Detection
```csv
image_id,class_label,confidence,xmin,ymin,xmax,ymax
img_001,car,0.95,10,20,100,120
```

### Segmentation
```csv
image_id,class_label,rle_mask
img_001,cat,1 10 15 5
```

## 🛠️ Tech Stack

- **Backend**: Django 6.x + PostgreSQL
- **Async Queue**: Django-Q2 (ORM Broker)
- **Frontend**: Tailwind CSS + DaisyUI + HTMX
- **ML Libraries**: pandas, numpy, scikit-learn, opencv-python-headless
- **Deployment**: Docker + Docker Compose
