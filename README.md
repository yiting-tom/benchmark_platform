# CV Benchmark Platform

An internal CV competition platform supporting Image Classification, Object Detection, and Image Segmentation tasks.

## 🚀 Quick Start

### Development Environment

# 1. Install dependencies
uv sync

# 2. Run migrations
uv run python manage.py migrate

# 3. Create superuser
uv run python manage.py createsuperuser

# 4. Start development server
uv run python manage.py runserver

# 5. (Another terminal) Start Q Worker
uv run python manage.py qcluster

Access http://127.0.0.1:8000/admin/ to enter the management back-end.

### Production Environment (Docker)

# 1. Copy environment variable template
cp .env.example .env

# 2. Edit .env with real settings

# 3. Start all services
docker-compose up -d

# 4. Create superuser
docker-compose exec web python manage.py createsuperuser

## 📁 Project Structure

├── config/                 # Django settings
├── competitions/           # Competition core app
│   ├── models.py          # Data models
│   ├── admin.py           # Admin interface
│   ├── views.py           # Participant interface
├── scoring/               # Scoring engine
│   │   ├── base.py        # Abstract base classes
│   └── tasks.py           # Django-Q2 async tasks
├── templates/             # HTML templates
├── tests/                 # Tests

## 🏆 Features

### Organizer (Admin)
- Create competitions in Django Admin
- Set task types (Classification/Detection/Segmentation)
- Upload Ground Truth
- Manage participant whitelist and specific time windows

### Participant
- Real-time countdown for remaining time
- Automatic scoring after prediction file upload
- View submission history and Public Score
- Select final submission version

### Validator
- Enter Private Score directly in Admin portal
- Announce final rankings after competition ends

## 📊 Supported Metrics

| Task Type | Metric |
|-----------|--------|
| Classification | Accuracy, F1-Score |
| Detection | mAP@0.5, mAP@[0.5:0.95] |
| Segmentation | mIoU |

## 📝 CSV Format
(Details of CSV format remain unchanged)
