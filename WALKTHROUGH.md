# EYE — Application Walkthrough

> **Preserve memories. Have intelligent conversations.**

EYE is an open-source, AI-first computer vision platform that transforms personal photos and videos into intelligent, searchable memories. It lets users upload media, have AI understand and describe it, and then converse with their memories using natural language — all running locally for maximum privacy.

**Author:** Anurag Atulya — *EYE for Humanity*
**License:** Coffee License ☕ (Free for personal use; commercial use requires buying the author good coffee)

---

## Table of Contents

1. [What is EYE?](#1-what-is-eye)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Core Modules Deep Dive](#3-core-modules-deep-dive)
   - [Backend (FastAPI)](#31-backend-fastapi)
   - [Frontend (Next.js)](#32-frontend-nextjs)
   - [Engines (ML/AI)](#33-engines-mlai)
   - [Orchestrator](#34-orchestrator)
   - [Storage](#35-storage)
4. [Infrastructure & Services](#4-infrastructure--services)
5. [API Reference Summary](#5-api-reference-summary)
6. [User-Facing Features](#6-user-facing-features)
7. [Data Flow & How It All Works Together](#7-data-flow--how-it-all-works-together)
8. [Configuration System](#8-configuration-system)
9. [Business Model & Roadmap](#9-business-model--roadmap)
10. [Running Modes](#10-running-modes)
11. [Getting Started (Prerequisites)](#11-getting-started-prerequisites)
12. [Directory Structure](#12-directory-structure)

---

## 1. What is EYE?

EYE is a **memory preservation platform** — think of it as a self-hosted, AI-powered photo and video manager that:

- **Ingests** your photos and videos and stores them securely.
- **Understands** them using computer vision (YOLO-E object detection with 4,000+ classes) and multimodal LLMs (Ollama + Gemma 3).
- **Lets you converse** with your memories — ask "What happened at the beach last summer?" and get AI-generated answers about your own photos.
- **Searches** your memories using natural language and semantic/vector similarity (FAISS).
- **Annotates** them with a full annotation pipeline (projects, tasks, labels, reviews, exports in COCO/YOLO/Pascal VOC formats).
- **Trains** custom models on your data using few-shot learning.

Everything runs **locally** — no cloud dependency, no data leaving your machine. Privacy-first by design.

---

## 2. System Architecture Overview

EYE is a **microservices-based** application with **7 Docker containers** orchestrated via Docker Compose:

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Docker Compose                              │
│                                                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │  Frontend    │   │   Backend   │   │   Ollama    │               │
│  │  (Next.js)   │──▶│  (FastAPI)  │──▶│  (LLM AI)  │               │
│  │  Port: 3003  │   │  Port: 8001 │   │  Port:11434│               │
│  └─────────────┘   └──────┬──────┘   └─────────────┘               │
│                           │                                          │
│              ┌────────────┼────────────┐                             │
│              ▼            ▼            ▼                             │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────┐                   │
│  │  PostgreSQL   │ │   Redis    │ │    MinIO      │                  │
│  │  Port: 5433   │ │  Port:6380 │ │ Port:9002/03  │                 │
│  └──────────────┘ └────────────┘ └──────────────┘                   │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐                                │
│  │  Prometheus   │   │   Worker     │                                │
│  │  Port: 9090   │   │ (Background) │                                │
│  └──────────────┘   └──────────────┘                                │
└──────────────────────────────────────────────────────────────────────┘
```

### Services at a Glance

| Service | Technology | Port | Purpose |
|---------|-----------|------|---------|
| **Frontend** | Next.js 14 + React 18 + TailwindCSS | 3003 | Web dashboard & UI |
| **Backend** | FastAPI (Python 3.11) + Uvicorn | 8001 | REST API server |
| **Database** | PostgreSQL 16 (Alpine) | 5433 | Persistent data storage (projects, annotations, memories) |
| **Redis** | Redis 7 (Alpine) | 6380 | Job queue, caching, real-time state |
| **MinIO** | MinIO (S3-compatible) | 9002 / 9003 | Object storage (images, datasets, model weights) |
| **Ollama** | Ollama (Gemma 3:12B) | 11434 | Local LLM for vision analysis & chat |
| **Prometheus** | Prometheus v2.55 | 9090 | Metrics collection & monitoring |
| **Worker** | Python (Redis worker) | — | Background job processing (YOLO-E training, inference) |

---

## 3. Core Modules Deep Dive

### 3.1 Backend (FastAPI)

**Location:** `backend/`

The backend is a FastAPI application that serves as the central API layer. It exposes **8 routers**, each handling a specific domain:

#### API Routers

| Router | Prefix | File | Responsibility |
|--------|--------|------|---------------|
| **Routes** | `/api/v1` | `api/routes.py` | Core endpoints: health, auth, image upload, CVAT integration |
| **Metrics** | `/api` | `api/metrics.py` | Prometheus metrics exposure |
| **Queue** | `/api` | `api/queue.py` | Job queue status and management |
| **Jobs** | `/api` | `api/jobs.py` | Training & processing job tracking |
| **YOLO-E** | `/api` | `api/yolo_e.py` | Object detection model management, training, inference, datasets |
| **Annotations** | `/api` | `api/annotations.py` | Full annotation lifecycle (projects → tasks → labels → annotations → reviews → exports) |
| **Ollama** | `/api/v1/ollama` | `api/ollama.py` | LLM chat, text generation, vision analysis, model management |
| **Memory** | `/api/v1/memory` | `api/memory.py` | Memory upload, search, retrieval, AI chat with memories |

#### Key Backend Services

| Service | File | Purpose |
|---------|------|---------|
| **CVAT Integration** | `services/cvat_integration.py` | Interface with CVAT annotation tool (creating projects/tasks, syncing annotations) |
| **Memory Service** | `services/memory_service.py` | Core memory CRUD, vector search with FAISS, semantic retrieval |
| **Memory Processing** | `services/memory_processing_service.py` | Background processing pipeline: extract metadata, generate embeddings, AI description |
| **Ollama Service** | `services/ollama_service.py` | Interface with Ollama LLM for chat, text generation, and vision tasks |
| **Queue Service** | `services/queue.py` | Redis-backed job queue for async task execution |
| **Jobs Service** | `services/jobs.py` | Job lifecycle management (create, track, complete, fail) |

#### Backend Configuration

Settings are loaded from environment variables using `pydantic-settings` (`settings.py`). All configuration values (database, redis, minio, JWT, CVAT) have sensible development defaults.

#### Key Dependencies (requirements.txt)

- **Web Framework:** FastAPI, Uvicorn, Pydantic
- **ML/AI:** PyTorch, TorchVision, Ultralytics (YOLO), OpenCV
- **Database:** SQLAlchemy, Alembic, psycopg2
- **Storage:** Boto3 (S3/MinIO), MinIO client
- **Search:** FAISS (vector similarity search), sentence-transformers (embeddings)
- **Monitoring:** prometheus-client
- **Annotation SDKs:** cvat-sdk, supervisely, labelbox

### 3.2 Frontend (Next.js)

**Location:** `frontend/`

The frontend is a **Next.js 14** application with the App Router pattern. It uses:

- **React 18** for UI components
- **TailwindCSS 3.4** for styling
- **Zustand 5** for client-side state management
- **TanStack Query (React Query) 5** for server state / API data fetching
- **Lucide React** for icons
- **React Dropzone** for file uploads

#### Pages (App Router)

| Route | Page | Description |
|-------|------|-------------|
| `/` | `page.tsx` | Homepage — hero section, feature cards, getting started guide, community section |
| `/eye-ai` | `eye-ai/page.tsx` | EYE AI conversational interface — chat with AI about images |
| `/memory` | `memory/page.tsx` | Memory vault — upload, browse, and search memories |
| `/annotation` | `annotation/page.tsx` | Annotation workspace — label images for training |
| `/inference` | `inference/page.tsx` | Run inference on images using trained models |
| `/training` | `training/page.tsx` | Train custom models on your datasets |

#### Feature Modules (11 total)

Each feature is a self-contained module in `frontend/src/features/`:

| Module | Purpose |
|--------|---------|
| `annotations` | Annotation UI components (canvas, tools, panels) |
| `datasets` | Dataset browser and uploader |
| `eye-ai` | AI chat interface (text + vision) |
| `inference` | Inference runner and results viewer |
| `jobs` | Job queue dashboard |
| `memory` | Memory upload, search, and browsing |
| `models` | Model registry and management |
| `projects` | Project workspace management |
| `settings` | Application settings |
| `training` | ML training dashboard |
| `yolo_e` | YOLO-E specific UI components |

#### Homepage Sections

The homepage (`page.tsx`) contains:
1. **Navigation bar** with links to all sections + GitHub + Auth
2. **Hero section** — "Preserve memories. Have intelligent conversations."
3. **Core Features grid** — Memory Ingestion, AI Understanding, Intelligent Conversations, Smart Search, Privacy Protection, Open Source
4. **EYE AI Showcase** — example conversation demo
5. **Open Source Community** — Contribute, Innovate, Collaborate
6. **Getting Started** — 3-step guide (Clone, Upload, Converse)
7. **Footer** with docs, API, community, and legal links

### 3.3 Engines (ML/AI)

**Location:** `engines/`

The engines module provides a **pluggable ML inference architecture**. All engines extend the `EngineNode` abstract base class:

```python
class EngineNode(ABC):
    name: str = "base"
    
    @abstractmethod
    def load(self, weights_path: str) -> None: ...
    
    @abstractmethod
    def infer(self, input_path: str) -> Dict[str, Any]: ...
```

#### Available Engines

| Engine | File | Status | Description |
|--------|------|--------|-------------|
| **YOLO-E** | `yolo_e_node.py` | ✅ Implemented | 4,000+ class object detection with few-shot learning |
| **Ultra Node** | `ultra_node.py` | 🔧 Scaffold | Placeholder for Ultralytics-based detection |
| **Forge Node** | `forge_node.py` | 🔧 Scaffold | Placeholder for custom forge models |
| **Spectra Node** | `spectra_node.py` | 🔧 Scaffold | Placeholder for spectral analysis |
| **Custom Node** | `custom_node.py` | 🔧 Scaffold | Template for user-defined engines |

#### YOLO-E Node (Primary Engine)

The `YOLOENode` is the main workhorse:

- **4,000+ base classes** out of the box
- **Few-shot learning** — train on custom objects with minimal examples
- **Batch processing** — efficiently process entire photo directories
- **PyTorch-based** — uses `torch` and `torchvision` transforms
- **Methods:** `load()`, `infer()`, `train_few_shot()`, `batch_process_images()`, `get_model_info()`

### 3.4 Orchestrator

**Location:** `orchestrator/`

The orchestrator manages background job execution:

| Component | File | Purpose |
|-----------|------|---------|
| **Dispatcher** | `dispatcher.py` | Routes task types to engine names (e.g., "detection" → "ultra_node") |
| **Scheduler** | `scheduler.py` | In-memory job queue with enqueue/dequeue operations |
| **Redis Worker** | `workers/redis_worker.py` | Listens to Redis queue, processes jobs asynchronously |
| **YOLO-E Worker** | `workers/yolo_e_worker.py` | Dedicated worker for YOLO-E training/inference jobs |
| **Simple Worker** | `workers/simple_worker.py` | Minimal worker for basic task processing |

**Flow:** API receives job request → Enqueues to Redis → Worker picks up → Dispatches to appropriate engine → Reports results back.

### 3.5 Storage

**Location:** `storage/`

- **S3 Adapter** (`storage/adapters/s3.py`) — Unified interface for S3-compatible storage (MinIO). Handles `upload_bytes`, `download`, `list_objects`, and `presigned_url` operations.
- **Directories:** `datasets/`, `outputs/`, `weights/` — organized storage for ML artifacts.

---

## 4. Infrastructure & Services

### Docker Compose Services

All services are defined in `docker-compose.yml`:

1. **`backend`** — Builds from `backend/Dockerfile` (Python 3.11 slim), installs OpenCV and ML dependencies, runs Uvicorn with hot-reload.

2. **`frontend`** — Builds from `frontend/Dockerfile` (Node 20 Alpine), runs `npm run dev` on port 3003.

3. **`db`** — PostgreSQL 16 Alpine. Database: `vision`, User: `vision`, Password: `vision`. Persisted via `pgdata` volume.

4. **`redis`** — Redis 7 Alpine on port 6380. Used for job queuing, caching, and real-time pub/sub.

5. **`minio`** — S3-compatible object storage. API on port 9002, Console on port 9003. Credentials: `miniokey` / `miniopass123`.

6. **`prometheus`** — Metrics collection server. Scrapes backend metrics. 2-day retention.

7. **`ollama`** — Local LLM inference server running Gemma 3:12B model. Supports GPU acceleration (NVIDIA) and CPU fallback.

8. **`worker`** — Background worker running `redis_worker.py` from the orchestrator module.

### Monitoring

- **Prometheus** scrapes the backend at regular intervals.
- Backend exposes metrics via `/api/metrics`.
- Health check: `GET /health` → `{"status": "ok"}`.

---

## 5. API Reference Summary

### Core APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/ping` | API ping |
| `POST` | `/api/v1/auth/token` | Issue JWT auth token |
| `POST` | `/api/v1/uploads/image` | Upload image to MinIO |

### Ollama / AI APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/ollama/health` | Ollama service health |
| `GET` | `/api/v1/ollama/models` | List available LLM models |
| `POST` | `/api/v1/ollama/chat` | Conversational chat with LLM |
| `POST` | `/api/v1/ollama/generate` | Text generation from prompt |
| `POST` | `/api/v1/ollama/vision/chat` | Vision analysis with images |
| `POST` | `/api/v1/ollama/vision/upload` | Upload image + ask question |
| `POST` | `/api/v1/ollama/models/pull` | Pull model from registry |

### Memory APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/memory/upload` | Upload memory (image + tags/notes) |
| `POST` | `/api/v1/memory/search` | Semantic search through memories |
| `GET` | `/api/v1/memory/` | List user's memories |
| `GET` | `/api/v1/memory/{id}` | Get specific memory |
| `GET` | `/api/v1/memory/image/{uuid}` | Get memory image file |
| `PUT` | `/api/v1/memory/{id}` | Update memory metadata |
| `DELETE` | `/api/v1/memory/{id}` | Delete memory |
| `GET` | `/api/v1/memory/stats` | Memory statistics |
| `GET` | `/api/v1/memory/jobs/{id}` | Processing job status |
| `POST` | `/api/v1/memory/chat` | Chat with AI using memories as context |

### Annotation APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/annotations/projects` | Create annotation project |
| `GET/PUT/DELETE` | `/api/v1/annotations/projects/{id}` | CRUD on projects |
| `POST` | `/api/v1/annotations/tasks` | Create annotation task |
| `POST` | `/api/v1/annotations/` | Create annotation (bbox, polygon, keypoint, mask) |
| `POST` | `/api/v1/annotations/reviews` | Submit annotation review |
| `POST` | `/api/v1/annotations/export/{project_id}` | Export annotations (COCO, YOLO, Pascal VOC, etc.) |
| `GET` | `/api/v1/annotations/stats` | Annotation statistics |

### YOLO-E APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/yoloe/load` | Load YOLO-E model |
| `POST` | `/api/yoloe/train` | Start training job |
| `GET` | `/api/yoloe/training/jobs` | List training jobs |
| `GET/DELETE` | `/api/yoloe/training/{id}` | Get/cancel training job |
| `POST` | `/api/yoloe/datasets/upload` | Upload training dataset |
| `GET` | `/api/yoloe/datasets` | List datasets |
| `GET` | `/api/yoloe/models` | List trained models |
| `POST` | `/api/yoloe/infer` | Run inference on image(s) |
| `POST` | `/api/yoloe/batch` | Batch process images |

---

## 6. User-Facing Features

### 📸 Memory Vault
Upload photos (drag-and-drop or batch), tag them, add notes. EYE automatically extracts metadata, generates AI descriptions, and creates vector embeddings for semantic search.

### 🧠 EYE AI (Chat Interface)
A conversational AI interface powered by Ollama (Gemma 3:12B). Users can:
- Upload images and ask questions about them
- Get scene descriptions, object identification, and emotional context
- Have multi-turn conversations about their memories
- Search memories using natural language

### 🔍 Smart Search
FAISS-based semantic search. Ask "photos from the beach" or "pictures with my dog" and get relevant results based on visual content and AI-generated descriptions.

### 🏷️ Annotation System
A full-featured annotation pipeline for labeling images:
- **Projects** — organize annotation work
- **Tasks** — assign work to team members with priorities and due dates
- **Labels** — custom label definitions with colors and categories
- **Annotation Types** — bounding boxes, polygons, keypoints, masks
- **Reviews** — QA workflow (approved, rejected, needs_revision) with scoring
- **Export** — COCO, YOLO, Pascal VOC, Supervisely, Labelbox formats

### 🎯 Object Detection (YOLO-E)
4,000+ class object detection with:
- Few-shot learning for custom objects
- Batch processing for large photo libraries
- Model training pipeline with progress tracking
- Dataset management

### 📊 Training Dashboard
Train custom ML models:
- Upload datasets with annotations
- Configure training hyperparameters (epochs, batch size, learning rate, image size)
- Monitor training progress in real-time
- Export trained models

### 🔧 Inference Runner
Run trained models on new images:
- Single image or batch inference
- Configurable confidence/IOU thresholds
- Visual results with bounding boxes

---

## 7. Data Flow & How It All Works Together

### Memory Upload Flow

```
User uploads photo
       │
       ▼
   Frontend (React Dropzone)
       │
       ▼
   POST /api/v1/memory/upload
       │
       ├──▶ Image saved to MinIO (S3)
       │
       ├──▶ Memory record created in PostgreSQL
       │
       └──▶ Background job enqueued to Redis
                    │
                    ▼
              Worker picks up job
                    │
                    ├──▶ Extract EXIF metadata (date, location, camera)
                    │
                    ├──▶ Generate embeddings (sentence-transformers → FAISS)
                    │
                    ├──▶ YOLO-E object detection (identify objects in image)
                    │
                    └──▶ Ollama AI description (natural language summary)
                              │
                              ▼
                    Memory record updated with:
                    - AI-generated description
                    - Detected objects
                    - Searchable embeddings
                    - Extracted metadata
```

### Chat with Memories Flow

```
User asks: "Show me photos from the beach"
       │
       ▼
   POST /api/v1/memory/chat
       │
       ├──▶ Query embedded via sentence-transformers
       │
       ├──▶ FAISS similarity search finds relevant memories
       │
       ├──▶ Relevant memories sent as context to Ollama
       │
       └──▶ Ollama generates natural language response
                    │
                    ▼
            User sees: AI response + relevant photos
```

### Training Flow

```
User uploads labeled dataset
       │
       ▼
   POST /api/yoloe/datasets/upload → MinIO
       │
       ▼
   POST /api/yoloe/train
       │
       ├──▶ Training job created in Redis
       │
       └──▶ YOLO-E Worker starts training
                    │
                    ├──▶ Load base model (4000+ classes)
                    │
                    ├──▶ Few-shot learning on custom data
                    │
                    ├──▶ Epoch-by-epoch progress updates → Redis
                    │
                    └──▶ Trained model weights saved → MinIO
                              │
                              ▼
                    Model available for inference
```

---

## 8. Configuration System

### Single Source of Truth: `config/eye.yaml`

All service configuration lives in one YAML file:

```yaml
app:
  name: "EYE"
  version: "0.1.0"

services:
  backend:   { port: 8001 }
  frontend:  { port: 3003 }
  database:  { host: "db", port: 5433, name: "vision" }
  redis:     { host: "redis", port: 6380 }
  minio:     { port: 9002, console_port: 9003 }
  prometheus: { port: 9090 }
  ollama:    { port: 11434, default_model: "gemma3:12b" }

ml:
  yolo_e:
    base_classes: 4000
    few_shot_epochs: 50
    confidence_threshold: 0.5

security:
  jwt:
    secret_key: "change-me-in-production"
    access_token_expire_minutes: 60
```

### Config Generation

Run `python scripts/generate-config.py` to generate Docker Compose, environment files, and service configs from `eye.yaml`.

### Environment Variables

The backend uses `pydantic-settings` to load env vars with the `EYE_` prefix:
- `EYE_DATABASE_HOST`, `EYE_DATABASE_PORT`, etc.
- `EYE_REDIS_HOST`, `EYE_REDIS_PORT`
- `EYE_MINIO_HOST`, `EYE_MINIO_PORT`
- `EYE_JWT_SECRET_KEY`

---

## 9. Business Model & Roadmap

### Business Model

EYE targets three customer segments:

| Segment | Target | Pricing (INR) | Value Proposition |
|---------|--------|--------------|-------------------|
| **Personal** | Individuals & families | ₹299–999/month | Memory preservation with AI |
| **Business** | Startups, CA firms | ₹999–4,999/month | Document intelligence & automation |
| **Retail** | Cafes, restaurants | ₹1,499–4,999/month | Camera-based intelligent automation |

### Running Modes (Planned)

| Mode | Target Users | Resources |
|------|-------------|-----------|
| **Light Mode** | Low-compute users | Basic inference, minimal GPU — runs on slowest laptops |
| **Heavy Load Mode** | Advanced users | Full GPU training and inference |
| **God Mode** | Power users | Full AI, best visual UI, everything unlocked |

### Roadmap Highlights (from TODO.md)

- **High Priority:** Local network access, bulk processing, video support, Graph DB for knowledge graph
- **Medium Priority:** Material-UI redesign, graph visualization, mobile support, dashboard scorecards
- **Future:** Gamified memory games, comprehensive testing, auth/authorization, analytics

---

## 10. Running Modes

### Current Prerequisites

- Docker Desktop installed and running
- For GPU acceleration: NVIDIA GPU + NVIDIA Container Toolkit (Linux/Windows only)
- On macOS: remove `gpus: all` directives from `docker-compose.yml` (Ollama/AI runs on CPU)

### Start the System

```bash
# 1. Generate config files
python scripts/generate-config.py

# 2. Start all services (builds containers on first run)
docker-compose up -d --build

# 3. Wait 2-3 minutes for all services to initialize

# 4. Check status
docker-compose ps

# 5. Access the dashboard
open http://localhost:3003
```

### Service URLs

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:3003 |
| **EYE AI** | http://localhost:3003/eye-ai |
| **Memory** | http://localhost:3003/memory |
| **Backend API** | http://localhost:8001 |
| **API Health** | http://localhost:8001/health |
| **MinIO Console** | http://localhost:9003 |
| **Prometheus** | http://localhost:9090 |

---

## 11. Getting Started (Prerequisites)

### Required

| Requirement | Details |
|-------------|---------|
| **Docker Desktop** | v20+ (must be running before `docker-compose up`) |
| **16 GB RAM** | Recommended (Ollama + YOLO-E can be memory-intensive) |
| **20 GB Disk** | For Docker images, model weights, and stored memories |

### Optional (for enhanced performance)

| Requirement | Details |
|-------------|---------|
| **NVIDIA GPU** | 8 GB+ VRAM for GPU-accelerated training/inference (Linux/Windows only) |
| **Python 3.11** | For running test scripts outside Docker |
| **Node.js 20** | For frontend development outside Docker |

### macOS Note

macOS does **not** support NVIDIA GPUs in Docker. The `gpus: all` directives in `docker-compose.yml` must be removed for Mac deployments. Ollama and YOLO-E will run on CPU, which is slower but functional.

---

## 12. Directory Structure

```
eye/
├── backend/                    # FastAPI backend server
│   ├── Dockerfile
│   ├── main.py                 # App entry point — registers all routers
│   ├── settings.py             # Pydantic settings (env vars)
│   ├── config.py               # Config loader
│   ├── requirements.txt        # Python dependencies
│   ├── api/                    # API route handlers
│   │   ├── routes.py           # Core API (auth, upload, CVAT)
│   │   ├── annotations.py     # Full annotation pipeline
│   │   ├── yolo_e.py           # YOLO-E endpoints
│   │   ├── ollama.py           # LLM/vision endpoints
│   │   ├── memory.py           # Memory management endpoints
│   │   ├── jobs.py             # Job tracking
│   │   ├── queue.py            # Queue management
│   │   └── metrics.py          # Prometheus metrics
│   ├── services/               # Business logic layer
│   │   ├── memory_service.py
│   │   ├── memory_processing_service.py
│   │   ├── ollama_service.py
│   │   ├── cvat_integration.py
│   │   ├── jobs.py
│   │   └── queue.py
│   ├── schemas/                # Pydantic models
│   └── auth/                   # Authentication module
│
├── frontend/                   # Next.js 14 frontend
│   ├── Dockerfile
│   ├── package.json
│   ├── src/
│   │   ├── app/                # App Router pages
│   │   │   ├── page.tsx        # Homepage
│   │   │   ├── layout.tsx      # Root layout
│   │   │   ├── eye-ai/         # AI chat page
│   │   │   ├── memory/         # Memory vault page
│   │   │   ├── annotation/     # Annotation page
│   │   │   ├── inference/      # Inference page
│   │   │   └── training/       # Training page
│   │   ├── features/           # Feature modules (11)
│   │   ├── shared/             # Shared hooks, utils
│   │   ├── components/         # Global components
│   │   └── styles/             # Global CSS
│   └── tailwind.config.js
│
├── engines/                    # ML engine nodes
│   ├── base.py                 # Abstract EngineNode
│   └── yolo_e_node.py          # YOLO-E implementation
│
├── orchestrator/               # Background job orchestration
│   ├── Dockerfile
│   ├── dispatcher.py           # Task → Engine routing
│   ├── scheduler.py            # In-memory job queue
│   └── workers/                # Background workers
│       ├── redis_worker.py
│       └── yolo_e_worker.py
│
├── storage/                    # Storage layer
│   └── adapters/
│       └── s3.py               # S3/MinIO adapter
│
├── config/                     # Configuration
│   ├── eye.yaml                # Single source of truth
│   └── PORT_MAPPING.md
│
├── monitoring/                 # Monitoring stack
│   └── prometheus/
│       ├── Dockerfile
│       └── prometheus.yml
│
├── business/                   # Business model & strategies
│   ├── README.md
│   ├── models/
│   ├── strategies/
│   ├── case-studies/
│   └── resources/
│
├── scripts/                    # Utility scripts
│   ├── generate-config.py      # Config generator
│   ├── setup_ollama.py         # Ollama model setup
│   └── test_*.py               # Various test scripts
│
├── docs/                       # Documentation
│   ├── API_REFERENCE.md
│   ├── CONFIGURATION.md
│   ├── DEPLOYMENT.md
│   ├── OLLAMA_INTEGRATION.md
│   └── YOLO_E_INTEGRATION.md
│
├── docker-compose.yml          # Service orchestration
├── QUICKSTART.md               # Quick start guide
├── TODO.md                     # Development roadmap
├── LICENSE                     # Coffee License ☕
└── README.md                   # Project overview
```

---

## Summary

EYE is a privacy-first, AI-powered memory platform with:

- **7 Docker services** working together seamlessly
- **Full ML pipeline** — from data ingestion to training to inference
- **Conversational AI** — chat with your memories using multimodal LLMs
- **Production-grade annotation** — label, review, and export for any ML format
- **Extensible engine architecture** — plug in any ML model
- **Clear business model** — targeting personal, business, and retail customers

The system is currently in **active development** (v0.1.0) with a clear roadmap of 22 planned tasks.

---

*Made with ❤️ and ☕ by Anurag Atulya — EYE for Humanity*
*"I love machines, AI, humans, coffee, leaf. 8 is my north star."* ☕
