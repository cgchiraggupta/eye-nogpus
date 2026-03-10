# eye-nogpus

## EYE — Application Walkthrough & Documentation (No-GPU Version)

This repository contains the **comprehensive walkthrough and documentation** for the [EYE](https://github.com/eight-atulya/eye) project — an AI-first, privacy-first memory preservation platform.

> **EYE** transforms your photos and videos into intelligent, searchable memories using local AI — no cloud, no data leaving your machine.

### 📖 What's Inside

- **[WALKTHROUGH.md](./WALKTHROUGH.md)** — Full application walkthrough covering:
  - System architecture (7 Docker services)
  - Backend API reference (FastAPI, 8 routers, 40+ endpoints)
  - Frontend structure (Next.js 14, 11 feature modules)  
  - ML engine architecture (YOLO-E with 4,000+ classes, few-shot learning)
  - Data flow diagrams (memory upload → AI processing → chat)
  - Orchestrator & background job system
  - Configuration guide
  - Business model & roadmap
  - Directory structure reference

### ⚠️ Why "nogpus"?

The original EYE repo requires **NVIDIA GPUs** (via Docker `gpus: all` directives) which don't work on macOS. This repo documents the application as-is while a lightweight, CPU-compatible version is planned.

### 🔗 Original Repo

- **Source:** [eight-atulya/eye](https://github.com/eight-atulya/eye)
- **Author:** Anurag Atulya — *EYE for Humanity*
- **License:** Coffee License ☕

---

*"Preserve memories. Have intelligent conversations."*
