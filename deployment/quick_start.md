# Quick Start Deployment Guide

**Description:** Get the HolographicMemory SOA system running in minutes  
**Estimated Time:** 5-10 minutes

## Prerequisites

- macOS 12.0+ (for Metal GPU support)
- Python 3.13+
- Xcode Command Line Tools
- 8GB+ RAM
- 2GB+ free disk space

## Deployment Steps


### Step 1: Clone Repository

**Commands:**

```bash
git clone https://github.com/your-org/HolographicMemory.git
cd HolographicMemory
```


### Step 2: Install Dependencies

**Commands:**

```bash
pip install -r requirements.txt
pip install -r services/holographic-memory/api/requirements.txt
```


### Step 3: Build GPU Components

**Commands:**

```bash
cd services/holographic-memory/core/native/holographic/build
make clean && make
```


### Step 4: Start System

**Commands:**

```bash
cd ../../../../..
python3.13 main.py
```


### Step 5: Verify Deployment

**Commands:**

```bash
curl http://localhost:8081/health
python3.13 benchmark_soa_system.py
```

