# 🧬 RFantibody GUI

[![GitHub](https://img.shields.io/badge/GitHub-RFantibody-blue?style=flat&logo=github)](https://github.com/RosettaCommons/RFantibody)

A user-friendly graphical interface for RFantibody. This web-based GUI makes it easy to access and utilize the powerful features of RFantibody.

## ✨ Key Features
- 🖥️ Intuitive Web Interface
- 🐳 Easy Installation and Execution with Docker
- 📊 Result Visualization and Analysis Tools

## 🚀 Getting Started

### Prerequisites
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Installation and Setup

1. Clone the repository
```bash
git clone https://github.com/Kangjuheon/rfantibody-ui.git
```

2. Navigate to project directory
```bash
cd rfantibody-ui
```

3. Prepare RFAntibody
```bash
cd third_party/RFantibody
bash include/download_weights.sh
cd ../..
docker compose --profile builder build rfantibody
```

3. Launch Docker containers
```bash
docker compose up -d --build
```

## 🌐 Access
Open your browser and visit:
```
http://localhost:2239
```

## 💡 Tips
- Initial startup might take some time due to Docker image downloads
- If you encounter any issues, check the Docker logs:
  ```bash
  docker compose logs
  ```