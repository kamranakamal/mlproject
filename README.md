# Students Score Prediction

FastAPI inference API and Streamlit UI for predicting student math scores. Includes training pipeline (ingestion → transformation → model selection/tuning), Dockerized API/UI, and a simple orchestrator script.

**Author:** kamran (kamranakmal6776@gmail.com)

## Project Structure
- API: [main.py](main.py)
- Streamlit UI: [prediction_gui.py](prediction_gui.py)
- Training pipeline: [src/components/data_ingestion.py](src/components/data_ingestion.py), [src/components/data_transformation.py](src/components/data_transformation.py), [src/components/model_trainer.py](src/components/model_trainer.py)
- Utilities: [src/utils.py](src/utils.py)
- Docker: [Dockerfile](Dockerfile) (API), [Dockerfile.ui](Dockerfile.ui) (UI), [docker-compose.yml](docker-compose.yml)
- Orchestrator: [run_app.py](run_app.py)
- Package metadata: [pyproject.toml](pyproject.toml)

## Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker and Docker Compose (for containerized run)

## Local Setup
1) **Install uv:** `curl -LsSf https://astral.sh/uv/install.sh | sh` (or see [uv docs](https://docs.astral.sh/uv/))
2) **Install dependencies:**
	- With uv: `uv pip install -e .` (installs from pyproject.toml)
	- Or: `uv pip install -r requirements.txt`
	- Fallback (pip): `pip install -r requirements.txt`
3) **Train pipeline** (produces artifacts/preprocessor.pkl and artifacts/model.pkl):
	- `python run_app.py --train`
4) **Run API locally:**
	- `uvicorn main:app --host 0.0.0.0 --port 8000`
	- Open docs: http://127.0.0.1:8000/docs
5) **Run Streamlit locally** (uses API_URL env or defaults to localhost):
	- `API_URL=http://127.0.0.1:8000/predict streamlit run prediction_gui.py`

## Docker (recommended)
1) **Build images:** `docker compose build`
2) **Run:** `docker compose up -d`
3) **Access:**
	- API: http://localhost:8000/docs
	- UI: http://localhost:8501 (calls API at http://api:8000/predict inside the compose network)

## Pushing images to Docker Hub
- Images are tagged in [docker-compose.yml](docker-compose.yml) as `7903954268/mlproject-api:latest` and `7903954268/mlproject-ui:latest`
- **Build and push:** `docker compose build && docker compose push`

## Troubleshooting
- Ensure artifacts exist before starting API/UI (run training once).
- If port 8000 or 8501 is busy, change the host mapping in [docker-compose.yml](docker-compose.yml).
- For uv issues, see [uv documentation](https://docs.astral.sh/uv/).
