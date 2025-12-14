# Student Performance ML App

FastAPI inference API and Streamlit UI for predicting student math scores. Includes training pipeline (ingestion → transformation → model selection/tuning), Dockerized API/UI, and a simple orchestrator script.

## Project Structure
- API: [main.py](main.py)
- Streamlit UI: [prediction_gui.py](prediction_gui.py)
- Training pipeline: [src/components/data_ingestion.py](src/components/data_ingestion.py), [src/components/data_transformation.py](src/components/data_transformation.py), [src/components/model_trainer.py](src/components/model_trainer.py)
- Utilities: [src/utils.py](src/utils.py)
- Docker: [Dockerfile](Dockerfile) (API), [Dockerfile.ui](Dockerfile.ui) (UI), [docker-compose.yml](docker-compose.yml)
- Orchestrator: [run_app.py](run_app.py)

## Prerequisites
- Python 3.11+
- pip
- Docker and Docker Compose (for containerized run)

## Local Setup
1) Install deps: `pip install -r requirements.txt`
2) Train pipeline (produces artifacts/preprocessor.pkl and artifacts/model.pkl):
	- `python run_app.py --train`
3) Run API locally:
	- `uvicorn main:app --host 0.0.0.0 --port 8000`
	- Open docs: http://127.0.0.1:8000/docs
4) Run Streamlit locally (uses API_URL env or defaults to localhost):
	- `API_URL=http://127.0.0.1:8000/predict streamlit run prediction_gui.py`

## Docker (recommended)
1) Build images: `docker compose build`
2) Run: `docker compose up -d`
3) Access:
	- API: http://localhost:8000/docs
	- UI: http://localhost:8501 (calls API at http://api:8000/predict inside the compose network)

## Pushing images to a registry (example)
- Tag (API): `docker tag mlproject-api:latest <registry>/mlproject-api:latest`
- Tag (UI): `docker tag mlproject-ui:latest <registry>/mlproject-ui:latest`
- Push: `docker push <registry>/mlproject-api:latest` and `docker push <registry>/mlproject-ui:latest`

## Troubleshooting
- Ensure artifacts exist before starting API/UI (run training once).
- If port 8000 or 8501 is busy, change the host mapping in [docker-compose.yml](docker-compose.yml).
