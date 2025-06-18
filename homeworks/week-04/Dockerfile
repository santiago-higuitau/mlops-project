FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY scoring_docker.py .

RUN pipenv install --system --deploy
# RUN pip install pandas pyarrow scikit-learn==1.5.0

ENTRYPOINT ["python", "scoring_docker.py"]