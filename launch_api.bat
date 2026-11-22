@echo off
title GeneWeaver REST API Microservice
echo ======================================================================
echo   GeneWeaver High-Throughput REST API Microservice
echo   Swagger UI Interactive Docs: http://127.0.0.1:8000/docs
echo   OpenAPI JSON Specification:  http://127.0.0.1:8000/openapi.json
echo ======================================================================

set PYTHONPATH=src;%PYTHONPATH%
python -m uvicorn geneweaver.api.app:app --host 127.0.0.1 --port 8000 --reload
pause
