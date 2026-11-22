# Launch GeneWeaver REST API Microservice
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  GeneWeaver High-Throughput REST API Microservice" -ForegroundColor Green
Write-Host "  Swagger UI Interactive Docs: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "  OpenAPI JSON Specification:  http://127.0.0.1:8000/openapi.json" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

$env:PYTHONPATH = "src;" + $env:PYTHONPATH
python -m uvicorn geneweaver.api.app:app --host 127.0.0.1 --port 8000 --reload
