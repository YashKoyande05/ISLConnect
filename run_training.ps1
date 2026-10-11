$ErrorActionPreference = "Stop"

Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Starting Landmark Extraction (this will process all images, taking a few minutes)..."
python -m training.extract_landmarks --input dataset/isl-words-2

Write-Host "Training the Model on full dataset..."
python -m training.train

Write-Host "Training complete! You can now start the application with 'python app.py'"
