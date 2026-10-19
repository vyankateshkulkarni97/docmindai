cd docmind-ai\backend

py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

