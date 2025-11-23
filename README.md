# Mini Elsa

Mini Elsa is a flashcard application with speech recognition capabilities.

## Production Setup

### Prerequisites
- Docker and Docker Compose installed.

### Configuration
Create a `.env` file in the root directory with the following variables:

```env
GOOGLE_CLIENT_ID=your_google_client_id
JWT_SECRET=your_strong_jwt_secret
JWT_ALGORITHM=HS256
JWT_EXPIRES_MINUTES=60
CORS_ORIGINS=http://your-frontend-domain.com,http://localhost:3000
```

### Running with Docker Compose

1. Build and start the container:
   ```bash
   docker-compose up -d --build
   ```

2. The API will be available at `http://localhost:8000`.

### Manual Production Run

If you prefer running without Docker:

1. Install dependencies:
   ```bash
   pip install .
   ```

2. Run with Uvicorn (ensure env vars are set):
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
