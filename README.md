# RAG Art Application

Art style identification and RAG-powered art analysis application.

## Deployment Guide

This application consists of two parts:
- **Frontend**: React application deployed on Vercel
- **Backend**: FastAPI application deployed on Railway

### Backend Deployment (Railway)

1. Create a new project in Railway and connect to your GitHub repository
2. Set the service to deploy from the project's root
3. Set up the following environment variables:
   ```
   OPENAI_SECRET_KEY=sk-youropenaikey
   PINECONE_API_KEY=yourpineconekey
   PINECONE_ENV=us-east1-gcp
   CORS_ALLOWED_ORIGINS=https://your-frontend-domain.vercel.app,http://localhost:3000
   ```
4. Railway will automatically use the `railway.json` configuration which points to the specialized Dockerfile.railway
5. Deploy the service
6. Note the deployed URL (e.g., `https://rag-art-production.up.railway.app`)

### Frontend Deployment (Vercel)

1. Create a new project in Vercel and connect to your GitHub repository
2. Set the following environment variables:
   ```
   REACT_APP_BACKEND_URL=https://your-backend-url-from-railway.app
   ```
3. Set the root directory to `frontend`
4. Deploy
5. Note the deployed URL (e.g., `https://rag-art.vercel.app`)

### Update Backend CORS Configuration

After deploying your frontend, update the `CORS_ALLOWED_ORIGINS` in your Railway environment variables to include your Vercel domain:

```
CORS_ALLOWED_ORIGINS=https://your-frontend-domain.vercel.app,http://localhost:3000
```

### Local Development

1. Clone the repository
2. Create `.env` files in both `/frontend` and `/backend` directories (use the `.env.example` files as templates)
3. Install dependencies:
   ```
   # Backend
   cd backend
   pip install -r requirements.txt
   
   # Frontend
   cd frontend
   npm install
   ```
4. Start the development servers:
   ```
   # Backend
   cd backend
   uvicorn main:app --reload
   
   # Frontend
   cd frontend
   npm start
   ```

## Troubleshooting

### Railway Deployment Issues

If you encounter the "externally-managed-environment" error with pip on Railway, it means Railway is using Nix to manage Python packages. Our configuration handles this by:

1. Using a custom `Dockerfile.railway` that properly installs dependencies
2. Setting the `--break-system-packages` flag when necessary
3. Properly configuring the environment in `railway.json`

### Vercel Deployment Issues

If environment variables aren't being correctly applied in your Vercel deployment:

1. Make sure they're correctly set in the Vercel project settings
2. Use the `build:production` script which explicitly passes environment variables to the build process
3. Check that `vercel.json` is properly configured

## Architecture

- **Frontend**: React with Material-UI
- **Backend**: FastAPI with OpenAI and Pinecone integrations
- **Database**: Pinecone vector database for art style embeddings

## Security Notes

- API keys and sensitive credentials are stored as environment variables
- CORS is configured to allow only specific origins
- Authentication is handled by Clerk