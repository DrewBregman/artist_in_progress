# RAG Art Application

Art style identification and RAG-powered art analysis application.

## Deployment Guide

This application consists of two parts:
- **Frontend**: React application deployed on Vercel
- **Backend**: FastAPI application deployed on Railway

### Backend Deployment (Railway)

1. Create a new project in Railway and connect to your GitHub repository
2. Set up the following environment variables:
   ```
   OPENAI_SECRET_KEY=sk-youropenaikey
   PINECONE_API_KEY=yourpineconekey
   PINECONE_ENV=us-east1-gcp
   CORS_ALLOWED_ORIGINS=https://your-frontend-domain.vercel.app,http://localhost:3000
   ```
3. Deploy the service
4. Note the deployed URL (e.g., `https://rag-art-backend.up.railway.app`)

### Frontend Deployment (Vercel)

1. Create a new project in Vercel and connect to your GitHub repository
2. Set the following environment variables:
   ```
   REACT_APP_BACKEND_URL=https://your-backend-url-from-railway.app
   ```
3. Set the root directory to `frontend`
4. Set the build command to `npm run build`
5. Set the output directory to `build`
6. Deploy
7. Note the deployed URL (e.g., `https://rag-art.vercel.app`)

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

## Architecture

- **Frontend**: React with Material-UI
- **Backend**: FastAPI with OpenAI and Pinecone integrations
- **Database**: Pinecone vector database for art style embeddings

## Security Notes

- API keys and sensitive credentials are stored as environment variables
- CORS is configured to allow only specific origins
- Authentication is handled by Clerk