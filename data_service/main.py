from fastapi import FastAPI
from data_service.api.routes import router # Import the router from api/routes

app = FastAPI(
    title="Data Service - AI Real Estate Assistant",
    description="Microservice for managing and providing real estate project data.",
    version="1.0.0"
)

# Include API routes
app.include_router(router)

# No need for @app.on_event("startup") explicitly here,
# as ProjectRepository is instantiated when api/routes.py is imported,
# triggering its data loading.

# You can add a simple root endpoint if desired
@app.get("/")
async def read_root():
    return {"message": "Data Service is running!"}