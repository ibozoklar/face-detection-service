"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from face_detection_service.api.routes import router
from face_detection_service.utils.logger import setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger = setup_logger()
    logger.info("Face Detection Service starting up")
    yield
    logger.info("Face Detection Service shutting down")


app = FastAPI(
    title="Face Detection Service",
    description="Modular face detection API with Haar Cascade, Dlib HOG, and MediaPipe detectors.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    from face_detection_service import config

    uvicorn.run(
        "face_detection_service.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        reload=False,
    )
