import logging
import os
import pathlib
import sys
from types import FrameType
from typing import List, cast

from loguru import logger
from pydantic import AnyHttpUrl, BaseSettings

# Project Directories
ROOT = pathlib.Path(__file__).resolve().parent.parent

APP_NAME = 'ml_api'

class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO  # logging levels are type int

class Config(BaseSettings):
    API_V1_STR: str = "/api/v1"
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # type: ignore
        "http://localhost:8000",  # type: ignore
        "https://localhost:3000",  # type: ignore
        "https://localhost:8000",  # type: ignore
    ]
    PROJECT_NAME: str = "Predicting customer churn API"
    SHADOW_MODE_ACTIVE = os.getenv('SHADOW_MODE_ACTIVE', True)
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:"
        f"{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    )
    DB_USER = os.getenv("DB_USER", "user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_PORT = os.getenv("DB_PORT", 6609)
    DB_HOST = os.getenv("DB_HOST", "0.0.0.0")
    DB_NAME = os.getenv("DB_NAME", "ml_api_dev")
    logging: LoggingSettings = LoggingSettings()

    class Config:
        case_sensitive = True

class DevelopmentConfig(Config):
    DEBUG = True
    ENV = "development"  # do not use in production!
    LOGGING_LEVEL = logging.DEBUG

class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    LOGGING_LEVEL = logging.DEBUG

    # DB config matches test docker container
    DB_USER = os.getenv("DB_USER", "test_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_PORT = os.getenv("DB_PORT", 6608)
    DB_HOST = os.getenv("DB_HOST", "0.0.0.0")
    DB_NAME = "ml_api_test"
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{DB_USER}:"
        f"{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

class ProductionConfig(Config):
    DB_USER = os.getenv("DB_USER", "user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_PORT = os.getenv("DB_PORT", 6609)
    DB_HOST = os.getenv("DB_HOST", "database")
    DB_NAME = os.getenv("DB_NAME", "ml_api")
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{DB_USER}:"
        f"{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )

def setup_app_logging(config: Config) -> None:
    LOGGERS = ("uvicorn.asgi", "uvicorn.access")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=config.logging.LOGGING_LEVEL)]

    logger.configure(
        handlers=[{"sink": sys.stderr, "level": config.logging.LOGGING_LEVEL}]
    )

settings = Config()
