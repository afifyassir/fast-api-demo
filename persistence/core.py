import logging
import os

import alembic.config
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy_utils import database_exists, create_database

from api.config import Config, ROOT

_logger = logging.getLogger('mlapi')

# Base class for SQLAlchemy models
Base = declarative_base()

def create_db_engine_from_config(*, config: Config) -> Engine:
    db_url = config.SQLALCHEMY_DATABASE_URI
    if not database_exists(db_url):
        create_database(db_url)
    engine = create_engine(db_url)

    _logger.info(f"creating DB conn with URI: {db_url}")
    return engine

def create_db_session(*, engine: Engine) -> scoped_session:
    return scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

def get_db_session(app: FastAPI = Depends()) -> Session:
    return app.db_session

def init_database(app: FastAPI, config: Config, db_session=None) -> None:
    if not db_session:
        engine = create_db_engine_from_config(config=config)
        db_session = create_db_session(engine=engine)

    @app.on_event("startup")
    def startup_event():
        app.db_session = db_session

    @app.on_event("shutdown")
    def shutdown_event():
        db_session.remove()

def run_migrations():
    os.chdir(str(ROOT))
    alembicArgs = ["--raiseerr", "upgrade", "head"]
    alembic.config.main(argv=alembicArgs)
