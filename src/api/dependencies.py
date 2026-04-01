from sqlmodel import SQLModel, create_engine, Session
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Database setup
DB_PATH = Path(__file__).parent.parent.parent / "data" / "centros.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# Templates setup
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
