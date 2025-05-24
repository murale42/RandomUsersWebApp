import os
import asyncio
import random
import httpx
from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, String, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from pydantic import BaseModel, EmailStr, ValidationError, ConfigDict
from contextlib import asynccontextmanager

DATABASE_URL = "postgresql+asyncpg://postgres:Qweras.1@localhost:5432/db"
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async_session = SessionLocal  

Base = declarative_base()

templates = Jinja2Templates(directory="templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("TESTING"):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with async_session() as db:
            result = await db.execute(select(User.email))
            users_emails = set(result.scalars().all())

            max_users = 1000
            if len(users_emails) > max_users:
                print("Удаляем лишних пользователей...")
                result = await db.execute(select(User).order_by(User.id.desc()).offset(max_users))
                for user in result.scalars().all():
                    await db.delete(user)
                await db.commit()

            to_add = max_users - len(users_emails)
            if to_add > 0:
                print(f"Загружаем {to_add} пользователей...")
                fetched_users = await fetch_unique_users(to_add, users_emails)
                new_users = [User(**u.model_dump()) for u in fetched_users]
                db.add_all(new_users)
                await db.commit()
                print(f"Добавлено {len(new_users)} пользователей.")
            else:
                print("Данные уже загружены.")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    gender = Column(String, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    location = Column(String, nullable=False)
    picture = Column(String, nullable=False)

class UserCreate(BaseModel):
    gender: str
    first_name: str
    last_name: str
    phone: str
    email: EmailStr
    location: str
    picture: str

class UserOut(UserCreate):
    id: int
    model_config = ConfigDict(from_attributes=True)

from collections.abc import AsyncGenerator

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session

async def fetch_users(n: int) -> list[UserCreate]:
    url = f"https://randomuser.me/api/?results={n}"
    retries = 0

    while True:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                users = []

                for u in data["results"]:
                    email = u["email"]
                    if ".." in email:
                        continue
                    try:
                        user = UserCreate(
                            gender=u["gender"],
                            first_name=u["name"]["first"],
                            last_name=u["name"]["last"],
                            phone=u["phone"],
                            email=email,
                            location=f"{u['location']['city']}, {u['location']['country']}",
                            picture=u["picture"]["thumbnail"],
                        )
                        users.append(user)
                    except ValidationError:
                        continue

                return users
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = min(60, (2 ** retries) * 5)
                print(f"429 Too Many Requests, ждем {wait_time} секунд...")
                await asyncio.sleep(wait_time)
                retries += 1
            else:
                raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=str(e))

async def fetch_unique_users(to_add: int, seen_emails: set) -> list[UserCreate]:
    fetched_users = []
    emails_collected = set(seen_emails)
    batch_size = 500

    while len(fetched_users) < to_add:
        batch_limit = min(batch_size, to_add - len(fetched_users))
        batch = await fetch_users(batch_limit)
        unique_batch = [u for u in batch if u.email not in emails_collected]
        emails_collected.update(u.email for u in unique_batch)
        fetched_users.extend(unique_batch)

    return fetched_users[:to_add]

@app.post("/load-users", response_model=list[UserOut])
async def load_users(total_count: int = Query(..., gt=0), db: AsyncSession = Depends(get_db)):
    max_users = 10000
    if total_count > max_users:
        raise HTTPException(status_code=400, detail=f"Максимальное количество пользователей: {max_users}")

    result = await db.execute(select(User.email))
    seen_emails = set(result.scalars().all())
    to_add = total_count - len(seen_emails)
    if to_add <= 0:
        return []

    fetched_users = await fetch_unique_users(to_add, seen_emails)
    new_users = [User(**u.model_dump()) for u in fetched_users]
    db.add_all(new_users)
    await db.commit()
    return new_users

@app.get("/users", response_model=list[UserOut])
async def list_users(skip: int = 0, limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).offset(skip).limit(limit))
    return result.scalars().all()

@app.get("/users/count")
async def get_users_count(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User.id))
    return {"count": len(result.scalars().all())}

@app.get("/random")
async def redirect_to_random_user(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User.id))
    ids = result.scalars().all()
    if not ids:
        raise HTTPException(status_code=404, detail="Пользователи не найдены")
    return RedirectResponse(url=f"/{random.choice(ids)}")

@app.get("/", response_class=HTMLResponse)
async def frontend_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/{user_id}", response_class=HTMLResponse)
async def user_detail(request: Request, user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return templates.TemplateResponse("user_detail.html", {"request": request, "user": user})
