import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app import app, get_db, User

DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def client_and_session():
    engine = create_async_engine(DATABASE_URL, future=True)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(User.metadata.create_all)

    async def override_get_db():
        async with async_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, async_session

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_users_empty(client_and_session):
    client, _ = client_and_session
    response = await client.get("/users")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_users_count_zero(client_and_session):
    client, _ = client_and_session
    response = await client.get("/users/count")
    assert response.status_code == 200
    assert response.json() == {"count": 0}


@pytest.mark.asyncio
async def test_load_users_with_mock(client_and_session, monkeypatch):
    client, session_maker = client_and_session

    async def mock_fetch_unique_users(count, seen_emails):
        from app import UserCreate
        return [
            UserCreate(
                gender="male",
                first_name=f"Test{i}",
                last_name="User",
                phone="1234567890",
                email=f"user{i}@example.com",
                location="TestCity, TestCountry",
                picture="http://example.com/pic.jpg"
            )
            for i in range(count)
        ]

    monkeypatch.setattr("app.fetch_unique_users", mock_fetch_unique_users)

    response = await client.post("/load-users?total_count=5")
    assert response.status_code == 200
    assert len(response.json()) == 5

    async with session_maker() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
        assert len(users) == 5


@pytest.mark.asyncio
async def test_redirect_to_random_user_404(client_and_session):
    client, _ = client_and_session
    response = await client.get("/random", follow_redirects=False)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_random_redirect_success(client_and_session):
    client, session_maker = client_and_session

    async with session_maker() as session:
        session.add(User(
            gender="male",
            first_name="John",
            last_name="Doe",
            phone="1234567890",
            email="john@example.com",
            location="TestCity",
            picture="http://example.com/pic.jpg"
        ))
        await session.commit()

    response = await client.get("/random", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"].startswith("/")


@pytest.mark.asyncio
async def test_user_detail_not_found(client_and_session):
    client, _ = client_and_session
    response = await client.get("/999")
    assert response.status_code == 404
