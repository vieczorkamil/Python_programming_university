from fastapi import FastAPI

app = FastAPI(
    title="Advanced programming in Python - FastAPI"
)

# app.include_router(ep_2.router)
# app.include_router(ep_1.router)
# app.include_router(ep_3.router)

@app.get("/", tags=["Default"])
async def read_root():
    return {"Hello": "World"}