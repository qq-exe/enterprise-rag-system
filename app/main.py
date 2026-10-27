import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

# 创建FastAPI实例
app = FastAPI(title="企业RAG知识库问答服务2026")

# 跨域设置，支持前端网页调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #生产环境建议改为具体的前端域名，不要用*
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 注册路由，将不同功能的接口挂载到主应用上
app.include_router(router)

if __name__ == "__main__":
    #启动服务，reload=True仅用于开发环境，生产环境应关闭
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)