"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import uvicorn
import json
from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from fastdeploy.utils import FlexibleArgumentParser, api_server_logger, is_port_available
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.entrypoints.openai.protocol import (
    CompletionRequest, 
    ChatCompletionRequest, 
    ErrorResponse, 
    ChatCompletionResponse, 
    CompletionResponse
)

from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat
from fastdeploy.entrypoints.openai.serving_completion import OpenAIServingCompletion

app = FastAPI()

llm_engine = None
chat_handler = None
completion_handler = None


def init_app(args):
    """
    init LLMEngine
    """
 
    global llm_engine
    global chat_handler
    global completion_handler

    engine_args = EngineArgs.from_cli_args(args)
    llm_engine = LLMEngine.from_engine_args(engine_args)
    if not llm_engine.start():
        api_server_logger.error("Failed to initialize FastDeploy LLM engine, service exit now!")
        return False

    chat_handler = OpenAIServingChat(llm_engine)
    completion_handler = OpenAIServingCompletion(llm_engine)
    api_server_logger.info(f"FastDeploy LLM engine initialized!")
    return True

@app.get("/health")
def health(request: Request) -> Response:
    """Health check."""
    status, msg = llm_engine.check_health()
    if not status:
        return Response(content=msg, status_code=404)
    return Response(status_code=200)


@app.get("/load")
async def list_all_routes():
    """
    列出所有以/v1开头的路由信息

    Args:
        无参数

    Returns:
        dict: 包含所有符合条件的路由信息的字典，格式如下:
            {
                "routes": [
                    {
                        "path": str,  # 路由路径
                        "methods": list,  # 支持的HTTP方法列表，已排序
                        "tags": list  # 路由标签列表，默认为空列表
                    },
                    ...
                ]
            }

    """
    routes_info = []

    for route in app.routes:
        # 直接检查路径是否以/v1开头
        if route.path.startswith("/v1"):
            methods = sorted(route.methods)
            tags = getattr(route, "tags", []) or []
            routes_info.append({
                "path": route.path,
                "methods": methods,
                "tags": tags
            })
    return {"routes": routes_info}




@app.api_route("/ping", methods=["GET", "POST"])
def ping(raw_request: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    return health(raw_request)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion for the provided prompt and parameters.
    """
    generator = await chat_handler.create_chat_completion(request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """
    Create a completion for the provided prompt and parameters.
    """

    generator = await completion_handler.create_completion(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


def launch_api_server(args) -> None:
    """
    启动http服务
    """
    if not is_port_available(args.host, args.port):
        api_server_logger.error(f"The parameter `port`:{args.port} is already in use.")
        return

    api_server_logger.info(f"launch Fastdeploy api server... port: {args.port}")
    api_server_logger.info(f"args: {args.__dict__}")

    if not init_app(args):
        api_server_logger.error("API Server launch failed.")
        return False

    try:
        uvicorn.run(app=app,
                    host=args.host,
                    port=args.port,
                    workers=args.workers,
                    log_level="info")  # set log level to error to avoid log
    except Exception as e:
        api_server_logger.error(f"launch sync http server error, {e}")


def main():
    """main函数"""
    parser = FlexibleArgumentParser()
    parser.add_argument("--port", default=9904, type=int, help="port to the http server")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="host to the http server")
    parser.add_argument("--workers", default=1, type=int, help="number of workers")
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    launch_api_server(args)
    

if __name__ == "__main__":
    main()
