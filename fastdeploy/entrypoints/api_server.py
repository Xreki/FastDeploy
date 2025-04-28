"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse

from fastdeploy.utils import FlexibleArgumentParser, api_server_logger, is_port_available
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine

app = FastAPI()

llm_engine = None

def init_app(args):
    """
    init LLMEngine
    """

    global llm_engine
    engine_args = EngineArgs.from_cli_args(args)
    llm_engine = LLMEngine.from_engine_args(engine_args)
    if not llm_engine.start():
        api_server_logger.error("Failed to initialize FastDeploy LLM engine, service exit now!")
        return False

    api_server_logger.info(f"FastDeploy LLM engine initialized!")
    return True


@app.get("/health")
def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
def generate(request: dict):
    """
    generate stream api
    """
    api_server_logger.info(f"receive request: {request}")
    stream = request.get("stream", 0)
    def event_generator():
        for result in llm_engine.generate(request, stream):
            yield json.dumps(result)
    return StreamingResponse(event_generator(), media_type="text/event-stream")



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
        return

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
