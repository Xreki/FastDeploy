# 日志说明

启动服务，在服务启动当前目录下，服务会自动创建log目录，并以追加的方式写入日志，具体日志文件包括
- backup_env.*.json
- data_processor.log
- default.*.log
- envlog.*
- api_server.log
- llm.log
- workerlog.*
- worker.log

其中在日志使用过程中，重点关注的日志文件为data_processor.log, api_server.log, llm.log, workerlog.*, worker.log
