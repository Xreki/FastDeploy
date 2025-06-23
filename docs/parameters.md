# FastDeploy参数说明

在使用FastDeploy部署模型（包括离线推理、服务化部署），涉及如下参数配置，其实需要注意，在使用离线推理时，各参数配置即为如下参数名；而在使用命令行启动服务时，相应参数中的分隔符需要从```_```修改为```-```，如```max_model_len```在命令行中则为```--max-model-len```。

| 参数名 | 类型 | 说明 |
| :---- | :---- | :----- |
| ```port``` | int | 仅服务化部署需配置，服务HTTP请求端口号 |
| ```metrics-port``` | int | 仅服务化部署需配置，服务监控Metrics端口号，默认 |
| ```engine_worker_queue_port``` | int | FastDeploy内部引擎进程通信端口 |
| ```cache_queue_port``` | int | FastDeploy内部KVCache进程通信端口 |
| ```max_model_len``` | int | 推理默认最大支持上下文长度，默认2048 |
| ```tensor_parallel_size``` | int | 模型默认张量并行数，默认1 |
| ```block_size``` | int | KVCache管理粒度(Token数)，推荐默认值64 |
| ```max_num_seqs``` | int | Decode阶段最大的并发数，默认 |
| ```mm_processor_kwars``` | dict[str] | xxxx |
| ```limit_mm_per_prompt``` | dict[str] | xxx |
| ```enable_mm``` | bool | 是否支持多模态数据（仅针对多模模型)，默认False |
| ```quantization``` | str | 模型量化策略，当在加载BF16 CKPT时，指定wint4或wint8时，支持无损在线4bit/8bit量化 |
| ```gpu_memory_utilization``` | float | GPU显存利用率，默认0.9 |
| ```num_gpu_blocks_override``` | int | 预分配KVCache块数，此参数可由FastDeploy自动根据显存情况计算，无需用户配置，默认为None |
| ```max_num_batched_tokens``` | int | Prefill阶段最大Batch的Token数量，默认为None(与max_model_len一致) |
| ```kv_cache_ratio``` | float | KVCache块按kv_cache_ratio比例分给Prefill阶段和Decode阶段 | 
| ```enable_prefix_caching``` | bool | 是否开启Prefix Caching，默认False |
| ```cpu_offload_gb``` | float | 开启Prefix Caching时，用于swap KVCache的CPU内存大小，单位GB，默认None |
| ```enable_chunk_prefill``` | bool | 开启Chunked Prefill，默认False |
| ```max_num_partial_prefills``` | int | 开启Chunked Prefill时，Prefill阶段的最大并发数，默认1 |
| ```max_long_partial_prefills``` | int | 开启Chunked Prefill时，Prefill阶段并发中包启的最多长请求数，默认1 |
| ```long_prefill_token_threshold``` | int | 开启Chunked Prefill时，请求Token数超过此值的请求被视为长请求，默认为max_model_len*0.04 |
| ```static_decode_blocks``` | int | 推理过程中，每条请求强制从Prefill的KVCache分配对应块数给Decode使用，默认2|


## 1. KVCache分配与```num_gpu_blocks_override```、```block_size```的关系？

FastDeploy在推理过程中，显存被```模型权重```、```预分配KVCache块```和```模型计算中间激活值```占用。其中预分配KVCache块由```num_gpu_blocks_override```决定，其单位为```block_size```(默认64），即一个块可以存储64个Token的KVCache。

在实际推理中，用户很难知道```num_gpu_blocks_override```到底该配置到多少合适，因此FastDeploy采用如下方式来自动推导并配置这个值，流程如下
> 1. 加载模型，在完成模型加载后，记录当前显存占用情况```total_memory_after_load```和FastDeploy框架占用的显存值```fd_memory_after_load```; 注意前者为GPU实际被占用显存（可能有其它进程也占用），后者是FD框架本身占用显存；
- 根据用户配置的```max_num_batched_tokens```(默认为```max_model_len```)，Fake相应长度的输入数据进行Prefill计算，记录当前FastDeploy框架显存最大分配值```fd_memory_after_prefill```，因此可以认为```模型计算中间激活值```为```fd_memory_after_prefill - fd_memory_after_load```;
- - 截止当前，认为GPU卡可以剩分配KVCache的显存(以A800 80G为例)为```80GB * gpu_memory_utilization - total_memory_after_load - (fd_memory_after_prefill - fd_memory_after_load)``` 
- - 根据模型KVCache的精度（如8bit/16bit)，计算一个block占用的KVCache大小，从而计算出总共可分配的block数量，赋值给```num_gpu_blocks_override```

   在服务启动日志中，我们可以在log/fastdeploy.log中找到```Reset block num, the total_block_num:17220, prefill_kvcache_block_num:12915```，其中```total_block_num```即为自动计算出来的KVCache block数量，将其乘以```block_size```即可知道整个服务可以缓存多少Token的KV值。

## 2. ```kv_cache_ratio```、```block_size```、```max_num_seqs```的关系？
   - FastDeploy里面将KVCache按照```kv_cache_ratio```分为Prefill阶段使用和Decode阶段使用，在配置这个参数时，可以按照```kv_cache_ratio = 平均输入Token数/(平均输入+平均输出Token数)```进行配置，常规情况输入是输出的3倍，因此可以配置成0.75
   - ```max_num_seqs```是Decode阶段的最大并发数，一般而言可以配置成最大值128，但用户也可以根据KVCache情况作调用，例如输出的KVCache Token量为```decode_token_cache = total_block_num * (1 - kv_cache_ratio) * block_size```，为了防止极端情况下的显存不足问题，可以配置```max_num_seqs = decode_token_cache / 平均输出Token数```，不高于128即可。

## 3. ```enable_chunked_prefill```参数配置说明
@程延福
