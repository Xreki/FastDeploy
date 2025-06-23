# Benchmark

FastDeploy基于[vLLM benchmark](https://github.com/vllm-project/vllm/blob/main/benchmarks/)脚本，增加了部分统计信息，可用于benchmark FastDeploy更详细的性能指标。

## 测试数据集

以下数据集来源于开源数据集
| 数据集 | 说明 |
| :----- | :--- |
| abc | abc |

## 测试方式

```
cd FastDeploy/benchmark
python -m pip install -r requirements.txt

# 启动服务
python -m fastdeploy.entrypoints.openai.api_server \
       --model Qwen2-Instruct-7B \
       --port 8188 \
       --tensor-parallel-size 1 \
       --max-model-len 8192

# 压测服务
python benchmark_serving.py \
  --backend openai-chat \
  --model Qwen2-Instruct-7B \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port 8192 \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --hyperparameter-path yaml/ernie_45_12k_80g_tp4.yaml \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --save-result
```
