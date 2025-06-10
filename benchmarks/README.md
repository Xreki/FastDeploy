### FastDeploy服务化性能压测工具

#### 数据集：

wget下载到本地用于性能测试

<table style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="width:15%; text-align: left;">Dataset</th>
      <th style="width:65%; text-align: left;">Data Path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>EB45T纯文 9903条</strong></td>
      <td><code>https://fastdeploy.bj.bcebos.com/eb_query/eb45t_spv4_dataserver_1w_fd</code></td>
    </tr>
    <tr>
      <td><strong>EB45T纯文 9903带外挂</strong></td>
      <td><code>https://fastdeploy.bj.bcebos.com/eb_query/eb45t_spv4_dataserver_1w_waigua_fd</code></td>
    </tr>
    <tr>
      <td><strong>X1-Turbo 5000条</strong></td>
      <td><code>https://fastdeploy.bj.bcebos.com/eb_query/0316_x1_45t_yiyan_nocot_5000_fd</code></td>
    </tr>
    <tr>
      <td><strong>纯文128K数据 2000条</strong></td>
      <td><code>https://fastdeploy.bj.bcebos.com/eb_query/mingming_0410_eb_2k_ds_108_32_128k_part1_fd</code></td>
    </tr>
    <tr>
      <td><strong>其他数据</strong></td>
      <td><code>https://fastdeploy.bj.bcebos.com/eb_query/0416_x1_45t_forqianfan_4998_fd
https://fastdeploy.bj.bcebos.com/eb_query/0419_api9_yiyan_spv5_forqianfan_4872_fd</code></td>
    </tr>
  </tbody>
</table>

#### 使用方式：

python -m pip install -r requirements.txt

##### （可选）

```
conda env压缩包：
放入miniconda3/envs解压即可使用
https://fastdeploy.bj.bcebos.com/py310-2.tar
```

##### 参数说明

```bash
--backend openai-chat：压测使用的后端接口，指定为"openai-chat"使用chat/completion接口
--model EB45T：模型名，任意取名，影响最后保存的结果文件名 EB45T \
--endpoint /v1/chat/completions：endpoint，用于组url
--host 0.0.0.0：服务ip地址，用于组url
--port 9812：服务HTTP端口，用于组url
--dataset-name EBChat：指定数据集类，指定为"EBChat"可读取转存的FD格式数据集
--dataset-path ./eb45t_spv4_dataserver_1w_waigua_fd：压测数据集路径
--hyperparameter-path EB45T.yaml：超参文件，请求时会更新进payload中
--percentile_metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len：性能结果中展示的指标集合
--metric_percentiles 80,95,99,99.9,99.95,99.99：性能结果中展示的性能指标分位值
--num-prompts 1：总计发送多少条请求
--max_concurrency 1：压测并发数
--save_result：开启结果保存，结果文件会存入json
```

##### /v1/chat/completions接口压测单条数据调试

```
python benchmark_serving.py \
  --backend openai-chat \
  --model EB45T \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port 9812 \
  --dataset-name EBChat \
  --dataset-path ./eb45t_spv4_dataserver_1w_waigua_fd \
  --hyperparameter-path EB45T.yaml \
  --percentile_metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric_percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 1 \
  --max_concurrency 1 \
  --save_result
```

##### /v1/chat/completions接口完整100并发 9903条压测

```
# 保存infer_log.txt
python benchmark_serving.py \
  --backend openai-chat \
  --model EB45T \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port 9812 \
  --dataset-name EBChat \
  --dataset-path ./eb45t_spv4_dataserver_1w_waigua_fd \
  --hyperparameter-path EB45T.yaml \
  --percentile_metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric_percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 9903 \
  --max_concurrency 100 \
  --save_result > infer_log.txt 2>&1 &
```

##### /v1/completions接口压测

修改endpoint为/v1/completions，backend为openai，会对/v1/completions接口进行压测

```
# 保存infer_log.txt
python benchmark_serving.py \
  --backend openai \
  --model EB45T \
  --endpoint /v1/completions \
  --host 0.0.0.0 \
  --port 9812 \
  --dataset-name EBChat \
  --dataset-path ./eb45t_spv4_dataserver_1w_waigua_fd \
  --hyperparameter-path EB45T.yaml \
  --percentile_metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric_percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 9903 \
  --max_concurrency 100 \
  --save_result > infer_log.txt 2>&1 &
```


