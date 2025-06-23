# 自定算子导入
自动算子全部导入到fastdeploy/model_executor/ops/xxx_gpu目录下
## 1. PaddleCustomDevice中自定义算子
如果自定义算子实现在PaddleCustomDevice中，需要此目录下对算子进行封装
以NPU flash_attention实现为例：
其在PaddleCustomDevice中实现,中的定义为：
调用实现：
```python
    from paddle.base import core
    attn_output = core.eager._run_custom_op(
        "flash_attention_npu",
        query_states,
        key_states,
        value_states,
        None,
        attention_mask,
        None,
        None,
        0.0,
        attention_mask is None,
        True,
        False,
        npu_is_casual,
        False,
    )[0]
```
添加

fastdeploy/model_executor/ops/npu/flash_attention_npu.py文件，实现内容为：
```python
from paddle.base import core

def fusion_flash_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    alibi=None,
    attn_mask_startend_row_indices=None,
    sequence_parallel=False,
    reshard_layer=None,
    npu_is_casual=False,
    skip_recompute=False,
):
    attn_output = core.eager._run_custom_op(
        "flash_attention_npu",
        query_states,
        key_states,
        value_states,
        None,
        attention_mask,
        None,
        None,
        0.0,
        attention_mask is None,
        True,
        False,
        npu_is_casual,
        False,
    )[0]
```

在fastdeploy/model_executor/ops/npu/__init__.py
文件中
```python
PACKAGE = "fastdeploy.model_executor.ops.npu"

from .fusion_flash_attention import fusion_flash_attention
```
