
# ERNIE-45-Turbo weight-only-int2 规范性说明

* quant_weight: 量化权重（即编码值）；(*.weight -> *.quant_weight)
  * 数据类型：int8；
  * 维度：按照group_size=64,进行pp-acc 2bit压缩，量化为int8存储；

* weight_scale：权重的量化scale；(*.weight  -> *.weight_scale), 如'ernie.layers.1.mlp.down_proj.weight_scale'
  * 数据类型：默认进行4bits 量化，pack成int8存储；
  * 维度：per-group

* super_scale：对权重scale进行二次量化的scale，(*.weight  -> *.super_scale), 如'ernie.layers.1.mlp.down_proj.super_scale'
  * 数据类型：bf16；
  * 维度：per-channel

* code_scale：编码聚类的量化scale；
  * 数据类型：fp32；
  * 维度：per-channel

* code_zp：编码聚类的量化zero point；
  * 数据类型：fp32；
  * 维度：per-channel

# 量化模型配置：config.json

```json
"is_quantized": true,
"quantization_config": {
    "dense_quant_type": "wint8",
    "moe_quant_type": "w4w2",
    "quantization": "wint2",
    "moe_quant_config": {
    "moe_w4_quant_config": {
        "quant_type": "wint4",
        "quant_granularity": "per_channel",
        "quant_start_layer": 0,
        "quant_end_layer": 6
    },
    "moe_w2_quant_config": {
        "quant_type": "wint2",
        "quant_granularity": "pp_acc",
        "quant_group_size": 64,
        "quant_start_layer": 7,
        "quant_end_layer": 53
    }
  }
}
```
