# ERNIE参数推荐

在使用ERNIE系列生成模型时，推荐使用如下请求参数配置

> 关于Diff率问题: 服务在部署后，无法保证同样的输入可以得到同样的输出，当且仅当如下情况，可以确保结果的稳定性
> 1. 服务当前仅单并发处理请求（即完成一个请求后，再发下一个请求），避免动态插入组batch，使得计算kernel的变化影响结果的稳定性
> 2. top_p参数设置为0，且seed参数固定

## ERNIE 4.5 Turbo

|    参数    | 推荐值 | 服务默认值 |
|    :---    | :----  | :------- |
| top_p | 0.8 | 0.7|
| temperature | 0.8 | 1.0 |
| max_tokens | 12288 | 2048 |
| repetition_penalty | 1.0 | 1.0 |
| frequency_penalty | 0.0 | 0.0 |
| presence_penalty | 0.0 | 0.0 |

## ERNIE 4.5 Turbo X1

|    参数    | 推荐值 | 服务默认值 |
|    :---    | :----  | :------- |
| top_p | 0.95 | 0.7|
| temperature | 0.6 | 1.0 |
| max_tokens | 32768 | 2048 |
| repetition_penalty | 1.0 | 1.0 |
| frequency_penalty | 0.0 | 0.0 |
| presence_penalty | 0.0 | 0.0 |

## ERNIE 4.5 Turbo VL(多模)

|    参数    | 推荐值 | 服务默认值 |
|    :---    | :----  | :------- |
| top_p | 0.8 | 0.7|
| temperature | 0.2 | 1.0 |
| max_tokens | 4096 | 2048 |
| repetition_penalty | 1.0 | 1.0 |
| frequency_penalty | 0.0 | 0.0 |
| presence_penalty | 0.0 | 0.0 |
