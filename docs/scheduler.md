# 调度器

FastDeploy 目前支持两种调度器 本地调度器 和 全局调度器 本地调度器 全局调度器 适用于大规模集群，基于各节点的实际负载在集群内部进行二次负载均衡。

## 调度策略

### 本地调度器
本地调度器可以等效于内存管理器，根据 任务队列长度 和 TTL 的配置进行内存淘汰。

### 全局调度器
全局调度器基于 Redis 实现，各个节点根据自身 GPU 负载情况，空闲时主动从其他节点偷取任务，然后将任务的执行结果推送回原节点。

## 配置参数
| 字段名                    | 字段类型 | 是否必填 | 默认值    | 生效范围     | 说明                                                             |
| ------------------------- | -------- | -------- | --------- | ------------ | ---------------------------------------------------------------- |
| scheduler_name            | str      | 否       | local     | local,global | 调度器名：local，global                                          |
| scheduler_max_size        | int      | 否       | -1        | local        | 最大任务队列长度                                                 |
| scheduler_ttl             | int      | 否       | 900       | local,global | 任务最大存活时间                                                 |
| scheduler_host            | str      | 否       | 127.0.0.1 | global       | redis服务地址                                                    |
| scheduler_port            | int      | 否       | 6379      | global       | redis服务端口                                                    |
| scheduler_db              | int      | 否       | 0         | global       | redis数据库序号                                                  |
| scheduler_password        | str      | 否       | ""        | global       | redis访问密码                                                    |
| scheduler_topic           | str      | 否       | default   | global       | 任务主题                                                         |
| scheduler_min_load_score  | float    | 否       | 1         | global       | 当节点的负载大于最小阈值时，若其他节点空闲则可以偷取该节点的任务 |
| scheduler_load_shards_num | int      | 否       | 1         | global       | 集群负载信息表的分片数                                           |