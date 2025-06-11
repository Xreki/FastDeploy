Append Attention文档整理

# 一、前言
对于Append Attention在使用过程中遇到的问题、各种参数的含义、用法，以及prefill与decode场景下append attention分别应该如何使用，需要注意的点以及踩坑部分，便于各位参考与借鉴。

截止目前，Append Attn有 **必填Tensor** 参数**21**个，**可选Tensor** 参数**12**个，**普通非Tensor** 参数**11**个，总计**44**个参数。



# 二、结构
由于CUDA与模板类的开发密不可分的缘故，整个Append Attention由于要生成各种各样的模板实例化，因此有各种if-else语句，阅读起来十分复杂，不利于大家直观的感受这个庞大kernel的结构，因此我绘制了一份包含关系图，以便各位抽丝剥茧的感受整个kernel的架构：

![图中，使用Times New Roman + 斜体的部分，表示Kernel中不同的片区；使用Arial + 粗体的部分，表示Kernel的名字](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=f1d5e51d5f91416ba9a94fad8cfcbb15&docGuid=c2lbwlsRaMMyNZ "图中，使用Times New Roman + 斜体的部分，表示Kernel中不同的片区；使用Arial + 粗体的部分，表示Kernel的名字")
****

这张图乍一看很复杂，为什么一个Attention里面有这么多组件？

实际上，在常规推理的时候，这个Kernel在**运行时**，一般来说，通常会是以下三种形态中的某一种：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=f7ef35689aec46dbab9971b07ca951c9&docGuid=c2lbwlsRaMMyNZ "")
这样看起来，一下子就清爽许多。我们可以做这样一个规律性的总结：

1. Append Attention总是由两部分组成:
    1. 一个部分叫做 Write Cache With RoPE
    2. 一个部分叫做 Cascade Append Attention




## 2.1 XxCoder Write Cache With RoPE
其实我们可以从名字观察出来：不管是Encoder还是Decoder，也不管Decoder有没有用投机解码，**Write Cache总是与RoPE融合在一起的**。这是kernel fusion的操作，也是以加速运算为目的。

我们可以进一步来看看，WriteCacheWithRoPEKernel里面是什么结构 (此处我们以Encoder Write Cache With RoPE为例)：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=3939c8395e78491ba31c4568ef0a8309&docGuid=c2lbwlsRaMMyNZ "")
Encoder Write Cache With RoPE里面有两个Kernel，**先是**apply RoPE，**然后**是Write KV Cache。

【**RoPE**】

Variable Length Rotary Kernel最后有一个**optional**，这个的意思是：如果在调用Append Attention的时候，不传入rotaty_embeds参数，就不会执行这个Variable Length Rotary Kernel。

*所以这其实是很灵活的，在Python层面就可以控制是否要进行RoPE。*



【**Write Cache**】

底层调用的kernel就是cache_kernel，负责更新kv cache。这一过程**十分重要**，因为在decoder阶段，k和v的来源是从kv cache中取得的，而不是依赖传入的qkv_out这个tensor，不是从qkv_out里面split出来q k v。所以在调用Append Attention之前，一般来说要使用Write Cache （后面会介绍prefill阶段的特殊性）。



## 2.2 Cascade Append Attention
首先一个问题：Cascade在这里是什么意思？

在计算机领域，**cascade**（级联）通常表示一种**分层或链式传递的关联关系**，在我们这里，就表示级联了一些小kernel。具体来说可以这样看级联的kernels:

* multi_query_append_attention_kernel
* merge_multi_chunks_decoder_kernel
* merge_multi_chunks_v2_kernel
* multi_query_append_attention_warp1_4_kernel

这是因为，append attention在保存KV Cache的时候，并不是像传统的kv cache一样的shape: 

**[****max_batch_size,       max_seq_len,      n_heads,         ****head_dim]，**

而是：

**[max_block_num,       n_heads,              block_size,     head_dim]**

*有关这两种KV Cache的换算方法，我们会在后面讲解*，目前聚焦于Append Attention的结构。



上述KV Cache的保存方法，折射出Append Attention处理文本的方案：**将一个sequence分为N个chunks，针对每一个chunk进行qkv计算。最后再merge到相应的位置。**



# 三、参数解读
在讲解计算过程之前，想先针对每一个参数进行解读，这样的阅读顺序有利于理解后面的内容。

## 3.1 基本参数（核心参数）
* **qkv**: 它的形状是 [total_seqlen, (num_q_head * head_dim_qk + num_kv_head * head_dim_qk + num_kv_head * head_dim_v)]
    * 这个shape是把 bsz 和seqlen融合起来、num_head和head_dim融合起来的结果。如果觉得不能理解，不妨我们抽丝剥茧进行推理：
        * 假设一般情况下，qkv的shape全都是：[bsz, seqlen, num_head, head_dim]
        * 如果这个时候，我们出现了变长seqlen的情况，也就是一个batch里面seqlen不一致，而且做padding又十分影响kernel效率的时候，就需要将batch_size和seqlen两个维度融合为一个维度，形成total_seqlen。于是有：$total\_seqlen = seqlen_1 + seqlen_2 + ... + seqlen_n$，这个时候，我们的qkv 的shape就变成了：[total_seqlen, num_head, head_dim]
        * 如果这个时候，我们出现了MQA、GQA的情况，num_head被分离为了num_q_head和num_kv_head的情况，那么我们的qkv就变成了：
            * q: [total_seqlen, num_q_head,   head_dim]
            * k: [total_seqlen, num_kv_head,  head_dim]
            * v: [total_seqlen, num_kv_head,  head_dim]

        * 如果这个时候，我们又出现了MLA的情况，head_dim被分离为了 head_dim_qk和head_dim_v的情况 （参考 [MLA论文](https://arxiv.org/pdf/2405.04434) 和 [MLA实现](https://github.com/deepseek-ai/DeepSeek-V3/blob/4cc6253d5c225e2c5fea32c54573449c1c46470a/inference/model.py#L463)）, 那么我们的qkv就变成了：
            * q: [total_seqlen, num_q_head,   head_dim_qk]
            * k: [total_seqlen, num_kv_head,  head_dim_qk]
            * v: [total_seqlen, num_kv_head,  head_dim_v  ]

        * 综上，Append Attention为了兼容这几种情况，又为了把num_head和head_dim两个维度融合在一起，并且又把qkv三个tensor融合到一起，最终才形成了最上面的shape。


* **cache_k**: 它的形状是 [max_block_num, num_kv_head, block_size, head_dim_qk]
    * block_size默认情况下定义为64
    * $max\_block\_num = div\_ceil(max\_input\_length, block\_size)$注意这个max_input_length并不是当前batch里面的最长sequence length，而是在启动程序的时候预设好的一个max_input_length。一般来说我们会设置为64K或者32K或者8K （视情况而定）
    * 划分block，主要也是为了利于长文本处理，将文本划分为chunks。

* **cache_v**: 它的形状是 [max_block_num, num_kv_head, block_size, head_dim_v]，其余部分同上。



接下来的参数，我们以实际例子来讲解。假设我们现在的输入是：

```
"2014年3月，大范围雾霾天气长时间影响我国东部地区，严重危害人体健康。造成雾霾天气的人为原因有____\r\n①工业生产中使用矿物作为燃料，大量排放污染物     ②汽车尾气的大量排放     \r\n③风力小，空气流动不畅     ④冬季取暖排放粉尘\nA. ①②③\nB. ②③④\nC. ①③④\nD. ①②④"
```
这一个sequence在tokenize之后，seqlen为：131。因此我们暂时假设prefill阶段的seqlen为131. 同时，batch_size = 2.



* **seq_lens_encoder**: 它的形状是：**[bsz, 1] （[2, 1]）**
    * 它的实际值是：[[131], [131]], 每一个元素表示该sequence的长度。
    * 在Encoder阶段，它的值会是 [[seq1], [seq2], [seq3], ..., [seqN]]
    * 在Decode阶段，它的值会是 [[0], [0], ..., [0]]

* **seq_lens_decoder**: 它的形状是：**[bsz, 1] （[2, 1]）**
    * 它的实际值是：[[131], [131]]，每一个元素表示该sequence在当前decoding阶段的激活token数目。
    * 在Encoder阶段，它的值会是 [[0], [0], ..., [0]]
    * 在Decode阶段，它的值会是 [[activated_seqlen_1], [activated_seqlen_2], ..., [activated_seqlen_N]]

* **seq_lens_this_time**: 它的形状是：**[bsz, 1] （[2, 1]）**
    * 它的实际值是：[[131], [131]] 或者 [[1], [1]]，每一个元素表示该sequence在当前阶段需要被kernel处理的token数目。
    * 在Encoder阶段，它的值会是 [[seq1], [seq2], [seq3], ..., [seqN]]
    * 在Decoder阶段，它的值会是 [[1], [1], ..., [1]]




***介绍完这些参数，我们对目前的Kernel运行时情况做一个总结。***

上述情况，描述的是离线推理模型时候的情况。（例如使用predictor.py进行推理）。如果我们在服务化Serving的情况下，这三个参数的值会多变起来，因为服务化有continuous batching (in-flight batching)的情况。

如果有continuous batching的情况下，整个Append Attention的Kernel形态会变成完全体：

![运行时的情况：既有seq在prefill，也有seq在进行decoding](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=26311291ca324a089cd459f903b5c9ff&docGuid=c2lbwlsRaMMyNZ "运行时的情况：既有seq在prefill，也有seq在进行decoding")
又有Encoder的Kernel又有Decoder的两个Kernel Module同时存在。

可以用这样一幅图来阐述推理时候发生的情况：

![最左边的seq_lens_this_time应该是 [[131], [130]]，笔误](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=2cf0c15937934ff1a3924d2bc0ba4a22&docGuid=c2lbwlsRaMMyNZ "最左边的seq_lens_this_time应该是 [[131], [130]]，笔误")
左四我们假设第一个seq已经完成了推理，被踢出batch，下一个新的seq进来开始进行prefill。所以下面三个数值就是这样变化的。



## 3.2 辅助参数（帮助kernel定位index）
辅助参数们，包括 

* cu_seqlens, padding_offsets, 
* cum_offsets, 
* block_tables, 
* encoder_batch_ids, 
* encoder_tile_ids_per_batch, 
* encoder_num_blocks, 
* kv_batch_ids, 
* kv_tile_ids_per_batch, 
* kv_num_blocks, 
* decoder_batch_ids, 
* decoder_tile_ids_per_batch, 
* decoder_num_blocks_cpu, 
* max_len_kv 
* excess_blocks（此处共14个参数）

这些参数的获取，可以通过如下方式来得到：

```
seq_lens_enc = [
    input_length,
] * bsz
seq_lens_dec = [
    0,
] * bsz
seq_lens_this_time = [
    input_length,
] * bsz
max_enc_len_this_time = max(seq_lens_enc)
max_dec_len_this_time = max(seq_lens_dec)
max_enc_len_this_time = paddle.to_tensor([max_enc_len_this_time], "int32", place=paddle.CPUPlace())
max_dec_len_this_time = paddle.to_tensor([max_dec_len_this_time], "int32", place=paddle.CPUPlace())
token_num = sum(seq_lens_this_time)
block_num_per_seq = (max_length + block_size - 1) // block_size
max_block_num = block_num_per_seq * bsz
free_list = list(range(max_block_num - 1, -1, -1))

# 在这里计算block tables
block_tables = paddle.zeros(shape=(bsz, block_num_per_seq), dtype="int32") * (-1)
for i in range(bsz):
    need_block_num = (seq_lens_enc[i] + max_dec_len + block_size - 1) // block_size
    for j in range(need_block_num):
        block_id = free_list.pop()
        block_tables[i, j] = block_id
        
# 在这里获得其他参数 (encoder_batch_ids, encoder_tile_ids_per_batch, encoder_num_blocks, kv_batch_ids, kv_tile_ids_per_batch, kv_num_blocks, decoder_batch_ids等)
(
    encoder_batch_ids,
    encoder_tile_ids_per_batch,
    encoder_num_blocks,
    kv_batch_ids,
    kv_tile_ids_per_batch,
    kv_num_blocks,
    decoder_batch_ids,
    decoder_tile_ids_per_batch,
    decoder_num_blocks,
    max_len_kv,
) = paddlenlp_ops.get_block_shape_and_split_kv_block(
    seq_lens_encoder,
    seq_lens_decoder,
    max_enc_len_this_time,
    max_dec_len_this_time,
    seq_lens_this_time,
    cum_offsets,
    num_q_head // num_kv_head,
    block_size,
    1,
)

# 通过这个函数获得padding offsets，cum_offset
def get_padding_offset(bsz, max_seq_len, seq_lens_this_time):
    cum_offsets_now = paddle.cumsum(max_seq_len - seq_lens_this_time)
    cum_offsets = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cum_offsets[1:] = cum_offsets_now
    token_num = paddle.sum(seq_lens_this_time)
    padding_offsets = paddle.zeros(shape=(token_num), dtype="int32")
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
    for i in range(bsz):
        seq_len_now = seq_lens_this_time[i]
        cum_offset = cum_offsets[i]
        for j in range(seq_len_now):
            padding_offsets[i * max_seq_len - cum_offset + j] = cum_offset
        cum_seq_len = (i + 1) * max_seq_len - cum_offsets[i + 1]
        cu_seqlens_q[i + 1] = cum_seq_len
        cu_seqlens_k[i + 1] = cum_seq_len
    return padding_offsets, cum_offsets[:-1], cu_seqlens_q, cu_seqlens_k
```




## 3.3 通用信息参数
通用信息参数包括：

* rotary_embs: optional<paddle::Tensor>
    * 如果提供，则进行旋转位置编码
    * 如果不提供，则默认不进行旋转位置编码

* attn_mask: optional<paddle::Tensor>
    * 如果提供，则按照attn_mask来处理mask
    * 如果不提供，优先参照参数causal。若为True，默认进行上三角mask；若为False，不进行任何掩码






## 3.4 量化参数
* qkv_bias
* qkv_out_scales
* cache_k_quant_scales
* cache_v_quant_scales
* cache_k_dequant_scales
* cache_v_dequant_scales
* cache_k_zp
* cache_v_zp
* out_linear_shifts
* out_linear_smooths



上述参数均为 paddle::optional<paddle::Tensor>类型，在16位推理的时候不需要提供。**在使用A8W8，FP8推理的时候，需要提供**，以在运行时进行量化与反量化。



* cache_quant_type_str: string
* quant_max_bound: float
* quant_min_bound: float
* out_linear_in_scale: float



这四个参数需要**在使用A8W8，FP8推理的时候提供。**

寻常情况下，传入None或者0.0即可。





## 3.5 其余信息flags
* use_neox_rotary_style: bool，是否使用neox形式的旋转位置编码风格
* max_input_length: int, 当前batch中最大的序列长度
* softmax_scale: float, softmax的scale，一般情况下是$\sqrt{d}$
* speculate_max_draft_token_num: int, 投机解码参数
* causal: bool，是否causal
* speculate_decoder: bool, 是否在decode阶段使用投机解码





# 四、Prefill的特殊性
Prefill阶段，主要是把输入的所有内容都转化为KV cache，这一步基本上是推理时候的overhead比较大的部分。

而且我们之前提到：在decoder阶段，k和v的来源是从kv cache中取得的，而不是依赖传入的qkv_out这个tensor。

但是，prefill阶段，又没有KV cache，总得有个办法获取input。**所以qkv_out这个参数，可以视为** **仅仅在Encoder阶段使用**，在Prefill的时候，qkv_out里面的信息是包含了qkv全量的，可以分别拆出来q,k,v 三个tensor。

Decoder阶段会用到qkv_out里面的q，KV是直接从kv_cache里面取用的。



附上一段代码来演示如何在算子中访问QKV（在prefill阶段）：

```
// 参数：meta_data, cu_seqlen, qkv
int batch_size = cu_seqlen.shape()[0] - 1;

const int num_q_head = meta_data.q_num_heads;
const int head_dim_qk = meta_data.head_dims;

const int num_kv_head = meta_data.kv_num_heads;
const int head_dim_v = meta_data.head_dims_v;

std::vector<paddle::Tensor>&& qkv_with_rope = paddle::split(qkv, {num_q_head * head_dim_qk, num_kv_head * head_dim_qk, num_kv_head * head_dim_v}, 1);
paddle::Tensor q = paddle::reshape(qkv_with_rope[0], {-1, num_q_head, head_dim_qk});
paddle::Tensor k = paddle::reshape(qkv_with_rope[1], {-1, num_kv_head, head_dim_qk});
paddle::Tensor v = paddle::reshape(qkv_with_rope[2], {-1, num_kv_head, head_dim_v});
```
至于如何在Python层面访问q、k、v，如法炮制即可。


再往下，是一些零散组件。大致有write_cache_with_rope，append_attn_cX_impl.cuh等。可以去append_attn/路径下细看。

* 一般来说，如果仅仅是修改append attention的入口、桥接一个新的attention进去，**只需要关注上面提到的前两层**即可。



* **注意**！有一份文件叫做**cpp_extensions.cu**，这份文件里面**声明了append attention的函数**。如果要修改函数入口的话，记得也要修改这份文件里面的函数声明！（否则会运行时报错undefined symbol）



## 
