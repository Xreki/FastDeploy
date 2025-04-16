import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet

from paddlenlp.trl import llm_utils
from paddlenlp.trl.llm_utils import get_rotary_position_embedding
import numpy as np
from fastdeployllm.worker.model_runner.model_runner_base import ModelRunnerBase
from fastdeployllm.worker.utils import PredictorArgument, ModelArgument

class ModelRunner(ModelRunnerBase):
    def __init__(self, config, args, nranks, rank):
        """
            Initializes the model and sets up the necessary parameters for distributed training.
        
        Args:
            config (DictConfig): Config dictionary for the model.
            args (argparse.Namespace): Arguments for the model.
            nranks (int): Number of GPUs used in parallel training.
            rank (int): Rank of the current GPU used in parallel training.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self.nranks = nranks
        self.rank = rank
        super().__init__(config, args)

    def _load_model(self, model_name):
        """
            加载模型，并设置缓存。
        
        Args:
            model_name (str): 模型名称或路径。
        
        Returns:
            None.
        """
        llm_utils.set_triton_cache(self.args.model_name_or_path, "dynamic")

        predictor_args = PredictorArgument()
        model_args = ModelArgument()

        predictor_args.model_name_or_path = self.args.model_name_or_path
        predictor_args.max_length = self.args.max_dec_len
        predictor_args.dtype = self.args.dtype
        predictor_args.total_max_length = self.args.max_seq_len
        predictor_args.inference_model = True
        predictor_args.mode = "dynamic"
        predictor_args.block_attn = True

        paddle.set_device(predictor_args.device)
        paddle.set_default_dtype(predictor_args.dtype)

        from paddlenlp.transformers import AutoConfig, AutoInferenceModelForCausalLM

        config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)
        self.model = AutoInferenceModelForCausalLM.from_pretrained(
            predictor_args.model_name_or_path,
            config=config,
            predictor_args=predictor_args,
            model_args=model_args,
            dtype=predictor_args.dtype,
            tensor_parallel_degree=self.nranks,
            tensor_parallel_rank=self.rank,
        )

    def init_rotary_position_embedding(self, max_seq_len):
        """
            初始化旋转位置嵌入，并将其保存在模型中。
        该函数会创建一个长度为max_seq_len的位置ID序列，并使用get_rotary_position_embedding函数生成相应的旋转位置嵌入。
        
        Args:
            max_seq_len (int): 最大序列长度。
        
        Returns:
            None. 直接修改模型中的share_inputs字典，添加名称为"rope_emb"的键值对，包含旋转位置嵌入。
        """
        tmp_position_ids = paddle.arange(max_seq_len).reshape((1, -1))
        self.share_inputs["rope_emb"] = get_rotary_position_embedding(
            tmp_position_ids,
            self.model_cfg.hidden_size // self.model_cfg.num_attention_heads,
            self.rope_theta,
            self.rope_scaling,
        )

    def _init_kvcache(self, max_block_num):
        """
        分享不拷贝数据
        """
        self.cache_kvs = {}

        if (
            hasattr(self.model_cfg, "num_key_value_heads")
            and hasattr(self.model_cfg, "num_key_value_heads")
            and self.model_cfg.num_key_value_heads is not None
            and int(self.model_cfg.num_key_value_heads) > 0
        ):
            kv_num_head = int(self.model_cfg.num_key_value_heads) // self.nranks
        else:
            kv_num_head = self.model_cfg.num_attention_heads // self.nranks

        for i in range(self.model_cfg.num_layers):
            cache_type = self.args.dtype
            self.cache_kvs["key_caches_{}".format(i)] = paddle.full(
                shape=[
                    self.args.max_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.hidden_size // self.model_cfg.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )
            self.cache_kvs["value_caches_{}".format(i)] = paddle.full(
                shape=[
                    self.args.max_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.hidden_size // self.model_cfg.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )

        self.share_inputs["cache_kvs"] = list(self.cache_kvs.values())
        for value in self.cache_kvs.values():
            del value

    def generate(self):
        self.model.generate(**self.share_inputs)

    def dy_input_preprocess(self, tasks):
        """
        dynamic insertion
        """
        for i in range(len(tasks)):
            task = tasks[i]
            idx = task["idx"]
            length = len(task["input_ids"])
            self.share_inputs["input_ids"][idx : idx + 1, :length] = np.array(task["input_ids"])
            if len(task["eos_token_ids"]) < self.args.eos_tokens_lens:
                task["eos_token_ids"].append(task["eos_token_ids"][0])
            self.share_inputs["eos_token_id"][:] = np.array(task["eos_token_ids"], dtype="int64").reshape(-1, 1)
            self.share_inputs["pre_ids"][idx : idx + 1] = -1
            self.share_inputs["top_p"][idx : idx + 1] = task.get("topp", 0.7)
            self.share_inputs["temperature"][idx : idx + 1] = task.get("temperature", 0.95)
            self.share_inputs["penalty_score"][idx : idx + 1] = task.get("penalty_score", 1.0)
            self.share_inputs["frequency_score"][idx : idx + 1] = task.get("frequency_score", 0.0)
            self.share_inputs["presence_score"][idx : idx + 1] = task.get("presence_score", 0.0)
            self.share_inputs["seq_lens_this_time"][idx : idx + 1] = length
            self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = length
            self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
            self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.share_inputs["step_idx"][idx : idx + 1] = 0
            self.share_inputs["min_length"][idx : idx + 1] = task.get("min_dec_len", 1)
            if "max_dec_len" in task:
                max_dec_len = task["max_dec_len"]
            elif "seq_len" in task:
                max_dec_len = task["seq_len"]
            else:
                max_dec_len = self.args.max_dec_len
            self.share_inputs["max_length"][idx : idx + 1] = max_dec_len
            self.share_inputs["stop_flags"][idx : idx + 1] = False

            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = length

            if "infer_seed" in task:
                self.share_inputs["infer_seed"][idx : idx + 1] = task["infer_seed"]

            encoder_block_num = len(task["block_tables"])
            self.share_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
            self.share_inputs["block_tables"][idx : idx + 1, :] = -1
            self.share_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                task["block_tables"], dtype="int32"
            )

            if "stop_seqs_len" in task:
                stop_seqs_num = len(task["stop_seqs_len"])
                for i in range(stop_seqs_num, self.model_cfg.max_stop_seqs_num):
                    task["stop_seqs_len"].append(0)
                self.share_inputs["stop_seqs_len"][:] = np.array(task["stop_seqs_len"], dtype="int32")
                self.share_inputs["stop_seqs"][:stop_seqs_num, : len(task["stop_seqs"][0])] = np.array(
                    task["stop_seqs"], dtype="int64"
                )