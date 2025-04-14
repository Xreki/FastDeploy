import argparse
import copy
import json
import os
import sys
import time

# from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.base.framework import use_pir_api
from paddlenlp_ops import speculate_step_paddle, step_paddle


from fastdeployllm.engine.config import ModelConfig
from fastdeployllm.utils import get_logger
from fastdeployllm.inter_communicator.task_queue_manager import TaskQueueManager

from paddlenlp.experimental.transformers import (
    EagleProposer,
    InferenceWithReferenceProposer,
)


from fastdeployllm.worker.model_runner.model_runner_paddlenlp import ModelRunnerTransformer

File_Path = os.path.realpath(sys.argv[0])
Dir_Path = os.path.dirname(File_Path)
logger = get_logger("infer_server", "infer.log")



class ModelExecutor:
    def __init__(self, args):
        """
            Initializes the InferenceEngine class.
        
        Args:
            args (argparse.Namespace): The parsed arguments from the command line.
                See `transformers.inference_utils.InferenceEngine.add_arguments` for details.
        
        Raises:
            None.
        
        Returns:
            None. (NoneType)
        """
        self.args = args
        self.MAX_INFER_SEED = 9223372036854775806


        self.model_cfg = ModelConfig(args.model_name_or_path)
        self.init_dist_env()

        self.format_print_configuration()

        self.helper_tensors = {}


        self.infer_engine = ModelRunnerTransformer(
            config=self.model_cfg,
            args=self.args,
            nranks=self.nranks,
            rank=self.rank
        )

        self.infer_queue = TaskQueueManager(rank=self.rank, mp_num=self.nranks, port=self.args.infer_port)

        self.init_health()



    def init_dist_env(self, seed=20):
        """
        init distributed env
        """

        self.nranks = dist.get_world_size()
        strategy = fleet.DistributedStrategy()

        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.nranks,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
        fleet.init(is_collective=True, strategy=strategy)
        self.rank = fleet.worker_index()


    def init_health(self):
        """
            初始化健康状态，包括共享内存的创建和初始化。
        共享内存用于同步模型推理过程中的各个进程。
        
        Args:
            None
        
        Returns:
            None
        
        Raises:
            None
        """
        flag_array = np.zeros([1], dtype=np.int32)
        self.shm_flag_broadcast = shared_memory.SharedMemory(
            name="shm_pd_infer_flag_broadcast")
        self.flag_broadcast_array = np.ndarray(flag_array.shape,
                                        dtype=flag_array.dtype,
                                        buffer=self.shm_flag_broadcast.buf)

        self.shm_flag_ready = shared_memory.SharedMemory(name="shm_flag_infer_ready")
        self.flag_ready_array = np.ndarray(flag_array.shape,
                                    dtype=flag_array.dtype,
                                    buffer=self.shm_flag_ready.buf)
        self.flag_ready_array[self.rank] = 1  # 已初始化完毕


        self.shm_flag_has_block_step = shared_memory.SharedMemory(name="shm_flag_has_block_step")
        self.flag_has_block_step_array = np.ndarray(flag_array.shape,
                                            dtype=flag_array.dtype,
                                            buffer=self.shm_flag_has_block_step.buf)



        self.infer_live_flag_shm = shared_memory.SharedMemory(create=True,
                                                        size=1,
                                                        name="shm_flag_infer_{}_live".format(self.rank))

    def format_print_configuration(self):
        """
        print model config
        """
        logger.info("===============   Model Information   ==============")
        for k, v in self.model_cfg.__dict__.items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=============== Service Configuration ===============")
        for k, v in vars(self.args).items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=====================================================\n")


    def step_cuda(self):
        """
        step cuda
        """
        step_paddle(
            self.infer_engine.share_inputs["stop_flags"],
            self.infer_engine.share_inputs["seq_lens_this_time"],
            self.infer_engine.share_inputs["step_seq_lens_encoder"],
            self.infer_engine.share_inputs["seq_lens_encoder"],
            self.infer_engine.share_inputs["seq_lens_decoder"],
            self.infer_engine.share_inputs["block_tables"],
            self.infer_engine.share_inputs["encoder_block_lens"],
            self.infer_engine.share_inputs["is_block_step"],
            self.infer_engine.share_inputs["step_block_list"],
            self.infer_engine.share_inputs["step_lens"],
            self.infer_engine.share_inputs["recover_block_list"],
            self.infer_engine.share_inputs["recover_lens"],
            self.infer_engine.share_inputs["need_block_list"],
            self.infer_engine.share_inputs["need_block_len"],
            self.infer_engine.share_inputs["used_list_len"],
            self.infer_engine.share_inputs["free_list"],
            self.infer_engine.share_inputs["free_list_len"],
            self.infer_engine.share_inputs["input_ids"],
            self.infer_engine.share_inputs["pre_ids"],
            self.infer_engine.share_inputs["step_idx"],
            self.infer_engine.share_inputs["next_tokens"],
            self.infer_engine.share_inputs["first_token_ids"],
            self.args.block_size,
            self.args.enc_dec_block_num,
        )


    def run(self):
        """
        主函数，用于执行模型的推理过程。
            该函数会不断地从队列中获取任务，并进行相应的处理，直到所有任务都被完成。
            在每次获取任务后，会将结果写入输出文件中。
        
            Args:
                None.
        
            Returns:
                None.
        
            Raises:
                None.
        """
        infer_seed_increment = paddle.full(shape=[self.args.max_batch_size, 1], fill_value=4, dtype="int64")
        self.nnode = 1
        while True:
            self.insert_step = False

            # self.engine_healthy_recorded_time_array[0] = time.time()
            mp_num_per_node = self.nranks

            if self.rank % mp_num_per_node == 0:
                if not self.infer_queue.empty():
                    if self.nnode > 1:
                        self.infer_queue.read_finish_flag.set(1)
                    else:
                        self.flag_broadcast_array[0] = 1

            if self.nranks > 1:
                paddle.distributed.barrier()

            if self.flag_broadcast_array[0] == 1 or self.infer_queue.read_finish_flag.get() == 1:
                logger.info(f"rank: {self.rank} start to get")
                self.insert_step = True

                tasks, read_finish = self.infer_queue.get()
                if read_finish:
                    self.flag_broadcast_array[0] = 0
                    self.infer_queue.read_finish_flag.set(0)

                req_dicts = []
                for req_dict, bsz in tasks:
                    real_bsz = int(bsz)
                    req_dicts.extend(req_dict)
                    logger.info(f"rank: {self.rank}, real_bsz: {real_bsz}, query_num: {len(req_dicts)}")

                self.infer_engine.dy_input_preprocess(req_dicts)
                self.infer_engine.share_inputs["not_need_stop"][0] = True

            if not self.infer_engine.share_inputs["not_need_stop"]:
                if self.nranks > 1:
                    paddle.distributed.barrier()

                time.sleep(0.001)
                continue



            self.infer_engine.model.generate(**self.infer_engine.share_inputs)
            self.infer_engine.share_inputs["infer_seed"].add_(infer_seed_increment)
            self.infer_engine.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED

            self.step_cuda()



def parse_args():
    """
    parse args from command line
    """
    parser = argparse.ArgumentParser("FastDeploy LLM Inference")
    parser.add_argument("-m", "--model_name_or_path", type=str, default="./output", help="model dir")
    parser.add_argument("-mp", "--mp_degree", type=int, default=1, help="mp degree")
    parser.add_argument("-mbs", "--max_batch_size", type=int, default=34, help="max batch size")
    parser.add_argument("--max_block_num", type=int, default=2000)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--infer_port", type=int, default=9923)
    parser.add_argument("--max_seq_len", type=int, default=3072, help="max_seq_len")
    parser.add_argument("--max_dec_len", type=int, default=1024, help="max_dec_len")
    parser.add_argument("--use_cache_kv_int8", type=int, default=0, help="use cache kv int8")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="input dtype")
    parser.add_argument("--enc_dec_block_num", type=int, default=1, help="encoder's decoder num")
    parser.add_argument("--block_ratio", type=float, default=0.7, help="block ratio")
    parser.add_argument("--first_token_id", type=int, default=1, help="first token id")
    parser.add_argument("--pad_token_id", type=int, default=-1, help="pad token id")
    parser.add_argument("--eos_tokens_lens", type=int, default=2, help="eos token lens")
    args = parser.parse_args()
    return args


def main():
    """
    start model executor
    """
    args = parse_args()
    model_executor = ModelExecutor(args)
    model_executor.run()


if __name__ == "__main__":
    main()
