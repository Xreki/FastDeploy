import paddle

from fastdeploy.config import GraphOptimizationConfig, LLMConfig
from fastdeploy.model_executor.graph_optimization.decorator import \
    support_graph_opt


@support_graph_opt
class TestModel(paddle.nn.Layer):
    """ Tast Model """

    def __init__(self, llm_config: LLMConfig, **kwargs):
        self.llm_config = llm_config

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        """前向传播"""
        input_ids: paddle.Tensor = kwargs["input_ids"]
        return input_ids + input_ids


if __name__ == '__main__':
    graph_opt_config = GraphOptimizationConfig()
    graph_opt_config.use_cudagraph = True
    graph_opt_config.cudagraph_capture_sizes = [1, 4]
    llm_config = LLMConfig(graph_opt_config=graph_opt_config)
    model = TestModel(llm_config=llm_config)

    output = model(input_ids=paddle.zeros([1, 8]))
    print(output)
    output = model(input_ids=paddle.ones([1, 8]))
    print(output)
    output = model(input_ids=paddle.zeros([4, 9]))
    print(output)
    output = model(input_ids=paddle.ones([4, 9]))
    print(output)
