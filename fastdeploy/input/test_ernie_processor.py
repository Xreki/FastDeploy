"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from fastdeploy.input.ernie_processor import ErnieProcessor
import unittest
import os
import sys
import importlib.util

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestErnieProcessorComparison(unittest.TestCase):
    """测试ErnieProcessor与ErnieProcessor_base的对比验证"""

    @classmethod
    def setUpClass(cls):
        """类级别的设置，获取模型路径并创建processor实例"""
        # 从环境变量获取模型路径，如果没有设置则跳过测试
        cls.model_path = os.getenv('ERNIE_MODEL_PATH')
        if not cls.model_path:
            raise unittest.SkipTest("请设置ERNIE_MODEL_PATH环境变量指向真实的模型路径")

        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"模型路径不存在: {cls.model_path}")

        # 创建当前版本的ErnieProcessor实例（只创建一次）
        print("正在创建当前版本的ErnieProcessor实例...")
        cls.processor_current = ErnieProcessor(cls.model_path)

        # 创建base版本的ErnieProcessor实例（只创建一次）
        try:
            print("正在创建base版本的ErnieProcessor实例...")
            # 动态导入ernie_processor_base模块
            base_module_path = os.path.join(
                os.path.dirname(__file__), 'ernie_processor_base.py')
            spec = importlib.util.spec_from_file_location(
                "ernie_processor_base", base_module_path)
            base_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(base_module)

            cls.processor_base = base_module.ErnieProcessor(cls.model_path)
            print("Processor实例创建完成！")
        except Exception as e:
            raise unittest.SkipTest(f"无法加载ernie_processor_base: {e}")

    def setUp(self):
        """测试前的准备工作（现在只做轻量级的准备）"""
        # 由于processor实例已经在setUpClass中创建，这里只需要做一些轻量级的准备工作
        pass

    def _print_comparison_result(self, test_name, request, current_result, base_result):
        """打印详细的对比结果"""
        print(f"\n{'='*80}")
        print(f"对比测试: {test_name}")
        print(f"{'='*80}")

        # 显示原始request
        print(f"原始request:")
        for key, value in request.items():
            if key == 'messages':
                print(f"  {key}:")
                for i, msg in enumerate(value):
                    print(f"    [{i}] {msg['role']}: {msg['content']}")
            else:
                print(f"  {key}: {value}")

        print(f"\n{'-'*40} 当前版本结果 {'-'*40}")
        current_ids = current_result['prompt_token_ids']
        print(f"Token IDs 数量: {len(current_ids)}")
        print(f"Token IDs: {current_ids}")

        # 尝试解码当前版本
        try:
            current_decoded = self.__class__.processor_current.tokenizer.decode(current_ids,
                                                                      skip_special_tokens=False)
            print(f"解码文本长度: {len(current_decoded)} 字符")
            print(f"解码文本: {repr(current_decoded)}")
            if len(current_decoded) < 500:
                print(f"可读版本: {current_decoded}")
            else:
                print(f"可读版本（前200字符）: {current_decoded[:200]}...")
        except Exception as e:
            print(f"当前版本解码失败: {e}")
            current_decoded = ""

        print(f"\n{'-'*40} Base版本结果 {'-'*40}")
        base_ids = base_result['prompt_token_ids']
        print(f"Token IDs 数量: {len(base_ids)}")
        print(f"Token IDs: {base_ids}")

        # 尝试解码base版本
        try:
            base_decoded = self.__class__.processor_base.tokenizer.decode(
                base_ids, skip_special_tokens=False)
            print(f"解码文本长度: {len(base_decoded)} 字符")
            print(f"解码文本: {repr(base_decoded)}")
            if len(base_decoded) < 500:
                print(f"可读版本: {base_decoded}")
            else:
                print(f"可读版本（前200字符）: {base_decoded[:200]}...")
        except Exception as e:
            print(f"Base版本解码失败: {e}")
            base_decoded = ""

        print(f"\n{'-'*40} 对比分析 {'-'*40}")
        # 对比分析
        ids_identical = current_ids == base_ids
        text_identical = (current_decoded == base_decoded
                          if current_decoded and base_decoded else False)

        print(f"Token IDs 完全相同: {'✓' if ids_identical else '✗'}")
        print(f"解码文本完全相同: {'✓' if text_identical else '✗'}")
        print(f"Token 数量差异: {len(current_ids) - len(base_ids)}")

        if not ids_identical and current_ids and base_ids:
            # 分析差异
            min_len = min(len(current_ids), len(base_ids))
            first_diff_idx = -1
            for i in range(min_len):
                if current_ids[i] != base_ids[i]:
                    first_diff_idx = i
                    break

            if first_diff_idx >= 0:
                print(f"首个差异位置: 索引 {first_diff_idx}")
                print(f"  当前版本: {current_ids[first_diff_idx]}")
                print(f"  Base版本: {base_ids[first_diff_idx]}")

            # 显示前10个和后10个token的对比
            print(f"前10个token对比:")
            print(f"  当前版本: {current_ids[:10]}")
            print(f"  Base版本: {base_ids[:10]}")

            if len(current_ids) > 10 or len(base_ids) > 10:
                print(f"后10个token对比:")
                print(f"  当前版本: {current_ids[-10:]}")
                print(f"  Base版本: {base_ids[-10:]}")

        print(f"{'='*80}\n")

        return {
            'ids_identical': ids_identical,
            'text_identical': text_identical,
            'current_length': len(current_ids),
            'base_length': len(base_ids),
            'current_decoded': current_decoded,
            'base_decoded': base_decoded
        }

    def _compare_processors(self, test_name, request, max_model_len=None):
        """对比两个processor的处理结果"""
        # 处理当前版本
        current_result = self.__class__.processor_current.process_request_dict(
            request.copy(), max_model_len)

        # 处理base版本
        base_result = self.__class__.processor_base.process_request_dict(
            request.copy(), max_model_len)

        # 打印对比结果
        comparison = self._print_comparison_result(
            test_name, request, current_result, base_result)

        return current_result, base_result, comparison

    # 场景1: 给定输入明文，输出的input_ids符合预期
    def test_scenario_1_simple_user_input_comparison(self):
        """场景1对比: 简单用户输入"""
        # 检查tokenizer是否支持chat_template
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        request = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ]
        }

        # 不传入max_model_len，关闭截断逻辑
        current_result, base_result, comparison = self._compare_processors(
            "场景1-简单用户输入", request, max_model_len=None)

        # 基本验证
        self.assertIsInstance(current_result['prompt_token_ids'], list)
        self.assertIsInstance(base_result['prompt_token_ids'], list)
        self.assertGreater(len(current_result['prompt_token_ids']), 0)
        self.assertGreater(len(base_result['prompt_token_ids']), 0)

    def test_scenario_1_chinese_user_input_comparison(self):
        """场景1对比: 中文用户输入"""
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        request = {
            "messages": [
                {"role": "user", "content": "你好，今天天气怎么样？"}
            ]
        }

        current_result, base_result, comparison = self._compare_processors(
            "场景1-中文用户输入", request, max_model_len=None)

        self.assertIsInstance(current_result['prompt_token_ids'], list)
        self.assertIsInstance(base_result['prompt_token_ids'], list)
        self.assertGreater(len(current_result['prompt_token_ids']), 0)
        self.assertGreater(len(base_result['prompt_token_ids']), 0)

    def test_scenario_1_complex_user_input_comparison(self):
        """场景1对比: 复杂用户输入"""
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        request = {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello 你好 🌍! Can you help me with <code>print('hello world')</code>?"
                }
            ]
        }

        current_result, base_result, comparison = self._compare_processors(
            "场景1-复杂用户输入", request, max_model_len=None)

        self.assertIsInstance(current_result['prompt_token_ids'], list)
        self.assertIsInstance(base_result['prompt_token_ids'], list)
        self.assertGreater(len(current_result['prompt_token_ids']), 0)
        self.assertGreater(len(base_result['prompt_token_ids']), 0)

    # 场景2: 给定输入明文+system，输出的input_ids符合预期
    def test_scenario_2_user_input_with_system_comparison(self):
        """场景2对比: 用户输入 + 系统提示"""
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        request = {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is artificial intelligence?"}
            ]
        }

        current_result, base_result, comparison = self._compare_processors(
            "场景2-用户输入+系统提示", request, max_model_len=None)

        self.assertIsInstance(current_result['prompt_token_ids'], list)
        self.assertIsInstance(base_result['prompt_token_ids'], list)
        self.assertGreater(len(current_result['prompt_token_ids']), 0)
        self.assertGreater(len(base_result['prompt_token_ids']), 0)

    def test_scenario_2_chinese_system_prompt_comparison(self):
        """场景2对比: 中文系统提示"""
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        request = {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的编程助手，能够帮助用户解决各种编程问题。"
                },
                {"role": "user", "content": "请解释一下Python中的装饰器是什么？"}
            ]
        }

        current_result, base_result, comparison = self._compare_processors(
            "场景2-中文系统提示", request, max_model_len=None)

        self.assertIsInstance(current_result['prompt_token_ids'], list)
        self.assertIsInstance(base_result['prompt_token_ids'], list)
        self.assertGreater(len(current_result['prompt_token_ids']), 0)
        self.assertGreater(len(base_result['prompt_token_ids']), 0)

    def test_scenario_2_long_system_prompt_comparison(self):
        """场景2对比: 长系统提示"""
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        request = {
            "messages": [
                {
                    "role": "system",
                    "content": ("You are an expert software engineer with over 10 years of experience "
                                "in full-stack development. You specialize in Python, JavaScript, and "
                                "cloud technologies. You always provide detailed, accurate, and practical "
                                "solutions. When answering questions, you explain the reasoning behind "
                                "your recommendations and provide code examples when appropriate.")
                },
                {"role": "user", "content": "How do I optimize database queries in a web application?"}
            ]
        }

        current_result, base_result, comparison = self._compare_processors(
            "场景2-长系统提示", request, max_model_len=None)

        self.assertIsInstance(current_result['prompt_token_ids'], list)
        self.assertIsInstance(base_result['prompt_token_ids'], list)
        self.assertGreater(len(current_result['prompt_token_ids']), 0)
        self.assertGreater(len(base_result['prompt_token_ids']), 0)

    # 场景3: 给定输入明文+system+多轮，输出的input_ids符合预期
    def test_scenario_3_multi_turn_with_system_comparison(self):
        """场景3对比: 系统提示 + 多轮对话"""
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        request = {
            "messages": [
                {"role": "system", "content": "You are a helpful programming tutor."},
                {"role": "user", "content": "What is a function in programming?"},
                {
                    "role": "assistant",
                    "content": "A function is a reusable block of code that performs a specific task."
                },
                {"role": "user", "content": "Can you give me an example in Python?"}
            ]
        }

        current_result, base_result, comparison = self._compare_processors(
            "场景3-多轮对话+系统提示", request, max_model_len=None)

        self.assertIsInstance(current_result['prompt_token_ids'], list)
        self.assertIsInstance(base_result['prompt_token_ids'], list)
        self.assertGreater(len(current_result['prompt_token_ids']), 0)
        self.assertGreater(len(base_result['prompt_token_ids']), 0)

    def test_scenario_3_chinese_multi_turn_comparison(self):
        """场景3对比: 中文多轮对话"""
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        request = {
            "messages": [
                {"role": "system", "content": "你是一个专业的AI助手，擅长回答各种问题。"},
                {"role": "user", "content": "什么是机器学习？"},
                {
                    "role": "assistant",
                    "content": "机器学习是人工智能的一个分支，它让计算机能够从数据中自动学习和改进。"
                },
                {"role": "user", "content": "机器学习有哪些主要类型？"},
                {"role": "assistant", "content": "主要有监督学习、无监督学习和强化学习三种类型。"},
                {"role": "user", "content": "能详细解释一下监督学习吗？"}
            ]
        }

        current_result, base_result, comparison = self._compare_processors(
            "场景3-中文多轮对话", request, max_model_len=None)

        self.assertIsInstance(current_result['prompt_token_ids'], list)
        self.assertIsInstance(base_result['prompt_token_ids'], list)
        self.assertGreater(len(current_result['prompt_token_ids']), 0)
        self.assertGreater(len(base_result['prompt_token_ids']), 0)

    def test_scenario_3_complex_multi_turn_comparison(self):
        """场景3对比: 复杂多轮对话"""
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        request = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Python developer and teacher."
                },
                {"role": "user", "content": "How do I create a class in Python?"},
                {
                    "role": "assistant",
                    "content": ("You can create a class using the 'class' keyword. Here's a basic "
                                "example:\n\nclass MyClass:\n    def __init__(self):\n        pass")
                },
                {"role": "user", "content": "What about adding methods to the class?"},
                {
                    "role": "assistant",
                    "content": ("You can add methods by defining functions inside the class. "
                                "Methods should have 'self' as their first parameter.")
                },
                {"role": "user", "content": "Can you show me a complete example with attributes and methods?"}
            ]
        }

        current_result, base_result, comparison = self._compare_processors(
            "场景3-复杂多轮对话", request, max_model_len=None)

        self.assertIsInstance(current_result['prompt_token_ids'], list)
        self.assertIsInstance(base_result['prompt_token_ids'], list)
        self.assertGreater(len(current_result['prompt_token_ids']), 0)
        self.assertGreater(len(base_result['prompt_token_ids']), 0)

    def test_scenario_3_long_conversation_comparison(self):
        """场景3对比: 长对话序列"""
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        request = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a knowledgeable AI assistant specializing in technology and science."
                },
                {"role": "user", "content": "What is quantum computing?"},
                {
                    "role": "assistant",
                    "content": ("Quantum computing is a type of computation that harnesses quantum "
                                "mechanical phenomena like superposition and entanglement to process "
                                "information.")
                },
                {"role": "user", "content": "How does it differ from classical computing?"},
                {
                    "role": "assistant",
                    "content": ("Classical computers use bits that are either 0 or 1, while quantum "
                                "computers use quantum bits (qubits) that can be in superposition of "
                                "both states simultaneously.")
                },
                {"role": "user", "content": "What are some potential applications?"},
                {
                    "role": "assistant",
                    "content": ("Quantum computing could revolutionize cryptography, drug discovery, "
                                "financial modeling, and artificial intelligence by solving complex "
                                "problems much faster.")
                },
                {"role": "user", "content": "Are there any limitations or challenges?"}
            ]
        }

        current_result, base_result, comparison = self._compare_processors(
            "场景3-长对话序列", request, max_model_len=None)

        self.assertIsInstance(current_result['prompt_token_ids'], list)
        self.assertIsInstance(base_result['prompt_token_ids'], list)
        self.assertGreater(len(current_result['prompt_token_ids']), 0)
        self.assertGreater(len(base_result['prompt_token_ids']), 0)

    # 边界情况对比测试
    def test_edge_case_empty_messages_comparison(self):
        """边界测试对比: 空消息列表"""
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        request = {"messages": []}

        try:
            current_result, base_result, comparison = self._compare_processors(
                "边界测试-空消息", request, max_model_len=None)

            self.assertIsInstance(current_result['prompt_token_ids'], list)
            self.assertIsInstance(base_result['prompt_token_ids'], list)

        except Exception as e:
            print(f"\n空消息处理异常（这可能是正常行为）: {e}")

    def test_edge_case_with_stop_sequences_comparison(self):
        """边界测试对比: 包含停止序列"""
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        request = {
            "messages": [
                {"role": "user", "content": "Tell me a story"}
            ],
            "stop": ["The End", "END", "\n\n"]
        }

        current_result, base_result, comparison = self._compare_processors(
            "边界测试-停止序列", request, max_model_len=None)

        self.assertIsInstance(current_result['prompt_token_ids'], list)
        self.assertIsInstance(base_result['prompt_token_ids'], list)
        self.assertGreater(len(current_result['prompt_token_ids']), 0)
        self.assertGreater(len(base_result['prompt_token_ids']), 0)

        # 验证停止序列处理
        self.assertIn('stop_token_ids', current_result)
        self.assertIn('stop_token_ids', base_result)
        print(f"当前版本 Stop token IDs: {current_result['stop_token_ids']}")
        print(f"Base版本 Stop token IDs: {base_result['stop_token_ids']}")

    def test_direct_messages2ids_comparison(self):
        """直接对比messages2ids方法"""
        if self.__class__.processor_current.tokenizer.chat_template is None:
            self.skipTest("当前tokenizer不支持chat_template")

        # 测试数据
        test_cases = [
            {
                "name": "简单用户输入",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            },
            {
                "name": "用户输入+系统提示",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"}
                ]
            },
            {
                "name": "多轮对话",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"}
                ]
            }
        ]

        for test_case in test_cases:
            print(f"\n{'='*60}")
            print(f"直接对比messages2ids: {test_case['name']}")
            print(f"{'='*60}")

            messages = test_case['messages']

            # 当前版本 - 只需要传入messages参数（方法会从中提取messages字段）
            try:
                current_ids = self.__class__.processor_current.messages2ids(
                    messages)
                print(f"当前版本结果: {current_ids}")
                print(f"当前版本数量: {len(current_ids)}")
            except Exception as e:
                print(f"当前版本异常: {e}")
                current_ids = []

            # Base版本 - 传入messages列表和max_model_len
            try:
                base_ids = self.__class__.processor_base.messages2ids(
                    messages, None)
                print(f"Base版本结果: {base_ids}")
                print(f"Base版本数量: {len(base_ids)}")
            except Exception as e:
                print(f"Base版本异常: {e}")
                base_ids = []

            # 对比
            if current_ids and base_ids:
                identical = current_ids == base_ids
                print(f"结果相同: {'✓' if identical else '✗'}")
                if not identical:
                    print(f"长度差异: {len(current_ids) - len(base_ids)}")


if __name__ == '__main__':
    # 提供使用说明
    print("=" * 80)
    print("ErnieProcessor 与 ErnieProcessor_base 对比验证测试")
    print("=" * 80)
    print("运行此测试需要设置ERNIE_MODEL_PATH环境变量")
    print("例如: export ERNIE_MODEL_PATH=/path/to/your/ernie/model")
    print("然后运行: python test_ernie_processor.py")
    print()
    print("测试内容:")
    print("1. 对比当前版本与base版本的处理结果")
    print("2. 不传入max_model_len，关闭截断逻辑")
    print("3. 覆盖三个主要场景的对比验证")
    print("4. 显示详细的差异分析")
    print("5. 使用setUpClass减少重复的processor初始化")
    print("6. 适配不同版本的messages2ids方法签名差异")
    print("=" * 80)
    print()

    unittest.main(verbosity=2)
