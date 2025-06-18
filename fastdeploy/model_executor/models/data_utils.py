#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""Useful data utility."""

import json
import re
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import numpy as np
from paddleformers.utils.log import logger

INF = 1000000
OPT_MULTI_OF = 256


@dataclass
class Example:
    """Data format for raw SFT example."""

    src: List[str]
    tgt: List[str]
    label: List[int]
    is_memory: int
    is_system: int
    is_q2code: int
    math_is_end: int
    ctxt_src: List[str]
    ctxt_tgt: List[str]
    disable_pseudo_multi_turn: int
    source: str
    prefix: str
    suffix: str
    output: str
    data_format: str


# 不能去掉 K 单独训练tgt 的 markup 类型
NO_OPT_MARKUPS = [
    "[<citation>]",
    "[<citation-ref>]",
    "[<kg>]",
    "[<kg-res>]",
    "[<retrieve>]",
    "[<retrieve-ref>]",
]
KG_RES_MARKUPS = [
    "[<kg-res>]",
    "[</kg-res>]",
    "[<kg-yes>]",
    "[</kg-yes>]",
    "[<kg-cs-yes>]",
    "[</kg-cs-yes>]",
    "[<kg-cs-no>]",
    "[</kg-cs-no>]",
]


def get_markup_tokens():
    """Get all special markup tokens for example with K."""
    markups = ["kg", "prompt", "search"]
    markup_tokens = []
    for markup_token in markups:
        markup_tokens.extend(
            [
                f"[<{markup_token}>]",
                f"[</{markup_token}>]",
                f"[<{markup_token}-res>]",
                f"[</{markup_token}-res>]",
            ]
        )
    markup_tokens.extend(
        [
            "[<citation>]",
            "[</citation>]",
            "[<citation-ref>]",
            "[</citation-ref>]",
            "[<retrieve>]",
            "[<retrieve-ref>]",
        ]
    )
    return markup_tokens


def contains_markup(text, special_markups):
    """Checks for the presence of markup tokens in the text."""
    for sp_token in special_markups:
        for x in text:
            if sp_token in x:
                return True
    return False


def contains_same_tgt(tgt, pseudo_example_list):
    """Checks for the presence of tgt in pseudo examples"""
    tgt_text_set = set()
    for pseudo_example in pseudo_example_list:
        for x in pseudo_example.tgt:
            tgt_text_set.add(x.strip())
    for x in tgt:
        if x.strip() in tgt_text_set:
            return True
    return False


def convert_pseudo_example_list_to_example(
    previous_pseudo_example_list,
    current_pseudo_example_list,
    tokenizer,
    stop_by_k=False,
):
    """Convert multiple pseudo examples into one example."""
    multi_turn_src, multi_turn_tgt, multi_turn_label = [], [], []
    for i, example in enumerate(previous_pseudo_example_list):
        # 历史多轮处理
        # 如果math_is_end=1, 作为上文的时候, 要使用ctxt_tgt, ctxt_src代替tgt, src
        if example.math_is_end == 1:
            assert len(example.src) == len(example.ctxt_src)
            assert len(example.tgt) == len(example.ctxt_tgt)
            example = replace(example, src=example.ctxt_src, tgt=example.ctxt_tgt)

        if contains_markup(example.src, tokenizer.markup_tokens):
            # 包含K，则去掉K
            src = example.src[:-2] + [example.src[-2]]
            tgt = example.tgt[:-2] + [example.tgt[-1]]

            label = [
                x and int(not stop_by_k) for x in example.label[:-2]
            ]  # 因为k停止，则不优化
            label.append(
                int(not stop_by_k and not contains_markup(example.src, NO_OPT_MARKUPS))
            )  # 非k停止，并且不包含不能舍弃k单独优化的样本，才进行优化
        else:
            src = example.src
            tgt = example.tgt

            label = [
                x and int(not stop_by_k) for x in example.label
            ]  # 因为k停止，则不优化

        # 涉黄问题前序轮去掉prompt
        new_src = []
        for x in src:
            x = x.replace("\n这是一个涉习问题", "")
            x = x.replace("\n这是一个涉政问题", "")
            x = x.replace("\n这是一个涉黄问题", "")
            x = x.replace("\n这是一个违法犯罪问题", "")
            new_src.append(x)

        multi_turn_src.extend(new_src)
        multi_turn_tgt.extend(tgt)
        multi_turn_label.extend(label)

    for i, example in enumerate(current_pseudo_example_list):
        # 当前轮处理，只有最后一轮可能有K
        src = example.src
        tgt = example.tgt
        if i != len(current_pseudo_example_list) - 1:
            # 非最后一个
            # 如果math_is_end=1, 作为上文的时候, 要使用ctxt_tgt, ctxt_src代替tgt, src
            if example.math_is_end == 1:
                assert len(example.src) == len(example.ctxt_src)
                assert len(example.tgt) == len(example.ctxt_tgt)
                example = replace(example, src=example.ctxt_src, tgt=example.ctxt_tgt)
            label = [
                x and int(not stop_by_k) for x in example.label
            ]  # 因为k停止，则不优化

            # 涉黄问题前序轮去掉prompt
            new_src = []
            for x in src:
                x = x.replace("\n这是一个涉习问题", "")
                x = x.replace("\n这是一个涉政问题", "")
                x = x.replace("\n这是一个涉黄问题", "")
                x = x.replace("\n这是一个违法犯罪问题", "")
                new_src.append(x)
            src = new_src
        else:
            # 最后一个
            if contains_markup(src, tokenizer.markup_tokens):
                # 包含K，只优化KB，前序的不优化
                label = [0] * len(src[:-2])
                label.extend([1] * (len(src) - len(src[:-2])))
            else:
                label = example.label

        multi_turn_src.extend(src)
        multi_turn_tgt.extend(tgt)
        multi_turn_label.extend(label)
    new_example = Example(
        src=multi_turn_src,
        tgt=multi_turn_tgt,
        label=multi_turn_label,
        is_memory=0,
        is_system=0,
        is_q2code=0,
        math_is_end=2,
        ctxt_src=[],
        ctxt_tgt=[],
        disable_pseudo_multi_turn=0,
        source=example.source,
        prefix=None,
        suffix=None,
        output=None,
        data_format="sft",
    )
    return new_example


def get_length(example, tokenizer, eb_markup_rounter, add_number=2):
    """
    计算样本总长度
        add_number: <sep> <cls> 预留长度

    返回：
        cur_len_w_k 包含k的总长度
        cur_len_wo_k 不包含k的总长度
    """
    cur_len_w_k = 0
    cur_len_wo_k = 0
    if contains_markup(example.src, tokenizer.markup_tokens) and contains_markup(
        example.tgt, tokenizer.markup_tokens
    ):
        try:
            tokens_src, tokens_target, _, _ = eb_markup_rounter.encode(
                example.src[-2:], example.tgt[-2:], INF
            )
        except Exception:
            raise RuntimeError(f"Parse example fail: {example}")
        cur_len_w_k += len(tokens_src) + len(tokens_target)

        for src, tgt in zip(example.src[:-2], example.tgt[:-2]):
            src_len = len(tokenizer.tokenize(src))
            tgt_len = len(tokenizer.tokenize(tgt))
            cur_len_w_k += src_len + tgt_len + add_number
            cur_len_wo_k += src_len + tgt_len + add_number

        cur_len_wo_k += (
            len(tokenizer.tokenize(example.src[-2]))
            + len(tokenizer.tokenize(example.tgt[-1]))
            + add_number
        )
    else:
        for src, tgt in zip(example.src, example.tgt):
            cur_len_wo_k += (
                len(tokenizer.tokenize(src)) + len(tokenizer.tokenize(tgt)) + add_number
            )
        cur_len_w_k = cur_len_wo_k

    return cur_len_w_k, cur_len_wo_k


def extract_knowledge(text):
    """Extract knowledge string from text."""
    if any(markup in text for markup in KG_RES_MARKUPS):
        for markup in KG_RES_MARKUPS + ["[<image>]", "[</image>]"]:
            text = text.replace(markup, "")
        text = f"知识库：{text.strip()}\n根据所提供的知识库信息，回答问题并补全对话："
        return text

    res = re.findall(
        r"\[<search-res>\](.*?)\[<\/search-res>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = f"{text.strip()}\n根据以上参考文章回答问题，补全对话"
        return text

    res = re.findall(
        r"\[<prompt-res>\](.*?)\[<\/prompt-res>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = text.strip()
        return text

    res = re.findall(
        r"\[<compute-res>\](.*?)\[<\/compute-res>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = f"参考文章1：{text.strip()}\n根据以上参考文章回答问题，补全对话"
        return text

    res = re.findall(
        r"\[<citation-ref>\](.*?)\[<\/citation-ref>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = (
            "请参考搜索结果回答下面问题并使用引用标记来标注回答内容参考的搜索结果序号，"
            "例如^[1]^ (引用单个搜索结果）,^[1][2]^（引用多个搜索结果），"
            "其中方括号中的数字是搜索结果序号。引用标记只能出现在句尾标点符号前。\n"
            "以下是搜索结果（每行开头[1]、[2]、...是搜索结果序号），"
            f"可以对答案中的核心部分进行markdown加粗（**加粗内容**）：\n{text.strip()}\n"
            "根据以上搜索结果回答问题并标注引用，补全对话"
        )
        return text

    res = re.findall(
        r"\[<retrieve-ref>\](.*?)\[<\/retrieve-ref>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = (
            "请你扮演一个专家，参考搜索结果中正确、可信、高质量的信息回答问题，并注明答案中引用的搜索结果，"
            "格式为^[2]^表示引用了第2条搜索结果，^[1][3]^表示引用第1和第3条搜索结果。"
            "每条搜索结果包含若干相关内容片段。同时你需要遵循以下原则回答问题：\n"
            "1. 严格遵循搜索结果作答，可以承认不知道答案，并尝试给出一些搜索结果中的相关背景信息。\n"
            "2. 如果搜索结果存在多种可能的答案，要罗列出每种情况。\n"
            "3. 如果问题涉及金融、医疗、法律等存在风险的领域，请在结尾提醒用户注意并进行免责说明。\n"
            f"搜索结果：\n{text.strip()}\n\n现在，请根据上面的搜索结果回答问题并标注引用，补全对话"
        )
        return text

    raise ValueError(f"Cannot extract knowledge from `{text}`")


def sampling_pseudo_examples(
    examples_all,
    examples_per_task,
    tokenizer,
    eb_markup_rounter,
    rng,
    max_seq_len,
    example_from_same_task_prob,
    pseudo_sampling_prob,
    trigger_data_prob,
):
    """Sample pseudo examples from a dataset"""
    # 构造伪多轮
    previous_pseudo_example_list = []
    current_pseudo_example_list = []
    total_len_wo_k = 2  # preserved tokens: <s> <cls>

    total_task_num = len(examples_per_task)
    FORCE_EXAMPLE_FROM_SAME_TASK = False
    task_index = 0
    while len(examples_all) > 0:
        # The ec3 dataformat just yield the example
        if examples_all[-1].data_format == "ec3_completion":
            example = examples_all.pop()
            yield example, 1
            continue
        if rng.random() >= pseudo_sampling_prob:
            # 不走伪多轮
            example = examples_all.pop()
            if (
                contains_markup(example.src, tokenizer.markup_tokens)
                and rng.random() >= trigger_data_prob
            ):
                # 如果是触发数据，并且采样为非触发，则跳过删除knowledge，当做普通QA训练
                if contains_markup(example.src, NO_OPT_MARKUPS):
                    # 部分数据不能去除knowledge，则跳过该样本
                    continue
                try:
                    example = Example(
                        src=example.src[:-1],
                        tgt=example.tgt[:-2] + example.tgt[-1:],
                        label=example.label[:-2] + example.label[-1:],
                        is_memory=example.is_memory,
                        source=example.source,
                    )
                except Exception:
                    # Invalid example with k
                    continue
            yield example, 1
            continue

        total_task_num = len(examples_per_task)
        # 如果examples_all没有值，则退出
        if (
            not FORCE_EXAMPLE_FROM_SAME_TASK
            and rng.random() < example_from_same_task_prob
            and total_task_num > 0
        ):
            # example_from_same_task_prob的概率开启连续从相同类型样本中的采样策略
            FORCE_EXAMPLE_FROM_SAME_TASK = True
            # 确定max_seq_len长度的伪多轮均来自于task_index
            task_index = rng.randint(0, total_task_num)

        if FORCE_EXAMPLE_FROM_SAME_TASK:
            # 从同类样本从采样
            example = examples_per_task[task_index].pop()
            if len(examples_per_task[task_index]) == 0:
                examples_per_task = (
                    examples_per_task[:task_index] + examples_per_task[task_index + 1 :]
                )  # 删除空列表
                FORCE_EXAMPLE_FROM_SAME_TASK = (
                    False  # 该任务已经采样完，则提前终止从相同类型样本中的采样策略
                )
        else:
            # 从正常的合并后的样本中采样
            example = examples_all.pop()

        # The ec3 dataformat just yield the example
        if example.data_format == "ec3_completion":
            yield example, 1
            continue
        # 强制不能组件伪多轮数据
        if example.disable_pseudo_multi_turn == 1:
            if (
                contains_markup(example.src, tokenizer.markup_tokens)
                and rng.random() >= trigger_data_prob
            ):
                # 如果是触发数据，并且采样为非触发，则跳过删除knowledge，当做普通QA训练
                if contains_markup(example.src, NO_OPT_MARKUPS):
                    # 部分数据不能去除knowledge，则跳过该样本
                    continue
                try:
                    example = Example(
                        src=example.src[:-1],
                        tgt=example.tgt[:-2] + example.tgt[-1:],
                        label=example.label[:-2] + example.label[-1:],
                        is_memory=example.is_memory,
                        source=example.source,
                    )
                except Exception:
                    # Invalid example with k
                    continue
            yield example, 1
            continue

        # 如果当前example存在相同的tgt在之前轮，则不加入
        if contains_same_tgt(
            example.tgt,
            previous_pseudo_example_list + current_pseudo_example_list,
        ):
            continue

        if (
            contains_markup(example.src, tokenizer.markup_tokens)
            and rng.random() >= trigger_data_prob
        ):
            # 如果是触发数据，并且采样为非触发，则跳过删除knowledge，当做普通QA训练
            if contains_markup(example.src, NO_OPT_MARKUPS):
                # 部分数据不能去除knowledge，则跳过该样本
                continue
            try:
                example = Example(
                    src=example.src[:-1],
                    tgt=example.tgt[:-2] + example.tgt[-1:],
                    label=example.label[:-2] + example.label[-1:],
                    is_memory=example.is_memory,
                    source=example.source,
                )
            except Exception:
                # Invalid example with k
                continue

        len_w_k, len_wo_k = get_length(example, tokenizer, eb_markup_rounter)
        if (
            total_len_wo_k + len_w_k > max_seq_len
            or example.is_memory
            or example.is_system
            or example.is_q2code
            or example.math_is_end == 0
        ):
            # 终止条件1:
            # 1. 超过最大长度限制，需清空历史伪多轮
            # 3. 遇到包含memory / system的样本，需清空历史伪多轮，因为memory / ststem必须为开头第一个样本
            # 4. 遇到包含只触发的数据，需清空历史伪多轮，如q2code

            # FIXME(hehuang)：有可能因为加入了 q2code 的数据，导致超过长度限制！
            if example.is_q2code or example.math_is_end == 0:
                current_pseudo_example_list.append(example)

            if len(current_pseudo_example_list) > 0:
                # 当前有新增需要优化的，则输出
                yield convert_pseudo_example_list_to_example(
                    previous_pseudo_example_list,
                    current_pseudo_example_list,
                    tokenizer,
                    stop_by_k=False,
                ), len(previous_pseudo_example_list) + len(current_pseudo_example_list)
                FORCE_EXAMPLE_FROM_SAME_TASK = False  # 重置从相同来源采样逻辑
            # 清空结果
            previous_pseudo_example_list, current_pseudo_example_list = [], []
            total_len_wo_k = 2  # preserved tokens: <s> <cls>

        if not (example.is_q2code or example.math_is_end == 0):
            # 非单独触发数据，才加入当前集合
            current_pseudo_example_list.append(example)

        if contains_markup(example.src, tokenizer.markup_tokens):
            # 终止条件2:
            # 2. 包含Markup

            yield convert_pseudo_example_list_to_example(
                previous_pseudo_example_list,
                current_pseudo_example_list,
                tokenizer,
                stop_by_k=True,
            ), len(previous_pseudo_example_list) + len(current_pseudo_example_list)

            # 新增加入历史伪多轮中
            previous_pseudo_example_list.extend(current_pseudo_example_list)
            current_pseudo_example_list = []
            FORCE_EXAMPLE_FROM_SAME_TASK = (
                False  # 重置从相同来源采样逻辑，相当于对带k样本去掉同源采样策略
            )

        if not (example.is_q2code or example.math_is_end == 0):
            total_len_wo_k += len_wo_k  # 更新当前样本长度

    # FIXME(hehuang): 目前未处理最后一个样本
    pass


def get_optimized_max_len(original_max_len):
    """Get the optimized max sequence length for computing device."""
    original_max_len = max(original_max_len, 1)
    optimized_max_len = (
        (original_max_len + OPT_MULTI_OF - 1) // OPT_MULTI_OF * OPT_MULTI_OF
    )
    return optimized_max_len


def pad_batch_data(
    insts,
    pad_idx=0,
    return_pos=False,
    max_seq_len=None,
    return_input_mask=False,
    return_max_len=False,
    return_num_token=False,
    return_seq_lens=False,
):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = (
        max_seq_len if max_seq_len is not None else max(len(inst) for inst in insts)
    )
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts]
    )
    return_list += [inst_data.astype("int64").reshape([-1, max_len])]

    # position data
    if return_pos:
        inst_pos = np.array(
            [
                list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
                for inst in insts
            ]
        )

        return_list += [inst_pos.astype("int64").reshape([-1, max_len])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array(
            [[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts]
        )
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


def get_function_call_instruction(tools, system=None):
    """build the function call instruction from a given set of tools"""
    if tools is not None:
        tools_prompt = "你是一个乐于助人的助手, 非常善于使用工具来解决用户的问题。"
        if system:
            tools_prompt += f"\n{system}"
        tools_str = json.dumps([func for func in tools], indent=2, ensure_ascii=False)
        tool_w_func = {
            "tool_calls": [
                {
                    "name": "function_name",
                    "arguments": {"arguments_name": "arguments_value"},
                }
            ]
        }
        tool_wo_func = {
            "tool_calls": [{"name": "GetFinalAnswer", "content": "直接回复用户的内容"}]
        }
        tool_w_func_str = json.dumps(tool_w_func, indent=2, ensure_ascii=False)
        tool_wo_func_str = json.dumps(tool_wo_func, indent=2, ensure_ascii=False)
        system = f"""{tools_prompt}
以下是你可以选择使用的所有工具，以 JSON 形式呈现，你可以按需使用。
{tools_str}
如果你需要调用工具，请输出以下 JSON：
{tool_w_func_str}
如果你不需要调用工具，请输出以下 JSON：
{tool_wo_func_str}"""
        return system
    else:
        return None


def check_tools_for_model(tools_for_model):
    """check the format of tools_for_model"""
    from jsonschema import ValidationError
    from openapi_schema_validator import validate

    schema_inside = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name", "description"],
        },
    }
    # 校验数据
    try:
        validate(tools_for_model, schema_inside)
    except ValidationError:
        # print(f"{tools_for_model} -> 数据格式错误: {e}")
        return False
    finally:
        del schema_inside

    return True


def build_fc_instruction(functions, system=None):
    """convert, build and check the function call instruction from a given set of tools"""
    # 转换
    tools_for_model = []
    for item in functions:
        function = item["function"]
        tools_for_model.append(function)
    instruction = None
    if check_tools_for_model(tools_for_model):
        instruction = get_function_call_instruction(tools_for_model, system)
    return instruction


def get_infer_data_type(input_data):
    """check the format of function call infer data"""
    if isinstance(input_data, dict) and all(
        isinstance(value, list) and all(isinstance(i, dict) for i in value)
        for value in input_data.values()
    ):
        return "fc_data"
    else:
        return "qa_data"


def convert_fc_infer_data(input_data):
    """convert the function call infer data"""
    output_data = []
    output_data.append(input_data["tools"])

    for message in input_data["messages"]:
        new_message = {}
        new_message["role"] = message["role"]
        new_message["utterance"] = message["content"]
        output_data.append(new_message)

    return output_data


def insert_fc_instruction(dial, fc_instruction_input):
    """build and insert the function call instruction into the dial"""
    dial = [
        item for item in dial if not isinstance(item, list) and item["role"] != "system"
    ]
    fc_instruction = build_fc_instruction(
        fc_instruction_input["tools"], fc_instruction_input["system"]
    )
    if not isinstance(fc_instruction, str):
        raise ValueError("Function call instruction build fail.")
    dial.insert(0, {"role": "system", "utterance": fc_instruction})
    return dial


def convert_to_tokens_for_pt(
    dial: List[dict],
    tokenizer,
    append_bos_token,
    max_src_len,
):
    """Convert a dial to tokens for PT model."""
    # [utterance_1, "\n",  utterance_2, "\n", utterance_3]
    sentence = "\n".join([x["utterance"] for x in dial])
    tokens = tokenizer.tokenize(sentence)
    if append_bos_token:
        tokens = [tokenizer.bos_token] + tokens

    if len(tokens) > max_src_len:
        logger.warning(
            f"The length of text ({len(tokens)}) cannot "
            f"be greater than max input length \
            ({max_src_len}). \
            We will truncate it."
        )
        # NOTE: LLM lost in middle
        tokens = tokens[: max_src_len // 2] + tokens[-max_src_len // 2 :]

    return tokens


def preprocess_data(dial, generation_mode=True, return_query_list=False):
    """
    Convert data format
    Return
    dial(list): odd number list for generation mode=True, even number list for generation mode=False
    knowledge(str): knowledge
    system(str): system
    """
    knowledge = None
    system = None
    query_list = []
    if isinstance(dial, list) and len(dial) == 1:
        dial = dial[0]
    # Uniform input key name to src and tgt
    if isinstance(dial, list):
        assert "role" in dial[0], f"Unsupported data: {dial}."
        for data in dial:
            if data["role"] == "system":
                system = data["utterance"]
            else:
                query_list.append(data["utterance"])
        dial = {}
        dial["src"] = query_list
    elif isinstance(dial, dict):
        if "query" in dial:
            query_list = dial["query"]
            del dial["query"]
            dial["src"] = query_list
        elif "src" in dial:
            query_list = dial["src"]
        elif "query_list" in dial:
            query_list = dial["query_list"]
            del dial["query_list"]
            dial["src"] = query_list
        elif "utterance" in dial:
            query_list = [dial["utterance"]]
            del dial["utterance"]
            dial["src"] = query_list
        else:
            raise ValueError(f"Unsupported data: {dial}")

    if isinstance(query_list, str):
        query_list = [query_list]
        dial["src"] = query_list

    # Extract tgt or multi-turn response
    tgt_list = []
    if "tgt" in dial:
        tgt_list = dial["tgt"]
        if isinstance(tgt_list, str):
            tgt_list = [tgt_list]
    if "response" in dial:
        response_list = dial["response"]
        sort = dial["sort"]
        if len(query_list) == len(tgt_list) + 1:
            if len(sort) == 2:
                chosen = response_list[sort.index(max(sort))]
                tgt_list.extend([chosen])
            elif sort == [1]:
                tgt_list.extend(response_list)
            else:
                raise ValueError(f"Unsupported data: {dial}.")

    # Extract knowledge
    if contains_markup([query_list[-1]], get_markup_tokens()):
        knowledge = query_list.pop(-1)
        knowledge = extract_knowledge(knowledge)
    elif dial.get("knowledge", "") != "":
        knowledge = dial["knowledge"]

    # Extract system
    if dial.get("is_system", 0) == 1 and len(query_list) >= 2:
        system = query_list.pop(0)
        # pop response of system from tgt_list
        tgt_list.pop(0)
    elif dial.get("system", "") != "":
        system = dial["system"]

    dial["src"] = query_list
    if tgt_list:
        new_tgt_list = tgt_list
        for tgt in tgt_list:
            if contains_markup([tgt], get_markup_tokens()):
                new_tgt_list.remove(tgt)
        tgt_list = new_tgt_list
        dial["tgt"] = tgt_list

    new_dial = []
    for idx, query in enumerate(query_list):
        new_dial.append(query)
        if generation_mode:
            if tgt_list and idx < len(query_list) - 1:
                new_dial.append(tgt_list[idx])
        else:
            if tgt_list and idx < len(tgt_list):
                new_dial.append(tgt_list[idx])

    if not return_query_list:
        return new_dial, knowledge, system
    else:
        return dial, knowledge, system, query_list, tgt_list


def convert_to_tokens_for_sft(
    dial: List[dict],
    extra_info: dict,
    tokenizer,
    append_bos_token,
    max_src_len,
    generation_mode=True,
    system_prompt_version="V1",
):
    """Convert a dial to tokens for SFT model.

    Concat each part in the follow priority:
    1. The current usert query (Truncate the left side if too long).
    2. The system setting (Skip the system turn if too long).
    3. The knowledge (Truncate the right side if too long).
    4. The dialogue history (Truncate the oldest turn if too long).
    """
    # extract dial, knowledge system from dial
    preprocessed_dial = preprocess_data(dial, generation_mode=generation_mode)

    preprocessed_dial, knowledge, system = preprocessed_dial
    partial_mode = False
    if len(preprocessed_dial) % 2 == 0 and generation_mode:
        logger.info(
            f"The dialogue has an even number of turns and geneartion_mode=True, "
            f"the partial mode will be applied. dial: {dial}\n"
            f"preprocessed_dial: {preprocessed_dial}."
        )
        partial_mode = True
    if len(preprocessed_dial) % 2 != 0 and not generation_mode:
        logger.warning(
            f"Unsupported data when geneartion_mode=False. "
            f"dial: {dial}\n"
            f"preprocessed_dial: {preprocessed_dial}. "
            f"Dial should be even number."
        )
        tgt = ""
    if len(preprocessed_dial) % 2 == 0 and not generation_mode:
        tgt = preprocessed_dial.pop(-1)
    extra_info["knowledge"] = (
        knowledge if knowledge is not None else extra_info.get("knowledge", "")
    )
    extra_info["system"] = (
        system if system is not None else extra_info.get("system", "")
    )

    # Step 0: Initialize prefix tokens.
    if append_bos_token:
        prefix_tokens = [tokenizer.bos_token, tokenizer.cls_token]
    else:
        prefix_tokens = [tokenizer.cls_token]

    # Step 1: Add user utterance.
    if partial_mode:
        context_tokens = (
            tokenizer.tokenize(preprocessed_dial[-2])
            + [tokenizer.sep_token]
            + tokenizer.tokenize(preprocessed_dial[-1])
        )
    else:
        context_tokens = tokenizer.tokenize(preprocessed_dial[-1]) + [
            tokenizer.sep_token
        ]
    if len(prefix_tokens) + len(context_tokens) > max_src_len:
        logger.warning(
            f"The length of the last user utterance ({len(prefix_tokens) + len(context_tokens)}) "
            f"is greater than max input length ({max_src_len}). We will truncate it."
        )
        # NOTE(hehuang): Truncate the left side of the the last user utterance now.
        context_tokens = context_tokens[-(max_src_len - len(prefix_tokens)) :]

    # Step 2: Add system.
    sys_turn_tokens = []
    if extra_info.get("system", "") != "":
        if system_prompt_version == "V1":
            from dataset.finetuning import SYSTEM_DEFAULT_TGT

            sys_turn_tokens = tokenizer.tokenize(extra_info["system"]) + [
                tokenizer.sep_token
            ]
            sys_turn_tokens += tokenizer.tokenize(SYSTEM_DEFAULT_TGT) + [
                tokenizer.cls_token
            ]
        elif system_prompt_version == "V2":
            sys_turn_tokens = (
                [tokenizer.sys_start_token]
                + tokenizer.tokenize(system)
                + [tokenizer.sys_end_token]
            )
        if (
            len(prefix_tokens) + len(sys_turn_tokens) + len(context_tokens)
            > max_src_len
        ):
            logger.warning(
                "The length of the last user utterance and system "
                f"({len(prefix_tokens) + len(sys_turn_tokens) + len(context_tokens)}) "
                f"is greater than max input length({max_src_len}). We will ignore system setting."
            )
            sys_turn_tokens = []

    # Step 3: Add knowledge.
    knowledge_tokens = []
    if extra_info.get("knowledge", "") != "":
        knowledge_tokens = tokenizer.tokenize(extra_info["knowledge"])
        if (
            len(prefix_tokens)
            + len(sys_turn_tokens)
            + len(knowledge_tokens)
            + len(context_tokens)
            > max_src_len
        ):
            # NOTE(hehuang): Truncate the right side of the knowledge now.
            knowledge_tokens = knowledge_tokens[
                : max_src_len
                - len(prefix_tokens)
                - len(sys_turn_tokens)
                - len(context_tokens)
            ]

    if system_prompt_version == "V1":
        # knowledge <mask:0> system <|endofprompt|> system_ans <mask:0>
        # query_0 <|endofprompt|> answer_0 <mask:0> query_1 <|endofprompt|>
        prefix_tokens.extend(sys_turn_tokens)
        if append_bos_token:
            prefix_tokens = prefix_tokens[:1] + knowledge_tokens + prefix_tokens[1:]
        else:
            prefix_tokens = knowledge_tokens + prefix_tokens
    elif system_prompt_version == "V2":
        # <mask:4> system <mask:5> knowledge <mask:0> query_0 <|endofprompt|>
        # answer_0 <mask:0> query_1 <|endofprompt|>
        # NOTE(xuchang): system_prompt_version V2 not used with append_bos_token = True
        prefix_tokens = sys_turn_tokens + knowledge_tokens + prefix_tokens
    else:
        raise ValueError(f"Unknown system_prompt_version: {system_prompt_version}")

    # Step 4: Add dialogue history.
    if partial_mode:
        dialogue_history_range = range(len(preprocessed_dial) - 3, -1, -2)
    else:
        dialogue_history_range = range(len(preprocessed_dial) - 2, -1, -2)
    for idx in dialogue_history_range:
        cur_turn_tokens = tokenizer.tokenize(preprocessed_dial[idx - 1]) + [
            tokenizer.sep_token
        ]
        cur_turn_tokens += tokenizer.tokenize(preprocessed_dial[idx]) + [
            tokenizer.cls_token
        ]
        if (
            len(prefix_tokens) + len(context_tokens) + len(cur_turn_tokens)
        ) > max_src_len:
            logger.debug(f"Truncate dialogue into: {preprocessed_dial[idx + 1:]}")
            break
        context_tokens = cur_turn_tokens + context_tokens

    if not generation_mode:
        context_tokens += tokenizer.tokenize(tgt)

    if (len(prefix_tokens) + len(context_tokens)) > max_src_len:
        logger.warning(f"Truncate dialogue into: {max_src_len}")
        context_tokens = context_tokens[: max_src_len - len(prefix_tokens)]

    return prefix_tokens + context_tokens


def convert_to_tokens_for_ec_completion(
    dial: Union[List[dict], dict],
    tokenizer,
    data_format,
    max_src_len,
    generation_mode=True,
):
    """Convert a dial to tokens for EC completion model."""
    if isinstance(dial, list):
        dial = dial[0]

    code_prefix = dial.get("prefix", "")
    code_suffix = dial.get("suffix", "")
    if not generation_mode:
        if "output" in dial:
            tgt = dial["output"]
        elif "code" in dial:
            tgt = dial["code"]
        else:
            raise ValueError(f"No suitable output or code in dial keys: {dial.keys()}")

    eos_token = tokenizer.eos_token
    if data_format == "ec2_completion":
        prefix_prepend_token = "<mask:0>"
        middle_prepend_token = "<mask:1>"
        suffix_prepend_token = "<mask:2>"
    elif data_format == "ec3_completion":
        prefix_prepend_token = "<|prefixoftext|>"
        middle_prepend_token = "<|middleoftext|>"
        suffix_prepend_token = "<|suffixoftext|>"
    else:
        raise ValueError(f"Invalid data_format: {data_format}")

    input_tokens = []
    if len(code_suffix) == 0:
        # code completion
        if data_format == "ec2_completion":
            input_tokens.extend(
                [
                    prefix_prepend_token,
                    suffix_prepend_token,
                    middle_prepend_token,
                ]
            )
        input_tokens.extend(tokenizer.tokenize(code_prefix))
    else:
        # code infilling
        input_tokens.append(prefix_prepend_token)
        input_tokens.extend(tokenizer.tokenize(code_prefix))
        input_tokens.append(suffix_prepend_token)
        input_tokens.extend(tokenizer.tokenize(code_suffix))
        input_tokens.append(middle_prepend_token)

    if not generation_mode:
        input_tokens.extend(tokenizer.tokenize(tgt))
        input_tokens.append(eos_token)

    # NOTE(hehuang): Truncation need optimize.
    if len(input_tokens) > max_src_len:
        input_tokens = input_tokens[:max_src_len]
        logger.warning(
            f"The query length is greater than max input length {max_src_len}."
            "We will truncate it."
        )

    return input_tokens


def convert_to_tokens_for_rm(dial: List[dict], tokenizer, max_len, use_cls=True):
    """Convert a dial to tokens for Reward Model."""
    if len(dial["src"]) != len(dial["tgt"]) + 1:
        raise ValueError(
            "Invalid dialog: the number of strings in src is one more than the number of strings in tgt"
        )

    context = []
    for src, tgt in zip(dial["src"][:-1], dial["tgt"]):
        context.append(src)
        context.append(tgt)
    context.append(dial["src"][-1])

    if len(dial["response"]) > 1:
        raise ValueError("Diag should only include one response")
    response = dial["response"]

    context.append(response[0])
    prefix_tokens = [tokenizer.bos_token, tokenizer.cls_token]
    postfix_tokens = [tokenizer.eos_token] if not use_cls else []
    context_tokens = []
    for idx in range(len(context) - 1, -1, -2):
        turn_tokens = tokenizer.tokenize(context[idx - 1]) + [tokenizer.sep_token]
        turn_tokens += tokenizer.tokenize(context[idx]) + [tokenizer.cls_token]
        if (
            len(prefix_tokens)
            + len(context_tokens)
            + len(turn_tokens)
            + len(postfix_tokens)
            > max_len
        ):
            break
        context_tokens = turn_tokens + context_tokens

    context.pop()
    tokens = prefix_tokens + context_tokens + postfix_tokens

    return tokens


def convert_to_input_ids(
    dials: List[List[dict]],
    tokenizer,
    data_format,
    append_bos_token,
    max_src_len,
    extra_infos: Optional[List[dict]] = None,
    generation_mode=True,
    system_prompt_version="V1",
    rm_use_cls=True,
) -> Tuple[List[List[int]], int]:
    """Convert batch dialogue into input_ids.

    The API support multiple data format: `pt`, `sft`, `rm`, `ec2_completion` and `ec3_completion`.

    Args:
        dials (List[List[dict]]): A batch of dialogue.
        tokenizer (ErnieBotTokenizer): The used tokenizer.
        data_format (str): The data format for converting dialogue to input_ids,
            support `pt`, `sft`, `rm`, `ec2_completion` and `ec3_completion`.
        append_bos_token (bool): Whether to append bos token id ahead of input_ids.
        max_src_len (int): The maximum length of input_ids.
        extra_infos (Optional[List[dict]]): The extra information for each dialogue
            which is useful for tokens predictions.

    Returns:
        input_ids (List[List[int]]): The raw input_ids with truncation, but without padding.
        num_input_tokens (int): The total input tokens in a batch.

    Raises:
        ValueError: Invalid data format.
    """
    batch_tokens = []
    if data_format == "pt":
        for dial in dials:
            tokens = convert_to_tokens_for_pt(
                dial, tokenizer, append_bos_token, max_src_len
            )
            batch_tokens.append(tokens)
    elif data_format == "sft":
        for i, dial in enumerate(dials):
            if extra_infos is not None and extra_infos[i] is not None:
                extra_info = extra_infos[i]
            else:
                extra_info = {}
            tokens = convert_to_tokens_for_sft(
                dial,
                extra_info,
                tokenizer,
                append_bos_token,
                max_src_len,
                generation_mode=generation_mode,
                system_prompt_version=system_prompt_version,
            )
            batch_tokens.append(tokens)
    elif data_format in ("ec2_completion", "ec3_completion"):
        for dial in dials:
            tokens = convert_to_tokens_for_ec_completion(
                dial,
                tokenizer,
                data_format,
                max_src_len,
                generation_mode=generation_mode,
            )
            batch_tokens.append(tokens)
    elif data_format == "rm":
        for dial in dials:
            tokens = convert_to_tokens_for_rm(dial, tokenizer, max_src_len, rm_use_cls)
            batch_tokens.append(tokens)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    # generate input_ids
    input_ids = []
    num_input_tokens = 0
    for tokens in batch_tokens:
        input_ids.append(tokenizer.convert_tokens_to_ids(tokens))
        num_input_tokens += len(input_ids[-1])
    return input_ids, num_input_tokens
