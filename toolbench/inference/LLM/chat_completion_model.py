#!/usr/bin/env python
# coding=utf-8
from typing import Optional, List, Mapping, Any
from termcolor import colored
import ast
import json
import random
import openai
import traceback
from typing import Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from toolbench.model.model_adapter import get_conversation_template
from toolbench.inference.utils import SimpleChatIO, react_parser

BASE_SYSTEM_MESSAGE = """Answer the following questions as best you can. Specifically, you have access to the following APIs:

{func_str}

Use the following format:
{{FORMAT_INSTRUCTIONS}}

Remember: (1) Follow the format, i.e,
{{FORMAT}}
(2) The Action: MUST be one of the following: {func_list}
(3) If you believe that you have obtained enough information (which can be judge from the history observations) that can answer the task, please call:
{{FINISH}}
"""

JSON_AS_ACTION_SYSTEM_MESSAGE = BASE_SYSTEM_MESSAGE.replace(
    "{{FORMAT_INSTRUCTIONS}}",
    (
        "Thought: you should always think about what to do\n"
        "Action: the action to take, should be one of {func_list}\n"
        "Action Input: the input to the action\n"
        "End Action\n"
    )
).replace(
    "{{FORMAT}}",
    (
        "Thought:\n"
        "Action:\n"
        "Action Input:\n"
        "End Action\n"
    )
).replace(
    "{{FINISH}}",
    (
        "Action: Finish\n"
        "Action Input: {\"return_type\": \"give_answer\", \"final_answer\": your answer string}.\n"
    )
)

CODE_AS_ACTION_SYSTEM_MESSAGE = BASE_SYSTEM_MESSAGE.replace(
    "{{FORMAT_INSTRUCTIONS}}",
    (
        "Thought: you should always think about what to do\n"
        "Action: one-line python code (the action to take), should call one function from {func_list}\n"
        "End Action\n"
    )
).replace(
    "{{FORMAT}}",
    (
        "Thought:\n"
        "Action:\n"
        "End Action\n"
    )
).replace(
    "{{FINISH}}",
    (
        "Action: Finish(return_type=\"give_answer\", final_answer=\"your answer string\")\n"
    )
)


# For prediction parsing, into ReACT format
def react_parser(string):
    thought = [string[string.find("Thought: ") + len("Thought: "): string.find("\nAction: ")]]
    action = [string[string.find("Action: ") + len("Action: "): string.find("\nAction Input: ")]]
    action_input = [string[string.find("Action Input: ") + len("Action Input: "):]]
    return thought[0], action[0], action_input[0]

def code_parser(string):
    thought = [string[string.find("Thought: ") + len("Thought: "): string.find("\nAction: ")]]
    _end_pos = string.find("\nEnd Action")
    if _end_pos == -1:
        _end_pos = len(string)
    action = [string[string.find("Action: ") + len("Action: "): _end_pos]]
    return thought[0], action[0]

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    key,
    messages,
    model="gpt-3.5-turbo-16k-0613",
    stop=None,
    **args):

    use_messages = []
    for message in messages:
        if not("valid" in message.keys() and message["valid"] == False):
            use_messages.append(message)

    json_data = {
        "model": model,
        "messages": use_messages,
        "max_tokens": 1024,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        **args
    }
    if stop is not None:
        json_data.update({"stop": stop})

    try:
        response = openai.ChatCompletion.create(
            api_key=key,
            **json_data,
        )
        return response["choices"][0]["message"], response["usage"]

    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"OpenAI calling Exception: {e}")
        return e

class ChatCompletion:
    def __init__(
        self,
        model,
        openai_key,
        action_mode = "json_as_action",
    ) -> None:
        super().__init__()
        self.model = model
        self.openai_key = openai_key
        self.action_mode = action_mode
        assert self.action_mode in ["json_as_action", "code_as_action"]
    
    def convert_function_call_message(self, message):
        if message["role"] == "function":
            return {
                "role": "user",
                "content": f"Observation: {message['content']}"
            }
        if "function_call" in message.keys():
            return {
                "role": "assistant",
                "content": message["function_call"]["raw_msg"],
            }
        return message


    def add_message(self, message):
        self.conversation_history.append(
            self.convert_function_call_message(message)
        )

    def change_messages(self,messages):
        self.conversation_history = [
            self.convert_function_call_message(message)
            for message in messages
        ]

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)


    def _build_fn_str_and_list_json(self, functions):
        func_str = ""
        func_list = []
        func_name_to_args = {}
        for function_dict in functions:
            param_str = ""
            api_name = function_dict["name"]
            func_list.append(api_name)
            args = []
            if "Finish" in api_name:
                param_str = f'"return_type": string, "final_answer": string, '
                args = [
                    {
                        "name": "return_type",
                        "type": "string",
                    },
                    {
                        "name": "final_answer",
                        "type": "string",
                    }
                ]
                api_desc = "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. ALWAYS call this function at the end of your attempt to answer the question finally."
                func_str += f"{api_name}: {api_desc}. Your input should be a json (args json schema): {param_str} The Action to trigger this API should be {api_name} and the input parameters should be a json dict string. Pay attention to the type of parameters.\n\n"
            else:
                api_desc = function_dict["description"][function_dict["description"].find("The description of this function is: ")+len("The description of this function is: "):]
                for param_name in function_dict["parameters"]["properties"]:
                    data_type = function_dict["parameters"]["properties"][param_name]["type"]
                    param_str += f'"{param_name}": {data_type}, '
                    args.append({
                        "name": param_name,
                        "type": data_type,
                    })
                param_str = "{{" + param_str + "}}"
                func_str += f"{api_name}: {api_desc}. Your input should be a json (args json schema): {param_str} The Action to trigger this API should be {api_name} and the input parameters should be a json dict string. Pay attention to the type of parameters.\n\n"
            func_name_to_args[api_name] = args
        func_list = str(func_list)
        return func_str, func_list, func_name_to_args

    def _build_fn_str_and_list_code(self, functions):
        func_str = ""
        func_list = []
        func_name_to_args = {}
        for function_dict in functions:
            param_str = ""
            api_name = function_dict["name"]
            func_list.append(api_name)
            args = []
            if "Finish" in api_name:
                param_str = f'return_type: str, final_answer: str, '
                args = [
                    {
                        "name": "return_type",
                        "type": "str",
                    },
                    {
                        "name": "final_answer",
                        "type": "str",
                    }
                ]
                api_desc = "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. ALWAYS call this function at the end of your attempt to answer the question finally."
            else:
                api_desc = function_dict["description"][function_dict["description"].find("The description of this function is: ")+len("The description of this function is: "):]
                for param_name in function_dict["parameters"]["properties"]:
                    assert " " not in param_name
                    assert "-" not in param_name
                    data_type = function_dict["parameters"]["properties"][param_name]["type"]
                    data_type = {
                        "string": "str",
                        "integer": "int",
                        "boolean": "bool",
                    }[data_type]
                    param_str += f'{param_name}: {data_type}, '
                    args.append({
                        "name": param_name,
                        "type": data_type,
                    })
            func_str += f"{api_name}: {api_desc}. Your action should be a one-line Python code: {api_name}({param_str}). Pay attention to the type of parameters.\n\n"
            func_name_to_args[api_name] = args
        func_list = str(func_list)
        return func_str, func_list, func_name_to_args

    def build_system_message(self, functions):
        action_mode = self.action_mode
        if action_mode == "json_as_action":
            func_str, func_list, func_name_to_args = self._build_fn_str_and_list_json(functions)
            base_template = JSON_AS_ACTION_SYSTEM_MESSAGE
        elif action_mode == "code_as_action":
            func_str, func_list, func_name_to_args = self._build_fn_str_and_list_code(functions)
            base_template = CODE_AS_ACTION_SYSTEM_MESSAGE
        else:
            raise NotImplementedError
        return base_template.replace("{func_str}", func_str).replace("{func_list}", func_list), func_name_to_args

    def parse(self,functions,process_id,**args):
        conv = get_conversation_template("tool-llama-single-round")
        roles = {
            "system": conv.roles[0],
            "user": conv.roles[1],
            "function": conv.roles[2],
            "assistant": conv.roles[3]
        }
        conversation_history = self.conversation_history
        question = ''
        for message in conversation_history:
            role = roles[message['role']]
            content = message['content']
            if role == "User":
                question = content
                break

        system_message, func_name_to_args = self.build_system_message(functions)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Please answer the question: {question}"},
        ]

        resp, usage = chat_completion_request(self.openai_key, messages, model=self.model)
        content = resp["content"]
        print(f"RAW Response: {content}")
        
        extra_function_call_kwargs = {}
        if self.action_mode == "json_as_action":
            # react format prediction
            thought, action, action_input = react_parser(content)
        
        elif self.action_mode == "code_as_action":
            thought, action_raw = code_parser(content)
            # parse a function call from the action with ast
            try:
                parsed = ast.parse(action_raw)
                assert len(parsed.body) == 1
                parsed = parsed.body[0]
                assert isinstance(parsed, ast.Expr)
                action = parsed.value.func.id
                # action_input should be keyword arguments in a json
                action_input = {}
                # Handle positional arguments
                if action in func_name_to_args:
                    cur_func_args = func_name_to_args[action]
                    for i, pos_arg in enumerate(parsed.value.args):
                        assert isinstance(pos_arg, ast.Constant)
                        action_input[cur_func_args[i]["name"]] = pos_arg.value
                        print(f"**** Assign pos ID {cur_func_args[i]['name']}: {pos_arg.value}")
                else:
                    print(f"**** No args for {action}")

                for keyword in parsed.value.keywords:
                    assert isinstance(keyword, ast.keyword)
                    assert isinstance(keyword.arg, str)
                    assert isinstance(keyword.value, ast.Constant)
                    action_input[keyword.arg] = keyword.value.value
                action_input = json.dumps(action_input)
            except SyntaxError:
                traceback.print_exc()
                action = "SyntaxError"
                # get the last line
                action_input = json.dumps({"raw_code": action_raw})

            extra_function_call_kwargs["raw_code"] = action_raw
        extra_function_call_kwargs["raw_msg"] = content

        message = {
            "role": "assistant",
            "content": thought,
            "function_call": {
                "name": action,
                "arguments": action_input,
                **extra_function_call_kwargs,
            }
        }
        return message, 0, usage["total_tokens"]
