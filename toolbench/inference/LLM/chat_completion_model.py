#!/usr/bin/env python
# coding=utf-8
from typing import Optional, List, Mapping, Any
from termcolor import colored
import re
import openai
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
(2) {{ACTION_DESC}}
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
    "{{ACTION_DESC}}",
    "Action: MUST be one of the following: {func_list}"
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
        "Code: executable Python code (the action to take), should call function from {func_list}\n"
        "End Action\n"
    )
).replace(
    "{{FORMAT}}",
    (
        "Thought:\n"
        "Code:\n"
        "End Action\n"
    )
).replace(
    "{{ACTION_DESC}}",
    "Code: MUST be valid Python code (i.e., can be executed directly). Be sure to print out function return values if you want to see them."
).replace(
    "{{FINISH}}",
    (
        "Code: Finish(return_type=\"give_answer\", final_answer=\"your answer string\")\n"
    )
)


# For prediction parsing, into ReACT format
def react_parser(string):
    thought = [string[string.find("Thought: ") + len("Thought: "): string.find("\nAction: ")]]
    action = [string[string.find("Action: ") + len("Action: "): string.find("\nAction Input: ")]]
    action_input = [string[string.find("Action Input: ") + len("Action Input: "):]]
    return thought[0], action[0], action_input[0]

def code_parser(string):
    thought = string[
        string.find("Thought:") + len("Thought:"):
        string.find("Code:")
    ].strip()
    action = string[
        string.find("Code:") + len("Code:"):
    ].strip().rstrip("End Action")
    return thought, action

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    key,
    messages,
    model="gpt-3.5-turbo-16k-0613",
    stop=None,
    **args):

    json_data = {
        "model": model,
        "messages": messages,
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

FINISH_FUNC_DESC = """If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer (set return_type to \"give_answer\"). Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart (set return_type to \"give_up_and_restart\"). Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information"""

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
            # print(f"role==function, message={message}")
            return {
                "role": "user",
                "content": f"Observation: {message['content']}"
            }
        if "function_call" in message.keys():
            # print(f"function_call in message, message={message}")
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
        for function_dict in functions:
            param_str = ""
            api_name = function_dict["name"]
            func_list.append(api_name)
            if "Finish" in api_name:
                param_str = f'"return_type": string, "final_answer": string, '
                api_desc = FINISH_FUNC_DESC
                func_str += f"{api_name}: {api_desc}. Your input should be a json (args json schema): {param_str} The Action to trigger this API should be {api_name} and the input parameters should be a json dict string. Pay attention to the type of parameters.\n\n"
            else:
                api_desc = function_dict["description"][function_dict["description"].find("The description of this function is: ")+len("The description of this function is: "):]
                for param_name in function_dict["parameters"]["properties"]:
                    data_type = function_dict["parameters"]["properties"][param_name]["type"]
                    param_str += f'"{param_name}": {data_type}, '
                param_str = "{{" + param_str + "}}"
                func_str += f"{api_name}: {api_desc}. Your input should be a json (args json schema): {param_str} The Action to trigger this API should be {api_name} and the input parameters should be a json dict string. Pay attention to the type of parameters.\n\n"
        func_list = str(func_list)
        return func_str, func_list

    def _build_fn_str_and_list_code(self, functions):
        func_str = ""
        func_list = []
        for function_dict in functions:
            param_str = ""
            api_name = function_dict["name"]
            func_list.append(api_name)
            if "Finish" in api_name:
                param_str = f'return_type: str, final_answer: str, '
                api_desc = FINISH_FUNC_DESC
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
            func_str += f"{api_name}: {api_desc}. Your action should be a one-line Python code: {api_name}({param_str}). Pay attention to the type of parameters.\n\n"
        func_list = str(func_list)
        return func_str, func_list

    def build_system_message(self, functions):
        action_mode = self.action_mode
        if action_mode == "json_as_action":
            func_str, func_list = self._build_fn_str_and_list_json(functions)
            base_template = JSON_AS_ACTION_SYSTEM_MESSAGE
        elif action_mode == "code_as_action":
            func_str, func_list = self._build_fn_str_and_list_code(functions)
            base_template = CODE_AS_ACTION_SYSTEM_MESSAGE
        else:
            raise NotImplementedError
        return base_template.replace("{func_str}", func_str).replace("{func_list}", func_list)

    def build_initial_messages(self, functions, question):
        system_message = self.build_system_message(functions)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Please answer the question: {question}"},
        ]
        return messages

    def parse(self,functions,process_id,**args):
        messages = self.conversation_history
        resp, usage = chat_completion_request(self.openai_key, messages, model=self.model)
        content = resp["content"]
        print(f"RAW Response:\n{content}")
        
        extra_function_call_kwargs = {"raw_msg": content}
        if self.action_mode == "json_as_action":
            # react format prediction
            thought, action, action_input = react_parser(content)
            message = {
                "role": "assistant",
                "content": thought,
                "function_call": {
                    "type": "json_as_action",
                    "name": action,
                    "arguments": action_input,
                    **extra_function_call_kwargs,
                }
            }
        elif self.action_mode == "code_as_action":
            thought, raw_code = code_parser(content)
            # parse a function call from the action with ast
            message = {
                "role": "assistant",
                "content": thought,
                "function_call": {
                    "type": "code_as_action",
                    "code": raw_code,
                    **extra_function_call_kwargs,
                }
            }

        return message, 0, usage["total_tokens"]
