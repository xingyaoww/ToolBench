import re
import json
from Tree.Tree import my_tree, tree_node
from Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION, FORMAT_INSTRUCTIONS_USER_FUNCTION
from Algorithms.base_search import base_search_method
from toolbench.inference.LLM.chat_completion_model import ChatCompletion
from repl import PythonREPL
from copy import deepcopy
from termcolor import colored

class single_chain(base_search_method):
    """Implement of CoT method
    """
    def __init__(self,llm,io_func,extra_prefix="",process_id=0,start_message_list=None):
        """extra_prefix and start_message_list is used in Reflection Algo"""
        super(single_chain, self).__init__(llm,io_func, process_id, callbacks=None)
        self.io_func = io_func
        self.llm = llm
        self.extra_prefix = extra_prefix
        self.start_message_list = start_message_list
        self.process_id = process_id

        self.restart()
    def restart(self):
        self.status = 0
        self.try_list = []
        self.terminal_node = []

        self.query_count = 0 # number of interactions with openai
        self.total_tokens = 0
        self.success_count = 0

    def to_json(self, answer=False,process=True):
        if process:
            json_obj = {
                "win": self.status == 1,
                "try_count": len(self.try_list),
                "trys": self.try_list,
                "compare_candidates": [],
                "forward_args":self.forward_args,
            }
            for node in self.terminal_node:
                if node.pruned == False: # has final answer
                    json_obj["compare_candidates"].append(node.get_chain_result_from_this_node(use_messages=False))
        else:
            json_obj = {}

        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "final_answer": "",
                "function": self.io_func.functions,
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "train_messages": [],
                "chain": [],
            }
            for node in self.terminal_node:
                if node.pruned == False:
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["final_answer"] = node.description
                    json_obj["answer_generation"]["train_messages"] = node.get_train_messages_from_this_node()
                    break
        return json_obj

    def to_json_single(self):
        """parse the last try
        Though the nodes are formed as a tree, We still know they are actually a chain
        """
        json_obj = {}
        tree_obj = self.terminal_node[-1].get_chain_result_from_this_node()
        json_obj["chain"] = tree_obj
        json_obj["win"] = self.status == 1
        return json_obj

    def start(self,single_chain_max_step,pass_at=1,answer=1):
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")

        for i in range(pass_at):
            if self.process_id == 0:
                print(f"[single_chain]try for the {i+1} time")
            self.tree = my_tree()
            self.tree.root.node_type = "Action Input"
            self.tree.root.io_state = deepcopy(self.io_func)
            out_node = self.do_chain(self.tree.root, single_chain_max_step)
            self.terminal_node.append(out_node)
            self.try_list.append(self.to_json_single())
            if out_node.io_state.check_success() == 1:
                self.status = 1
                self.success_count += 1
                if self.success_count >= answer:
                    return 1
        return 0

    def construct_func_name_to_args(self, functions):
        # For code_as_input, we need to use these to construct wrapper functions
        func_name_to_args = {}
        for function_dict in functions:
            args = []
            name = function_dict["name"]
            if "Finish" in name:
                name = "Finish"
                args = [
                    {"name": "return_type", "type": "str"},
                    {"name": "final_answer", "type": "str"}
                ]
            else:
                for param_name in function_dict["parameters"]["properties"]:
                    assert " " not in param_name
                    assert "-" not in param_name
                    data_type = function_dict["parameters"]["properties"][param_name]["type"]
                    data_type = {
                        "string": "str",
                        "integer": "int",
                        "boolean": "bool",
                    }[data_type]
                    args.append({
                        "name": param_name,
                        "type": data_type,
                    })
            func_name_to_args[name] = args
        return func_name_to_args

    def do_chain(self,now_node,single_chain_max_step):
        func_name_to_args = self.construct_func_name_to_args(self.io_func.functions)

        if isinstance(self.llm, ChatCompletion):
            # special case for chat completion (with json/code as input)
            initial_message = self.llm.build_initial_messages(
                self.io_func.functions,
                self.io_func.input_description
            )
            # func_name_to_args is a dict mapping function name to its arguments
            # used to construct wrapper functions latter
            print("** initial_message **")
            for message in initial_message:
                color_converter = {
                    "system": "yellow",
                    "user": "yellow",
                }
                print(
                    colored(f"{message['role']}: {message['content']}",
                              color = color_converter[message['role']])
                )
                self.tree.root.messages.append(message)
        elif self.start_message_list == None:
            system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
            system = system.replace("{task_description}",self.io_func.task_description)
            self.tree.root.messages.append({"role":"system","content":system})

            user = FORMAT_INSTRUCTIONS_USER_FUNCTION
            user = user.replace("{input_description}",self.io_func.input_description)
            self.tree.root.messages.append({"role":"user","content":user})
        else:
            """In Reflection Algo, we startswith former trials and reflections, so the caller will give the start messages"""
            self.tree.root.messages = self.start_message_list

        now_node = self.tree.root
        while True:
            # recursively parse message into nodes
            self.llm.change_messages(now_node.messages)
            new_message,error_code,total_tokens = self.llm.parse(
                functions=self.io_func.functions,
                process_id=self.process_id
            )
            self.total_tokens += total_tokens
            self.query_count += 1
            assert new_message["role"] == "assistant"
            if "content" in new_message.keys() and new_message["content"] != None:
                temp_node = tree_node()
                temp_node.node_type = "Thought"
                temp_node.description = new_message["content"]
                child_io_state = deepcopy(now_node.io_state)
                
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 
                temp_node.messages = now_node.messages.copy()
                temp_node.father = now_node
                now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                now_node = temp_node

                if error_code != 0:
                    now_node.observation_code = error_code
                    now_node.pruned = True

            if "function_call" in new_message.keys():

                # Handle json as action
                if new_message["function_call"]["type"] == "json_as_action":
                    function_name = new_message["function_call"]["name"]
                    temp_node = tree_node()
                    temp_node.node_type = "Action"
                    temp_node.description = function_name
                    child_io_state = deepcopy(now_node.io_state)
                    
                    temp_node.io_state = child_io_state
                    temp_node.is_terminal = child_io_state.check_success() != 0 
                    temp_node.messages = now_node.messages.copy()
                    temp_node.father = now_node
                    now_node.children.append(temp_node)

                    temp_node.print(self.process_id)
                    now_node = temp_node

                    function_input = new_message["function_call"]["arguments"]
                    temp_node = tree_node()
                    temp_node.node_type = "Action Input"
                    temp_node.description = function_input
                    child_io_state = deepcopy(now_node.io_state)
                    observation, status = child_io_state.step(action_name=now_node.description, action_input=function_input)
                
                # Handle code as action
                elif new_message["function_call"]["type"] == "code_as_action":
                    code = new_message["function_call"]["code"]
                    
                    temp_node = tree_node()
                    temp_node.node_type = "Code Action"
                    temp_node.description = code
                    child_io_state = deepcopy(now_node.io_state)
                    
                    # Wrap `child_io_state.step` into functions for REPL
                    # observation, status = child_io_state.step(action_name=now_node.description, action_input=function_input)
                    user_ns = {}
                    for function_name in func_name_to_args:
                        cur_args = func_name_to_args[function_name]

                        def wrap_func(function_name, cur_args):
                            def wrapper(*args, **kwargs):
                                action_input = {}
                                # process positional arguments
                                for i, arg in enumerate(args):
                                    action_input[cur_args[i]["name"]] = arg
                                # add keyword arguments
                                for key, value in kwargs.items():
                                    if key in action_input:
                                        raise ValueError(f"Duplicated argument name: {key}")
                                    action_input[key] = value

                                # call the function
                                observation, status = child_io_state.step(
                                    action_name=function_name,
                                    action_input=json.dumps(action_input)
                                )

                                # throw exception if the function call failed
                                if status != 0:
                                    error_msg = f"Function call {function_name} failed with status {status}: {observation}"
                                    raise ValueError(error_msg)

                                # return the observation if the function call succeeded
                                return observation
                            return wrapper
                        user_ns[function_name] = wrap_func(function_name, cur_args)

                    # execute the code
                    repl = PythonREPL(
                        user_ns=user_ns,
                        max_observation_length=self.io_func.max_observation_length,
                    )
                    observation = repl(code)
                    status = 0

                    # use regex to extract the observation for status if any
                    status_code = re.findall(r"Function call .* failed with status (\d+): .*", observation)
                    print(f"Parsed status code: {status_code}")
                    if len(status_code) > 0:
                        status = int(status_code[0])
                    if status == 4:
                        assert "give_up_and_restart" in code
                else:
                    raise NotImplementedError

                temp_node.observation = observation
                temp_node.observation_code = status

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 
                temp_node.messages = now_node.messages.copy()
                temp_node.father = now_node
                now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                now_node = temp_node

                if status != 0:
                    # return code refers to Downstream_tasks/rapidapi
                    if status == 4:
                        now_node.pruned = True
                    elif status == 1: # hallucination api name
                        assert "function_call" in new_message.keys()
                        new_message["function_call"]["name"] = "invalid_hallucination_function_name"
            
            now_node.messages.append(new_message)
            if now_node.node_type == "Action Input" or now_node.node_type == "Code Action":
                # The message will get converted by ChatCompletion into compatible format
                if "name" in new_message["function_call"].keys():
                    now_node.messages.append({
                        "role":"function",
                        "name": new_message["function_call"]["name"],
                        "content": now_node.observation,
                    })
                elif "type" in new_message["function_call"].keys():
                    now_node.messages.append({
                        "role": "user",
                        "content": f"Observation: {now_node.observation}"
                    })
            if now_node.get_depth() >= single_chain_max_step and not (now_node.is_terminal):
                now_node.pruned = True
            
            if now_node.pruned or now_node.is_terminal:
                return now_node

    
