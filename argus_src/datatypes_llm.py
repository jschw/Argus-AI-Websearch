from enum import IntEnum
from datetime import datetime
import time

class MsgType(IntEnum):
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3
    CONTEXT = 4

class Message():
    def __init__(self, type: MsgType, msg: str, timestamp: int = int(time.time())):
        self.content = msg
        self.type_int = int(type)
        self.timestamp = timestamp

        if self.type_int == (1 or 4):
            self.type_str = "system"
        elif self.type_int == 2:
            self.type_str = "user"
        elif self.type_int == 3:
            self.type_str = "assistant"
        else:
            self.type_str = "system"

    def get_type(self) -> int:
        return self.type_int
    
    def get_msg(self) -> str:
        return self.content
    
    def get_timestamp_int(self) -> int:
        return self.timestamp
    
    def get_timestamp_dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)
    
    def get_timestamp_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%d.%m.%Y, %H:%M:%S")
    
    def get_dict(self) -> dict:
        return {"role": self.type_str, "content": self.content}
    
    def get_chatml(self) -> str:
        # TODO
        return


class Conversation():

    def __init__(self):
        # messages = conversational content. Types: user, assistant
        # [sequence_num, Message]
        self.messages = []
        # general_context = context added to each conversation prompt. Type: system
        self.general_context = []
        # general_instructions = Instructions added to each conversation prompt. Type: system
        self.general_instructions = []
        # sequence_num = A counter that represents the current part of the conversation and all its components
        self.sequence_num = 0

    
    def create_prompt_chatml(self, sequences=[]):
        # defining the user input and the system message
        # user_input = "<your user input>" 
        # system_message = f"<|im_start|>system\n{'<your system message>'}\n<|im_end|>"

        # creating a list of messages to track the conversation
        # messages = [{"sender": "user", "text": user_input}]

        tmp_prompt = ""
        for msg in self.messages:
            tmp_prompt += f"\n<|im_start|>{msg['sender']}\n{msg['text']}\n<|im_end|>"

        tmp_prompt += "\n<|im_start|>assistant\n"
        return tmp_prompt
    
    def create_prompt_dict(self, sequences=[], exclude_context=True):
        tmp_prompt = []
        for msg in self.messages:
            
            # Check if sequence filter is set
            if len(sequences) > 0:
                if msg[0] not in sequences:
                    continue

            # Check if context messages should be excluded
            if exclude_context and msg[1].get_type() == 4:
                continue

            tmp_prompt.append(msg[1].get_dict())

        return tmp_prompt

    def add_message(self, msg : Message, finish_sequence=False):
        self.messages.append([self.sequence_num, msg])
        if finish_sequence: self.shift_sequence()

    def delete_sequence(self, idx : int):
        self.messages = list(filter(lambda a: a[0] != idx, self.messages))

    def get_sequence(self, idx : int):
        return filter(lambda a: a[0] == idx, self.messages)

    def shift_sequence(self):
        self.sequence_num += 1

    def get_last_prompt(self) -> str:
        for msg in self.messages:
            if msg[0] == self.sequence_num and msg[1].get_type() == 2:
                # If message is current sequence num and type 2 = USER
                return msg[1].get_msg()
        return []
            
    def get_last_output(self) -> str:
        for msg in self.messages:
            if msg[0] == self.sequence_num and msg[1].get_type() == 3:
                # If message is current sequence num and type 3 = ASSISTANT
                return msg[1].get_msg()
        return []
    
    def output_msg_store_cli(self):
        for msg in self.messages:
            print(f"Idx: {msg[0]} , Content: {msg[1].get_dict()}\n")

