from enum import IntEnum

class MsgType(IntEnum):
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3

class Message():
    def __init__(self, type: MsgType, msg):
        self.content = msg
        self.type_int = int(type)
        self.flags = []

        if self.type_int == 1:
            self.type_str = "system"
        elif self.type_int == 2:
            self.type_str = "user"
        elif self.type_int == 3:
            self.type_str = "assistant"
        else:
            self.type_str = "system"

        self.assets = []

    def add_asset(self, src: list):
        # https://platform.openai.com/docs/guides/prompt-engineering/tactic-instruct-the-model-to-answer-using-a-reference-text
        pass

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
        self.sequence_num = 1

    
    def create_prompt_chatml(self, system_message, messages):
        # defining the user input and the system message
        # user_input = "<your user input>" 
        # system_message = f"<|im_start|>system\n{'<your system message>'}\n<|im_end|>"

        # creating a list of messages to track the conversation
        # messages = [{"sender": "user", "text": user_input}]
            
        prompt = system_message
        for message in messages:
            prompt += f"\n<|im_start|>{message['sender']}\n{message['text']}\n<|im_end|>"
        prompt += "\n<|im_start|>assistant\n"
        return prompt
    
    def create_prompt_dict(self, sequences=[]):
        tmp_prompt = []
        for msg in self.messages:

            if len(sequences) > 0:
                if msg[0] not in sequences:
                    continue

            tmp_prompt.append(msg[1].get_dict())

        return tmp_prompt

    def add_message(self, msg : Message, finish_sequence=False):
        self.messages.append([self.sequence_num, msg])
        if finish_sequence: self.shift_sequence()

    def shift_sequence(self):
        self.sequence_num += 1
