import json
from enum import IntEnum
import openai
import os

class LlmConfiguration():

    def __init__(self, json_path: str):
        self.config_store = self.load_config(json_path)


    def load_config(self, json_path: str):
        f = open(json_path,)
        tmp_file = json.load(f)
        f.close()
        return tmp_file

    def get_config(self) -> dict:
        return {
                    "temp": float(self.config_store['settings'][0]['temperature']),
                    "model": self.config_store['settings'][0]['model_name'],
                    "top_p": float(self.config_store['settings'][0]['top_p']),
                    "freq_penalty": float(self.config_store['settings'][0]['frequency_penalty']),
                    "max_tokens": int(self.config_store['settings'][0]['max_tokens'])
                }
    
    def get_api_key(self) -> str:
        return self.config_store['settings'][0]['api_key']


class LlmInfTypes(IntEnum):
    OPENAI = 1
    MISTRAL = 2
    LLAMA2 = 3


class LlmInferenceEngine():

    def __init__(self, type: LlmInfTypes, model_path = None, api_key = None, enable_verbose_output = True):
        self.engine_type = int(type)
        self.api_key = api_key
        self.model_path = model_path
        self.model_instance = None
        self.enable_verbose_output = enable_verbose_output

        self.token_used_total = 0
        self.token_used_last = 0

        # Load local model to RAM
        match self.engine_type:
            case 1:
                # OpenAI API cloud inference
                # Set-up OpenAI API Key
                openai.api_key = self.api_key
                os.environ['OPENAI_API_KEY'] = self.api_key
                return None
            case 2:
                # Local Mistral inference
                return None
            case 3:
                # Local Llama2 inference
                return None

    def run_inference(self, messages : list, config : dict) -> str:
        match self.engine_type:
            case 1:
                # OpenAI API cloud inference
                if len(messages) > 0:

                    if self.enable_verbose_output:
                        print("Running inference...\n")

                    response = openai.chat.completions.create(model=config['model'],
                                                            max_tokens=config['max_tokens'],
                                                            temperature=config['temp'],
                                                            top_p=config['top_p'],
                                                            frequency_penalty=config['freq_penalty'],
                                                            messages=messages)

                    response_str = response.choices[0].message.content
                    tokens_used = int(response.usage.total_tokens)

                    self.token_used_last = tokens_used
                    self.token_used_total += tokens_used

                    return response_str, tokens_used
            case 2:
                # Local Mistral inference
                return 0
            case 3:
                # Local Llama2 inference
                return 0

