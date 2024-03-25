from enum import StrEnum
import jinja2

from examples.openai.gguf_kvs import GGUFKeyValues, Keys

def raise_exception(msg: str):
    raise Exception(msg)

class ToolStyle(StrEnum):
    # https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
    DEFAULT="Default",
    # https://github.com/MeetKai/functionary
    # TODO: look at https://github.com/ggerganov/llama.cpp/pull/5695
    # https://github.com/MeetKai/functionary/blob/main/functionary/prompt_template/prompt_template_v2.py
    FUNCTIONARY_V2="Functionary V2",
    # https://github.com/NousResearch/Hermes-Function-Calling
    NOUS_RESEARCH_HERMES="Nous-Research-Hermes-Function-Calling",

class ChatFormat: #(BaseModel):
    def __init__(self, template: str, eos_token: str, bos_token: str):
        env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True)
        self.template = env.from_string(template)
        self.eos_token = eos_token
        self.bos_token = bos_token

        self.strict_user_assistant_alternation = "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception" in template

        if "<|recipient|>' + tool_call['function']['name']" in template:
            self.tool_style = ToolStyle.FUNCTIONARY_V2
        else:
            self.tool_style = ToolStyle.DEFAULT


    def __str__(self):
        return f"ChatFormat(template={self.template}, eos_token={self.eos_token}, bos_token={self.bos_token})"


    @staticmethod
    def from_gguf(metadata: GGUFKeyValues):
        return ChatFormat(
            template = metadata[Keys.Tokenizer.CHAT_TEMPLATE],
            bos_token = metadata[Keys.Tokenizer.BOS_ID],
            eos_token = metadata[Keys.Tokenizer.EOS_ID])
    # @staticmethod
    # def from_gguf(model: Path):
    #     reader = GGUFReader(model.as_posix())
    #     return ChatFormat(
    #         template = reader.fields[Keys.Tokenizer.CHAT_TEMPLATE].read(),
    #         bos_token = reader.fields[Keys.Tokenizer.BOS_ID].read(),
    #         eos_token = reader.fields[Keys.Tokenizer.EOS_ID].read())
    
    def render(self, messages: list[dict], add_generation_prompt: bool, omit_bos: bool = False):
        return self.template.render(
            messages=messages,
            eos_token=self.eos_token,
            bos_token='' if omit_bos else self.bos_token,
            raise_exception=raise_exception,
            add_generation_prompt=add_generation_prompt,
        )
