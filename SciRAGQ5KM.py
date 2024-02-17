import os
from ctransformers import AutoModelForCausalLM
# Requires SCIPHI_API_KEY in the environment
from agent_search import SciPhi

def initialise():
    SCIPHI_API_KEY = "528d08dc3ed417f32954509131952c5a"
    sciphi_api_key = os.environ("SCI_PHI_API_KEY")

'''def get_chat_completion(
    self, conversation: list[dict], generation_config: GenerationConfig
) -> str:
    self._check_stop_token(generation_config.stop_token)
    prompt = ""
    added_system_prompt = False
    for message in conversation:
        if message["role"] == "system":
            prompt += f"### System:\n{SciPhiLLMInterface.ALPACA_CHAT_SYSTEM_PROMPT}. Further, the assistant is given the following additional instructions - {message['content']}\n\n"
            added_system_prompt = True
        elif message["role"] == "user":
            last_user_message = message["content"]
            prompt += f"### Instruction:\n{last_user_message}\n\n"
        elif message["role"] == "assistant":
            prompt += f"### Response:\n{message['content']}\n\n"

    if not added_system_prompt:
        prompt = f"### System:\n{SciPhiLLMInterface.ALPACA_CHAT_SYSTEM_PROMPT}.\n\n{prompt}"

    context = self.rag_interface.get_contexts([last_user_message])[0]
    prompt += f"### Response:\n{SciPhiFormatter.RETRIEVAL_TOKEN} {SciPhiFormatter.INIT_PARAGRAPH_TOKEN}{context}{SciPhiFormatter.END_PARAGRAPH_TOKEN}"
    latest_completion = self.model.get_instruct_completion(
        prompt, generation_config
    ).strip()

    return SciPhiFormatter.remove_cruft(latest_completion)
'''
def perform_search(client):
    # Perform a search
    search_response = client.search(query='Quantum Field Theory', search_provider='agent-search')
    print(search_response)
    # example: [{ 'score': '.89', 'url': 'https://...', 'metadata': {...} }

    # Generate a RAG response
    rag_response = client.get_search_rag_response(query='latest news', search_provider='bing', llm_model='SciPhi/Sensei-7B-V1')
    print(rag_response)
    # example: { 'response': '...', 'other_queries': '...', 'search_results': '...' }


if __name__ == "__main__":

    initialise()

    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    llm = AutoModelForCausalLM.from_pretrained("models/", model_file="sciphi-self-rag-mistral-7b-32k.Q5_K_M.gguf", model_type="mistral", gpu_layers=50)

    print(llm("In 2024 AI is going to"))

    perform_search(client)
