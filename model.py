from time import perf_counter as timer
import torch
from sentence_transformers import util, SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig


DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def check_gpu_memory():
    print("Your device is:", DEVICE)

    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes / (2**30))
    print(f"Available GPU memory: {gpu_memory_gb} GB")

    return gpu_memory_gb


def get_modelid_quantconfig() -> tuple[str, bool]:
    """ Return model name and boolean of use quantization config """
    model_id = None
    gpu_memory_gb = check_gpu_memory()

    if gpu_memory_gb < 5.1:
        print(f"Your available GPU memory is {
              gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
        use_quantization_config = True

    elif gpu_memory_gb < 8.1:
        print(f"GPU memory: {
              gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
        use_quantization_config = True
        model_id = "google/gemma-2b-it"

    elif gpu_memory_gb < 19.0:
        print(f"GPU memory: {
              gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
        use_quantization_config = False
        model_id = "google/gemma-2b-it"

    elif gpu_memory_gb > 19.0:
        print(f"GPU memory: {
              gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
        use_quantization_config = False
        model_id = "google/gemma-7b-it"

    print("You are using", model_id,
          "Use quantization config is", use_quantization_config)
    return model_id, use_quantization_config


def get_fask_attention():
    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"

    print("Your fask attention is", attn_implementation)
    return attn_implementation


def prepare_model(name=None, use_quantization=None) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """ Return tokenizer and llm model"""
    model_id, use_quantization = get_modelid_quantconfig()
    model_id = model_id if name is None else name
    use_quantization = use_quantization if use_quantization is None else use_quantization

    attn_implementation = get_fask_attention()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if use_quantization else None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     torch_dtype=torch.float16,
                                                     quantization_config=quantization_config,
                                                     low_cpu_mem_usage=False,
                                                     attn_implementation=attn_implementation)

    if not quantization_config:
        llm_model.to(DEVICE)

    return tokenizer, llm_model


def _retrieve_relevant_resources(query: str,
                                 embeddings: torch.tensor,
                                 model: SentenceTransformer,
                                 n_resources_to_return: int = 5,
                                 print_time: bool = True):

    query_embedding = model.encode(query, convert_to_tensor=True)

    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {
              len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

    return scores, indices


def _prompt_formatter(query: str,
                      context_items: list[dict], tokenizer: AutoTokenizer) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"]
                                 for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
You must follow this context to answer: {context}
Question: {query}
"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
         "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt


def ask(query, pages_and_chunks, embeddings,
        tokenizer: AutoTokenizer, llm_model: AutoModelForCausalLM,
        temperature=0.7, max_new_tokens=512,
        format_answer_text=True, return_answer_only=True):

    scores, indices = _retrieve_relevant_resources(
        query=query, embeddings=embeddings)

    context_items = [pages_and_chunks[i] for i in indices]

    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu()

    prompt = _prompt_formatter(query=query, context_items=context_items)

    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = llm_model.generate(**input_ids, temperature=temperature,
                                 do_sample=True, max_new_tokens=max_new_tokens)

    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace(
            "<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

    if return_answer_only:
        return output_text

    return output_text, context_items


if __name__ == "__main__":
    ...