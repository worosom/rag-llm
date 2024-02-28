import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from langchain_community.llms import HuggingFacePipeline

# ### Load quantized Mixtral 8x7B

#################################################################
# Tokenizer
#################################################################

def load_llms(model_name='mistralai/Mixtral-8x7B-Instruct-v0.1'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    #################################################################
    # bitsandbytes parameters
    #################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    #################################################################
    # Set up quantization config
    #################################################################
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    #################################################################
    # Load pre-trained config
    #################################################################
    mistral_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir="/raid/alex/cache",
        revision="refs/pr/5",
    )


    # ### Count number of trainable parameters

    # In[ ]:


    def print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


    # ### Build Mistral text generation pipelines

    # In[5]:


    standalone_query_generation_pipeline = pipeline(
     model=mistral_model,
     tokenizer=tokenizer,
     task="text-generation",
     temperature=0.0,
     repetition_penalty=1.1,
     return_full_text=True,
     max_new_tokens=1000,
    )
    standalone_query_generation_llm = HuggingFacePipeline(pipeline=standalone_query_generation_pipeline)

    response_generation_pipeline = pipeline(
     model=mistral_model,
     tokenizer=tokenizer,
     task="text-generation",
     temperature=0.2,
     do_sample=True,
     repetition_penalty=1.1,
     return_full_text=True,
     max_new_tokens=1000,
    )
    response_generation_llm = HuggingFacePipeline(pipeline=response_generation_pipeline)

    return standalone_query_generation_llm, response_generation_llm
