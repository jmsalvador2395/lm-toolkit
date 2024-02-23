import peft
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model(cfg):

    model_name = cfg.params['base_model']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cfg.paths['cache'],
    )
    #self.tokenizer.add_special_tokens({'pad_token': '<pad>'})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_name == 'mistralai/Mistral-7B-v0.1':
        target_modules = ['q_proj', 'k_proj']
    elif 'gpt2' in model_name:
        target_modules = ['c_proj']
    else:
        raise ValueError(f'invalid model name {model_name}')

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=target_modules,
        r=cfg.params['lora_r'],
        lora_alpha=cfg.params['lora_alpha'],
        lora_dropout=cfg.params['lora_dropout'],
    )
        
    model = peft.get_peft_model(base_model, peft_config)

    return tokenizer, model
