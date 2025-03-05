import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import LLM_MODEL_NAME

def load_model_on_gpus():
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)

    # If multiple GPUs are available, wrap the model in DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer
