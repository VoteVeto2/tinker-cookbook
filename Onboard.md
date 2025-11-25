# Onboard file 

## Setting up `uv` env

It's recommended to use either `conda` or `uv`, we will stick to `uv` for this repo. Follow the steps below:

```bash
uv venv --python 3.11 .venv
uv pip install tinker
uv pip install -e . # e flag installs the package in **e**ditable mode
```

## Testing API Key 

1. Set up your api key in `.env`:
```bash
TINKER_API_KEY="Your-api-keys"
```

2. Run the following commands:
```bash
# Activate the uv venv (Windows)
.venv\Scripts\activate

# Run training recipes
uv run --env-file .env python -m tinker_cookbook.recipes.rl_basic
uv run --env-file .env python -m tinker_cookbook.recipes.sl_basic
```

---

## Saved Checkpoints

### Qwen3-4B-Instruct-2507
- **State path** (for continuing training): `tinker://f0297fe1-2fac-5f60-b2ed-8f3ff82c5dc4:train:0/weights/final`
- **Sampler path** (for inference): `tinker://f0297fe1-2fac-5f60-b2ed-8f3ff82c5dc4:train:0/sampler_weights/final`

### Llama-3.2-1B
- **State path** (for continuing training): `tinker://ddc36680-656b-5749-ac14-757cf8bbdcfe:train:0/weights/final`
- **Sampler path** (for inference): `tinker://ddc36680-656b-5749-ac14-757cf8bbdcfe:train:0/sampler_weights/final`

---

## Using the Trained Model

After training a model with Tinker, you have two options for inference:
1. **Remote inference** - Use Tinker's sampling client (recommended for quick testing)
2. **Local inference** - Download weights and run locally with transformers/PEFT

---

## Option 1: Remote Inference (Tinker Sampling Client)

Run inference directly on Tinker's servers without downloading weights. This is the fastest way to test your trained model.

### Using the Chat CLI

```bash
# Chat with Qwen3-4B-Instruct model
uv run --env-file .env python -m tinker_cookbook.chat_app.tinker_chat_cli \
    model_path=tinker://f0297fe1-2fac-5f60-b2ed-8f3ff82c5dc4:train:0/sampler_weights/final \
    base_model=Qwen/Qwen3-4B-Instruct-2507

# Chat with Llama-3.2-1B model
uv run --env-file .env python -m tinker_cookbook.chat_app.tinker_chat_cli \
    model_path=tinker://ddc36680-656b-5749-ac14-757cf8bbdcfe:train:0/sampler_weights/final \
    base_model=meta-llama/Llama-3.2-1B
```

**Chat CLI Options:**
- `max_tokens` (default: 512) - Maximum tokens to generate
- `temperature` (default: 0.7) - Controls randomness (higher = more random)
- `top_p` (default: 0.9) - Nucleus sampling threshold

### Custom Remote Inference Script

For programmatic access, create a script like this:

```python
import asyncio
import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer

async def remote_inference():
    # Configuration
    base_model = "Qwen/Qwen3-4B-Instruct-2507"  # or your model
    sampler_path = "tinker://f0297fe1-2fac-5f60-b2ed-8f3ff82c5dc4:train:0/sampler_weights/final"
    
    # Create sampling client
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=base_model,
        model_path=sampler_path,
    )
    
    # Setup renderer
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(get_recommended_renderer_name(base_model), tokenizer)
    
    # Build prompt
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    model_input = renderer.build_generation_prompt(messages)
    
    # Generate response
    sampling_params = types.SamplingParams(
        max_tokens=512,
        temperature=0.7,
        stop=renderer.get_stop_sequences(),
    )
    
    response = await sampling_client.sample_async(
        prompt=model_input, num_samples=1, sampling_params=sampling_params
    )
    
    # Parse and print response
    parsed_message, _ = renderer.parse_response(response.sequences[0].tokens)
    print(f"Response: {parsed_message['content']}")

if __name__ == "__main__":
    asyncio.run(remote_inference())
```

---

## Option 2: Local Inference (Download Weights)

Download the trained LoRA weights to run inference locally. This is useful for offline usage or integration into other pipelines.

### Step 1: Download Weights

Use the `rl_basic_Qwen3-4B-Instruct_inference.py` script to download the checkpoint:

```bash
# Make sure your .env file has TINKER_API_KEY set
uv run --env-file .env python rl_basic_Qwen3-4B-Instruct_inference.py
```

**What the script does:**
1. Connects to Tinker using your API key
2. Gets the download URL for the checkpoint archive
3. Downloads `model-checkpoint.tar.gz` to the current directory

**To download a different model**, edit the `sampler_path` variable in the script:
```python
# For Llama-3.2-1B:
sampler_path = "tinker://ddc36680-656b-5749-ac14-757cf8bbdcfe:train:0/sampler_weights/final"
```

### Step 2: Extract the Weights

```bash
# Extract the checkpoint archive
tar -xzf model-checkpoint.tar.gz -C model-Qwen3-4B-Instuct-checkpoint
```

The extracted folder should contain:
- `adapter_config.json` - LoRA adapter configuration
- `adapter_model.safetensors` - LoRA weights
- `checkpoint_complete` - Marker file indicating successful download

### Step 3: Run Local Inference

Use the `test-Qwen3-4B.py` script to test the model locally:

```bash
# Install required packages (if not already installed)
uv pip install transformers peft torch accelerate

# Run the test script
uv run python test-Qwen3-4B.py
```

**What the script does:**
1. Loads the base model (`Qwen/Qwen3-4B-Instruct-2507`)
2. Applies the LoRA adapter from `./model-Qwen3-4B-Instuct-checkpoint`
3. Generates responses to test prompts

**To modify for different models:**

```python
# Edit these variables in test-Qwen3-4B.py:
BASE_MODEL = "meta-llama/Llama-3.2-1B"  # Change to your base model
LORA_PATH = "./model-Llama3.2-1B-checkpoint"  # Path to extracted weights
```

### Local Inference Template

Here's a minimal template for local inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Configuration
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_PATH = "./model-Qwen3-4B-Instuct-checkpoint"

# Load model with LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# Generate response
def generate(prompt: str, max_new_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# Test it
print(generate("What is machine learning?"))
```


---

