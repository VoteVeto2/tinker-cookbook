# Onboard file 

## Seting up `uv` env. 

It's recommended to use either `conda` or `uv`, we will stick to `uv` for this repo. Follow the steps below

```bash
uv venv --python 3.11 .venv
uv pip install tinker
uv pip install -e . # e flag installs the package in **e**ditable mode
```
## Testing api-key 

1. Set up your api key in `.env` 
```bash
TINKER_API_KEY="Your-api-keys"
```
2. run the following commands 
```bash
# Activate the uv venv. 
.venv\Scripts\activate

uv run --env-file .env python -m tinker_cookbook.recipes.rl_basic

uv run --env-file .env python -m tinker_cookbook.recipes.sl_basic
```
