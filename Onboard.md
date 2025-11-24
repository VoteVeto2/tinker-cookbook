# Onboard file 

## Seting up `uv` env. 

It's recommended to use either `conda` or `uv`, we will stick to `uv` for this repo. Follow the steps below

```bash
uv venv --python 3.11 .venv
uv pip install tinker
uv pip install -e . # e. means
```
## Testing api-key 

1. Set up your api key in `.env` 
2. run the following commands 

    - Activate the uv venv. 
    ```bash
    .venv\Scripts\activate
    ```

    - run the following python scripts
    ```bash
    uv run --env-file .env python -m tinker_cookbook.recipes.rl_basic
    

    or 

    
    uv run --env-file .env python -m tinker_cookbook.recipes.sl_basic
    ```


  $$\nabla \mathbb{E}{x\sim p\theta}\bigl[r(x) \bigr] = \mathbb{E}{x\sim q}\Bigl[r(x) \cdot \frac{\nabla p\theta(x)}{q(x)}\Bigr]$$