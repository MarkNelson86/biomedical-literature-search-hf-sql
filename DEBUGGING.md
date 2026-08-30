# Debugging Log

## Environment setup

### Issue 1 — Jupyter kernel used wrong Python environment

- Status: Resolved
- Error: `ModuleNotFoundError: No module named 'sentence_transformers'`
- Cause: Jupyter was using `/Users/mheado86/anaconda3/bin/python` rather than the project's `.venv`.
- Fix: Registered the project virtual environment as a Jupyter kernel and selected it for the notebook.
- Verification: `sys.executable` now points to the project's `.venv`.


### Issue 2 — Incompatible ML dependency stack on Intel macOS

- Status: Resolved
- Symptoms:
  - PyTorch 2.2.2 failed against NumPy 2.x.
  - Current Transformers required PyTorch >= 2.5.
  - `pip install --upgrade torch` could not obtain a newer compatible macOS x86-64 build.
- Cause: Unconstrained dependencies produced mutually incompatible package versions on Intel macOS.
- Fix: Used a compatible dependency stack with PyTorch 2.2.2, NumPy <2, Transformers 4.46.3, and Sentence Transformers 3.4.1.


### Issue 3 — Missing DeBERTa tokenizer dependencies

- Status: Resolved
- Error: DeBERTa tokenizer failed to initialize because `protobuf` was unavailable.
- Cause: The zero-shot classification model requires tokenizer dependencies that were not declared in the project environment.
- Fix: Added `protobuf` and `sentencepiece` to the environment and verified that the zero-shot Hugging Face pipeline loads successfully.
