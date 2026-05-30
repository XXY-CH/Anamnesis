# Tests

This directory contains unit and integration smoke tests.

## Files

- [test_aligned_architecture.py](file:///Users/xiexingyu/Documents/项目/Resources/tests/test_aligned_architecture.py) - Verification of Engram, Block AttnRes, milestone snapshot, full model, and toy training-step checks.
- [test_budget_contracts.py](file:///Users/xiexingyu/Documents/项目/Resources/tests/test_budget_contracts.py) - Mathematical budget contracts validation for learning and parameter bounds.
- [test_paper_faithfulness.py](file:///Users/xiexingyu/Documents/项目/Resources/tests/test_paper_faithfulness.py) - Faithfulness check to verify the implementation matches the LaTeX draft formulations.
- [test_recurrent_step.py](file:///Users/xiexingyu/Documents/项目/Resources/tests/test_recurrent_step.py) - Recurrent step consistency checks, comparing parallel, recurrent, and chunkwise modes.
- [test_retention.py](file:///Users/xiexingyu/Documents/项目/Resources/tests/test_retention.py) - Retention decay mechanism, multi-head gamma configurations, and state shape checks.
- [conftest.py](file:///Users/xiexingyu/Documents/项目/Resources/tests/conftest.py) - Test import path and environment setup.

## Running Tests

Run all tests with:

```bash
pytest
```

