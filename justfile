default: test

# run all tests
test:
    uv run --group dev pytest
