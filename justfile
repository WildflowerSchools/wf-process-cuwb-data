build:
    poetry build

publish: build
    poetry publish

install:
    poetry install

lint:
    pylint process_cuwb_data

test:
    pytest -s

format:
    black process_cuwb_data

version:
    poetry version