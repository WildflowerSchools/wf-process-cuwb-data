build:
    poetry build

publish: build
    poetry publish

install-dev:
    poetry install

test:
    pytest -s

fmt:
    black process_cuwb_data
