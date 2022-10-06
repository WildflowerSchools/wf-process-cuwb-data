build:
    poetry build

publish: build
    poetry publish

install-dev:
    poetry install

test:
    pytest -s

fmt:
    autopep8 --aggressive --recursive --in-place ./process_cuwb_data/
