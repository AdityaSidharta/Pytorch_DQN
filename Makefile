SHELL := /bin/bash

help:
	@echo "setup - setup pyenv and pipenv"

setup:
	bash libs/setup.sh
	pipenv shell

format:
	black .
	nbstripout */*