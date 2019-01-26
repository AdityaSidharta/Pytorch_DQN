#!/usr/bin/env bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev
sudo apt install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

# Install pyenv and its dependencies
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
pyenv update

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

pyenv install 3.6.7
pyenv local 3.6.7
pip install pipenv
pipenv install --dev

pipenv run pip install pip==18.0
pipenv run pip install black
pipenv run pip install 'gym[box2d]'
pipenv run pip install ipywidgets
pipenv run pip install nbstripout

pipenv run python -m ipykernel install --user --name=Pytorch_DQN
pipenv run jupyter nbextension enable --py widgetsnbextension
