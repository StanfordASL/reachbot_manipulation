# Installing Pyenv

```
curl https://pyenv.run | bash
```

Then, set up `~/.bashrc` -- Make sure the following lines are included

```
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

If the Pyenv Python install fails and warns about things not being installed, run this command to make sure the dependencies are up to date. (Then, retry the command that failed)

```
sudo apt-get update; sudo apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```
