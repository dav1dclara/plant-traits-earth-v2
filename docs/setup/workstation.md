# Connection to work station

We have got access to the workstation `pf-pc23`. Connect to it from the terminal with:
```bash
ssh <username>@pf-pc23.ethz.ch
```

## Cloning the repo

We have created a directory `/scratch/plant-traits-v2/` for this project. Inside that directory, each one of us has a folder, e.g. `dclara/` where each one of us should clone the repo from github.

First, generate a new ssh key by running:
```bash
ssh-keygen -t ed25519 -C <your_email>
```

To see the public key, run:
```bash
cat ~/.ssh/id_ed25519.pub
```

Copy it and paste it on Github -> Settings -> SSH and GPG keys -> New SSH key. You can then test the connection with

```bash
ssh -T git@github.com
```

You can now clone the repo to your personal folder on the scratch with:
```bash
git clone git@github.com:dav1dclara/plant-traits-earth-v2.git
```

## Install miniconda

Everyone has to install miniconda. Go to your dir on the scratch, e.g., `/scratch/plant-traits-v2/dclara/` and run the following commands:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

In the installation process, change the miniconda installation dir to `/scratch/plant-traits-v2/dclara/miniconda3`. Then initialize and reload the bash:

```bash
/scratch/plant-traits-v2/<user>/miniconda3/bin/conda init
source ~/.bashrc
```

Create a new env for yourself following the instructions in the README.md.
