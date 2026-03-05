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

## Connecting with vs code
