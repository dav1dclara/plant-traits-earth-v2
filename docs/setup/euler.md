# Instructions for Euler setup

VPN is required when working from off-campus. From a PRS workstation, you can skip the VPN.

## 1. Euler connection

Generate an SSH key on your laptop (skip if you already have one):

```bash
ssh-keygen -t ed25519 -C "<username>@laptop"
```

Copy it to Euler:

```bash
ssh-copy-id <username>@euler.ethz.ch
```

Test:

```bash
ssh <username>@euler.ethz.ch
```

## 2. Project storage

Shared data lives in `/cluster/work/igp_psr/plant-traits-earth-v2`. All three of us have access via ACLs.

Where things go:

- Code → `$HOME` (backed up, fast for git)
- Datasets, checkpoints, outputs → `/cluster/work/igp_psr/plant-traits-earth-v2/`
- Heavy I/O during jobs → `$TMPDIR` on the compute node
- Quick scratch → `/cluster/scratch/$USER` (auto-deleted after 15 days)

Store datasets as `.zip` archives, not loose files — LUSTRE handles large files well but degrades with millions of small ones. In Python, `zipfile.Path` is a drop-in replacement for `pathlib.Path`.

Check quota with `lquota /cluster/work/igp_psr`.


## 3. Cloning the repo

On Euler, generate a separate SSH key for GitHub:

```bash
ssh-keygen -t ed25519 -C "<username>@euler"
cat ~/.ssh/id_ed25519.pub
```

Copy the output and add it on GitHub under Settings → SSH and GPG keys.

Test and clone into your home directory:

```bash
ssh -T git@github.com
cd ~
git clone git@github.com:dav1dclara/plant-traits-earth-v2.git
```

## 4. Environment

Load the required modules:

```bash
module purge
module load stack/2025-06
module load gcc/12.2.0
module load python/3.13.0
```

Save the module set so you don't have to retype it every session:

```bash
module save plant-traits
```

Create a venv in your home directory and activate it:

```bash
python -m venv ~/venvs/plant-traits
source ~/venvs/plant-traits/bin/activate
```

Install dependencies from the project root (run this on the **login node**, not a compute node — compute nodes don't have internet access):

```bash
cd ~/plant-traits-earth-v2
pip install -r requirements_euler.txt
```

Verify the install:

```bash
python -c "import torch; print(torch.__version__)"
```

**Each new session**, restore your environment with:

```bash
module restore plant-traits
source ~/venvs/plant-traits/bin/activate
```

## 5. Euler Tunnel for VS Code

Never connect VS Code directly to `euler.ethz.ch` — it overloads the login node. Use Euler Tunnel to route into a compute node instead.

To configure Euler tunnels, run on Euler:

```bash
euler-tunnel config
```

This prints two blocks. On your laptop, append the first block to `~/.ssh/known_hosts` and paste the second block at the bottom of `~/.ssh/config`:

```
Host euler-tunnel
   User <username>
   ServerAliveInterval 10
   ServerAliveCountMax 10
   ProxyCommand ssh <username>@euler.ethz.ch euler-tunnel connect
   ControlMaster auto
   ControlPath ~/.ssh/cs-%r@%h:%p
   ControlPersist 15
```

Windows users: remove the three `Control*` lines.

### Daily routine

SSH into Euler and start a tunnel job:

```bash
ssh euler
euler-tunnel start --time=8:00:00 --cpus-per-task=4 --mem-per-cpu=4G
euler-tunnel status
```

Add `--gpus=1` if you need a GPU. Wait until status is `R` (running), then you can exit Euler — the job keeps running.

From your laptop:

```bash
ssh euler-tunnel
```

Or in VS Code: `Cmd/Ctrl + Shift + P` → "Remote-SSH: Connect to Host..." → `euler-tunnel`.

The job ends automatically at the `--time=` limit. To kill it early: `scancel <jobid>`.

## 5. Environment

Load the required modules:

```bash
module purge
module load stack/2025-06
module load gcc/12.2.0
module load python/3.13.0
```

Create a venv in your home directory and activate it:

```bash
python -m venv ~/venvs/plant-traits
source ~/venvs/plant-traits/bin/activate
```

## Cheatsheet

```bash
squeue -u $USER        # your running/queued jobs
scancel <jobid>        # cancel a job
sshare -u $USER        # Fairshare score
lquota <path>          # storage quota
euler-tunnel status    # check tunnel job
```


## More information

- [HPC documentation](https://docs.hpc.ethz.ch)
- [Detailed IT Onboarding @ PRS](https://docs.google.com/document/d/1u6oCZzSluQupjBEezT37qsEkngOVFssF8n4TQuPwSCU/edit?tab=t.0#heading=h.dbg4wjyk2hti)
- [PRS Euler usage guidelines](https://www.notion.so/PRS-Euler-usage-guidelines-33410860e8dd80b388acf8e1147b7f05)
