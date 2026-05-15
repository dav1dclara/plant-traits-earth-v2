# Why we should clean up git history

Our `.git` folder is currently **215MB** — almost entirely due to ML result artifacts
(`scripts/__temp_dev/`) and viz outputs that were committed before the gitignore rules
were in place. The files are already gone from the working tree, but git still stores
every blob in its history.

## Impact

- **Slow git operations** — every `git status`, branch switch, or clone has to scan
  through 210MB of packfiles containing numpy arrays, model checkpoints, and notebook outputs.
- **Slow onboarding** — new contributors clone the full history, downloading 200MB+ of
  files they'll never use.
- **Wasted storage** — on every machine and on the remote.

## The fix

A one-time history rewrite with `git filter-repo` drops the large blobs permanently.
Expected result: repo shrinks from ~215MB to a few MB.

**What's needed:**
1. One person runs `git filter-repo` and force-pushes.
2. All collaborators re-clone (old clones will have diverged history).

The files themselves don't need to move — they're already ignored and absent from the
working tree. This is purely a git housekeeping task.
