# patches

Patches applied to `extern/` submodules. We do **not** edit submodule source in-tree; changes live here and are applied via `scripts/apply_patches.sh`.

Layout:

- `psi0/` — patches against `extern/psi0`
- `simple/` — patches against `extern/simple`

Each patch should be a unified diff (`git format-patch` or `git diff`) relative to the submodule root.
