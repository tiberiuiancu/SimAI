# Build Scripts

## Path Configuration

The SimAI C++ code originally hardcoded log and result paths to `/etc/astra-sim/`. To make these paths configurable at runtime, we use a build-time patching approach.

### Files Patched

- `patch_paths.sh` - Patches vendor files to use relative paths (applied during build in GitHub Actions)
- `restore_paths.sh` - Restores vendor files from backups (for local testing)

### How It Works

During the GitHub Actions build process:
1. The `patch_paths.sh` script temporarily modifies vendor files to:
   - Replace hardcoded `/etc/astra-sim/` with `./results/`
   - Add conditional compilation guards (`#ifndef`) to C++ defines
   - Update config files to use relative paths

2. The binaries are built with these relative paths

3. At runtime, paths are relative to the working directory where the binary is executed

### Usage

For local testing:
```bash
# Apply patches
./scripts/patch_paths.sh

# Build your code...

# Restore original vendor files
./scripts/restore_paths.sh
```

For custom paths at compile time:
```bash
# Apply patches with custom paths
./scripts/patch_paths.sh vendor/simai/astra-sim-alibabacloud "./custom/path/" "./custom/results/"
```

### Note

The vendor directory is **not** modified in git - only temporarily during builds. This keeps the vendor submodule clean while allowing runtime path configuration.
