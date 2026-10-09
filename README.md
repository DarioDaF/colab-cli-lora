# Run sequence

```sh
. .dev.bashrc # Include `.bin` in PATH
uv run run_colab.py --sid qlora-session --start --keep # Start a GPU instance, upload and run the code
colab-rsync qlora-session remote:/artifacts/ ./artifacts # Download results
colab --session qlora-session stop # Close colab session
```

Staring the sketch in an already running session can be done by not passing `--start` for iterating
