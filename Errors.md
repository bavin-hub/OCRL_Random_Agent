This file contains all error that were face along with their solutions



https://github.com/julienokumu/unitree_rl_mugym
1. in unitree_rl_mugym the rsl_rl version was new which incompatible. has to reinstall pytorch and rsl_rl again

	-> pip uninstall rsl_rl -y
	-> pip install git+https://github.com/leggedrobotics/rsl_rl.git@v3.1.0
	-> pip uninstall torch torchvision torchaudio -y
	-> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
