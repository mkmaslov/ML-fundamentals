# Machine Learning fundamentals

This repository contains implementations of common machine learning algorithms. By default, most algorithms use Kaggle competition data for training (requires a [kaggle.com](https://www.kaggle.com) account).

To run the code, you would need:
- a Linux system or a [Windows Subsystem for Linux](https://ubuntu.com/desktop/wsl)
- a local Python virtual environment<br>
	(can be configured using [`./resources/configure-venv.sh`](https://github.com/mkmaslov/ML-fundamentals/blob/main/resources/configure-venv.sh) script)
- Kaggle API token (to create it, go to: `kaggle.com → Account → Settings → API → Create New Token`)
- use [`./resources/kaggle.sh`]() script to download the competition data:
	```
	./resources/kaggle.sh <competition_handle>
	```
	To make the script executable, run `chmod +x ./resources/kaggle.sh`. Data will be stored in `./data` directory.

**WARNING: before pushing to the repository, make sure that `./data` and `./.kaggle` folders are in the `.gitignore` file (as well as other private/useless data, like a `./.vscode` folder)!**
