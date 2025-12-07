# Machine Learning fundamentals

This repository contains implementations of common machine learning algorithms. By default, most algorithms use Kaggle competition data for training (requires a [kaggle.com](https://www.kaggle.com) account).

To run the code, you would need:
- a Linux system or a [Windows Subsystem for Linux](https://ubuntu.com/desktop/wsl)
- a local Python virtual environment<br>
	(can be configured using [`./resources/configure-venv.sh`](https://github.com/mkmaslov/ML-fundamentals/blob/main/resources/configure-venv.sh) script)
- Kaggle config file in `./.kaggle/kaggle.json` with the following contents:
	```
	{"username":"<kaggle_username>","key":"<kaggle_token>"}
	```
	To get a new Kaggle token, go to: `kaggle.com → Account → Settings → API → Create New Token`. After creating the file, run `chmod 600 ./.kaggle/kaggle.json` to restrict access permissions to the current user.
- use [`./resources/kaggle.sh`]() script to download the competition data:
	```
	./resources/kaggle.sh <competition_handle>
	```
	To make the script executable, run `chmod +x ./resources/kaggle.sh`. Data will be stored in `./data` directory.

**WARNING: before pushing to the repository, make sure that `./data` and `./.kaggle` folders are in the `.gitignore` file (as well as other private/useless data)!**
