# use this script to freeze all packages in the virtual environment

import os
import platform
import subprocess
import sys

ENV_NAME = '.venv'

def get_pip_path():
    os_type = platform.system()
    if os_type == "Windows":
        pip_path = os.path.join(ENV_NAME, "Scripts", "pip.exe")
    else:
        pip_path = os.path.join(ENV_NAME, "bin", "pip")

    if not os.path.isfile(pip_path):
        print("Could not find pip in your virtual environment.")
        print(f"Make sure the venv {ENV_NAME} is created in '{ENV_NAME}/'")
        sys.exit(1)

    return pip_path


def freeze_requirements(pip_path):
    print(f"Freezing packages in requirements.txt")
    with open("requirements.txt", "w") as f:
        subprocess.run([pip_path, "freeze"], stdout=f, check=True)


def main():
    pip_path = get_pip_path()
    freeze_requirements(pip_path)


if __name__ == '__main__':
    main()