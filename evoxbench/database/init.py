import json
import os
import sys
from pathlib import Path
import traceback

import django

CONFIG_FILE_DIR = Path.home() / '.config' / 'evoxbench'
CONFIG_FILE_PATH = CONFIG_FILE_DIR / 'config.json'


def auto_config():
    if not CONFIG_FILE_PATH.exists() or not CONFIG_FILE_PATH.is_file():
        print("First time running evoxbench, please config the database path manually!")
        return
    try:
        with open(CONFIG_FILE_PATH, "r", encoding='utf-8') as f:
            if not f.readable():
                print(f"Failed to read auto configuration file, path {CONFIG_FILE_PATH}")
                return

            config = json.load(f)
            os.environ.setdefault("EVOXBENCH_DATA", config['database'])
            if "model" in config:
                os.environ.setdefault("EVOXBENCH_MODEL", config['model'])
            sys.path.append(config['database'])
            os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', "ORM.settings")
            django.setup()
            print(f"Auto Configuration Succeed!, Using database {config['database']}.")
    except Exception as e:
        traceback.print_exc()
        print(f"Auto Configuration file corrupted, path {CONFIG_FILE_PATH}, reason: {str(e)}")


def init(cwd: str = ""):
    cwd = cwd or str(Path(os.getcwd()) / "data")
    os.environ.setdefault("EVOXBENCH_DATA", str(cwd))
    sys.path.append(cwd)
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', "ORM.settings")
    django.setup()


def config(database_path: str, data_path: str):
    CONFIG_FILE_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE_PATH.touch()
    if CONFIG_FILE_PATH.is_file():
        try:
            config = {
                'database': database_path,
                'model': data_path
            }
            with open(CONFIG_FILE_PATH, "w", encoding='utf-8') as f:
                json.dump(config, f)
                print('Configuration Succeed!')
                init(database_path)
                return
        except Exception as e:
            raise e
            print(f"Configuration Failed!\n{str(e)}")
    print(f"Configuration Failed, it's not a regular file")
