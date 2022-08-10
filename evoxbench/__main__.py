import argparse
import os, sys
from pathlib import Path
import json
from django.core.management import execute_from_command_line

CONFIG_FILE_DIR = Path.home() / '.config' / 'evoxbench'
CONFIG_FILE_PATH = CONFIG_FILE_DIR / 'config.json'


def get_config():
    conf = CONFIG_FILE_PATH
    if conf.exists():
        with open(conf, 'r') as f:
            return json.load(f)
    else:
        with open(conf, 'w') as f:
            json.dump(dict(), f)
    return dict()


def save_config(new_conf):
    conf = CONFIG_FILE_PATH
    with open(conf, 'w') as f:
        json.dump(new_conf, f)


def config_callback(args):
    if args.database:
        conf = get_config()
        conf['database'] = args.path
        save_config(conf)
    else:
        conf = get_config()
        conf['model'] = args.path
        save_config(conf)


def manage_callback(_):
    execute_from_command_line(sys.argv[1:])


def server_callback(args):
    config = get_config()
    data_path = config.get("database", "")
    if not data_path:
        print("You need to config database first!\n Please run evobench config --database PATH and try again!")
        exit(0)
    print(f"RPC Server running on {args.address}:{args.port}")
    from evoxbench.api import rpc
    # from evoxbench.database import init
    rpc.start_server(data_path, (args.address, args.port))


def get_args():
    description = "A benchmark for NAS algorithms"
    main_parser = argparse.ArgumentParser(description=description)
    # main_parser.add_argument('type', choices=['manage', 'server', 'config'])
    subparsers = main_parser.add_subparsers()

    config = subparsers.add_parser("config")
    config_group = config.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--database", action='store_true')
    config_group.add_argument("--model", action='store_true')
    config.add_argument("--path", default=str(os.getcwd()), required=False)
    config.set_defaults(func=config_callback)

    server = subparsers.add_parser("server")
    server.add_argument("--address", default="127.0.0.1")
    server.add_argument("--port", type=int, default=9876)
    server.set_defaults(func=server_callback)

    manage = subparsers.add_parser("manage")
    manage.set_defaults(func=manage_callback)
    print(sys.argv)

    return main_parser.parse_known_args()


def main():
    _ = get_args()[0]
    try:
        _.func(_)
    except Exception:
        print("Wrong use of command line!")


if __name__ == '__main__':
    x = get_args()[0]
    print(x)
    print("out")
    x.func(x)
    # print(x.config)
