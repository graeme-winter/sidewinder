import configparser
import os
import subprocess


def get_config(config_name):
    config = configparser.ConfigParser()
    config.read(
        os.path.join(os.environ["HOME"], ".config", "sidewinder", "spot_finder.conf")
    )
    return config[config_name]


if __name__ == "__main__":
    host = (
        subprocess.run([b"uname", b"-n"], stdout=subprocess.PIPE)
        .stdout.decode()
        .strip()
        .split(".")[0]
        .lower()
    )

    config = get_config(host)

    print(f"Host: {host}")
    for item in "nproc", "devices", "work":
        print(f"{item} = {config[item]}")
