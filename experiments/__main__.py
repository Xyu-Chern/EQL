"""Run a training protocol from the repository root."""
import argparse
import runpy
import sys
from pkgutil import iter_modules

import experiments


def main():
    commands = sorted(item.name for item in iter_modules(experiments.__path__) if not item.name.startswith("_"))
    parser = argparse.ArgumentParser(description=__doc__, usage="python -m experiments PROTOCOL [OPTIONS]")
    parser.add_argument("protocol", choices=commands)
    parser.add_argument("options", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module = f"experiments.{args.protocol}"
    sys.argv = [module, *args.options]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
