# SPDX-License-Identifier: BSD-3-Clause
import argparse
import sys

import ernie


def main() -> int:
    parser = argparse.ArgumentParser(prog="ernie")
    parser.add_argument("--version", action="version", version=f"ernie {ernie.__version__}")
    parser.parse_args()
    for adapter in ernie.list_adapters():
        unsupported = f", unsupported: {adapter.unsupported}" if adapter.unsupported else ""
        print(f"{adapter.alias}: {adapter.name} ({ernie.backend_name(adapter.backend)}){unsupported}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
