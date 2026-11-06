# SPDX-License-Identifier: BSD-3-Clause
import sys

from ernie._ernie import run_cli


def main() -> int:
    return run_cli(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
