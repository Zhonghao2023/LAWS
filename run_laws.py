#!/usr/bin/env python3
"""

LAWS is based on and extends the Community Water Model (CWatM). It is
distributed under the GNU General Public License, version 3 or later.
"""

from laws import (
    __author__,
    __copyright__,
    __date__,
    __email__,
    __maintainer__,
    __status__,
    __version__,
)
from laws.run_laws import GNU, main, mainwarm, parse_args, usage

__all__ = [
    "GNU",
    "main",
    "mainwarm",
    "parse_args",
    "usage",
]


if __name__ == "__main__":
    settings, arguments = parse_args()
    main(settings, arguments)
