# SPDX-FileCopyrightText: 2022-present manuel-k2 <kreutle@princeton.edu>
#
# SPDX-License-Identifier: MIT

"""
__init__.py
====================================
The core module of my example project
"""

from __future__ import annotations

from typing import TypeVar

Self = TypeVar("Self", bound="LabUnc")


class LabUnc:
    """An example docstring for a class definition."""

    @staticmethod
    def combine(a: float, b: float) -> float:
        """
        Computes a+b.

        :param a: The input value.
        :param b: The input value.
        :return: sum of a and b, often called y.

        Usage::

            >>> a = 1, b = 2
            >>> y = 3
        """
        return a + b

    rounding_rule = 1.0
    "This is the number to round at for display,"
    "lab rule is 1, particle physics uses 3.54"
