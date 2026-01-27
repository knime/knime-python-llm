# -*- coding: utf-8 -*-
import sys

# AP-25545: Monkey-patch scipy.stats.stats to fix Giskard import error with newer scipy versions
# The error is: ImportError: cannot import name 'Ks_2sampResult' from 'scipy.stats.stats'
def apply_patch():
    try:
        import scipy.stats.stats as stats
    except ImportError:
        from types import ModuleType

        stats = ModuleType("scipy.stats.stats")
        sys.modules["scipy.stats.stats"] = stats

    if not hasattr(stats, "Ks_2sampResult"):
        try:
            from scipy.stats._stats_py import Ks_2sampResult

            stats.Ks_2sampResult = Ks_2sampResult
        except ImportError:
            pass


apply_patch()
