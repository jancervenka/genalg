
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pkg_resources

__module_name__ = 'genalg'

try:
    __version__ = pkg_resources.get_distribution(__module_name__).version
except Exception:
    __version__ = 'unknown'
