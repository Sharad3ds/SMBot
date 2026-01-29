"""
Python Import Statements - Brief Overview

1. Basic Import
   import module_name
   # Access: module_name.function()

2. Import Specific Items
   from module_name import function_name, Class_name
   # Access directly: function_name()

3. Import All (use sparingly)
   from module_name import *
   # Imports all public names

4. Import with Alias
   import module_name as alias
   # Access: alias.function()

5. Import Specific with Alias
   from module_name import function_name as fn
   # Access: fn()

Common Examples:
--------------
import math
from datetime import datetime, date
import pandas as pd
from os import path as p
from collections import *

Best Practices:
---------------
- Use specific imports when possible
- Avoid import * (can cause namespace pollution)
- Use aliases for long module names
- Group imports: standard library, third-party, local
"""
