"""Public PlantEGP entry point.

The NAR-oriented interface is the default public application. The prior
interface is preserved verbatim in ``legacy_app.py`` for audit and rollback;
it is not the public NAR entry point.
"""

# Importing the Streamlit script executes its page definition in this process.
from nar_app import *  # noqa: F401,F403
