
import logging
logger = logging.getLogger("tinytroupe")

###########################################################################
# Exposed API
###########################################################################
from .proposition import Proposition, check_proposition, compute_score

__all__ = ["Proposition", "check_proposition", "compute_score"]