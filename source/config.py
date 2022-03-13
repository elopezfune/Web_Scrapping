from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent

# Data Path
# =========
PATH_TO_DATA = PROJECT / "data/events_log.csv"

# Seed for reproducibility
# ========================
SEED = 42


# Time variable
# =============
VAR_TIME = 'Timestamp'


# CTR variable
# ============
ACTION = 'Action'


# Session ID variable
# ===================
SESSION_ID = 'Session_Id'


# Visit Page variable
# ===================
VISIT_PAGE = 'visitPage'


# Search Result Page variable
# ===========================
SEARCH_RESULT_PAGE = 'searchResultPage'

# Number of Results variable
# ==========================
NUM_RESULTS = 'N_Results'





# Models
# ======
# Click Through Rate Models
CTR_MODEL_ALL = PROJECT / "models/ctr_all.sav"
CTR_MODEL_GRA = PROJECT / "models/ctr_gra.sav"
CTR_MODEL_GRB = PROJECT / "models/ctr_grb.sav"

# First Results Models
FR_MODEL_R01 = PROJECT / "models/fr_r01.sav"
FR_MODEL_R02 = PROJECT / "models/fr_r02.sav"
FR_MODEL_R03 = PROJECT / "models/fr_r03.sav"

# Zero Result Rate Models
ZRR_MODEL_ALL = PROJECT / "models/zrr_all.sav"
ZRR_MODEL_GRA = PROJECT / "models/zrr_gra.sav"
ZRR_MODEL_GRB = PROJECT / "models/zrr_grb.sav"