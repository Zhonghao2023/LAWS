from laws.laws_initial import LAWSModel_ini
from laws.laws_dynamic import LAWSModel_dyn

class LAWSModel(LAWSModel_ini, LAWSModel_dyn):
    """
    Combined initialization and dynamic components of LAWS.

    * The initialization component handles non-temporal setup procedures.
    * The dynamic component advances the model through time.
    """
