from datetime import date
from importlib.metadata import version

__name__ = "plastburstalign"
__author__ = "Michael Gruenstaeudl, PhD"
__email__ = "m_gruenstaeudl@fhsu.edu"

try:
    __version__ = version(__name__)
except ModuleNotFoundError:
    __version__ = date.today()

from .user_parameters import UserParameters
from .seqfeature_ops import PlastidData
from .extraction_ops import ExtractAndCollect, DataCleaning
from .alignment_ops import AlignmentCoordination
from .plast_burst_align import PlastBurstAlign
from .alignment_tools import MAFFT, ClustalOmega, MUSCLE
__all__ = ['PlastBurstAlign', 'UserParameters', 'PlastidData',
           'ExtractAndCollect', 'DataCleaning', 'AlignmentCoordination', 'MAFFT', 'ClustalOmega', 'MUSCLE' ]
