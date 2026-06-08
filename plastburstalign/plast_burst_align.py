from typing import Optional, Any, Dict

# Package imports
from .user_parameters import UserParameters
from .seqfeature_ops import PlastidData
from .extraction_ops import ExtractAndCollect, DataCleaning
from .alignment_ops import AlignmentCoordination


class PlastBurstAlign:

    def __init__(self, user_params: Optional[Dict[str, Any]] = None):
        """
        Coordinates extraction of sequence annotations from GenBank flatfiles,
        the alignment of those sequences grouped by region name,
        and writing the concatenated MSA results to file in NEXUS and FASTA formats.

        Args:
            user_params: Specifications for how the execution should be performed.
        """
        if not user_params or not isinstance(user_params, dict):
            self.user_params: Dict[str, Any] = UserParameters()
        else:
            self.user_params = user_params
        self.plastid_data = PlastidData(self.user_params)

    def execute(self):
        """
        Extracts all genes, introns, or intergenic spacers from a set of plastid genomes in GenBank flatfile format,
        aligns the corresponding regions, and then saves the alignments.
        """
        extractor = ExtractAndCollect(self.plastid_data, self.user_params)
        extractor.extract()

        cleaner = DataCleaning(self.plastid_data, self.user_params)
        cleaner.clean()

        aligncoord = AlignmentCoordination(self.plastid_data, self.user_params)
        aligncoord.perform_MSAs()
        if self.user_params.get("concat"):
            aligncoord.concat_MSAs()
