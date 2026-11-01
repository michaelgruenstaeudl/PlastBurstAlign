import os
from collections import OrderedDict, defaultdict
from re import compile as re_compile
from re import findall, match, sub
from typing import Any, ClassVar, Optional

from Bio import SeqUtils
from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.SeqFeature import ExactPosition, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from .extraction_report import ExtractionReport
from .logging_ops import logger as log
from .storage import DiskBackedFeatureStore, FeatureStoreView
from .user_parameters import SelectMode

# Anything outside the IUPAC nucleotide alphabet cannot be aligned: such a feature
# is skipped (and reported) instead of poisoning its region.
_INVALID_RESIDUES = re_compile(r"[^ACGTURYSWKMBDHVNacgturyswkmbdhvn]")
# Region names become file names; real gene names are far shorter.
MAX_REGION_NAME_CHARS = 100


class PlastidData:
    mode: SelectMode
    files: list[str]
    num_taxa: int
    order: str
    store: DiskBackedFeatureStore
    nucleotides: FeatureStoreView
    proteins: FeatureStoreView | None
    order_map: dict[str, int]
    extraction_report: ExtractionReport

    def __init__(self, user_params: dict[str, Any]):
        """
        Data structure to hold the extracted annotations.
        """
        self.user_params = user_params
        self._set_mode(user_params)
        self._set_files(user_params)
        self._set_order_fun(user_params)
        # filled by `ExtractAndCollect.extract()`; empty until then
        self.extraction_report = ExtractionReport()

        db_path = os.path.join(user_params.get("out_dir", "."), "feature_store.db")
        self.store = DiskBackedFeatureStore(db_path)

        self._set_nucleotides()
        self._set_proteins()

    def _set_mode(self, user_params: dict[str, Any]):
        self.mode = user_params.get("select_mode")  # type: ignore[assignment]

    def _set_nucleotides(self):
        self.nucleotides = FeatureStoreView(self.store, "nucl")

    def _set_proteins(self):
        self.proteins = (
            FeatureStoreView(self.store, "prot")
            if self.mode.collects_proteins()
            else None
        )

    def _set_files(self, user_params: dict[str, Any]):
        """Collects the input files in a deterministic (sorted by name) order.

        `os.listdir` order is filesystem dependent; the position of a file in this
        list becomes the `file_index` of its records and thereby the row order of
        every alignment and of the concatenated supermatrix.

        Cluster worker and finalizer runs operate on the existing feature store
        only, so a missing input directory is tolerated there.
        """
        in_dir = user_params.get("in_dir")
        is_cluster_worker_or_final = (
            user_params.get("align_partition") is not None
            or user_params.get("finalize_slurm") is True
        )
        try:
            names = os.listdir(in_dir)
        except OSError:
            if not is_cluster_worker_or_final:
                raise
            names = []
        fileext = user_params.get("fileext")
        self.files = [
            os.path.join(in_dir, f) for f in sorted(names) if f.endswith(fileext)
        ]
        self.num_taxa = len(self.files)

    def _set_order_fun(self, user_params: dict[str, Any]):
        self.order = user_params.get("order")

    def add_nucleotides(
        self, pdict: Any, inv_map: dict[str, str] | None = None
    ) -> None:
        """Merges extracted dictionary into the disk store."""
        if not pdict:
            return
        if inv_map is None:
            is_igs = isinstance(pdict, IntergenicDict)
            inv_map = getattr(pdict, "feat_name_inv_map", {}) if is_igs else None

        with self.store.transaction():
            self.store.add_pdict(pdict, seq_type="nucl", inv_map=inv_map)

    def add_proteins(self, pdict: Any) -> None:
        """Merges extracted proteins into the disk store."""
        if not pdict or not self.mode.collects_proteins():
            return
        with self.store.transaction():
            self.store.add_pdict(pdict, seq_type="prot", inv_map=None)

    def set_order_map(self):
        order_list = self.store.get_active_regions(self.order)
        self.order_map = {nuc: index for index, nuc in enumerate(order_list)}

    def remove_nuc(self, feat_name: str):
        self.store.remove_region(feat_name)


class PlastidDict(OrderedDict):
    def __init__(self):
        """
        A dictionary used to store extracted sequence annotations.
        """
        super().__init__()
        self.plastome_feats = defaultdict(set)

    def _not_id(self, feat_name: str, rec_name: str) -> bool:
        rec_set = self.plastome_feats.get(rec_name)
        return True if not rec_set else feat_name not in rec_set

    def _is_feat(self, feat_name: str) -> bool:
        return feat_name in self.keys()

    def add_feature(self, feature: "PlastidFeature"):
        """
        Adds an annotation to the dictionary. If an annotation with the same name has already been added
        for the plastome (record name), it will not be added.

        Args:
            feature: A plastid feature annotation.
        """
        if feature.seq_obj is None:
            feature.log_exception()
            return

        is_feat = self._is_feat(feature.feat_name)
        not_id = self._not_id(feature.feat_name, feature.rec_name)
        # if nuc list exists, and this nuc has not been added for the plastome
        if is_feat and not_id:
            self[feature.feat_name].append(feature.get_record())
        # if the nuc list does not exist
        elif not is_feat:
            self[feature.feat_name] = [feature.get_record()]
        # record that the feature for the plastome has been added
        self.plastome_feats[feature.rec_name].add(feature.feat_name)


class IntergenicDict(PlastidDict):
    def __init__(self):
        """
        A dictionary used to store extracted intergenic regions.
        """
        super().__init__()
        self.feat_name_inv_map = {}

    def add_feature(self, igs: "IntergenicFeature"):
        super().add_feature(igs)
        self.feat_name_inv_map[igs.feat_name] = igs.inv_feat_name


# -----------------------------------------------------------------#


class GeneNameCleaner:
    _rrna_gene_names: ClassVar[dict[str, str]] = {
        "16s": "rrn16",
        "16srrna": "rrn16",
        "16sribosomalrna": "rrn16",
        "rrna16": "rrn16",
        "rrna16s": "rrn16",
        "rns": "rrn16",
        "ssu": "rrn16",
        "ssurrna": "rrn16",
        "ssurdna": "rrn16",
        "rrn16": "rrn16",
        "rrn16s": "rrn16",
        "23s": "rrn23",
        "23srrna": "rrn23",
        "23sribosomalrna": "rrn23",
        "rrna23": "rrn23",
        "rrna23s": "rrn23",
        "rnl": "rrn23",
        "lsu": "rrn23",
        "lsurrna": "rrn23",
        "lsurdna": "rrn23",
        "rrn23": "rrn23",
        "rrn23s": "rrn23",
        "4.5s": "rrn4.5",
        "4.5srrna": "rrn4.5",
        "4.5sribosomalrna": "rrn4.5",
        "rrna4.5": "rrn4.5",
        "rrna4.5s": "rrn4.5",
        "rrn45": "rrn4.5",
        "rrn45s": "rrn4.5",
        "rrn4.5": "rrn4.5",
        "rrn4.5s": "rrn4.5",
        "5s": "rrn5",
        "5srrna": "rrn5",
        "5sribosomalrna": "rrn5",
        "rrna5": "rrn5",
        "rrna5s": "rrn5",
        "5srdna": "rrn5",
        "rrn5": "rrn5",
        "rrn5s": "rrn5",
    }

    def __init__(self, table_id: int = 11):
        """
        Extracts gene names from `SeqFeature` objects and standardizes gene names.
        tRNA names use amino acid abbreviations and a specified codon table to add
        anticodon information. The four plastid rRNA names are standardized to
        `rrn16`, `rrn23`, `rrn4.5`, and `rrn5`.
        The default codon table used is INSDC table 11.

        Args:
            table_id: Genetic code table ID to use for mapping from amino acids to codons.
                For more, reference https://www.insdc.org/submitting-standards/genetic-code-tables/.
        """
        self.codon_table = CodonTable.unambiguous_dna_by_id[table_id]

    def _amino3_to_anticodon(self, amino_code: str) -> Optional[str]:
        if len(amino_code) != 3:
            return None
        amino_char = SeqUtils.seq1(amino_code)
        codon = self.codon_table.back_table.get(amino_char)
        if codon:
            anticodon = Seq(codon).reverse_complement_rna()
            return str(anticodon)
        else:
            return None

    @staticmethod
    def get_gene(feat: SeqFeature) -> Optional[str]:
        """
        Finds and returns the gene name for a feature. If the feature is not a gene, `None` will be returned.

        Args:
            feat: `Biopython` feature.

        Returns:
            The gene name.

        """
        return feat.qualifiers["gene"][0] if feat.qualifiers.get("gene") else None

    def clean_trna_gene(
        self, gene: Optional[str], cut_trna: bool = False
    ) -> Optional[str]:
        """
        Standardizes a tRNA gene name by case, delimiter usage, and anticodon.
        If there is no initial gene name (`None`), `None` will be returned.

        Args:
            gene: Gene name.
            cut_trna: Option to disregard additional qualifiers from the name (default is `False`).

        Returns:
            Cleaned gene name.

        """
        if not gene:
            return None

        # look for standardized gene name
        gene_pattern = r"^[a-zA-Z]{3}[a-zA-Z0-9]+"
        gene_sub = match(gene_pattern, gene)
        # if non-standard, just return a "safe" version of the name
        if not gene_sub:
            return sub(r"\W+", "_", gene.replace("-", "_"))
        # the first match is the beginning of the cleaned name
        cleaned_gene = gene_sub.group()[0:3].lower() + gene_sub.group()[3:].upper()
        if cut_trna:
            return cleaned_gene

        # look for tRNA qualifiers
        gene = gene[len(cleaned_gene) :]
        trna_pattern = r"([a-zA-Z]{3})"
        qual_subs = findall(trna_pattern, gene)
        # if not tRNA, return the gene
        if not qual_subs:
            return cleaned_gene

        # handle each tRNA qualifier appropriately
        codon_pattern = r"\b[ACGTUacgtu]{3}\b"
        anticodon = None
        while not anticodon and qual_subs:
            qualifier = qual_subs.pop()
            # qualifier is codon
            if match(codon_pattern, qualifier):
                anticodon = qualifier.upper()
            # qualifier is amino acid?
            else:
                anticodon = self._amino3_to_anticodon(qualifier)
        if anticodon:
            cleaned_gene += "_" + anticodon
        return cleaned_gene

    def clean_rrna_gene(self, gene: Optional[str]) -> Optional[str]:
        """Standardizes recognized plastid rRNA names to their canonical gene names."""
        if not gene:
            return None
        normalized_gene = sub(r"[^a-z0-9.]", "", gene.casefold())
        canonical_name = self._rrna_gene_names.get(normalized_gene)
        if canonical_name:
            return canonical_name
        return self.clean_gene(gene)

    def clean_gene(self, gene: Optional[str], cut_trna: bool = False) -> Optional[str]:
        """Standardizes a gene name using the legacy general/tRNA cleaning behavior."""
        return self.clean_trna_gene(gene, cut_trna)

    def get_safe_gene(self, feat: SeqFeature) -> Optional[str]:
        """
        Finds and returns the cleaned gene name for a feature. If the feature is not a gene, `None` will be returned.

        Args:
            feat: `Biopython` feature.

        Returns:
            Cleaned gene name.

        """
        gene = self.get_gene(feat)
        if feat.type == "tRNA":
            return self.clean_trna_gene(gene)
        if feat.type == "rRNA":
            return self.clean_rrna_gene(gene)
        if gene and gene.casefold().startswith("trn"):
            return self.clean_trna_gene(gene)
        if gene and sub(r"[^a-z0-9.]", "", gene.casefold()) in self._rrna_gene_names:
            return self.clean_rrna_gene(gene)
        return self.clean_gene(gene)


# -----------------------------------------------------------------#


class PlastidFeature:
    name_cleaner = GeneNameCleaner()

    _type: str = "plastid feature"
    _default_exception: str = "exception"
    _log_fun = log.warning

    def __init__(self, record: SeqRecord, feature: SeqFeature):
        """
        Primarily a wrapper for a `SeqRecord` that can be accessed by `get_record()`.
        It contains other variables and methods that document the creation of the `SeqRecord`.

        Args:
            record: The record that the feature is being extracted from.
            feature: The feature to be extracted.
        """
        self._exception = None
        self.seq_obj = None

        self._set_feat_name(feature)
        self._set_rec_name(record)
        self._set_feature(feature)
        self._set_seq_obj(record)
        self._set_seq_name()

    def _set_feat_name(self, feature: SeqFeature):
        self.feat_name = self.name_cleaner.get_safe_gene(feature)

    def _set_rec_name(self, record: SeqRecord):
        self.rec_name = record.name

    def _set_feature(self, feature: SeqFeature):
        self.feature = feature

    def _set_seq_obj(self, record: SeqRecord):
        try:
            seq = self.feature.extract(record).seq
            bad = _INVALID_RESIDUES.search(str(seq))
            if bad:
                raise ValueError(
                    f"invalid nucleotide character {bad.group()!r} at position"
                    f" {bad.start() + 1} of the feature"
                )
            self.seq_obj = seq
        except Exception as e:
            self._set_exception(e)

    def _set_seq_name(self):
        self.seq_name = f"{self.feat_name}_{self.rec_name}"
        if (
            isinstance(self.feat_name, str)
            and len(self.feat_name) > MAX_REGION_NAME_CHARS
        ):
            self.seq_obj = None
            self._set_exception(
                ValueError(
                    f"region name of {len(self.feat_name)} characters exceeds the"
                    f" {MAX_REGION_NAME_CHARS} character limit"
                )
            )

    def _set_exception(self, exception: Exception | None = None):
        if exception is None:
            self._exception = self._default_exception
            self._log_fun = lambda x: None
        else:
            self._exception = exception
            self._log_fun = PlastidFeature._log_fun

    def status_str(self) -> str:
        """
        Retrieves a string that describes the status of the contained `SeqRecord`.

        Returns:
            A string describing the `SeqRecord` status.

        """
        message = (
            f"skipped due to {self._exception}" if self._exception else "successful"
        )
        status = (
            f"parsing of {self._type} '{self.feat_name}' in {self.rec_name} {message}"
        )
        return status

    def log_exception(self):
        """
        Logs a string that describes the exception status of the contained `SeqRecord`.
        If the `SeqRecord` was successfully extracted, or if the issue that prevented
        extraction is not noteworthy, nothing will be logged. To unconditionally access
        a string representation of the `SeqRecord` for logging purposes, use `status_str()`.
        """
        if self._exception:
            self._log_fun(self.status_str())

    def get_record(self) -> Optional[SeqRecord]:
        """
        Instantiates the contained `SeqRecord` and returns it.
        If the `SeqRecord` could not be successfully extracted, `None` is returned.
        Additional information about the `None` return can be accessed by `status_str()`.

        Returns:
            The instantiated `SeqRecord`.

        """
        if self.seq_obj:
            return SeqRecord(self.seq_obj, id=self.seq_name, name="", description="")
        return None


class GeneFeature(PlastidFeature):
    """
    Primarily a wrapper for a gene `SeqRecord` that can be accessed by `get_record()`.
    It contains other variables and methods that document the creation of the `SeqRecord`.
    """

    _type = "gene"
    _default_exception = (
        "potential reading frame error of this feature in input file; "
        "associated protein will be skipped as well"
    )

    def _set_seq_obj(self, record: SeqRecord):
        super()._set_seq_obj(record)
        self._trim_mult_three()

    def _trim_mult_three(self):
        """
        Accesses if a proper reading frame exists for the gene.
        """
        if self.seq_obj is None:  # extraction failed; the exception is recorded
            return
        trim_char = len(self.seq_obj) % 3
        if trim_char > 0 and self.seq_obj[:3] == "ATG":
            self.seq_obj = self.seq_obj[:-trim_char]
        elif trim_char > 0:
            self.seq_obj = None
            self._set_exception()
            self._log_fun = GeneFeature._log_fun


class ProteinFeature(GeneFeature):
    _type = "protein"
    _default_exception = "potential reading frame error of this feature in input file"

    def __init__(
        self,
        record: SeqRecord = None,
        feature: SeqFeature = None,
        gene: GeneFeature = None,
    ):
        """
        Primarily a wrapper for a protein `SeqRecord` that can be accessed by `get_record()`.
        It contains other variables and methods that document the creation of the `SeqRecord`.

        If an existing gene feature (`gene`) is provided instead of a `Biopython` `record` and `feature`,
        the object will be copied and the contained nucleotide sequence will be translated to a protein sequence.
        This is how the class is used within the module, but the standard `PlastidFeature` constructor
        exists for use cases where the nucleotide sequence of a feature is not used.

        Args:
            record: The record that the feature is being extracted from (default is `None`).
            feature: The feature to be extracted (default is `None`).
            gene: An existing gene feature (default is `None`).
        """
        # if we provide a GeneFeature to the constructor, we can just copy the attributes
        if gene is not None:
            self.__dict__.update(gene.__dict__)
        else:
            super().__init__(record, feature)
        self._set_prot_obj()

    def _set_prot_obj(self):
        if self.seq_obj is None:
            self._set_exception()
            return

        try:
            self.seq_obj = self.seq_obj.translate(
                table=11
            )  # Getting error TTA is not stop codon.
        except Exception as e:
            self._set_exception(e)


class IntronFeature(PlastidFeature):
    _type = "intron"

    def __init__(self, record: SeqRecord, feature: SeqFeature, offset: int = 0):
        """
        Primarily a wrapper for an intron `SeqRecord` that can be accessed by `get_record()`.
        It contains other variables and methods that document the creation of the `SeqRecord`.

        Args:
            record: The record that the feature is being extracted from.
            feature: The feature to be extracted.
            offset: An index offset for which intron is being considered (default is 0).
        """
        self.offset = offset
        super().__init__(record, feature)

    def _set_feat_name(self, feature: SeqFeature):
        super()._set_feat_name(feature)
        self.feat_name += "_intron" + str(self.offset + 1)

    def _set_feature(self, feature: SeqFeature):
        super()._set_feature(feature)
        exon_1 = self.feature.location.parts[self.offset]
        exon_2 = self.feature.location.parts[self.offset + 1]
        in_order = exon_2.start >= exon_1.end

        if in_order:
            self.feature.location = FeatureLocation(exon_1.end, exon_2.start)
        else:
            self.feature.location = FeatureLocation(exon_2.start, exon_1.end)


class IntergenicFeature(PlastidFeature):
    _type = "intergenic spacer"
    _default_exception = "negative intergenic length in input file"

    def __init__(
        self, record: SeqRecord, current_feat: SeqFeature, subsequent_feat: SeqFeature
    ):
        """
        Primarily a wrapper for an intergenic spacer `SeqRecord` that can be accessed by `get_record()`.
        It contains other variables and methods that document the creation of the `SeqRecord`.

        Args:
            record: The record that the feature is being extracted from.
            current_feat: The feature that borders the beginning of the intergenic space.
            subsequent_feat: The feature that borders the end of the intergenic space.
        """
        self._set_sub_feat(subsequent_feat)
        super().__init__(record, current_feat)

    def _set_sub_feat(self, subsequent_feat: SeqFeature):
        self.subsequent_feat = subsequent_feat
        self.subsequent_name = self.name_cleaner.get_safe_gene(subsequent_feat)

    def _set_feature(self, current_feat: SeqFeature):
        # Note: It's unclear if +1 is needed here.
        self.start_pos = ExactPosition(current_feat.location.end)  # +1)
        self.end_pos = ExactPosition(self.subsequent_feat.location.start)
        self.feature = (
            FeatureLocation(self.start_pos, self.end_pos)
            if self.start_pos < self.end_pos
            else None
        )
        self.default_exception = (
            f"{IntergenicFeature._default_exception}"
            f" (start pos: {self.start_pos}, end pos:{self.end_pos})"
        )

    def _set_seq_obj(self, record: SeqRecord):
        if self.feature is None:
            self._set_exception()
        else:
            super()._set_seq_obj(record)

    def _set_seq_name(self):
        self.inv_feat_name = f"{self.subsequent_name}_{self.feat_name}"
        self.feat_name += "_" + self.subsequent_name
        super()._set_seq_name()
