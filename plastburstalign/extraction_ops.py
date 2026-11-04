import bisect
import os
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Any, NamedTuple

from Bio import SeqIO
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from .helpers import bounded_submit_iterator, split_list
from .logging_ops import Logger
from .logging_ops import logger as log
from .task_queue import get_mp_context

# Package imports
from .seqfeature_ops import (
    GeneFeature,
    IntergenicDict,
    IntergenicFeature,
    IntronFeature,
    PlastidData,
    PlastidDict,
    PlastidFeature,
    ProteinFeature,
)


class ExtractAndCollect:
    def __init__(self, plastid_data: PlastidData, user_params: dict[str, Any]):
        """
        Coordinates the parsing of GenBank flatfiles, the extraction of the contained sequence annotations,
        and the collection of the resulting sequence records in a data structure.

        Args:
            plastid_data: Contains the locations of the files to be parsed,
                and where the extracted records will be stored.
            user_params: Specifications for how the extraction should be performed.
                These are `num_threads`, `out_dir`, `verbose`, `exclude_fullcds`,
                `include_trna`, and `include_rrna`.
        """
        self.plastid_data = plastid_data
        self.num_threads = user_params.get("num_threads")
        self.out_dir = user_params.get("out_dir")
        self.verbose = user_params.get("verbose")
        self.exclude_cds = user_params.get("exclude_fullcds")
        self.include_trna = user_params.get("include_trna", True)
        self.include_rrna = user_params.get("include_rrna", True)
        self.exclude_region = set(user_params.get("exclude_region") or [])
        min_len = user_params.get("min_seq_length")
        self.min_seq_length = min_len if min_len is not None else 3
        self._set_extract_fun()

    def _is_region_excluded(self, *names: str | None) -> bool:
        if not self.exclude_region:
            return False
        return any(name in self.exclude_region for name in names if name)

    def _set_extract_fun(self):
        if self.plastid_data.mode == "cds":
            self._extract_fun = self._extract_cds
        elif self.plastid_data.mode == "igs":
            self._extract_fun = self._extract_igs
        elif self.plastid_data.mode == "int":
            self._extract_fun = self._extract_int
        elif self.plastid_data.mode == "rna":
            self._extract_fun = self._extract_rna
        else:
            log.critical("Undefined extraction mode")
            raise ValueError

    @staticmethod
    def _worker_wrapper(args_tuple):
        """Unpacks arguments from the iterator into the static worker."""
        return ExtractAndCollect._extract_recs_static(*args_tuple)

    def extract(self):
        """
        Parses all GenBank flatfiles of a given folder and extracts
        all sequence annotations of the type specified by the user for each flatfile.
        """
        log.info(
            f"parsing GenBank flatfiles and extracting their sequence annotations using "
            f"{self.num_threads} processes"
        )

        if not self.plastid_data.files:
            log.critical("No input files found for extraction")
            raise FileNotFoundError("No input files found for extraction")

        # Resolve mode primitives once upfront so they are available regardless of ordering
        mode_str = getattr(self.plastid_data.mode, "mode", str(self.plastid_data.mode))
        collects_proteins = self.plastid_data.mode.collects_proteins()

        # Step 0. Extract first genome in list for feature ordering
        if self.plastid_data.order == "seq":
            first_genome = self.plastid_data.files.pop(0)

            nuc_dict, prot_dict = self._extract_recs_static(
                [first_genome],
                mode_str,
                collects_proteins,
                self.exclude_cds,
                self.include_trna,
                self.include_rrna,
                self.min_seq_length,
                self.verbose,
            )
            self.plastid_data.add_nucleotides(nuc_dict)
            self.plastid_data.add_proteins(prot_dict)

        # Step 1. Chunk remaining files for parallel workers
        file_lists = split_list(self.plastid_data.files, self.num_threads * 2)

        # Step 2. Use ProcessPoolExecutor to parallelize extraction with bounded queue
        mp_context = get_mp_context()
        max_in_flight = self.num_threads * 2

        with ProcessPoolExecutor(max_workers=self.num_threads, mp_context=mp_context) as executor:
            args_generator = (
                (
                    chunk,
                    mode_str,
                    collects_proteins,
                    self.exclude_cds,
                    self.include_trna,
                    self.include_rrna,
                    self.min_seq_length,
                    self.verbose,
                )
                for chunk in file_lists
            )
            for _, future in bounded_submit_iterator(
                executor, self._worker_wrapper, args_generator, max_in_flight=max_in_flight
            ):
                nuc_dict, prot_dict = future.result()
                self.plastid_data.add_nucleotides(nuc_dict)
                self.plastid_data.add_proteins(prot_dict)

        # Step 3. Stop execution if no nucleotides were extracted
        if not self.plastid_data.nucleotides.items():
            log.critical(f"No items in main dictionary: {self.out_dir}")
            raise Exception()

    @staticmethod
    def _extract_recs_static(
        files: list[str],
        mode_str: str,
        collects_proteins: bool,
        exclude_cds: list[str] | None,
        include_trna: bool,
        include_rrna: bool,
        min_seq_length: int,
        verbose: bool,
    ) -> tuple[PlastidDict, PlastidDict | None]:
        """Static worker function isolated from sqlite connections."""

        class MockMode:
            def __init__(self, mode: str, collects: bool):
                self.mode = mode
                self._collects = collects

            def collects_proteins(self) -> bool:
                return self._collects

            def __str__(self):
                return self.mode

        class MockData:
            def __init__(self):
                self.mode = MockMode(mode_str, collects_proteins)

        def extract_rec(file: str, n_dict: PlastidDict, p_dict: PlastidDict | None):
            try:
                log.info(f" parsing {os.path.basename(file)}")
                record = SeqIO.read(file, "genbank")

                extractor = ExtractAndCollect.__new__(ExtractAndCollect)
                extractor.plastid_data = MockData()
                extractor.exclude_cds = exclude_cds or []
                extractor.exclude_region = []  # Add missing property since __init__ was skipped
                extractor.include_trna = include_trna
                extractor.include_rrna = include_rrna
                extractor.min_seq_length = min_seq_length

                if mode_str == "SelectMode.CDS":
                    extractor._extract_cds(record, n_dict, p_dict)
                elif mode_str == "SelectMode.IGS":
                    extractor._extract_igs(record, n_dict)
                elif mode_str == "SelectMode.INT":
                    extractor._extract_int(record, n_dict)
                elif mode_str == "SelectMode.RNA":
                    extractor._extract_rna(record, n_dict)

            except Exception as e:
                log.error(" {!r} generated an exception: {}".format(os.path.basename(file), e))

        Logger.reinitialize_logger(verbose)
        nuc_dict = IntergenicDict() if mode_str == "SelectMode.IGS" else PlastidDict()
        prot_dict = PlastidDict() if collects_proteins else None

        for f in files:
            extract_rec(f, nuc_dict, prot_dict)
        return nuc_dict, prot_dict

    @staticmethod
    def _worker_wrapper(args_tuple):
        """Unpacks arguments from the iterator into the static worker."""
        return ExtractAndCollect._extract_recs_static(*args_tuple)

    def _extract_cds(
        self, rec: SeqRecord, nuc_dict: PlastidDict, protein_dict: PlastidDict | None = None
    ):
        """
        Extracts all CDS (coding sequences = genes) from a given sequence record
        """
        for feature in RecordFeatures(rec, self.exclude_cds).cds_feats:
            if len(feature.location) < self.min_seq_length:
                continue
            feat_name = PlastidFeature.name_cleaner.get_safe_gene(feature)
            if self._is_region_excluded(feat_name):
                continue

            # Step 1. Extract nucleotide sequence of each gene
            gene = GeneFeature(rec, feature)
            if gene.seq_obj is None or len(gene.seq_obj) < self.min_seq_length or self._is_region_excluded(gene.feat_name):
                continue

            # Step 2. Translate nucleotide sequence to protein when in CDS mode
            if protein_dict is not None:
                protein = ProteinFeature(gene=gene)
                if protein.seq_obj is None:
                    continue
                clean_aa = str(protein.seq_obj).rstrip("*")
                # Ensure the sequence contains valid residues for alignment (cannot be empty)
                min_aa = max(1, self.min_seq_length // 3)
                if len(clean_aa) < min_aa:
                    continue
                protein_dict.add_feature(protein)

            nuc_dict.add_feature(gene)

    def _extract_rna(self, rec: SeqRecord, nuc_dict: PlastidDict):
        """
        Extracts annotated tRNA and rRNA sequences without reading-frame checks or translation.
        """
        for feature in RecordFeatures(rec, self.exclude_cds).rna_feats:
            if (feature.type == "tRNA" and self.include_trna) or (
                feature.type == "rRNA" and self.include_rrna
            ):
                if len(feature.location) < self.min_seq_length:
                    continue
                feat_name = PlastidFeature.name_cleaner.get_safe_gene(feature)
                if self._is_region_excluded(feat_name):
                    continue

                feat = PlastidFeature(rec, feature)
                if feat.seq_obj is None:
                    nuc_dict.add_feature(feat)
                    continue
                if len(feat.seq_obj) < self.min_seq_length or self._is_region_excluded(
                    feat.feat_name
                ):
                    continue

                nuc_dict.add_feature(feat)

    def _extract_igs(self, rec: SeqRecord, nuc_dict: PlastidDict):
        """
        Extracts all IGS (intergenic spacers) from a given sequence record
        """
        all_genes = RecordFeatures(rec, self.exclude_cds).igs_feats
        try:
            subsequent_feat = next(all_genes)
        except StopIteration:
            return
        # Step 1. Loop through genes
        for feature in all_genes:
            current_feat = subsequent_feat
            subsequent_feat = feature

            # Early pre-checks before constructing IntergenicFeature
            curr_name = PlastidFeature.name_cleaner.get_safe_gene(current_feat)
            sub_name = PlastidFeature.name_cleaner.get_safe_gene(subsequent_feat)
            if curr_name and sub_name:
                igs_name = f"{curr_name}_{sub_name}"
                inv_name = f"{sub_name}_{curr_name}"
                if self._is_region_excluded(igs_name, inv_name):
                    continue

            start_pos = int(current_feat.location.end)
            end_pos = int(subsequent_feat.location.start)
            if start_pos < end_pos and (end_pos - start_pos) < self.min_seq_length:
                continue

            # Step 2. Make IGS SeqFeature
            igs = IntergenicFeature(rec, current_feat, subsequent_feat)
            if igs.seq_obj is None:
                nuc_dict.add_feature(igs)
                continue
            if len(igs.seq_obj) < self.min_seq_length:
                continue
            if self._is_region_excluded(igs.feat_name, getattr(igs, "inv_feat_name", None)):
                continue

            # Step 3. Attach IGS to growing dictionary
            nuc_dict.add_feature(igs)

    def _extract_int(self, rec: SeqRecord, nuc_dict: PlastidDict):
        """
        Extracts all INT (introns) from a given sequence record
        """

        def _add_intron(feature_obj: SeqFeature, offset: int):
            exon_1 = feature_obj.location.parts[offset]
            exon_2 = feature_obj.location.parts[offset + 1]
            intron_len = (
                abs(int(exon_2.start) - int(exon_1.end))
                if exon_2.start >= exon_1.end
                else abs(int(exon_1.start) - int(exon_2.end))
            )
            if intron_len < self.min_seq_length:
                return

            gene_name = PlastidFeature.name_cleaner.get_safe_gene(feature_obj)
            intron_name = f"{gene_name}_intron{offset + 1}" if gene_name else None
            if self._is_region_excluded(intron_name):
                return

            intron = IntronFeature(rec, feature_obj, offset)
            if intron.seq_obj is None:
                nuc_dict.add_feature(intron)
                return
            if len(intron.seq_obj) < self.min_seq_length or self._is_region_excluded(
                intron.feat_name
            ):
                return
            nuc_dict.add_feature(intron)

        for feature in RecordFeatures(rec, self.exclude_cds).int_feats:
            # Step 1.a. If one intron in gene:
            if len(feature.location.parts) == 2:
                _add_intron(feature, 0)

            # Step 1.b. If two introns in gene:
            elif len(feature.location.parts) == 3:
                feature_copy = deepcopy(
                    feature
                )  # Important b/c feature is overwritten in extract_internal_intron()
                _add_intron(feature, 0)
                _add_intron(feature_copy, 1)


class RecordFeatures:
    def __init__(self, record: SeqRecord, exclude_list: list[str] | None = None):
        """
        Returns generators for features in a `SeqRecord`.
        For all generators, ORFs and genes listed in `exclude_list` are filtered out.

        Args:
            record: A genome `SeqRecord`.
            exclude_list: A list of gene names to filter out.
        """
        self._features = record.features
        self.rec_name = record.name
        self.exclude_set = set(exclude_list) if exclude_list else set()

    @property
    def cds_feats(self) -> Generator[SeqFeature, None, None]:
        """
        A generator of CDS features in the record.
        """
        for f in self._features:
            if f.type == "CDS" and self._not_exclude(f):
                yield f

    @property
    def rna_feats(self) -> Generator[SeqFeature, None, None]:
        """
        A generator of tRNA and rRNA features in the record.
        """
        for f in self._features:
            if f.type in ("tRNA", "rRNA") and self._not_exclude(f):
                yield f

    @property
    def igs_feats(self) -> Generator[SeqFeature, None, None]:
        """
        A generator of IGS features in the record.
        """
        # Note: No need to include "if feature.type=='tRNA'", because all tRNAs are also annotated as genes
        features = [f for f in self._features if f.type == "gene" and self._not_exclude(f)]
        # Handle spliced exons
        handler = ExonSpliceHandler(features, self.rec_name)
        handler.resolve()
        yield from features

    @property
    def int_feats(self) -> Generator[SeqFeature, None, None]:
        """
        A generator of IGS features in the record.
        """
        for f in self._features:
            if (f.type == "CDS" or f.type == "tRNA") and self._not_exclude(f):
                yield f

    def _not_exclude(self, feature: SeqFeature) -> bool:
        gene = PlastidFeature.name_cleaner.get_safe_gene(feature)
        return gene and gene not in self.exclude_set and "orf" not in gene


# -----------------------------------------------------------------#


class ExonSpliceHandler:
    def __init__(self, genes: list[SeqFeature], rec_name: str):
        """
        Coordinates the handling of spliced exons.
        Cis and trans-spliced genes are considered separately.

        Args:
            genes: List of gene features.
            rec_name: Name of the record that contains the genes.
        """
        self.feat_rearr = FeatureRearranger(genes, rec_name)

    def resolve(self):
        """
        The handling of compound locations begins with merging the locations of
        cis-spliced genes. During this process, compound location genes qualified as "trans_splicing"
        or containing rps12 are identified and removed from the genes list, as they will be handled next.

        In the second step, the genes identified as trans-spliced are split into multiple simple location
        features. Each of these splits results in a simple location feature for each contiguous location
        of the compound location. All of these simple location features are inserted (or merged) into their
        expected location in the gene list.
        """
        log.info(f"  resolving genes with compound locations in {self.feat_rearr.rec_name}")
        log.info(f"   resolving cis-spliced genes in {self.feat_rearr.rec_name}")
        merger = ExonSpliceMerger(self.feat_rearr)
        merger.merge()
        log.info(f"   resolving trans-spliced genes in {self.feat_rearr.rec_name}")
        insertor = ExonSpliceInsertor(self.feat_rearr)
        insertor.insert()


class FeatureRearranger:
    class FeatureTuple(NamedTuple):
        """
        A tuple that contains a `SeqFeature` as well as other variables
        to assist with manipulating the feature within a features list.
        """

        feature: SeqFeature | None = None
        gene: str | None = None
        is_trans: bool = False
        is_compound: bool = False
        index: int = -1

    def __init__(self, features: list[SeqFeature], rec_name: str):
        """
        An interface to modify the features of a genome record.

        Args:
            features: List of features.
            rec_name: Name of the record that contains the features.
        """
        self._features = features
        self.rec_name = rec_name
        self._set_end_positions()
        self._compound_features: list[SeqFeature] = []
        self._compounds_removed = False
        self._is_sorted = False

    def _set_end_positions(self):
        self._end_positions = [f.location.end for f in self._features]

    def _size(self) -> int:
        return len(self._features)

    def _sort(self):
        if not self._is_sorted:
            self._features.sort(key=lambda gene: gene.location.end)
            self._set_end_positions()
            self._is_sorted = True

    def _get_insert_index(self, feature: SeqFeature) -> int:
        # feature list has to be sorted before being able to find index
        self._sort()
        return bisect.bisect_left(self._end_positions, feature.location.end)

    @staticmethod
    def _get_is_trans(feature: SeqFeature, gene: str) -> bool:
        trans_genes = ["rps12"]
        return bool(feature.qualifiers.get("trans_splicing")) or gene in trans_genes

    def _tuple_from_feature(self, feature: SeqFeature, index: int | None = None) -> FeatureTuple:
        gene = PlastidFeature.name_cleaner.get_safe_gene(feature)
        is_trans = self._get_is_trans(feature, gene)
        is_compound = type(feature.location) is CompoundLocation
        index = self._get_insert_index(feature) if index is None else index
        return self.FeatureTuple(feature, gene, is_trans, is_compound, index)

    def _tuple_from_index(self, index: int) -> FeatureTuple:
        if index < 0 or index >= len(self._features):
            return self.FeatureTuple()
        else:
            feature = self._features[index]
            return self._tuple_from_feature(feature, index)

    def feature_tuples(self, reverse: bool = True) -> Generator[FeatureTuple, None, None]:
        """
        A generator of feature tuples corresponding to the features list.
        By default, the features are returned in the reverse order that they appear in the file.

        Args:
            reverse: A boolean that indicates if the features should be returned in reverse order.

        Returns:
            A generator of feature tuples.
        """
        start = self._size() - 1 if reverse else 0
        end = -1 if reverse else self._size()
        increment = -1 if reverse else 1
        for index in range(start, end, increment):
            yield self._tuple_from_index(index)

    def _remove_compounds(self):
        # find the compound features and remove them from the gene list
        if not self._compounds_removed:
            self._compound_features.extend(
                f for f in self._features if type(f.location) is CompoundLocation
            )
            self._features = [f for f in self._features if f not in self._compound_features]
            self._compounds_removed = True

    def simple_features(self) -> Generator[FeatureTuple, None, None]:
        """
        A generator of simple feature tuples created from compound location features.
        If there are simple features, the features list will be sorted by end
        location so that the expected insertion locations can be determined.

        Returns:
            A generator of simple feature tuples.
        """
        self._remove_compounds()
        for f in self._compound_features:
            for p in f.location.parts:
                yield self._tuple_from_feature(
                    SeqFeature(location=p, type=f.type, id=f.id, qualifiers=f.qualifiers)
                )

    def insert(self, feature: FeatureTuple):
        """
        Inserts the feature into the features list.

        Args:
            feature: A feature tuple.
        """
        self._features.insert(feature.index, feature.feature)
        self._end_positions.insert(feature.index, feature.feature.location.end)

    def remove(self, feature: FeatureTuple):
        """
        Removes the feature from the features list.
        Attempting to remove a feature that is not in the list will result in no change.

        Args:
            feature: A feature tuple.
        """
        current = self.get_current(feature)
        if feature == current:
            del self._features[feature.index]
            del self._end_positions[feature.index]
            if feature.is_compound:
                self._compound_features.append(feature.feature)

    def get_previous(self, feature: FeatureTuple) -> FeatureTuple:
        """
        Returns the feature that comes immediately before `feature`.

        Args:
            feature: A feature tuple.

        Returns:
            The previous feature.

        """
        return self._tuple_from_index(feature.index - 1)

    def get_current(self, feature: FeatureTuple) -> FeatureTuple:
        """
        Returns the feature at the current or desired location of `feature`.
        For example, if feature is in the list, what is returned will also be `feature`.
        If the feature is a candidate feature that is not in the list, what will
        be returned is the feature at the location that `feature` should be inserted based on location.

        Args:
            feature:

        Returns:
            The current feature.

        """
        return self._tuple_from_index(feature.index)

    def get_subsequent(self, feature: FeatureTuple) -> FeatureTuple:
        """
        Returns the feature that comes immediately after `feature`.

        Args:
            feature: A feature tuple.

        Returns:
            The subsequent feature.

        """
        return self._tuple_from_index(feature.index + 1)

    def log_position(self, current: FeatureTuple, merge: bool = True):
        """
        Logs the position of `current` within the features list with gene name and location information.

        Args:
            current: A feature tuple.
            merge: A boolean indicating that the associated operation was a merge.
        """
        previous = self.get_previous(current)
        previous_gene = "\t" if not previous.feature else previous.gene
        previous_location = "\t\t\t\t\t" if not previous.feature else previous.feature.location

        subsequent = self.get_subsequent(current)
        subsequent_gene = "\t" if not subsequent.feature else subsequent.gene
        subsequent_location = (
            "\t\t\t\t\t" if not subsequent.feature else subsequent.feature.location
        )

        message_sub = "Merging exons of" if merge else "Repositioning"
        message = f"{message_sub} {current.gene} within {self.rec_name}"
        log.debug(
            f"   {message}\n"
            "-----------------------------------------------------------\n"
            f"\t\t{previous_gene}\t\t\t\t{current.gene}\t\t\t\t{subsequent_gene}\n"
            f"\t{previous_location}\t{current.feature.location}\t{subsequent_location}\n"
            "-----------------------------------------------------------\n"
        )


class ExonSpliceMerger:
    def __init__(self, feat_rearr: FeatureRearranger):
        """
        Coordinates the handling of cis-spliced exons by merging.

        Args:
            feat_rearr: An interface for rearranging features.
        """
        self.feat_rearr = feat_rearr

    def merge(self):
        """
        Locations of compound cis-spliced genes are merged in-place as they occur within the source GenBank file
        (no repositioning of the annotations). Adjacent genes of the same name are assumed to be separately annotated
        cis-spliced genes and merged as well. During this process, compound location annotations qualified as
        "trans_splicing" or containing the gene rps12 (these are known to be trans but are not always qualified as such)
        are removed from the list of gene features, and retained in a separate list.
        """
        feat_tuples = self.feat_rearr.feature_tuples()
        # no initial subsequent feature
        subsequent = FeatureRearranger.FeatureTuple()
        for current in feat_tuples:
            if current.is_trans:
                self.feat_rearr.remove(current)
            elif current.is_compound:
                self._merge_cis_exons(current)
                # print new alignment
                self.feat_rearr.log_position(current)

            is_same_gene = current.gene == subsequent.gene
            if is_same_gene and not current.is_trans:
                self._merge_adj_exons(current, subsequent)
                # delete merged exon
                self.feat_rearr.remove(subsequent)
                # print new alignment
                self.feat_rearr.log_position(current)

            # update for next iteration
            subsequent = current

    @staticmethod
    def _merge_cis_exons(current: FeatureRearranger.FeatureTuple):
        loc_parts = current.feature.location.parts
        gene_start = min(p.start for p in loc_parts)
        gene_end = max(p.end for p in loc_parts)
        current.feature.location = FeatureLocation(gene_start, gene_end)

    @staticmethod
    def _merge_adj_exons(
        current: FeatureRearranger.FeatureTuple, subsequent: FeatureRearranger.FeatureTuple
    ):
        gene_start = min(current.feature.location.start, subsequent.feature.location.start)
        gene_end = max(current.feature.location.end, subsequent.feature.location.end)
        current.feature.location = FeatureLocation(gene_start, gene_end)


class ExonSpliceInsertor:
    class TestsTuple(NamedTuple):
        is_same_previous: bool
        is_same_current: bool
        is_after_previous: bool
        is_before_current: bool
        not_overlap: bool
        not_same: bool

    def __init__(self, feat_rearr: FeatureRearranger):
        """
        Coordinates the handling of compound location features using insertion and merging.

        Args:
            feat_rearr: An interface for rearranging features.
        """
        self.feat_rearr = feat_rearr

    def insert(self):
        """
        The genes identified as having compound locations are split into simple location
        features. Each of these splits results in a simple location feature for each contiguous location
        of the compound location.

        We then examine each simple location and insert it into its expected location
        in the gene list. The expected location is determined by comparing the simple feature's end location
        with the end locations of the features in the gene list. If the expected location has no
        overlap with the proceeding and succeeding genes, and the feature is a different gene from those two,
        it is directly inserted into that expected location. Alternatively, if the expected location of the feature
        results in a flanking gene (strictly adjacent or overlapping) to be the same as the gene to be
        inserted they are merged. The merge can occur on the proceeding side, succeeding side, or both.
        When merging the gene location, the start location is minimized and the end location is maximized.
        """
        for insert in self.feat_rearr.simple_features():
            # set appropriate adjacent features
            previous = self.feat_rearr.get_previous(insert)
            current = self.feat_rearr.get_current(insert)

            # checks for how to handle the insert feature
            tests = self._get_adj_tests(previous, insert, current)

            # directly reposition if possible
            if tests.not_overlap and tests.not_same:
                self.feat_rearr.insert(insert)
                self.feat_rearr.log_position(insert, False)
            # attempt to resolve by merging
            else:
                self._try_merging(previous, insert, current, tests)

    def _get_adj_tests(
        self,
        previous: FeatureRearranger.FeatureTuple,
        insert: FeatureRearranger.FeatureTuple,
        current: FeatureRearranger.FeatureTuple,
    ) -> TestsTuple:
        is_same_previous = previous.gene == insert.gene
        is_same_current = current.gene == insert.gene
        is_after_previous = (
            not previous.feature or previous.feature.location.end < insert.feature.location.start
        )
        is_before_current = (
            not current.feature or insert.feature.location.end < current.feature.location.start
        )
        not_overlap = is_after_previous and is_before_current
        not_same = not is_same_previous and not is_same_current
        tests_tuple = self.TestsTuple(
            is_same_previous,
            is_same_current,
            is_after_previous,
            is_before_current,
            not_overlap,
            not_same,
        )
        return tests_tuple

    def _try_merging(
        self,
        previous: FeatureRearranger.FeatureTuple,
        insert: FeatureRearranger.FeatureTuple,
        current: FeatureRearranger.FeatureTuple,
        tests: TestsTuple,
    ):
        is_merged = False
        gene_start = insert.feature.location.start
        gene_end = insert.feature.location.end

        # if insert and current feature are the same gene, and insert starts before current,
        # remove current feature, and update ending location
        if tests.is_same_current and insert.feature.location.start < current.feature.location.start:
            gene_end = current.feature.location.end
            self.feat_rearr.remove(current)
            is_merged = True

        # if insert and previous feature are the same gene,
        # use the smaller start location, and remove previous feature
        if tests.is_same_previous:
            gene_start = min(gene_start, previous.feature.location.start)
            self.feat_rearr.remove(previous)
            # elements in list will shift to left, so update index
            insert.index -= 1
            is_merged = True

        # perform insertion if needed
        if is_merged:
            # updated feature to insert
            insert.feature.location = FeatureLocation(gene_start, gene_end)
            # insert
            self.feat_rearr.insert(insert)
            self.feat_rearr.log_position(insert)


# -----------------------------------------------------------------#


class DataCleaning:
    def __init__(self, plastid_data: PlastidData, user_params: dict[str, Any]):
        self.plastid_data = plastid_data
        self.min_num_taxa = user_params.get("min_num_taxa")
        self.min_seq_length = user_params.get("min_seq_length")
        self.exclude_region = user_params.get("exclude_region") or []

    def clean(self):
        log.info("cleaning extracted sequence annotations")
        if self.exclude_region:
            log.info("  removing excluded regions")
        if isinstance(self.min_num_taxa, float):
            log.info(
                f"  removing annotations that occur in fewer than {self.min_num_taxa:.2f} of taxa"
            )
        else:
            log.info(f"  removing annotations that occur in fewer than {self.min_num_taxa} taxa")
        log.info(
            f"  removing annotations whose longest sequence is shorter than {self.min_seq_length} bp"
        )

        summary = self.plastid_data.store.get_region_summary("nucl")
        num_taxa = getattr(self.plastid_data, "num_taxa", None)

        for feat_name, stats in summary.items():
            if self.exclude_region and feat_name in self.exclude_region:
                log.info(f"    removing {feat_name} for being in exclusion list")
                self.plastid_data.remove_nuc(feat_name)
                continue

            if stats["max_len"] < self.min_seq_length:
                log.info(
                    f"    removing {feat_name} for not reaching the minimum sequence length defined"
                )
                self.plastid_data.remove_nuc(feat_name)
                continue

            count = stats["count"]
            if isinstance(self.min_num_taxa, float):
                if num_taxa is None or num_taxa <= 0:
                    log.critical("Total number of taxa is unavailable for frequency filtering")
                    raise AttributeError(
                        "Total number of taxa is unavailable for frequency filtering"
                    )
                if (count / num_taxa) < self.min_num_taxa:
                    log.info(
                        f"    removing {feat_name} for not reaching the minimum relative frequency of taxa defined"
                    )
                    self.plastid_data.remove_nuc(feat_name)
            else:
                if count < self.min_num_taxa:
                    log.info(
                        f"    removing {feat_name} for not reaching the minimum number of taxa defined"
                    )
                    self.plastid_data.remove_nuc(feat_name)
