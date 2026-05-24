import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import List, Tuple, Optional, Any, Dict, NamedTuple, Generator
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation, SeqFeature
from copy import deepcopy
import multiprocessing
import os

# Package imports
from .seqfeature_ops import (PlastidData, PlastidDict, IntergenicDict,
                             GeneFeature, ProteinFeature, IntergenicFeature,
                             IntronFeature, PlastidFeature)
from .logging_ops import Logger, logger as log
from .helpers import split_list


class ExtractAndCollect:
    def __init__(self, plastid_data: PlastidData, user_params: Dict[str, Any]):
        """
        Coordinates the parsing of GenBank flatfiles, the extraction of the contained sequence annotations,
        and the collection of the resulting sequence records in a data structure.

        Args:
            plastid_data: Contains the locations of the files to be parsed,
                and where the extracted records will be stored.
            user_params: Specifications for how the extraction should be performed.
                These are `num_threads`, `out_dir`, `verbose`, and `exclude_fullcds`.
        """
        self.plastid_data = plastid_data
        self.num_threads = user_params.get("num_threads")
        self.out_dir = user_params.get("out_dir")
        self.verbose = user_params.get("verbose")
        self.exclude_cds = user_params.get("exclude_fullcds")
        self._set_extract_fun()

    def _set_extract_fun(self):
        if self.plastid_data.mode == "cds":
            self._extract_fun = self._extract_cds
        elif self.plastid_data.mode == "igs":
            self._extract_fun = self._extract_igs
        elif self.plastid_data.mode == "int":
            self._extract_fun = self._extract_int
        else:
            log.critical("Undefined extraction mode")
            raise ValueError

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

        # Step 0. Extract first genome in list for feature ordering
        if self.plastid_data.order == "seq":
            first_genome = self.plastid_data.files.pop(0)
            nuc_dict, prot_dict = self._extract_recs([first_genome])
            self.plastid_data.add_nucleotides(nuc_dict)
            self.plastid_data.add_proteins(prot_dict)

        # Step 1. Create the data for each worker
        file_lists = split_list(self.plastid_data.files, self.num_threads * 2)

        # Step 2. Use ProcessPoolExecutor to parallelize extraction
        mp_context = multiprocessing.get_context()
        with ProcessPoolExecutor(max_workers=self.num_threads, mp_context=mp_context) as executor:
            future_to_recs = [
                executor.submit(self._extract_recs, file_list) for file_list in file_lists
            ]
            for future in as_completed(future_to_recs):
                nuc_dict, prot_dict = future.result()
                self.plastid_data.add_nucleotides(nuc_dict)
                self.plastid_data.add_proteins(prot_dict)

        # Step 3. Stop execution if no nucleotides were extracted
        if not self.plastid_data.nucleotides.items():
            log.critical(f"No items in main dictionary: {self.out_dir}")
            raise Exception()

    def _extract_recs(self, files: List[str]) -> Tuple[PlastidDict, PlastidDict]:
        def extract_rec(file: str):
            try:
                log.info(f" parsing {os.path.basename(file)}")
                record = SeqIO.read(file, "genbank")
                extract_fun(record)
            except Exception as e:
                log.error(" %r generated an exception: %s" % (os.path.basename(file), e))

        # new logger instance for this child process (multiprocessing assumed)
        Logger.reinitialize_logger(self.verbose)

        # "bind" plastid dictionaries to the record extraction function
        nuc_dict = IntergenicDict() if self.plastid_data.mode == 'igs' else PlastidDict()
        extract_fun = partial(self._extract_fun, nuc_dict=nuc_dict)
        if self.plastid_data.mode.collects_proteins():
            prot_dict = PlastidDict()
            extract_fun = partial(extract_fun, protein_dict=prot_dict)
        else:
            prot_dict = None

        # extract the annotations and collect them in the plastid dictionaries
        for f in files:
            extract_rec(f)
        return nuc_dict, prot_dict

    def _extract_cds(self, rec: SeqRecord, nuc_dict: PlastidDict, protein_dict: PlastidDict):
        """
        Extracts all CDS (coding sequences = genes) from a given sequence record
        """
        for feature in RecordFeatures(rec, self.exclude_cds).cds_feats:
            # Step 1. Extract nucleotide sequence of each gene and add to dictionary
            gene = GeneFeature(rec, feature)
            nuc_dict.add_feature(gene)

            # Step 2. Translate nucleotide sequence to protein and add to dictionary
            protein = ProteinFeature(gene=gene)
            protein_dict.add_feature(protein)

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

            # Step 2. Make IGS SeqFeature
            igs = IntergenicFeature(rec, current_feat, subsequent_feat)

            # Step 3. Attach IGS to growing dictionary
            nuc_dict.add_feature(igs)

    def _extract_int(self, rec: SeqRecord, nuc_dict: PlastidDict):
        """
        Extracts all INT (introns) from a given sequence record
        """
        for feature in RecordFeatures(rec, self.exclude_cds).int_feats:
            # Step 1.a. If one intron in gene:
            if len(feature.location.parts) == 2:
                intron = IntronFeature(rec, feature)
                nuc_dict.add_feature(intron)

            # Step 1.b. If two introns in gene:
            elif len(feature.location.parts) == 3:
                feature_copy = deepcopy(
                    feature
                )  # Important b/c feature is overwritten in extract_internal_intron()

                intron = IntronFeature(rec, feature)
                nuc_dict.add_feature(intron)

                intron = IntronFeature(rec, feature_copy, 1)
                nuc_dict.add_feature(intron)


class RecordFeatures:
    def __init__(self, record: SeqRecord, exclude_list: Optional[List[str]] = None):
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
    def igs_feats(self) -> Generator[SeqFeature, None, None]:
        """
        A generator of IGS features in the record.
        """
        # Note: No need to include "if feature.type=='tRNA'", because all tRNAs are also annotated as genes
        features = [
            f for f in self._features if f.type == "gene" and self._not_exclude(f)
        ]
        # Handle spliced exons
        handler = ExonSpliceHandler(features, self.rec_name)
        handler.resolve()
        for f in features:
            yield f

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
    def __init__(self, genes: List[SeqFeature], rec_name: str):
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
        feature: Optional[SeqFeature] = None
        gene: Optional[str] = None
        is_trans: bool = False
        is_compound: bool = False
        index: int = -1

    def __init__(self, features: List[SeqFeature], rec_name: str):
        """
        An interface to modify the features of a genome record.

        Args:
            features: List of features.
            rec_name: Name of the record that contains the features.
        """
        self._features = features
        self.rec_name = rec_name
        self._set_end_positions()
        self._compound_features: List[SeqFeature] = []
        self._compounds_removed = False
        self._is_sorted = False

    def _set_end_positions(self):
        self._end_positions = [
            f.location.end for f in self._features
        ]

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
        trans_genes = [
            "rps12"
        ]
        return bool(feature.qualifiers.get("trans_splicing")) or gene in trans_genes

    def _tuple_from_feature(self, feature: SeqFeature, index: Optional[int] = None) -> FeatureTuple:
        gene = PlastidFeature.name_cleaner.get_safe_gene(feature)
        is_trans = self._get_is_trans(feature, gene)
        is_compound = type(feature.location) is CompoundLocation
        index = self._get_insert_index(feature) if index is None else index
        return self.FeatureTuple(
            feature,
            gene,
            is_trans,
            is_compound,
            index
        )

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
            self._features = [
                f for f in self._features if f not in self._compound_features
            ]
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
        subsequent_location = "\t\t\t\t\t" if not subsequent.feature else subsequent.feature.location

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
    def _merge_adj_exons(current: FeatureRearranger.FeatureTuple, subsequent: FeatureRearranger.FeatureTuple):
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
            current: FeatureRearranger.FeatureTuple
    ) -> TestsTuple:
        is_same_previous = previous.gene == insert.gene
        is_same_current = current.gene == insert.gene
        is_after_previous = not previous.feature or previous.feature.location.end < insert.feature.location.start
        is_before_current = not current.feature or insert.feature.location.end < current.feature.location.start
        not_overlap = is_after_previous and is_before_current
        not_same = not is_same_previous and not is_same_current
        tests_tuple = self.TestsTuple(
            is_same_previous,
            is_same_current,
            is_after_previous,
            is_before_current,
            not_overlap,
            not_same
        )
        return tests_tuple

    def _try_merging(
            self,
            previous: FeatureRearranger.FeatureTuple,
            insert: FeatureRearranger.FeatureTuple,
            current: FeatureRearranger.FeatureTuple,
            tests: TestsTuple
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
    def __init__(self, plastid_data: PlastidData, user_params: Dict[str, Any]):
        """
        Coordinates the cleaning (removal) of dictionary regions based on properties of the regions.

        Args:
            plastid_data: Plastid data to be cleaned.
            user_params: Specifications for how the cleaning process should be performed.
                These are `min_seq_length`, `min_num_taxa`, and `exclude_region`.
        """
        self.plastid_data = plastid_data
        self.min_num_taxa = user_params.get("min_num_taxa")
        self.min_seq_length = user_params.get("min_seq_length")
        self.exclude_region = user_params.get("exclude_region")

    def clean(self):
        """
        Cleans the nucleotide and protein dictionaries according to user specifications.
        Specifically, this removes regions that are below the threshold of
        `min_seq_length` or `min_num_taxa`.
        Additionally, any regions specified in `exclude_region` are removed.
        """
        log.info("cleaning extracted sequence annotations")
        if self.exclude_region:
            log.info(
                f"  removing excluded regions"
            )
        if isinstance(self.min_num_taxa, float):
            log.info(
                f"  removing annotations that occur in fewer than {self.min_num_taxa:.2f} of taxa"
            )
        else:
            log.info(
                f"  removing annotations that occur in fewer than {self.min_num_taxa} taxa"
            )
        log.info(
            f"  removing annotations whose longest sequence is shorter than {self.min_seq_length} bp"
        )
        for feat_name, rec_list in list(self.plastid_data.nucleotides.items()):
            self._remove_excluded(feat_name)
            self._remove_infreq(feat_name, rec_list)
            self._remove_short(feat_name, rec_list)

    def _remove_short(self, feat_name: str, rec_list: List[SeqRecord]):
        longest_seq = max([len(s.seq) for s in rec_list])
        if longest_seq < self.min_seq_length:
            log.info(f"    removing {feat_name} for not reaching the minimum sequence length defined")
            self.plastid_data.remove_nuc(feat_name)

    def _remove_infreq(self, feat_name: str, rec_list: List[SeqRecord]):
        if isinstance(self.min_num_taxa, float):
            num_taxa = getattr(self.plastid_data, "num_taxa", None)
            if num_taxa is None:
                log.critical("Total number of taxa is unavailable for frequency filtering")
                raise AttributeError
            frequency = len(rec_list) / num_taxa
            if frequency < self.min_num_taxa:
                log.info(
                    f"    removing {feat_name} for not reaching the minimum relative frequency of taxa defined"
                )
                self.plastid_data.remove_nuc(feat_name)
        else:
            if len(rec_list) < self.min_num_taxa:
                log.info(f"    removing {feat_name} for not reaching the minimum number of taxa defined")
                self.plastid_data.remove_nuc(feat_name)

    def _remove_excluded(self, feat_name: str):
        if feat_name in self.exclude_region:
            log.info(f"    removing {feat_name} for being in exclusion list")
            self.plastid_data.remove_nuc(feat_name)
