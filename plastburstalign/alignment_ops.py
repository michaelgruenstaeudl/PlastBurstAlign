from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import Any

from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Data.CodonTable import ambiguous_generic_by_id
from Bio.Nexus.Nexus import Nexus as NexusAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .alignment_tools import get_alignment_tool
from .helpers import bounded_submit_iterator, split_list
from .logging_ops import logger as log
from .nexus_ops import ConcatenationError, IncrementalNexusAccumulator
from .seqfeature_ops import PlastidData

# ===================================================================
# Module-level helper functions for multiprocessing
# ===================================================================


class AlignmentCoordination:
    def __init__(self, plastid_data: PlastidData, user_params: dict[str, Any]):
        """
        Coordinates the alignment of nucleotide or protein sequences.

        Args:
            plastid_data: Plastid data to be aligned.
            user_params: Specifications for how the alignment should be performed.
                These are `num_threads`, `out_dir`, `verbose`, and `concat`.
        """
        self.plastid_data = plastid_data
        self.user_params = user_params
        self.success_list = []
        self._unaligned_saved = False
        self._aligned_saved = False

        # Create the appropriate alignment tool
        tool_name = (user_params or {}).get("alignment_tool", "mafft") or "mafft"
        tool_path = user_params.get("alignment_tool_path") if user_params else None
        self.aligner = get_alignment_tool(tool_name, user_params, tool_path)

        if not self.aligner:
            log.error("Failed to initialize alignment tool, falling back to MAFFT")
            self.aligner = get_alignment_tool("mafft", user_params, None)

    def save_unaligned(self):
        """
        For each region in the plastid data, an unaligned nucleotide matrix of sequences is saved to file.
        These are saved in FASTA format, located in the path specified by `user_params.out_dir`.
        If the plastid data contains protein sequences, those are saved as unaligned protein matrices.
        """
        log.info("saving individual regions as unaligned nucleotide matrices")
        collect_proteins = self.plastid_data.mode.collects_proteins()
        out_dir = self.user_params.get("out_dir")
        store = self.plastid_data.store

        for feat_name in self.plastid_data.nucleotides.keys():
            try:
                out_fn_unalign_nucl = os.path.join(out_dir, f"nucl_{feat_name}.unalign.fasta")
                store.export_fasta(feat_name, "nucl", out_fn_unalign_nucl)

                if collect_proteins:
                    out_fn_unalign_prot = os.path.join(out_dir, f"prot_{feat_name}.unalign.fasta")
                    store.export_fasta(feat_name, "prot", out_fn_unalign_prot)
            except Exception as e:
                log.error(f"Error saving {feat_name}: {e}")

        self._unaligned_saved = True

    def perform_MSA(self):
        """
        For each region in the plastid data, the sequences are aligned and saved to file as a nucleotide matrix.
        These are saved in FASTA format, located in the path specified by `user_params.out_dir`.

        In the nucleotide case, this is the direct process of aligning the unaligned matrix of each region
        using a third-party software tool. For the protein case, each unaligned matrix is aligned, and
        then back-translated to a nucleotide matrix before saving to file.
        """
        if not self.aligner or not self.aligner.exec_path:
            log.error("unable to align regions; alignment tool is not properly configured")
            return
        elif not self._unaligned_saved:
            self.save_unaligned()

        log.info("conducting the alignment of extracted sequences")
        if self.plastid_data.mode.collects_proteins():
            self._prot_MSA()
        else:
            self._nuc_MSA()
        self._aligned_saved = True

    def _nuc_MSA(self):
        log.info(
            f"conducting multiple sequence alignments based on nucleotide sequence data using "
            f"{self.user_params.get('num_threads')} threads"
        )

        ### Inner Function - Start ###
        def single_nuc_MSA(k: str):
            log.debug(f" aligning {k}")
            # Define input and output names
            out_fn_unalign_nucl = os.path.join(
                self.user_params.get("out_dir"), f"nucl_{k}.unalign.fasta"
            )
            fasta_dir = os.path.join(self.user_params.get("out_dir"), "fasta")
            os.makedirs(fasta_dir, exist_ok=True)
            out_fn_aligned_nucl = os.path.join(fasta_dir, f"nucl_{k}.aligned.fasta")
            # Step 1. Align matrices via third-party alignment tool
            self.aligner.align(out_fn_unalign_nucl, out_fn_aligned_nucl)
            # Report whether the aligner produced output
            if os.path.exists(out_fn_aligned_nucl) and os.path.getsize(out_fn_aligned_nucl) > 0:
                log.debug(f"Alignment successful for `{k}` -> {out_fn_aligned_nucl}")
            else:
                log.warning(f"Alignment for `{k}` produced no output: {out_fn_aligned_nucl}")

        ### Inner Function - End ###

        # Step 2. Use ThreadPoolExecutor to parallelize alignment
        num_workers = self.user_params.get("num_threads")
        # num_workers = 1  # multiprocessing can cause issues on some platforms, so default to single-threaded for now
        if self.plastid_data.nucleotides.items():
            max_in_flight = num_workers * 2
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for k, future in bounded_submit_iterator(
                    executor,
                    single_nuc_MSA,
                    self.plastid_data.nucleotides.keys(),
                    max_in_flight=max_in_flight,
                ):
                    try:
                        future.result()
                    except Exception as e:
                        log.error(f"{k!r} generated an exception: {e}")
        else:
            log.critical("No items in nucleotide main dictionary to process")
            raise RuntimeError("No items in nucleotide main dictionary to process")

    def _prot_MSA(self):
        log.info(
            f" conducting multiple sequence alignments based on protein sequence data, "
            f"followed by back-translation to nucleotides using {self.user_params.get('num_threads')} threads"
        )

        ### Inner Function - Start ###
        def single_prot_MSA(k: str):
            log.debug(f" aligning {k}")
            # Define input and output names
            out_fn_unalign_prot = os.path.join(
                self.user_params.get("out_dir"), f"prot_{k}.unalign.fasta"
            )
            out_fn_aligned_prot = os.path.join(
                self.user_params.get("out_dir"), f"prot_{k}.aligned.fasta"
            )
            out_fn_unalign_nucl = os.path.join(
                self.user_params.get("out_dir"), f"nucl_{k}.unalign.fasta"
            )
            fasta_dir = os.path.join(self.user_params.get("out_dir"), "fasta")
            os.makedirs(fasta_dir, exist_ok=True)
            out_fn_aligned_nucl = os.path.join(fasta_dir, f"nucl_{k}.aligned.fasta")
            # Step 1. Align matrices based on their PROTEIN sequences via third-party alignment tool
            self.aligner.align(out_fn_unalign_prot, out_fn_aligned_prot)
            # Report whether the protein aligner produced output
            if os.path.exists(out_fn_aligned_prot) and os.path.getsize(out_fn_aligned_prot) > 0:
                log.debug(f"Protein alignment successful for `{k}` -> {out_fn_aligned_prot}")
            else:
                log.warning(
                    f"Protein alignment for `{k}` produced no output: {out_fn_aligned_prot}"
                )
                return
            # Step 2. Conduct actual back-translation from PROTEINS TO NUCLEOTIDES
            try:
                translator = BackTranslation(
                    k, out_fn_aligned_prot, out_fn_unalign_nucl, out_fn_aligned_nucl, 11
                )
                translator.backtranslate()
                # Report whether back-translation produced nucleotide alignment
                if os.path.exists(out_fn_aligned_nucl) and os.path.getsize(out_fn_aligned_nucl) > 0:
                    log.debug(f"Back-translation successful for `{k}` -> {out_fn_aligned_nucl}")
                else:
                    log.warning(
                        f"Back-translation produced no output for `{k}`: {out_fn_aligned_nucl}"
                    )
            except Exception as err:
                log.warning(f"Unable to conduct back-translation of `{k}`. Error message: {err}.")

        ### Inner Function - End ###

        # Step 2. Use ThreadPoolExecutor to parallelize alignment and back-translation with bounded queue
        num_workers = self.user_params.get("num_threads")
        max_in_flight = num_workers * 2
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for k, future in bounded_submit_iterator(
                executor,
                single_prot_MSA,
                self.plastid_data.nucleotides.keys(),
                max_in_flight=max_in_flight,
            ):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"{k} generated an exception: {e}")

    def perform_MSAs(self):
        """
        For each region in the plastid data, the sequences are aligned and saved to file in FASTA and NEXUS format
        in the path specified by `user_params.out_dir`.
        """

        if not self._aligned_saved:
            self.perform_MSA()

        # log.info(
        #     f"collecting all successful alignments using {self.user_params.get('num_threads')} processes"
        # )
        log.info(
            f"converting all successful alignments to NEXUS format using {self.user_params.get('num_threads')} threads"
        )
        msa_lists = split_list(
            list(self.plastid_data.nucleotides.keys()), self.user_params.get("num_threads") * 2
        )
        # Use ThreadPoolExecutor for I/O-bound file conversions
        num_workers = self.user_params.get("num_threads")
        # num_workers = 1  # multiprocessing can cause issues on some platforms, so default to single-threaded for now

        max_in_flight = num_workers * 2
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _, future in bounded_submit_iterator(
                executor, self.convert_MSAs, msa_lists, max_in_flight=max_in_flight
            ):
                success_list = future.result()
                if len(success_list) > 0:
                    self.success_list.extend(success_list)

    def convert_MSAs(self, msa_list: list[str]) -> list[tuple[str, str]]:
        """Convert each region's aligned FASTA file to NEXUS format and collect successful conversions."""

        def _convert_FASTA_to_NEXUS(msa_name: str) -> str | None:
            fasta_dir = os.path.join(self.user_params.get("out_dir"), "fasta")
            aligned_nucl_fasta = os.path.join(fasta_dir, f"nucl_{msa_name}.aligned.fasta")
            nexus_dir = os.path.join(self.user_params.get("out_dir"), "nexus")
            os.makedirs(nexus_dir, exist_ok=True)
            aligned_nucl_nexus = os.path.join(nexus_dir, f"nucl_{msa_name}.aligned.nexus")
            try:
                AlignIO.convert(
                    aligned_nucl_fasta,
                    "fasta",
                    aligned_nucl_nexus,
                    "nexus",
                    molecule_type="DNA",
                )
                return aligned_nucl_nexus
            except Exception:
                log.warning(f"Unable to convert alignment of `{msa_name}` from FASTA to NEXUS.")
                return None

        success_list = []
        for msa_name in msa_list:
            nexus_path = _convert_FASTA_to_NEXUS(msa_name)
            if nexus_path is not None:
                success_list.append((msa_name, nexus_path))
        return success_list

    def concat_MSAs(self):
        """
        Concatenated region alignments are saved to file in both NEXUS and FASTA formats,
        located in the path specified by `user_params.out_dir`.
        """
        num_successful = len(self.success_list)
        if num_successful == 0:
            log.warning("No successful alignments found to concatenate.")
            return

        def _concat_sync():
            try:
                with ThreadPoolExecutor(max_workers=2) as write_pool:
                    fut_nexus = write_pool.submit(
                        alignm_concat.write_nexus_data,
                        filename=out_fn_nucl_concat_nexus,
                    )
                    fut_fasta = write_pool.submit(
                        alignm_concat.export_fasta,
                        filename=out_fn_nucl_concat_fasta,
                    )
                    fut_nexus.result()
                    fut_fasta.result()
            except Exception as e:
                log.error(f"Writer thread failure while exporting alignments: {e}")
                raise ConcatenationError(
                    f"Failure while writing concatenated alignments: {e}"
                ) from e

        def _load_nexus_alignment(msa_name: str, nexus_path: str) -> NexusAlignment:
            # Parse nexus directly from disk without StringIO round-trips
            alignm_nexus = NexusAlignment(nexus_path)
            prefix = f"{msa_name}_"
            r_prefix = f"_R_{msa_name}_"

            new_matrix = {}
            new_taxlabels = []
            for t in alignm_nexus.taxlabels:
                if t.startswith(r_prefix):
                    clean_t = "concat_" + t[len(r_prefix):]
                elif t.startswith(prefix):
                    clean_t = "concat_" + t[len(prefix):]
                else:
                    clean_t = t
                new_matrix[clean_t] = alignm_nexus.matrix[t]
                new_taxlabels.append(clean_t)

            alignm_nexus.matrix = new_matrix
            alignm_nexus.taxlabels = new_taxlabels
            return alignm_nexus

        log.info(f"concatenating all successful alignments in `{self.plastid_data.order}` order")
        self.plastid_data.set_order_map()
        self.success_list.sort(key=lambda t: self.plastid_data.order_map[t[0]])

        out_fn_nucl_concat_fasta = os.path.join(
            self.user_params.get("out_dir"), f"nucl_{num_successful}concat.aligned.fasta"
        )
        out_fn_nucl_concat_nexus = os.path.join(
            self.user_params.get("out_dir"), f"nucl_{num_successful}concat.aligned.nexus"
        )

        try:
            accumulator = IncrementalNexusAccumulator()
            for msa_name, nexus_path in self.success_list:
                single_matrix = _load_nexus_alignment(msa_name, nexus_path)
                accumulator.add(msa_name, single_matrix)
            alignm_concat = accumulator.finalize()
        except Exception as e:
            log.critical(f"Unable to concatenate alignments: {e}")
            raise ConcatenationError(f"Alignment concatenation failed: {e}") from e

        if alignm_concat is None:
            log.warning("Concatenation produced empty output.")
            return

        log.info(" writing concatenation to file in NEXUS and FASTA formats")
        _concat_sync()


# -----------------------------------------------------------------#


class BackTranslation:
    def __init__(
        self,
        prot_name: str,
        prot_align_file: str,
        nuc_fasta_file: str,
        nuc_align_file: str,
        table: int = 0,
        align_format: str = "fasta",
        gap: str = "-",
    ):
        """
        Coordinate the back-translation of protein sequences to nucleotide sequences.

        Args:
            prot_name: Name of the protein being back-translated
            prot_align_file: Path to the file containing the aligned protein sequences.
            nuc_fasta_file: Path to the file containing the unaligned nucleotide sequences.
            nuc_align_file: Path to the output file for the back-translated nucleotide sequences.
            table: Genetic code table number (default is 0).
            align_format: Format of the alignment file (default is 'fasta').
            gap: String that designates a nucleotide gap (default is '-').
        """
        self.prot_name = prot_name
        self.align_format = align_format
        self.prot_align_file = prot_align_file
        self.nuc_fasta_file = nuc_fasta_file
        self.nuc_align_file = nuc_align_file
        self.table = table
        self.gap = gap

    def _evaluate_nuc(self, identifier: str, nuc: Seq, prot: Seq) -> Seq:
        """
        Returns nucleotide sequence if works (can remove trailing stop).
        """
        if len(nuc) % 3:
            log.debug(
                f"Nucleotide sequence for {identifier} is length {len(nuc)} (not a multiple of three)"
            )

        p = str(prot).upper().replace("*", "X")
        t = str(nuc.translate(self.table)).upper().replace("*", "X")
        if (
            len(t) == len(p) + 1
            and str(nuc)[-3:].upper() in ambiguous_generic_by_id[self.table].stop_codons
        ):
            # Allow this...
            t = t[:-1]
            nuc = nuc[:-3]  # edit return value
        if len(t) != len(p):
            debug = (
                f"Inconsistent lengths for {identifier}, ungapped protein {len(p)}, "
                f"tripled {len(p) * 3} vs ungapped nucleotide {len(nuc)}."
            )
            if t.endswith(p):
                debug += f"\nThere are {len(t) - len(p)} extra nucleotides at the start."
            elif t.startswith(p):
                debug += f"\nThere are {len(t) - len(p)} extra nucleotides at the end."
            elif p in t:
                debug += "\nHowever, protein sequence found within translated nucleotides."
            elif p[1:] in t:
                debug += "\nHowever, ignoring first amino acid, protein sequence found within translated nucleotides."
            log.debug(debug)

        if t == p:
            return nuc
        elif p.startswith("M") and "M" + t[1:] == p:
            if str(nuc[0:3]).upper() in ambiguous_generic_by_id[self.table].start_codons:
                return nuc
            else:
                log.debug(
                    f"translation for {identifier} would match if {nuc[0:3].upper()} "
                    f"was a start codon (check correct table used)"
                )
                log.warning(f"translation check failed for {identifier}")

        else:
            m = "".join("." if x == y else "!" for (x, y) in zip(p, t))
            if len(prot) < 70:
                log.debug(
                    f"translation mismatch for {identifier} [0:{len(prot)}]\n"
                    f"protein:     {p}\n"
                    f"             {m}\n"
                    f"translation: {t}\n"
                )
            else:
                for offset in range(0, len(p), 60):
                    log.debug(
                        f"translation mismatch for {identifier} [{offset}:{offset + 60}]\n"
                        f"protein:     {p[offset : offset + 60]}\n"
                        f"             {m[offset : offset + 60]}\n"
                        f"translation: {t[offset : offset + 60]}\n"
                    )
            log.warning(f"translation check failed for {identifier}")

    def _backtrans_seq(
        self,
        identifier: str,
        aligned_protein_record: SeqRecord,
        unaligned_nucleotide_record: SeqRecord,
    ) -> SeqRecord:
        # Per https://biopython.org/docs/1.81/api/Bio.Seq.html,
        # `replace` is proper replacement for depreciated method `ungap`.
        try:
            ungapped_protein = aligned_protein_record.seq.replace(self.gap, "")
        except AttributeError:
            ungapped_protein = aligned_protein_record.seq.ungap(self.gap)

        ungapped_nucleotide = unaligned_nucleotide_record.seq
        if self.table:
            ungapped_nucleotide = self._evaluate_nuc(
                identifier, ungapped_nucleotide, ungapped_protein
            )
        elif len(ungapped_protein) * 3 != len(ungapped_nucleotide):
            log.debug(
                f"Inconsistent lengths for {aligned_protein_record.id}, "
                f"ungapped protein {len(ungapped_protein)}, "
                f"tripled {len(ungapped_protein) * 3} vs "
                f"ungapped nucleotide {len(ungapped_nucleotide)}"
            )
            log.warning(f"Backtranslation failed for {identifier} due to ungapped length mismatch")
        if ungapped_nucleotide is None:
            return None

        seq = []
        nuc = str(ungapped_nucleotide)
        gap_codon = self.gap * 3
        for amino_acid in aligned_protein_record.seq:
            if amino_acid == self.gap:
                seq.append(gap_codon)
            else:
                seq.append(nuc[:3])
                nuc = nuc[3:]
        if len(nuc) > 0:
            log.debug(
                f"Nucleotide sequence for {unaligned_nucleotide_record.id} "
                f"longer than protein {aligned_protein_record.id}"
            )
            log.warning(f"Backtranslation failed for {identifier} due to unaligned length mismatch")
            return None

        aligned_nuc = unaligned_nucleotide_record[:]  # copy for most annotation
        aligned_nuc.letter_annotation = {}  # clear this
        aligned_nuc.seq = Seq(
            "".join(seq)
        )  # , alpha)  # Modification on 09-Sep-2022 by M. Gruenstaeudl
        if len(aligned_protein_record.seq) * 3 != len(aligned_nuc):
            log.debug(
                f"Nucleotide sequence for {aligned_nuc.id} is length {len(aligned_nuc)} "
                f"but protein sequence {aligned_protein_record.seq} is length {len(aligned_protein_record.seq)} "
                f"(3 * {len(aligned_nuc)} != {len(aligned_protein_record.seq)})"
            )
            log.warning(f"Backtranslation failed for {identifier} due to aligned length mismatch")
            return None

        return aligned_nuc

    def _backtrans_seqs(
        self,
        protein_alignment: MultipleSeqAlignment,
        nucleotide_records: Mapping,
        key_function: Callable[[str], str] = lambda x: x,
    ):
        """
        Thread nucleotide sequences onto a protein alignment.
        """
        aligned = []
        for protein in protein_alignment:
            identifier = self._get_identifier(protein.id)
            try:
                nucleotide = nucleotide_records[key_function(protein.id)]
            except KeyError:
                raise ValueError(f"Could not find nucleotide sequence for {identifier}")
            sequence = self._backtrans_seq(identifier, protein, nucleotide)
            if sequence is not None:
                aligned.append(sequence)
        return MultipleSeqAlignment(aligned)

    def _get_identifier(self, rec_id: str) -> str:
        rec_name = rec_id.replace(self.prot_name + "_", "")
        identifier = f"protein '{self.prot_name}' in {rec_name}"
        return identifier

    def backtranslate(self):
        """
        Perform back-translation of a protein alignment to nucleotides.
        """
        # Step 1. Load the protein alignment
        prot_align = AlignIO.read(self.prot_align_file, self.align_format)
        # Step 2. Index the unaligned nucleotide sequences
        nuc_dict = SeqIO.index(self.nuc_fasta_file, "fasta")
        # Step 3. Perform back-translation
        nuc_align = self._backtrans_seqs(prot_align, nuc_dict)
        log.debug(
            f"  back-translation of `{self.prot_name}` complete, "
            f"{len(nuc_align)} sequences of length {nuc_align.get_alignment_length()}"
        )
        # Step 4. Write the back-translated nucleotide alignment to a file
        with open(self.nuc_align_file, "w") as output_handle:
            AlignIO.write(nuc_align, output_handle, self.align_format)
