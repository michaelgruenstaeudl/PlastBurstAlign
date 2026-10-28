import os
import tempfile
import unittest

from plastburstalign.alignment_tools import CustomAligner, get_alignment_tool


class CustomAlignerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "in.fasta")
        self.output_file = os.path.join(self.temp_dir.name, "out.fasta")
        with open(self.input_file, "w") as f:
            f.write(">taxon1\nATGC\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_placeholder_substitution_and_execution(self):
        # Emulate alignment via copy command with shell redirection
        template = "cp %INPUT% %OUTPUT%"
        aligner = CustomAligner(command_template=template)
        success = aligner.align(self.input_file, self.output_file)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.output_file))
        with open(self.output_file) as f:
            self.assertEqual(f.read(), ">taxon1\nATGC\n")

    def test_thread_placeholder_interpolation(self):
        template = "echo threads=%THREADS% input=%INPUT% output=%OUTPUT% > %OUTPUT%"
        params = {"num_threads": 4, "alignment_command": template}
        aligner = CustomAligner(user_params=params)
        success = aligner.align(self.input_file, self.output_file)
        self.assertTrue(success)
        with open(self.output_file) as f:
            content = f.read().strip()
            self.assertIn("threads=4", content)
            self.assertIn(self.input_file, content)

    def test_factory_resolves_custom_aligner(self):
        params = {"alignment_command": "cp %INPUT% %OUTPUT%"}
        aligner = get_alignment_tool("mafft", user_params=params)
        self.assertIsInstance(aligner, CustomAligner)

    def test_failure_on_non_zero_exit_code(self):
        template = "false %INPUT% %OUTPUT%"
        aligner = CustomAligner(command_template=template)
        success = aligner.align(self.input_file, self.output_file)
        self.assertFalse(success)

    def test_failure_on_empty_output(self):
        template = "touch %OUTPUT%"
        aligner = CustomAligner(command_template=template)
        success = aligner.align(self.input_file, self.output_file)
        self.assertFalse(success)

    def test_cli_parsing_alignment_command(self):
        import sys
        from unittest.mock import patch
        from plastburstalign.user_parameters import UserParametersScript

        out_dir = os.path.join(self.temp_dir.name, "cli_out")
        test_args = [
            "plastburstalign",
            "-i",
            self.temp_dir.name,
            "-o",
            out_dir,
            "-s",
            "cds",
            "--alignmentcmd",
            "mafft --thread %THREADS% %INPUT% > %OUTPUT%",
        ]
        with patch.object(sys, "argv", test_args):
            params = UserParametersScript()
            self.assertEqual(
                params["alignment_command"],
                "mafft --thread %THREADS% %INPUT% > %OUTPUT%",
            )


if __name__ == "__main__":
    unittest.main()
