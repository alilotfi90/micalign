"""
Runnable example (not collected by pytest — name does not start with test_).

    python run_example.py
"""
from micalign import AlignmentProcessor

if __name__ == "__main__":
    AlignmentProcessor(
        input_dir   = "samples images/samples",
        calibration = "samples images/caliberation",
        output_dir  = "aligned_out",
    ).run()
