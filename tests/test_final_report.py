from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts.make_final_report import research_status


def exact_joint(status: str):
    return {
        "execution_status": "complete",
        "status": status,
        "exact_job_matrix_verified": True,
        "image_checksums_verified": True,
        "protocol_fingerprint": "same",
        "expected_protocol_fingerprint": "same",
    }


def test_final_status_requires_all_core_and_human_gates():
    selection = {"status": "selected"}
    calibration = {"gate_pass": True}
    assert research_status(exact_joint("validated"), calibration, [3], selection, {}, True) == "VALIDATED CANDIDATE SET"
    assert research_status(exact_joint("validated"), calibration, [3], selection, {}, False) == "NOT VALIDATED"
    assert research_status(exact_joint("validated"), {"gate_pass": False}, [3], selection, {}, True) == "NOT VALIDATED"


def test_final_status_preserves_valid_negative_and_no_go_outcomes():
    calibration = {"gate_pass": True}
    assert research_status(exact_joint("not_validated"), calibration, [3], {"status": "selected"}, {}, True) == "VALIDATED NEGATIVE JOINT RESULT"
    assert research_status(
        {},
        calibration,
        [],
        {"status": "no_go"},
        {"status": "validated_no_go", "selected_global_blocks": []},
        True,
    ) == "VALIDATED NO-GO"
