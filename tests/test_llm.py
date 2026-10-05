from app.llm import INSUFFICIENT_EVIDENCE, generate_answer


def test_no_evidence_never_calls_llm():
    answer, mode = generate_answer("Predict the 2030 price", [])
    assert answer == INSUFFICIENT_EVIDENCE
    assert mode == "insufficient_evidence"
