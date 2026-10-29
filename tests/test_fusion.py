from app.fusion import Hit, rrf


def h(cid, doc="d"):
    return Hit(cid, doc, f"text {cid}")


def test_item_in_both_lists_beats_item_in_one():
    dense = [h(1), h(2), h(3)]
    lexical = [h(3), h(4)]
    fused = rrf({"dense": dense, "lexical": lexical})
    assert fused[0].chunk_id == 3
    assert fused[0].ranks == {"dense": 3, "lexical": 1}


def test_scores_follow_the_formula():
    fused = rrf({"a": [h(1)], "b": [h(1)]}, k=60)
    assert abs(fused[0].score - 2 / 61) < 1e-12


def test_every_candidate_survives_fusion():
    fused = rrf({"a": [h(1), h(2)], "b": [h(3)]})
    assert {x.chunk_id for x in fused} == {1, 2, 3}


def test_weights_change_the_winner():
    a = [h(1)]
    b = [h(2)]
    assert rrf({"a": a, "b": b}, weights={"a": 2.0})[0].chunk_id == 1
    assert rrf({"a": a, "b": b}, weights={"b": 2.0})[0].chunk_id == 2


def test_empty_input():
    assert rrf({}) == []
    assert rrf({"a": []}) == []


def test_inputs_are_not_mutated():
    src = [h(1)]
    rrf({"a": src})
    assert src[0].score == 0.0 and src[0].ranks == {}
