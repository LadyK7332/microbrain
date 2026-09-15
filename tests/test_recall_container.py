from microbrain.memory.recall_container import (
    build_recall_container,
    build_recall_plan,
    classify_recall_route,
    infer_hit_organ,
)


def test_recall_route_specializes_surface_and_object_queries():
    assert classify_recall_route("Good evening", statement_kind="greeting") == "surface"
    assert classify_recall_route("What is this apple?") == "object"
    assert classify_recall_route("Why does this keep happening?", statement_kind="question") == "pattern"


def test_recall_container_keeps_bounded_active_index():
    plan = build_recall_plan("What is apple?", active_limit=15, candidate_limit=100)
    hits = [
        {
            "cell_id": f"cell:{idx}",
            "kind": "object_descriptor" if idx % 2 else "thought_template",
            "tier": "long",
            "score": 1.0 - (idx * 0.001),
            "anchor": {"kind": "object/apple", "ref": f"apple descriptor {idx}"},
        }
        for idx in range(40)
    ]
    container = build_recall_container(plan, hits)
    assert container["candidate_count"] == 40
    assert container["active_count"] == 15
    assert container["mutation"] is False
    assert len(container["active_evidence"]) == 15
    assert container["bookmark"]["last_cell_id"] == "cell:39"


def test_infer_hit_organ_marks_surface_memory():
    hit = {"kind": "surface_structure_v2", "anchor": {"kind": "language/surface", "ref": "hello"}}
    assert infer_hit_organ(hit) == "surface_memory"
