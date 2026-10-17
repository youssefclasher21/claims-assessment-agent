from src.orchestration import WorkerPlan, normalize_plan


def test_none_plan_runs_all_workers():
    selected, notes = normalize_plan(None)
    assert selected == ["policy", "history", "documentation"]
    assert notes


def test_mandatory_workers_are_added():
    selected, notes = normalize_plan(WorkerPlan(workers=["history"], reasoning="x"))
    assert selected == ["policy", "history", "documentation"]
    assert any("mandatory" in n for n in notes)


def test_unknown_and_duplicate_names_are_dropped():
    selected, notes = normalize_plan(
        WorkerPlan(workers=["policy", "policy", "hack", "documentation"])
    )
    assert selected == ["policy", "documentation"]
    assert any("hack" in n for n in notes)


def test_empty_plan_still_runs_mandatory_workers():
    selected, _ = normalize_plan(WorkerPlan(workers=[]))
    assert selected == ["policy", "documentation"]