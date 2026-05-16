"""Tests for vanilla_dag.py — engine logic only, no I/O."""

import pytest
from vanilla_dag import DAG, Status


# ── helpers ────────────────────────────────────────────────────────────────────

def ok(ctx):
    """Node that always succeeds."""
    pass

def fail(ctx):
    """Node that always raises."""
    raise RuntimeError("intentional failure")

def write(key, value):
    """Returns a node func that writes a value to the context."""
    def _node(ctx):
        ctx[key] = value
    return _node

def read_and_write(read_key, write_key):
    """Returns a node func that reads one key and writes another."""
    def _node(ctx):
        ctx[write_key] = ctx[read_key] + "_transformed"
    return _node


# ── topological sort ───────────────────────────────────────────────────────────

def test_topological_sort_linear():
    dag = DAG()
    dag.add_node("a", ok)
    dag.add_node("b", ok)
    dag.add_node("c", ok)
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")

    order = dag._topological_sort()
    names = [n.name for n in order]
    assert names.index("a") < names.index("b") < names.index("c")


def test_topological_sort_diamond():
    # a → b, a → c, b → d, c → d
    dag = DAG()
    for name in ["a", "b", "c", "d"]:
        dag.add_node(name, ok)
    dag.add_edge("a", "b")
    dag.add_edge("a", "c")
    dag.add_edge("b", "d")
    dag.add_edge("c", "d")

    order = dag._topological_sort()
    names = [n.name for n in order]
    assert names.index("a") < names.index("b")
    assert names.index("a") < names.index("c")
    assert names.index("b") < names.index("d")
    assert names.index("c") < names.index("d")


def test_topological_sort_cycle_raises():
    dag = DAG()
    dag.add_node("a", ok)
    dag.add_node("b", ok)
    dag.add_edge("a", "b")
    dag.add_edge("b", "a")

    with pytest.raises(ValueError, match="Cycle detected"):
        dag._topological_sort()


def test_topological_sort_independent_nodes():
    dag = DAG()
    dag.add_node("x", ok)
    dag.add_node("y", ok)
    dag.add_node("z", ok)

    order = dag._topological_sort()
    assert len(order) == 3


# ── run: status transitions ────────────────────────────────────────────────────

def test_run_all_success():
    dag = DAG()
    dag.add_node("a", ok)
    dag.add_node("b", ok)
    dag.add_edge("a", "b")
    dag.run()

    assert dag.nodes["a"].status == Status.SUCCESS
    assert dag.nodes["b"].status == Status.SUCCESS


def test_run_failed_node_is_marked():
    dag = DAG()
    dag.add_node("a", fail)
    dag.run()

    assert dag.nodes["a"].status == Status.FAILED
    assert dag.nodes["a"].error is not None


def test_run_failure_blocks_direct_downstream():
    dag = DAG()
    dag.add_node("a", fail)
    dag.add_node("b", ok)
    dag.add_edge("a", "b")
    dag.run()

    assert dag.nodes["a"].status == Status.FAILED
    assert dag.nodes["b"].status == Status.BLOCKED


def test_run_failure_blocks_transitive_downstream():
    # a(fail) → b → c — both b and c should be BLOCKED
    dag = DAG()
    dag.add_node("a", fail)
    dag.add_node("b", ok)
    dag.add_node("c", ok)
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    dag.run()

    assert dag.nodes["b"].status == Status.BLOCKED
    assert dag.nodes["c"].status == Status.BLOCKED


def test_run_failure_does_not_block_sibling_branch():
    # a → b(fail) → c
    # a → d → e
    # d and e should still succeed
    dag = DAG()
    dag.add_node("a", ok)
    dag.add_node("b", fail)
    dag.add_node("c", ok)
    dag.add_node("d", ok)
    dag.add_node("e", ok)
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    dag.add_edge("a", "d")
    dag.add_edge("d", "e")
    dag.run()

    assert dag.nodes["b"].status == Status.FAILED
    assert dag.nodes["c"].status == Status.BLOCKED
    assert dag.nodes["d"].status == Status.SUCCESS
    assert dag.nodes["e"].status == Status.SUCCESS


def test_run_root_with_no_edges():
    dag = DAG()
    dag.add_node("solo", ok)
    dag.run()

    assert dag.nodes["solo"].status == Status.SUCCESS


# ── context ────────────────────────────────────────────────────────────────────

def test_context_is_shared_between_nodes():
    dag = DAG()
    dag.add_node("producer", write("message", "hello"))
    dag.add_node("consumer", read_and_write("message", "result"))
    dag.add_edge("producer", "consumer")
    dag.run()

    assert dag.context["message"] == "hello"
    assert dag.context["result"] == "hello_transformed"


def test_context_preloaded_before_run():
    dag = DAG()
    dag.context["seed"] = 42

    def use_seed(ctx):
        ctx["doubled"] = ctx["seed"] * 2

    dag.add_node("a", use_seed)
    dag.run()

    assert dag.context["doubled"] == 84


# ── reprocess ──────────────────────────────────────────────────────────────────

def test_reprocess_resets_target_and_ancestors():
    dag = DAG()
    dag.add_node("a", ok)
    dag.add_node("b", ok)
    dag.add_node("c", ok)
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    dag.run()

    # manually corrupt b's status to simulate a stale/failed state
    dag.nodes["b"].status = Status.FAILED

    dag.reprocess("c")

    assert dag.nodes["a"].status == Status.SUCCESS
    assert dag.nodes["b"].status == Status.SUCCESS
    assert dag.nodes["c"].status == Status.SUCCESS


def test_reprocess_leaves_unrelated_nodes_intact():
    # a → b(fail) → c
    # a → d → e  (unrelated branch)
    dag = DAG()
    dag.add_node("a", ok)
    dag.add_node("b", fail)
    dag.add_node("c", ok)
    dag.add_node("d", ok)
    dag.add_node("e", ok)
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    dag.add_edge("a", "d")
    dag.add_edge("d", "e")
    dag.run()

    assert dag.nodes["d"].status == Status.SUCCESS
    assert dag.nodes["e"].status == Status.SUCCESS

    run_count = {"d": 0, "e": 0}

    def counting(name):
        def _node(ctx):
            run_count[name] += 1
        return _node

    dag.nodes["d"].func = counting("d")
    dag.nodes["e"].func = counting("e")

    # replace b with a passing func and reprocess c
    dag.nodes["b"].func = ok
    dag.reprocess("c")

    assert run_count["d"] == 0, "d should not have been re-executed"
    assert run_count["e"] == 0, "e should not have been re-executed"


def test_reprocess_failure_still_blocks_downstream():
    dag = DAG()
    dag.add_node("a", fail)
    dag.add_node("b", ok)
    dag.add_edge("a", "b")
    dag.run()

    dag.reprocess("b")

    assert dag.nodes["a"].status == Status.FAILED
    assert dag.nodes["b"].status == Status.BLOCKED


def test_reprocess_multiple_targets():
    dag = DAG()
    dag.add_node("a", ok)
    dag.add_node("b", ok)
    dag.add_node("c", ok)
    dag.run()

    dag.nodes["b"].status = Status.FAILED
    dag.nodes["c"].status = Status.FAILED

    dag.reprocess("b", "c")

    assert dag.nodes["b"].status == Status.SUCCESS
    assert dag.nodes["c"].status == Status.SUCCESS
