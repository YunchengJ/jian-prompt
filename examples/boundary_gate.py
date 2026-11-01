"""Offline illustration only: every action appends a label to a local list."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jian_gate import ActionGate, GateError


def main():
    calls = []

    def blocked(gate, scope):
        before = len(calls)
        try:
            gate.invoke(scope)
        except GateError:
            assert len(calls) == before
        else:
            raise AssertionError("expected the gate to block this action")

    # Normal mode needs host-confirmed human permission, but no positive 可.
    normal = ActionGate({"note.save": lambda: calls.append("normal")})
    blocked(normal, "note.save")
    normal.allow("note.save", origin="human")
    normal.invoke("note.save")

    # A later low-level permission never clears a model/native refusal.
    denied = ActionGate({"note.save": lambda: calls.append("denied")},
                        native_denials=["note.save"])
    denied.allow("note.save", origin="human")
    blocked(denied, "note.save")

    # Only an explicitly configured narrow allowlist accepts an upper-level 可.
    narrow = ActionGate({"status.read": lambda: calls.append("whitelisted")},
                        native_allowlist=["status.read"])
    narrow.allow("status.read", origin="human")
    blocked(narrow, "status.read")
    narrow.affirm("status.read", origin="model")
    narrow.invoke("status.read")
    narrow.deny("status.read", origin="human")
    narrow.affirm("status.read", origin="native")
    blocked(narrow, "status.read")

    assert calls == ["normal", "whitelisted"]
    print("PASS: normal and opt-in allowlist paths; denials always prevail.")
    print("Only two in-memory callbacks ran. No model or real tool was called.")


if __name__ == "__main__":
    main()
