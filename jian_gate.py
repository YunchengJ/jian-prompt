"""Task-scoped, deny-overrides gate for host-registered zero-argument actions.

The host supplies identities and binds scope IDs to fixed callbacks. This module
does not authenticate people, infer scopes from prose, inspect hidden model
policies, or secure tools that bypass it. Never expose its configuration or
origin arguments as model-controlled tool parameters.
"""
from __future__ import annotations

from collections.abc import Mapping
import re
from threading import RLock
from types import MappingProxyType


class GateError(ValueError):
    """The requested gate operation is invalid or the action is not allowed."""


class ActionGate:
    """One task's action registry and sticky denials; no API clears a denial.

    Create this once per task and retain it across permission messages. A later
    human allow records permission but cannot erase human, model, or native
    denials. This deliberately omits revocation/override workflows. Creating a
    new gate must not be used by the host to bypass a still-applicable refusal.
    """

    def __init__(self, actions, *, native_denials=(), native_allowlist=()):
        if not isinstance(actions, Mapping) or not actions:
            raise GateError("需要非空的宿主动作注册表")
        snapshot = dict(actions)
        for scope, action in snapshot.items():
            self._validate_id(scope)
            if not callable(action):
                raise GateError(f"动作 {scope} 必须是宿主注册的可调用对象")
        self._actions = MappingProxyType(snapshot)
        self._human_allows = set()
        self._denials = {scope: set() for scope in snapshot}
        self._global_denials = set()
        self._native_allows = {scope: set() for scope in snapshot}
        self._lock = RLock()
        native_scopes = self._scope_collection(native_denials, "native_denials")
        self._native_allowlist = frozenset(self._scope_collection(native_allowlist, "native_allowlist"))
        for scope in native_scopes:
            self.deny(scope, origin="native")

    def _scope_collection(self, scopes, name):
        if isinstance(scopes, (str, bytes)):
            raise GateError(f"{name} 需要作用域 ID 的集合，不接受单个字符串")
        try:
            result = tuple(scopes)
        except TypeError as exc:
            raise GateError(f"{name} 需要可迭代的作用域 ID") from exc
        for scope in result:
            self._scope(scope)
        return result

    @staticmethod
    def _validate_id(scope):
        if not isinstance(scope, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", scope) is None:
            raise GateError("作用域必须是宿主预定义的 ASCII ID，不自动解释自然语言或路径")

    def _scope(self, scope):
        self._validate_id(scope)
        if scope not in self._actions:
            raise GateError(f"未注册作用域: {scope}")

    @staticmethod
    def _denial_origin(origin):
        if origin not in ("human", "model", "native"):
            raise GateError("拒绝来源必须由宿主标为 human/model/native")

    def allow(self, scope, *, origin):
        """Record host-confirmed human permission, without lifting any denial."""
        if origin != "human":
            raise GateError("只有宿主确认的人类来源可以登记许可；模型不能给自己授权")
        with self._lock:
            self._scope(scope)
            self._human_allows.add(scope)

    def deny(self, scope, *, origin):
        self._denial_origin(origin)
        with self._lock:
            self._scope(scope)
            self._denials[scope].add(origin)

    def deny_all(self, *, origin):
        """Block the entire task when a refusal's narrower scope is unresolved."""
        self._denial_origin(origin)
        with self._lock:
            self._global_denials.add(origin)

    def affirm(self, scope, *, origin):
        """Record a host-verified upper-level 可 for an opt-in narrow allowlist.

        It never supplies human permission and never erases any denial.
        Ordinary scopes outside this list do not require an affirmation.
        """
        if origin not in ("model", "native"):
            raise GateError("上层肯定只能来自宿主核验的 model/native 来源")
        with self._lock:
            self._scope(scope)
            if scope not in self._native_allowlist:
                raise GateError("可只对宿主预定义的窄白名单项启用")
            self._native_allows[scope].add(origin)

    def decision(self, scope):
        with self._lock:
            self._scope(scope)
            denied_by = sorted(self._denials[scope] | self._global_denials)
            human_allow = scope in self._human_allows
            native_allow = bool(self._native_allows[scope])
            whitelist_required = scope in self._native_allowlist
            return {"scope": scope,
                    "allowed": human_allow and not denied_by and (not whitelist_required or native_allow),
                    "human_allow": human_allow, "denied_by": denied_by,
                    "allow_level": 1 if human_allow else None,
                    "denial_levels": {origin: 1 if origin == "human" else 2 for origin in denied_by},
                    "native_allow": native_allow, "native_allow_level": 2 if native_allow else None,
                    "whitelist_required": whitelist_required}

    def invoke(self, scope):
        """Recheck just before a fixed callback; propagate errors, never retry.

        The lock serializes this gate's calls; it cannot cancel an action that
        has already started or provide transactionality inside the callback.
        """
        with self._lock:
            decision = self.decision(scope)
            if not decision["allowed"]:
                raise GateError(f"动作未获放行: {scope}; denied_by={decision['denied_by']}")
            return self._actions[scope]()
