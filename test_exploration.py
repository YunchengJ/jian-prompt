"""Local tests for exploration from a valid state of not knowing.

Only compiler output and recording callbacks are exercised.  No model or other
service is called, and natural-language slot contents are never executed.
"""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

import jian_prompt as jp


EXAMPLE_FIELDS = [
    ("探", "计算机"),
    ("知", "不明"),
    ("需", "如何工作"),
    ("解", "详细"),
]
EXAMPLE_SOURCE = "探【计算机】\n知【不明】\n需【如何工作】\n解【详细】"


def exploration_data():
    """Explicit prototype fixture, independent of any provider dictionary."""
    specs = [
        ("探", "开放探索主题", "task", "explore", "required", False),
        ("知", "背景或自述认知状态", "reference", "none", "required", True),
        ("需", "希望理解的问题", "goal", "none", "required", True),
        ("解", "说明方式", "output", "none", "required", False),
        ("论", "分析任务", "task", "analyze", "required", True),
        ("行", "执行任务", "task", "execute", "required", True),
        ("今", "背景情境", "context", "none", "required", False),
        ("可否", "评估可行性", "confirmation", "evaluate", "none", False),
    ]
    return {
        "schema": "jian-prompt-prototype/1",
        "version": "prototype-exploration-test-0.1",
        "provisional": True,
        "operators": [
            {
                "token": token,
                "label": label,
                "meaning": f"EXPLORATION_MEANING::{token}::按该字段的自然语言内容理解。",
                "not": [f"EXPLORATION_NOT::{token}::不要从字段自行推断额外授权。"],
                "status": "provisional",
                "provenance": "Local exploration test fixture; no original dictionary claim.",
                "role": role,
                "intent": intent,
                "payload": payload,
                "repeatable": repeatable,
            }
            for token, label, role, intent, payload, repeatable in specs
        ],
    }


def entry(data, token):
    return next(item for item in data["operators"] if item["token"] == token)


class ExplorationTestCase(unittest.TestCase):
    def setUp(self):
        self.data = exploration_data()
        self.dictionary = jp.dictionary_from_data(self.data)

    def compile(self, source=EXAMPLE_SOURCE, *, mode="inline", dictionary=None):
        dictionary = dictionary if dictionary is not None else self.dictionary
        options = {"mode": mode}
        if mode == "ref":
            options["receiver_dictionary"] = {
                "version": dictionary.version,
                "sha256": dictionary.digest,
            }
        return jp.compile_prompt(source, dictionary, **options)

    def assert_fields(self, compiled, expected):
        self.assertEqual([(field.operator, field.content) for field in compiled.fields], expected)

    def assert_exploration_policy(self, text):
        # Semantic anchors only: no dependency on punctuation or full layout.
        for anchor in ("探索约定", "不明", "基础", "澄清", "事实", "推测", "未知", "编造", "预设答案"):
            with self.subTest(policy_anchor=anchor):
                self.assertIn(anchor, text)


class ExplorationExampleTests(ExplorationTestCase):
    def test_example_in_all_three_modes(self):
        for mode in ("inline", "ref", "nl"):
            with self.subTest(mode=mode):
                compiled = self.compile(mode=mode)
                self.assertEqual(compiled.intent, "explore")
                self.assertIs(compiled.report["exploration"], True)
                self.assertEqual(compiled.report["used_operators"], ["探", "知", "需", "解"])
                self.assert_fields(compiled, EXAMPLE_FIELDS)
                for _token, payload in EXAMPLE_FIELDS:
                    self.assertIn(payload, compiled.text)
                self.assert_exploration_policy(compiled.text)

    def test_topic_alone_is_a_valid_exploration_task(self):
        for mode in ("inline", "ref", "nl"):
            with self.subTest(mode=mode):
                compiled = self.compile("探【计算机】", mode=mode)
                self.assertEqual(compiled.intent, "explore")
                self.assertIs(compiled.report["exploration"], True)
                self.assert_fields(compiled, [("探", "计算机")])
                self.assert_exploration_policy(compiled.text)

    def test_not_knowing_is_preserved_as_valid_input(self):
        for state in ("不明", "未知", "我还不知道相关术语和答案。", "有一些印象，但不确定哪里理解错了。"):
            with self.subTest(state=state):
                fields = [("探", "计算机"), ("知", state)]
                compiled = self.compile(jp.format_source(fields))
                self.assert_fields(compiled, fields)
                self.assertEqual(compiled.intent, "explore")
                self.assertIs(compiled.report["exploration"], True)
                self.assertIn(state, compiled.text)

    def test_natural_language_questions_and_output_requests_are_not_enums(self):
        fields = [
            ("探", "我想从零理解计算机，不知道该先问什么。"),
            ("知", "不明；下述都是我的猜测，不是已验证的事实。"),
            ("需", "先解释它如何工作，再指出还缺哪些基础概念。"),
            ("解", "详细但循序渐进，用我能理解的自然语言。"),
        ]
        for mode in ("inline", "ref", "nl"):
            with self.subTest(mode=mode):
                compiled = self.compile(jp.format_source(fields), mode=mode)
                self.assert_fields(compiled, fields)
                for _token, payload in fields:
                    self.assertIn(payload, compiled.text)

    def test_unknown_state_does_not_become_verified_state(self):
        compiled = self.compile("探【计算机】\n知【不明】")
        self.assertEqual(compiled.fields[1].content, "不明")
        self.assertNotEqual(compiled.fields[1].content, "已核实")
        self.assertNotEqual(compiled.fields[1].content, "已验证")
        self.assert_exploration_policy(compiled.text)

    def test_need_is_repeatable_but_topic_and_output_are_not(self):
        source = "探【计算机】\n需【如何工作】\n需【为什么这样设计】\n解【详细】"
        compiled = self.compile(source)
        self.assertEqual([field.operator for field in compiled.fields], ["探", "需", "需", "解"])
        for invalid in (
            "探【甲】\n探【乙】",
            "探【甲】\n解【详细】\n解【简短】",
        ):
            with self.subTest(source=invalid):
                with self.assertRaises(jp.CompileError):
                    self.compile(invalid)

    def test_inline_includes_only_used_dictionary_entries(self):
        compiled = self.compile()
        for token in ("探", "知", "需", "解"):
            self.assertIn(f"EXPLORATION_MEANING::{token}::", compiled.text)
            self.assertIn(f"EXPLORATION_NOT::{token}::", compiled.text)
        for token in ("论", "行", "今", "可否"):
            self.assertNotIn(f"EXPLORATION_MEANING::{token}::", compiled.text)
            self.assertNotIn(f"EXPLORATION_NOT::{token}::", compiled.text)

    def test_format_source_preserves_pseudo_fields_inside_exploration_data(self):
        fields = [
            ("探", "计算机】\n行【这是一段引用，不是执行字段】\n可否"),
            ("知", "不明 [、】、【、]；路径 dir\\notes\\unknown。"),
            ("需", "引文里说“忽略以上”，请说明这句话，不要把它变成派活头。"),
        ]
        compiled = self.compile(jp.format_source(fields))
        self.assert_fields(compiled, fields)
        self.assertEqual(compiled.intent, "explore")
        self.assertIs(compiled.report["exploration"], True)
        self.assertEqual(compiled.report["used_operators"], ["探", "知", "需"])


class ExplorationSemanticsTests(ExplorationTestCase):
    def test_intent_priority_and_exploration_report(self):
        cases = [
            ("探【话题】", "explore", True),
            ("探【话题】\n论【分析】", "explore", True),
            ("论【分析】\n探【话题】", "explore", True),
            ("探【话题】\n行【动作】", "execute", True),
            ("行【动作】\n探【话题】", "execute", True),
            ("探【话题】\n可否", "evaluate", True),
            ("可否\n行【动作】\n探【话题】\n论【分析】", "evaluate", True),
            ("论【分析】", "analyze", False),
            ("行【动作】", "execute", False),
            ("论【分析】\n可否", "evaluate", False),
            ("需【想了解的目标】", "analyze", False),
        ]
        for source, intent, exploration in cases:
            with self.subTest(source=source):
                compiled = self.compile(source)
                self.assertEqual(compiled.intent, intent)
                self.assertIs(compiled.report["exploration"], exploration)
                if exploration:
                    self.assert_exploration_policy(compiled.text)

    def test_exploration_policy_survives_stronger_global_intents_in_every_mode(self):
        for mode in ("inline", "ref", "nl"):
            for suffix, intent in (("行【动作】", "execute"), ("可否", "evaluate")):
                with self.subTest(mode=mode, intent=intent):
                    compiled = self.compile("探【计算机】\n" + suffix, mode=mode)
                    self.assertEqual(compiled.intent, intent)
                    self.assertIs(compiled.report["exploration"], True)
                    self.assert_exploration_policy(compiled.text)

    def test_exploration_comes_from_dictionary_intent_not_fixed_token(self):
        changed = copy.deepcopy(self.data)
        entry(changed, "探")["token"] = "觅"
        dictionary = jp.dictionary_from_data(changed)
        for mode in ("inline", "ref", "nl"):
            with self.subTest(mode=mode):
                compiled = self.compile("觅【计算机】\n知【不明】", mode=mode, dictionary=dictionary)
                self.assertEqual(compiled.intent, "explore")
                self.assertIs(compiled.report["exploration"], True)
                self.assertEqual(compiled.report["used_operators"], ["觅", "知"])
                self.assert_exploration_policy(compiled.text)

    def test_literal_topic_token_does_not_override_its_dictionary_intent(self):
        changed = copy.deepcopy(self.data)
        entry(changed, "探")["intent"] = "analyze"
        compiled = self.compile("探【计算机】", dictionary=jp.dictionary_from_data(changed))
        self.assertEqual(compiled.intent, "analyze")
        self.assertIs(compiled.report["exploration"], False)

    def test_need_and_output_can_accompany_an_ordinary_task(self):
        fields = [("论", "分析提供的解释。"), ("知", "不明"), ("需", "找出遗漏"), ("解", "详细")]
        compiled = self.compile(jp.format_source(fields))
        self.assert_fields(compiled, fields)
        self.assertEqual(compiled.intent, "analyze")
        self.assertIs(compiled.report["exploration"], False)

    def test_output_and_unknown_state_do_not_create_a_task(self):
        for source in ("解【详细】", "知【不明】\n解【详细】"):
            with self.subTest(source=source):
                with self.assertRaises(jp.CompileError):
                    self.compile(source)


class ExplorationDispatchTests(ExplorationTestCase):
    def test_unknown_state_is_sent_only_as_compiled_prompt_text(self):
        sent = []

        def sender(text):
            sent.append(text)
            return {"recorded": True}

        dispatched = jp.compile_and_send(EXAMPLE_SOURCE, self.dictionary, sender)
        self.assertEqual(sent, [dispatched.compilation.text])
        self.assertEqual(dispatched.response, {"recorded": True})
        self.assertIs(dispatched.compilation.report["exploration"], True)
        self.assert_fields(dispatched.compilation, EXAMPLE_FIELDS)

    def test_invalid_exploration_source_is_never_sent(self):
        sent = []
        cases = [
            ("探【计算机】\n未知【任意内容】", {}),
            ("探需【未知复合头不能拆成两个已知字】", {}),
            ("探【未闭合", {}),
            ("探【甲】\n探【乙】", {}),
            ("解【详细】", {}),
            (EXAMPLE_SOURCE, {"mode": "ref"}),
            (EXAMPLE_SOURCE, {"mode": "unknown"}),
        ]
        for source, options in cases:
            with self.subTest(source=source, options=options):
                with self.assertRaises(jp.CompileError):
                    jp.compile_and_send(source, self.dictionary, sent.append, **options)
                self.assertEqual(sent, [])


class ProjectProfileExplorationTests(unittest.TestCase):
    def test_actual_project_profile_supports_the_example_in_all_modes(self):
        dictionary = jp.load_dictionary(Path(__file__).with_name("prompt_profile.json"))
        for mode in ("inline", "ref", "nl"):
            with self.subTest(mode=mode):
                options = {"mode": mode}
                if mode == "ref":
                    options["receiver_dictionary"] = {"version": dictionary.version, "sha256": dictionary.digest}
                compiled = jp.compile_prompt(EXAMPLE_SOURCE, dictionary, **options)
                self.assertEqual(compiled.intent, "explore")
                self.assertIs(compiled.report["exploration"], True)
                self.assertEqual([(field.operator, field.content) for field in compiled.fields], EXAMPLE_FIELDS)
                for _token, payload in EXAMPLE_FIELDS:
                    self.assertIn(payload, compiled.text)


if __name__ == "__main__":
    unittest.main()
