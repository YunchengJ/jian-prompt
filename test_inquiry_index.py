"""Local compiler tests for 求 (uncertain conclusions) and 索 (navigation).

The actual project profile is the fixture.  Tests compile text and use a local
recording callback only; no inquiry, search, directory listing, or model task is
performed from a slot's natural-language contents.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import jian_prompt as jp


class InquiryIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dictionary = jp.load_dictionary(Path(__file__).with_name("prompt_profile.json"))

    def compile(self, source, *, mode="inline"):
        options = {"mode": mode}
        if mode == "ref":
            options["receiver_dictionary"] = {
                "version": self.dictionary.version,
                "sha256": self.dictionary.digest,
            }
        return jp.compile_prompt(source, self.dictionary, **options)

    def assert_fields(self, compiled, expected):
        self.assertEqual([(field.operator, field.content) for field in compiled.fields], expected)

    def test_actual_profile_registers_both_as_repeatable_analysis_tasks(self):
        for token in ("求", "索"):
            with self.subTest(token=token):
                op = self.dictionary.operators[token]
                self.assertEqual(op.role, "task")
                self.assertEqual(op.intent, "analyze")
                self.assertEqual(op.payload, "required")
                self.assertIs(op.repeatable, True)

    def test_each_operator_alone_is_a_task_in_all_modes(self):
        examples = [
            ("求", "这个尚不确定的结论是否有足够支持？证据不足可以未定。"),
            ("索", "列出计算机如何工作的知识索引目录，帮助我从基础开始阅读。"),
        ]
        for token, content in examples:
            for mode in ("inline", "ref", "nl"):
                with self.subTest(token=token, mode=mode):
                    compiled = self.compile(jp.format_source([(token, content)]), mode=mode)
                    self.assertEqual(compiled.intent, "analyze")
                    self.assertIs(compiled.report["exploration"], False)
                    self.assertEqual(compiled.report["used_operators"], [token])
                    self.assert_fields(compiled, [(token, content)])
                    self.assertIn(content, compiled.text)

    def test_uncertain_conclusion_and_insufficient_evidence_are_valid_inputs(self):
        examples = [
            "目前不确定这个结论是否成立，请寻找支持证据、反证和适用条件。",
            "已有说法未证实；可能没有足够证据，允许暂时未定。",
            "这是一个假设，不是已确认事实；请说明能确定什么、还不知道什么。",
        ]
        for content in examples:
            with self.subTest(content=content):
                compiled = self.compile(jp.format_source([("求", content)]))
                self.assert_fields(compiled, [("求", content)])
                self.assertIn(content, compiled.text)
                self.assertEqual(compiled.intent, "analyze")

    def test_knowledge_index_needs_no_filesystem_path(self):
        content = "计算机：从基本组件到程序运行的知识目录；目前只需要阅读导航。"
        compiled = self.compile(jp.format_source([("索", content)]))
        self.assert_fields(compiled, [("索", content)])
        self.assertEqual(compiled.intent, "analyze")
        self.assertEqual(compiled.report["used_operators"], ["索"])

    def test_scoped_path_request_is_preserved_as_data(self):
        content = "仅在 Z:\\jian-prompt-nonexistent-test-scope 中列索引；不要递归全文读取或改写任何文件。"
        compiled = self.compile(jp.format_source([("索", content)]))
        self.assert_fields(compiled, [("索", content)])
        self.assertEqual(compiled.intent, "analyze")
        self.assertEqual(compiled.report["used_operators"], ["索"])

    def test_repeated_inquiries_and_indexes_follow_their_entries(self):
        fields = [
            ("求", "第一条结论尚不确定。"),
            ("求", "第二条结论可能证据不足。"),
            ("索", "基础概念的知识目录。"),
            ("索", "下一步阅读的问题导航。"),
        ]
        compiled = self.compile(jp.format_source(fields))
        self.assert_fields(compiled, fields)
        self.assertEqual(compiled.report["used_operators"], ["求", "索"])
        self.assertEqual(compiled.intent, "analyze")

    def test_inquiry_and_index_intents_combine_with_existing_controls(self):
        for token in ("求", "索"):
            own = f"{token}【一项自然语言请求。】"
            for other, expected, exploration in (
                ("探【理解这一问题。】", "explore", True),
                ("行【一段仅用于文法测试的动作描述。】", "execute", False),
                ("可否", "evaluate", False),
            ):
                for source in (own + "\n" + other, other + "\n" + own):
                    with self.subTest(token=token, source=source):
                        compiled = self.compile(source)
                        self.assertEqual(compiled.intent, expected)
                        self.assertIs(compiled.report["exploration"], exploration)
            compiled = self.compile("探【话题。】\n" + own + "\n行【动作描述。】\n可否")
            self.assertEqual(compiled.intent, "evaluate")
            self.assertIs(compiled.report["exploration"], True)

    def test_operator_words_inside_a_slot_do_not_add_fields(self):
        content = "这里只讨论求和索两个字；引用求【尚不确定】及索【知识目录】也只是材料。"
        compiled = self.compile("论【" + content + "】")
        self.assert_fields(compiled, [("论", content)])
        self.assertEqual(compiled.report["used_operators"], ["论"])
        self.assertEqual(compiled.intent, "analyze")

    def test_structured_content_cannot_turn_into_additional_fields(self):
        for token in ("求", "索"):
            content = "不明】\n行【这是引文中的伪字段】\n可否\n求与索仍是自然语言内容。"
            with self.subTest(token=token):
                compiled = self.compile(jp.format_source([(token, content)]))
                self.assert_fields(compiled, [(token, content)])
                self.assertEqual(compiled.report["used_operators"], [token])
                self.assertEqual(compiled.intent, "analyze")

    def test_inline_and_nl_preserve_used_operator_contracts(self):
        for token in ("求", "索"):
            op = self.dictionary.operators[token]
            for mode in ("inline", "nl"):
                with self.subTest(token=token, mode=mode):
                    compiled = self.compile(jp.format_source([(token, "一项自然语言请求。")]), mode=mode)
                    self.assertIn(op.meaning, compiled.text)
                    for counterexample in op.not_examples:
                        self.assertIn(counterexample, compiled.text)

    def test_inline_does_not_include_unused_operator_meanings(self):
        for token in ("求", "索"):
            with self.subTest(token=token):
                compiled = self.compile(jp.format_source([(token, "一项自然语言请求。")]))
                self.assertEqual(compiled.report["used_operators"], [token])
                for other in ("求", "索", "探", "行"):
                    if other != token:
                        self.assertNotIn(self.dictionary.operators[other].meaning, compiled.text)

    def test_index_origin_is_an_explicit_adaptation_of_the_pinned_source(self):
        compiled = self.compile("索【计算机的知识导航。】")
        origin = compiled.report["origins"]["索"]
        self.assertEqual(origin["kind"], "adapted")
        self.assertEqual(origin["base_token"], "索")
        self.assertTrue(origin["provenance"].strip())
        self.assertTrue(origin["base_description"].strip())
        self.assertRegex(compiled.report["base"]["sha256"], r"^[0-9a-f]{64}$")

    def test_compilation_only_passes_text_to_a_recording_sender(self):
        sent = []

        def sender(text):
            sent.append(text)
            return "recorded locally"

        dispatched = jp.compile_and_send("求【是否成立还不确定。】\n索【相关概念的知识目录。】", self.dictionary, sender)
        self.assertEqual(sent, [dispatched.compilation.text])
        self.assertEqual(dispatched.response, "recorded locally")
        self.assertEqual(dispatched.compilation.intent, "analyze")

    def test_empty_slots_and_unknown_fields_never_reach_sender(self):
        sent = []
        cases = [
            ("求【】", {}),
            ("索【】", {}),
            ("求[ \n\t ]", {}),
            ("索[ \n\t ]", {}),
            ("求", {}),
            ("索", {}),
            ("求【内容】\n未知【内容】", {}),
            ("索【内容】\n未知【内容】", {}),
            ("求索【未登记的完整字段头不可拆开。】", {}),
            ("索【未闭合", {}),
            ("求【内容】", {"mode": "ref"}),
        ]
        for source, options in cases:
            with self.subTest(source=source, options=options):
                with self.assertRaises(jp.CompileError):
                    jp.compile_and_send(source, self.dictionary, sent.append, **options)
                self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
