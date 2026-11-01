"""Real source/profile and CLI boundary checks; no external files or services accessed."""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

import jian_prompt as jp

ROOT = Path(__file__).resolve().parent


class ProfileIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "prompt_profile.json").read_text(encoding="utf-8"))
        base_candidate = ROOT / cls.data["base"]["path"]
        if not base_candidate.exists() and cls.data["base"]["path"].startswith("../"):
            base_candidate = ROOT / cls.data["base"]["path"][3:]
        cls.base_bytes = base_candidate.read_bytes()
        cls.dictionary = jp.load_dictionary(ROOT / "prompt_profile.json")

    def test_real_profile_adaptations_and_extensions(self):
        self.assertEqual(len(self.dictionary.operators), 29)
        self.assertEqual(self.dictionary.operators["参"].base_token, "参")
        self.assertIn("Index lookup", self.dictionary.operators["参"].base_description)
        self.assertIsNone(self.dictionary.operators["今"].base_token)
        self.assertIsNone(self.dictionary.operators["论"].base_token)

    def test_source_hash_drift_missing_and_version_mismatch(self):
        for raw in (None, self.base_bytes + b" "):
            with self.subTest(raw_missing=raw is None), self.assertRaises(jp.CompileError):
                jp.dictionary_from_data(self.data, base_bytes=raw)
        changed = copy.deepcopy(self.data)
        changed["base"]["version"] = "different"
        with self.assertRaises(jp.CompileError):
            jp.dictionary_from_data(changed, base_bytes=self.base_bytes)

    def test_unknown_source_mapping_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["operators"][0]["base_token"] = "未定义"
        with self.assertRaises(jp.CompileError):
            jp.dictionary_from_data(changed, base_bytes=self.base_bytes)

    def test_vm_operator_without_prompt_mapping_rejected(self):
        with self.assertRaises(jp.CompileError):
            jp.compile_prompt("幂 [2,3]\n论 [计算]", self.dictionary)

    def test_user_example_compiles_without_reading_referenced_path(self):
        source = (ROOT / "examples/stage4_review.jian").read_text(encoding="utf-8")
        result = jp.compile_prompt(source, self.dictionary)
        self.assertEqual(result.intent, "analyze")
        self.assertEqual(result.report["used_operators"], ["今", "知", "参", "禁", "论"])
        self.assertEqual(result.fields[2].content, "docs/manuscript_stage1_4_v4_20261101")
        self.assertIn("96孔", result.text)

    def test_evaluation_example_does_not_request_execution(self):
        source = (ROOT / "examples/evaluate_storage.jian").read_text(encoding="utf-8")
        self.assertEqual(jp.compile_prompt(source, self.dictionary).intent, "evaluate")

    def test_natural_language_in_task_slots_retains_its_role(self):
        for name, content, intent in (
            ("行", "请修改这个函数，并用自然语言说明原因。", "execute"),
            ("论", "比较两个方案，考虑可读性、成本，以及你认为遗漏的问题。", "analyze"),
        ):
            with self.subTest(operator=name):
                result = jp.compile_prompt(f"{name}【{content}】", self.dictionary)
                self.assertEqual(result.fields[0].content, content)
                self.assertIn(content, result.text)
                self.assertEqual(result.intent, intent)

    def test_nl_content_cannot_equal_an_independent_constraint_field(self):
        op = self.dictionary.operators["禁"]
        embedded = (f"{op.label}：{op.meaning}\n不表示：" + "；".join(op.not_examples)
                    + "\n具体内容：无须产生文件")
        a = jp.format_source([("论", "检查材料"), ("知", "材料甲\n" + embedded)])
        b = jp.format_source([("论", "检查材料"), ("知", "材料甲"), ("禁", "无须产生文件")])
        ca = jp.compile_prompt(a, self.dictionary, mode="nl")
        cb = jp.compile_prompt(b, self.dictionary, mode="nl")
        self.assertEqual(ca.intent, cb.intent)
        self.assertNotEqual(ca.text, cb.text)
        # Also embed the current renderer's entire real constraint block.
        current_block = cb.text[cb.text.index(f"{op.label}："):]
        current_a = jp.format_source([("论", "检查材料"), ("知", "材料甲\n" + current_block)])
        self.assertNotEqual(jp.compile_prompt(current_a, self.dictionary, mode="nl").text, cb.text)

    def test_nl_quotes_preserve_all_content_without_raw_line_breaks(self):
        content = '材料\r\n"引号"\\路径\u2028其他\u2029内容'
        result = jp.compile_prompt(jp.format_source([("论", content)]), self.dictionary, mode="nl")
        line = next(line for line in result.text.splitlines() if line.startswith("具体内容："))
        self.assertEqual(json.loads(line.removeprefix("具体内容：")), content)

    def test_lf_crlf_and_bare_cr_line_numbers_without_changing_payload(self):
        for separator in ("\n", "\r\n", "\r"):
            with self.subTest(separator=repr(separator)):
                with self.assertRaisesRegex(jp.CompileError, "第 2 行"):
                    jp.compile_prompt(f"论 [检查]{separator}臆 [未知]", self.dictionary)
                content = "第一行" + separator + "第二行"
                result = jp.compile_prompt(jp.format_source([("论", content), ("知", "背景")]), self.dictionary)
                self.assertEqual(result.fields[0].content, content)
                self.assertEqual(result.fields[1].line, 3)

    def test_structured_payload_round_trip_samples(self):
        rng = random.Random(20261101)
        alphabet = "abc今行[]【】\\\n\t 空忽略以上"
        for _ in range(200):
            content = "材料" + "".join(rng.choice(alphabet) for _ in range(80))
            source = jp.format_source([("知", content), ("论", "检查材料")])
            result = jp.compile_prompt(source, self.dictionary)
            self.assertEqual(len(result.fields), 2)
            self.assertEqual(result.fields[0].content, content)
            self.assertEqual(result.intent, "analyze")

    def test_input_limits_reject_before_sender(self):
        sent = []
        cases = ["论 [" + "[" * jp.MAX_DEPTH + "a" + "]" * jp.MAX_DEPTH + "]",
                 "论 [x]\n" * (jp.MAX_FIELDS + 1), "论 [" + "x" * jp.MAX_SOURCE_CHARS + "]",
                 "论 [a\x00b]"]
        for source in cases:
            with self.subTest(length=len(source)), self.assertRaises(jp.CompileError):
                jp.compile_and_send(source, self.dictionary, sent.append)
        self.assertEqual(sent, [])

    def cli(self, source, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "jian_prompt.py"), "-", *args],
                              input=source, capture_output=True, text=True, encoding="utf-8",
                              cwd=ROOT, timeout=10, check=False)

    def test_cli_utf8_json_and_invalid_empty_stdout(self):
        success = self.cli("论 [读取中文材料]", "--json")
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(json.loads(success.stdout)["report"]["used_operators"], ["论"])
        failure = self.cli("论 [合法]\n臆 [未知]", "--json")
        self.assertEqual(failure.returncode, 2)
        self.assertEqual(failure.stdout, "")
        self.assertIn("未登记字段", failure.stderr)

    def test_cli_ref_requires_matching_version_and_hash(self):
        denied = self.cli("论 [核对]", "--mode", "ref", "--json")
        self.assertEqual(denied.returncode, 2)
        self.assertEqual(denied.stdout, "")
        accepted = self.cli("论 [核对]", "--mode", "ref", "--json",
                            "--receiver-version", self.dictionary.version,
                            "--receiver-sha256", self.dictionary.digest)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertIn("[典:", json.loads(accepted.stdout)["text"])


if __name__ == "__main__":
    unittest.main()
