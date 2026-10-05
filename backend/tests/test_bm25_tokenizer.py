import re
import unittest

from app.services.advanced_retrieval import chinese_bm25_tokenizer


class ChineseBM25TokenizerTests(unittest.TestCase):
    def test_business_tokens(self):
        cases = [
            ("管理员账号的密码多久换一次？", {"管理员", "账号", "密码", "多久"}),
            ("VPN 连接失败以后账号会锁多久？", {"vpn", "连接", "失败", "账号", "多久"}),
            ("Git/API 接口需要经过网关鉴权。", {"git", "api", "接口"}),
            ("密码至少 12 位，且不得复用。", {"密码", "12", "复用"}),
            ("# VPN 管理规范", {"vpn", "管理", "规范"}),
        ]
        for text, required_tokens in cases:
            with self.subTest(text=text):
                tokens = chinese_bm25_tokenizer(text)
                self.assertGreater(len(tokens), 1)
                self.assertTrue(required_tokens.issubset(tokens))
                self.assertTrue(
                    all(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", token) for token in tokens)
                )

    def test_lowercases_english(self):
        tokens = chinese_bm25_tokenizer("VPN Access API")
        self.assertIn("vpn", tokens)
        self.assertNotIn("VPN", tokens)

    def test_handles_empty_and_whitespace(self):
        self.assertEqual(chinese_bm25_tokenizer(""), [])
        self.assertEqual(chinese_bm25_tokenizer("   \t\n"), [])


if __name__ == "__main__":
    unittest.main()
