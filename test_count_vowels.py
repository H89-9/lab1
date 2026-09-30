import unittest

from grader_contracts.python_basics import TextInput
from python_basics import count_vowels


class CountVowelsTests(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(count_vowels(TextInput("")), 0)

    def test_no_vowels(self):
        self.assertEqual(count_vowels(TextInput("bcdfgxyz")), 0)

    def test_all_lowercase_vowels(self):
        self.assertEqual(count_vowels(TextInput("aeiou")), 5)

    def test_all_uppercase_vowels(self):
        self.assertEqual(count_vowels(TextInput("AEIOU")), 5)

    def test_mixed_case(self):
        self.assertEqual(count_vowels(TextInput("HeLLo")), 2)

    def test_repeated_vowels(self):
        self.assertEqual(count_vowels(TextInput("aAaA")), 4)

    def test_single_letter(self):
        self.assertEqual(count_vowels(TextInput("I")), 1)
        self.assertEqual(count_vowels(TextInput("y")), 0)


if __name__ == "__main__":
    unittest.main()
