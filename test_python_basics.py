import unittest

from python_basics import count_vowels
from grader_contracts.python_basics import TextInput


class TestPythonBasics(unittest.TestCase):

    def test_count_vowels(self):
        self.assertEqual(count_vowels(TextInput("hello")), 2)
        self.assertEqual(count_vowels(TextInput("AEIOU")), 5)
        self.assertEqual(count_vowels(TextInput("bcdfg")), 0)
        self.assertEqual(count_vowels(TextInput("")), 0)


if __name__ == "__main__":
    unittest.main()