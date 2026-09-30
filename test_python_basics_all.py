import unittest

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    mse,
    prime_factorization,
    pyramid,
    is_balanced_number,
)


class PythonBasicsTests(unittest.TestCase):
    def test_count_vowels(self):
        self.assertEqual(count_vowels(TextInput("Hello")), 2)
        self.assertEqual(count_vowels(TextInput("AEIOU")), 5)
        self.assertEqual(count_vowels(TextInput("")), 0)

    def test_unique_characters(self):
        self.assertTrue(has_unique_characters(TextInput("abc")))
        self.assertTrue(has_unique_characters(TextInput("")))
        self.assertFalse(has_unique_characters(TextInput("abca")))

    def test_count_one_bits(self):
        self.assertEqual(count_one_bits(PositiveIntegerInput(1)), 1)
        self.assertEqual(count_one_bits(PositiveIntegerInput(13)), 3)
        self.assertEqual(count_one_bits(PositiveIntegerInput(255)), 8)

    def test_multiplicative_persistence(self):
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(4)), 0)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(39)), 3)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(999)), 4)

    def test_mse(self):
        self.assertEqual(mse(VectorPairInput([1, 2, 3], [1, 2, 3])), 0.0)
        self.assertAlmostEqual(
            mse(VectorPairInput([1, 2, 3], [1, 4, 2])),
            5 / 3,
        )

    def test_prime_factorization(self):
        self.assertEqual(
            prime_factorization(PositiveIntegerInput(86240)),
            "(2**5)(5)(7**2)(11)",
        )
        self.assertEqual(prime_factorization(PositiveIntegerInput(13)), "(13)")
        self.assertEqual(prime_factorization(PositiveIntegerInput(1)), "(1)")

    def test_pyramid(self):
        self.assertEqual(pyramid(PositiveIntegerInput(1)), 1)
        self.assertEqual(pyramid(PositiveIntegerInput(5)), 2)
        self.assertEqual(pyramid(PositiveIntegerInput(14)), 3)
        self.assertEqual(pyramid(PositiveIntegerInput(15)), "It is impossible")

    def test_balanced_number(self):
        self.assertTrue(is_balanced_number(PositiveIntegerInput(1234006)))
        self.assertFalse(is_balanced_number(PositiveIntegerInput(123456)))
        self.assertTrue(is_balanced_number(PositiveIntegerInput(7)))


if __name__ == "__main__":
    unittest.main()
