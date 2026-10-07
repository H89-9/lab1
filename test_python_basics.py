import unittest
from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    mse,
    prime_factorization,
    pyramid,
    is_balanced_number,)

from grader_contracts.python_basics import (
    TextInput,
    PositiveIntegerInput,
    VectorPairInput,)
class TestPythonBasics(unittest.TestCase):

    def test_count_vowels(self):
        self.assertEqual(count_vowels(TextInput("hello")), 2)
        self.assertEqual(count_vowels(TextInput("AEIOU")), 5)
        self.assertEqual(count_vowels(TextInput("bcdfg")), 0)
        self.assertEqual(count_vowels(TextInput("")), 0)

    def test_has_unique_characters(self):
        self.assertTrue(has_unique_characters(TextInput("abc")))
        self.assertFalse(has_unique_characters(TextInput("hello")))
        self.assertTrue(has_unique_characters(TextInput("")))
        self.assertFalse(has_unique_characters(TextInput("aa")))

    def test_count_one_bits(self):
        self.assertEqual(count_one_bits(PositiveIntegerInput(1)), 1)
        self.assertEqual(count_one_bits(PositiveIntegerInput(5)), 2)
        self.assertEqual(count_one_bits(PositiveIntegerInput(7)), 3)
        self.assertEqual(count_one_bits(PositiveIntegerInput(8)), 1)

    def test_multiplicative_persistence(self):
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(39)), 3)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(4)), 0)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(999)), 4)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(10)), 1)

    def test_mse(self):
        self.assertAlmostEqual(mse(VectorPairInput([1, 2, 3], [1, 3, 5])), 5 / 3)
        self.assertEqual(mse(VectorPairInput([1, 1], [1, 1])), 0)
        self.assertEqual(mse(VectorPairInput([0], [2])), 4)

    def test_prime_factorization(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(86240)), "(2**5)(5)(7**2)(11)")
        self.assertEqual(prime_factorization(PositiveIntegerInput(12)), "(2**2)(3)")
        self.assertEqual(prime_factorization(PositiveIntegerInput(13)), "(13)")
        self.assertEqual(prime_factorization(PositiveIntegerInput(1)), "")

    def test_pyramid(self):
        self.assertEqual(pyramid(PositiveIntegerInput(1)), 1)
        self.assertEqual(pyramid(PositiveIntegerInput(5)), 2)
        self.assertEqual(pyramid(PositiveIntegerInput(14)), 3)
        self.assertEqual(pyramid(PositiveIntegerInput(10)), "It is impossible")

    def test_is_balanced_number(self):
        self.assertTrue(is_balanced_number(PositiveIntegerInput(1234006)))
        self.assertFalse(is_balanced_number(PositiveIntegerInput(123456)))
        self.assertTrue(is_balanced_number(PositiveIntegerInput(121)))
        self.assertTrue(is_balanced_number(PositiveIntegerInput(1)))
if __name__ == "__main__":
    unittest.main()