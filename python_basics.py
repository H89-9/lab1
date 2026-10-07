"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    count = 0

    for letter in text.lower():
        if letter in "aeiou":
            count += 1
    return count


def has_unique_characters(data: TextInput) -> bool:
    text = data.value

    for symbol in text:
        if text.count(symbol) > 1:
            return False
    return True


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    binary = bin(number)
    return binary.count("1")
    

def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0

    while number >= 10:
        product = 1

        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    total = 0

    for i in range(len(predicted)):
        difference = predicted[i] - expected[i]
        total += difference ** 2

    return total / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    divisor = 2
    result = ""

    while number > 1:
        power = 0

        while number % divisor == 0:
            number //= divisor
            power += 1

        if power == 1:
            result += f"({divisor})"
        elif power > 1:
            result += f"({divisor}**{power})"

        divisor += 1
    return result
    

def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0

    while total < cube_count:
        k += 1
        total += k ** 2
    if total == cube_count:
        return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    raise NotImplementedError  # TODO
