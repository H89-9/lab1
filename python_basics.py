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
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


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

    if len(predicted) != len(expected):
        raise ValueError("Vectors must have the same length")
    if len(predicted) == 0:
        raise ValueError("Vectors must not be empty")

    error_sum = 0
    for predicted_value, expected_value in zip(predicted, expected):
        error_sum += (predicted_value - expected_value) ** 2

    return error_sum / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value

    if number == 1:
        return "(1)"

    result = []
    divisor = 2

    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1

        if power == 1:
            result.append(f"({divisor})")
        elif power > 1:
            result.append(f"({divisor}**{power})")

        divisor += 1

    if number > 1:
        result.append(f"({number})")

    return "".join(result)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    level = 0
    total = 0

    while total < cube_count:
        level += 1
        total += level ** 2

    if total == cube_count:
        return level
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = str(data.value)
    length = len(digits)

    if length % 2 == 0:
        half = length // 2
        left = digits[:half - 1]
        right = digits[half + 1:]
    else:
        half = length // 2
        left = digits[:half]
        right = digits[half + 1:]

    left_sum = sum(int(digit) for digit in left)
    right_sum = sum(int(digit) for digit in right)

    return left_sum == right_sum
