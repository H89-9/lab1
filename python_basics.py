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
    count = 0

    while number >= 10:
        result = 1
        for digit in str(number):
            result *= int(digit)
        number = result
        count += 1

    return count


def mse(data: VectorPairInput) -> float:
    predicted = data.predicted
    expected = data.expected

    total = 0
    for i in range(len(predicted)):
        total += (predicted[i] - expected[i]) ** 2

    return total / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value

    if number == 1:
        return "(1)"

    answer = ""
    divisor = 2

    while number > 1:
        power = 0

        while number % divisor == 0:
            number //= divisor
            power += 1

        if power == 1:
            answer += f"({divisor})"
        elif power > 1:
            answer += f"({divisor}**{power})"

        divisor += 1

    return answer


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
    middle = len(digits) // 2

    if len(digits) % 2 == 0:
        left = digits[:middle - 1]
        right = digits[middle + 1:]
    else:
        left = digits[:middle]
        right = digits[middle + 1:]

    left_sum = sum(int(x) for x in left)
    right_sum = sum(int(x) for x in right)

    return left_sum == right_sum
