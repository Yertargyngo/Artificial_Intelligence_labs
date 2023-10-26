# Кодирование и декодирование секретных сообщений: Используйте
# эволюционные алгоритмы для оптимизации ключа шифрования.

import random
import string

# Задаем исходное сообщение
message = "Yertargyn"

# Шифр Виженера
def vigenere_cipher(text, key):
    result = ""
    for i in range(len(text)):
        char = text[i]
        if char.isalpha():
            ascii_offset = ord('a') if char.islower() else ord('A')
            key_char = key[i % len(key)]
            key_offset = ord(key_char.lower()) - ord('a')
            result += chr((ord(char) - ascii_offset + key_offset) % 26 + ascii_offset)
        else:
            result += char
    return result

# Генерация случайного ключа
def generate_random_key(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

# Оценка приспособленности
def fitness_score(decoded_message, target):
    score = 0
    for i in range(len(target)):
        if decoded_message[i] == target[i]:
            score += 1
    return score

# Эволюционный алгоритм
def evolve(message, iterations, key_length):
    best_key = generate_random_key(key_length)
    best_message = vigenere_cipher(message, best_key)
    best_score = fitness_score(best_message, message)

    for _ in range(iterations):
        new_key = generate_random_key(key_length)
        new_message = vigenere_cipher(message, new_key)
        new_score = fitness_score(new_message, message)

        if new_score > best_score:
            best_score = new_score
            best_key = new_key
            best_message = new_message

    return best_key, best_message

# Кодирование и вывод
key_length = 5
key, encoded_message = evolve(message, 1000, key_length)
print(f"Original Message: {message}")
print(f"Encoded Message: {encoded_message}")
print(f"Key: {key}")

# Декодирование
decoded_message = vigenere_cipher(encoded_message, key)
print(f"Decoded Message: {decoded_message}")
