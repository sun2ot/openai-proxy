def multiply_large_integers(num1, num2):
    result = [0] * (len(num1) + len(num2))

    for i in range(len(num1) - 1, -1, -1):
        carry = 0
        for j in range(len(num2) - 1, -1, -1):
            product = int(num1[i]) * int(num2[j]) + carry + result[i + j + 1]
            carry = product // 10
            result[i + j + 1] = product % 10

        result[i] += carry

    # 移除前导0
    result_str = ''.join(map(str, result))
    result_str = result_str.lstrip('0') or '0'

    return result_str

num1 = "123456789"
num2 = "987654321"
result = multiply_large_integers(num1, num2)
print(result)
