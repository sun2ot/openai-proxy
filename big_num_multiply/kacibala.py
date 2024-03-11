def karatsuba_multiply(num1, num2):
    if len(num1) == 1 or len(num2) == 1:
        return str(int(num1) * int(num2))

    n = max(len(num1), len(num2))
    n_half = n // 2

    a, b = num1[:-n_half], num1[-n_half:]
    c, d = num2[:-n_half], num2[-n_half:]

    ac = int(karatsuba_multiply(a, c))
    bd = int(karatsuba_multiply(b, d))
    ad_bc = int(karatsuba_multiply(str(int(a) + int(b)), str(int(c) + int(d)))) - ac - bd

    result = ac * 10**(2 * n_half) + ad_bc * 10**n_half + bd

    return str(result)

num1 = "123456789"
num2 = "987654321"
result = karatsuba_multiply(num1, num2)
print(result)
