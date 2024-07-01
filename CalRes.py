"""
Descripttion: 
version: 1.0
Author: hwzhao
Date: 2022-09-20 20:38:56
"""

res = [
    0.97123,
    0.95328,
    0.95328,
    0.94862,
    0.95035,
    0.95155,
    0.94839,
    0.94163,
    0.94381,
    0.93795,
]

val = 0
for i in range(len(res)):
    val += res[i]
print(val / len(res))
