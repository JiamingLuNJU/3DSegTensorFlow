# tf.random_normal always has 0.1-0.2 derivation of mean, and its sum does not equal zero.

import tensorflow as tf
import math

W = tf.random_normal([5,6],0,1)
#W = tf.truncated_normal([5,6],0,1)

mySession = tf.Session()
Warray = mySession.run(W)

print("Warray shape:",Warray.shape)

print(Warray)

sum = 0.0
nCount = 0
for i in range(Warray.shape[0]):
    for j in range(Warray.shape[1]):
        sum +=Warray[i][j]
        nCount +=1
print("Warray[2][3]:", Warray[2][3])


print("nCount:", nCount)
mean = sum/nCount;
print("sum:", sum)
print("mean:", mean)

Var = 0;
for i in range(Warray.shape[0]):
    for j in range(Warray.shape[1]):
        Var +=(Warray[i][j]-mean)**2

Var = Var/nCount
stddev = math.sqrt(Var)
print("stddev:",stddev)





