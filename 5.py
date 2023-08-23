for i in range(1000,10000):
    a=(i//1000)*((i//100)%10)
    b=((i//10)%10)*(i%10)

    if a<b:
        s=str(b) + str(a)

    else:
        s=str(a) + str(b)

    if int(s) ==124:
        print(i)
        break
