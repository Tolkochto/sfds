print('x y z')
for x in range(0,2):
    for y in range(0,2):
        for z in range(0,2):
            F=(x*z)+(x*(not y)*(not z))
            if F==1:
                print(x,y,z,)
