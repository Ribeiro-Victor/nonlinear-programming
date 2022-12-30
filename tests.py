from optmengine import *
print("\n TESTANDO \n")
#mocando valores:

pontoCANTO = {'startingPoint': (0.5,0.5), 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': (.5,.5), 'currentValue': funcValue(.5,.5), 'residual': 0}

#testando evaluação:
print("\n TESTANDO FUNÇÃO\n")
print(funcValue(1000,200))

#testando gradiente
print("\n TESTANDO GRADIENTE\n")
print(gradValue(.1,.1))
print(gradValue(.5,.5))
print(gradValue(.9,.9))
print(gradValue(1.1,1.1))
print(gradValue(300,300))
print(gradValue(.1,300))
print(gradValue(300,.1))
print(gradValue(5,10))
print(gradValue(.1,1))
print(gradValue(.999999,1))




#testando direção:
print("\n TESTANDO DIREÇÃO\n")
print(direction(300,300))
print(direction(.001,.001))
print(direction(0,300))
print(direction(.5,0.8660254038))
print(direction(-20,-20))


#testando passo de armijo:
print("\n TESTANDO ARMIJO\n")
print(stepSize(pontoCANTO,direction(3,2),10,.5,.5))

#testando método gradiente

print("\n TESTANDO METODO GRADIENTE\n")
print(gradientMethod((.1,.2)))
print("\n")
print(gradientMethod((5,4)))
print("\n")
print(gradientMethod((50,40)))
print("\n")
print(gradientMethod((500,400)))
print("\n")
print(gradientMethod((1000,1)))
