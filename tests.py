from optmengine import *
from random import randint

def getPoint():
    return 1.0/randint(1,10) + 1.0*randint(0,10), 1.0/randint(1,10) + 1.0*randint(0,10)


def getCANTO():
    return 1.0/randint(1,100), 1.0/randint(1,100)

def getPAREDE1():
    return 1.0*randint(1,100), 1.0/randint(1,100)

def getPAREDE2():
    return 1.0/randint(1,100), 1.0*randint(1,100)

def getPLANALTO():
    return 1.0*randint(1,500),1.0*randint(1,500)


print("\n TESTANDO \n")
#mocando valores:

pontoCANTO = {'startingPoint': (0.5,0.5), 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': (.5,.5), 'currentValue': funcValue(.5,.5), 'residual': 0}

canto, parede1, parede2, planalto = getCANTO(), getPAREDE1(), getPAREDE2(), getPLANALTO()
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

#testando hessiana
print("\nTESTANDO HESSIANA\n\n")
evCANTO = eigenValues(hessianValue(canto[0],canto[1]))
evPAREDE1 = eigenValues(hessianValue(parede1[0],parede1[1]))
evPAREDE2 = eigenValues(hessianValue(parede2[0],parede2[1]))
evPLANALTO = eigenValues(hessianValue(planalto[0],planalto[1]))

checkConvexity(evCANTO[0],evCANTO[1])
checkConvexity(evPAREDE1[0],evPAREDE1[1])
checkConvexity(evPAREDE2[0],evPAREDE2[1])
checkConvexity(evPLANALTO[0],evPLANALTO[1])



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

#testando método gradiente:
'''
print("\n TESTANDO METODO GRADIENTE\n")
print(gradientMethod((.1,.2)))
print("\n")
print(gradientMethod((5,4)))
print("\n")
print(gradientMethod((50,40)))
print("\n")
print(gradientMethod((500,400)))
print("\n")
print(gradientMethod((1000,1.1)))
print("\n")
print(gradientMethod((1000,1000)))
'''
print("\n")
print(gradientMethod(getPoint()))
#testando tabela:
print("\n"*3)
#simulate([getPoint() for i in range(10)],gradientMethod,"Tabela 1 Método Gradiente")







simulate([canto,parede1,parede2,planalto,getPoint()],gradientMethod,"TABELA 1 MÉTODO GRADIENTE")

simulate([canto,parede1,parede2,planalto,getPoint()],newtonMethod,"TABELA 2 MÉTODO DE NEWTON")

simulate([canto,parede1,parede2,planalto,getPoint()],mockMethod,"TABELA 3 MÉTODO DAS DIREÇÕES FIXAS")

"""
while(True):
    testPoint = getPLANALTO()
    gradientValue = gradValue(testPoint[0],testPoint[1])
    hessian = hessianValue(testPoint[0],testPoint[1])
    evs = eigenValues(hessian)
    inverted = np.linalg.inv(hessian)
    descendValue = np.dot(inverted, gradientValue)
    print(descendValue)
"""

  
#print(algoGenSearch((.5,.5)))

#print(algoGenSearch((.5,.5)))