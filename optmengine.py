import numpy as np
from tabulate import tabulate
from operator import add


def funcValue(x1, x2):
    ''' Dado dois pontos x1 e x2, retorna o valor da funcao f(x1, x2)'''
    return np.sqrt(np.log(x1)**2 + np.log(x2)**2)

def direction(x,y):
    denom = sqrt(x**2 + y**2)
    return x/denom , y/denom
        
def  stepSize(state,direction, S, beta, sigma,m = 0, iterationCap = 100):
    iterationRem = iterationCap - 1
    if iterationRem < 1:
        return "Limite de iteracoes alcancado - Tamanho do Passo"
    state['stepSizeCalls'] += 1
    newPoint = map(add,list(state['currentPoint']),[i * sigma*(beta**m)*S for i in list(direction)])
    newValue = funcValue(newPoint[0],newPoint[1])
    if(state['currentValue'] < newValue):
        m += 1
        return stepSize(state, direction, S, beta, sigma, m, iterationRem)
    else:
        state['currentPoint'] = (newPoint[0],newPoint[1])
        state['currentValue'] = newValue
        return state 

def gradientMethod(startingPoint, function):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': function(startingPoint), 'residual': 0}

def newtonMethod(startingPoint, function):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': function(startingPoint), 'residual': 0}

def otherMethod(startingPoint, function):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': function(startingPoint), 'residual': 0}
    
def mockMethod(startingPoint, function):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': function(startingPoint), 'residual': 0}
    


def simulate(startingPointList,function, method, header):
    finalList = []
    for point in startingPointList:
        lineOutput = method(point, function)
        finalList += [[lineOutput['startingPoint'],lineOutput['iterations'],lineOutput['stepSizeCalls'],lineOutput['currentPoint'],lineOutput['currentValue'],lineOutput['residual']]]
    finalTable = tabulate(finalList, headers=["Ponto Inicial", "# de Iteracoes", "# de Cham de Armijo", "Ponto Otimo", "Avaliacao do Ponto Otimo", "Erro de Aproximacao"])
    print(header)
    print(finalTable)