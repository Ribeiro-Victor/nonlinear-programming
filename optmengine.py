import numpy as np
from tabulate import tabulate
from operator import add


def funcValue(x1, x2):
    #print(f'valorando para {x1} e {x2}')
    try:
        return np.sqrt(np.log(x1)**2 + np.log(x2)**2)
    except:
        print("fora do domínio")

def gradValue(x1,x2):
    denom = funcValue(x1,x2)
    try:
        return np.log(x1)/(denom*x1), np.log(x2)/(denom*x2)
    except(error):
        print(error)

def direction(x1,x2):
    denom = np.sqrt(x1**2 + x2**2)
    return x1/denom , x2/denom

def truncate(extensive, decimalSize):
    pointer = 0.1**decimalSize
    reduced = (extensive//pointer)*pointer
    residue = extensive - reduced
    return reduced, residue

def sizeEvaluation(x1,x2,y1,y2):
    s1 = y1 - x1
    s2 = y2 - x2
    return np.sqrt(s1**2 + s2**2)
        
def stepSize(state,direction, S = 1000, beta = .5, sigma = .5,m = 0, iterationCap = 100):
    iterationRem = iterationCap - 1
    if iterationRem < 1:
        return stepSize(state,direction,np.sqrt(S),beta,sigma**2)
    state['stepSizeCalls'] += 1
    stepLength = sigma*(beta**m)*S
    newPoint = list(map(add,list(state['currentPoint']),[i * stepLength for i in list(direction)]))
    if(newPoint[0] > 0 and newPoint[1] > 0): 
        newValue = funcValue(newPoint[0],newPoint[1]) 
    else: newValue = np.inf
    
    if(state['currentValue'] < newValue):
        m += 1
        return stepSize(state, direction, S, beta, sigma, m, iterationRem)
    else:
        state['currentPoint'] = (newPoint[0],newPoint[1])
        state['currentValue'] = newValue
        return state, stepLength

def gradientMethod(startingPoint, iteractionCap = 10000, epsolon = .000001):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': funcValue(startingPoint[0],startingPoint[1]), 'residual': 0}
    searching = True
    while(searching):
        state['iterations'] += 1
        point = state['currentPoint']
        startingValue = state['currentValue']
        grdientValue = gradValue(point[0],point[1])
        descendValue = [i *(-1) for i in grdientValue]
        descendDirection = direction(descendValue[0],descendValue[1])
        state, stepDiff = stepSize(state, descendDirection)
        #print(point) - #pra ver o ponto se aproximar do otimo
        if state['iterations'] > iteractionCap: 
            searching = False 
            #print('iteration limit')
        valueDiff = startingValue - state['currentValue']
        if ( valueDiff < epsolon and stepDiff < epsolon ): 
            searching = False
            #print("optimal")
    
    splitValue = truncate(state['currentValue'],5)
    state['currentValue'] = splitValue[0]
    state['residual'] = splitValue[1]
    return state

def newtonMethod(startingPoint):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': function(startingPoint), 'residual': 0}

def otherMethod(startingPoint):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': function(startingPoint), 'residual': 0}
    
def mockMethod(startingPoint):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': function(startingPoint), 'residual': 0}
    


def simulate(startingPointList, method, header):
    finalList = []
    for point in startingPointList:
        lineOutput = method(point)
        finalList += [[lineOutput['startingPoint'][0],lineOutput['startingPoint'][1],lineOutput['iterations'],lineOutput['stepSizeCalls'],lineOutput['currentPoint'][0],lineOutput['currentPoint'][1],lineOutput['currentValue'],lineOutput['residual']]]
    finalTable = tabulate(finalList, headers=["Ponto Inicial(x1)","(x2)", "# de Iteracoes", "# de Cham de Armijo", "Ponto Otimo(x1)","(x2)", "Avaliacao do Ponto Otimo", "Erro de Aproximacao"],floatfmt=[".2f",".2f","","",".6f",".6f",".7f"])
    print(header)
    print(finalTable)