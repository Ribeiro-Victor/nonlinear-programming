import numpy as np
from tabulate import tabulate
from operator import add
from random import random


def funcValue(x1, x2):
        return np.sqrt(np.log(x1)**2 + np.log(x2)**2)

def gradValue(x1,x2):
    denom = funcValue(x1,x2)
    return np.log(x1)/(denom*x1), np.log(x2)/(denom*x2)

def hessianValue(x1,x2):
    if(x1 <= 0 or x2 <= 0 ): return (np.inf)*3
    log1, log2 = np.log(x1), np.log(x2)
    func, func3 = funcValue(x1,x2), (log1**2 + log2**2)**(1.5)
    a1 = (1/x1**2)*(((1-log1)/(func))-(log1**2/func3))
    a2 = (1/x2**2)*(((1-log2)/(func))-(log2**2/func3))
    b = (-1)*(log1*log2)/(x1*x2*func3)
    return [[a1, b],[b,a2]]

def eigenValues(matrix):
    hdelta, avg = np.sqrt((matrix[0][0]-matrix[1][1])**2 + matrix[0][1]**2)/2, (matrix[0][0] + matrix[1][1])/2
    ev1 = avg + hdelta
    ev2 = avg - hdelta
    return ev1, ev2

def checkConvexity(ev1,ev2):
    if (ev1 > 0 and ev2 > 0): 
        #print("def POSITIVA")
        return True
    #elif (ev1 < 0 and ev2 < 0): print("def NEGATIVA")
    #else: print("indef")
    return False


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
        
def stepSize(state,direction, S , beta , sigma ,m = 0, iterationCap = 100):
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

def gradientMethod(startingPoint, iteractionCap = 10000, epsolon = .000001, S = 2.34, beta = .45, sigma = .1):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': funcValue(startingPoint[0],startingPoint[1]), 'residual': 0}
    searching = True
    stepDiff = S
    while(searching):
        state['iterations'] += 1
        point = state['currentPoint']
        startingValue = state['currentValue']
        grdientValue = gradValue(point[0],point[1])
        descendValue = [i *(-1) for i in grdientValue]
        descendDirection = direction(descendValue[0],descendValue[1])
        state, stepDiff = stepSize(state, descendDirection,stepDiff * 10, beta, sigma)
        #print(point) - #pra ver o ponto se aproximar do otimo
        if state['iterations'] > iteractionCap: 
            searching = False 
            #print('iteration limit')
        valueDiff = startingValue - state['currentValue']
        if ( valueDiff < epsolon and stepDiff < epsolon ): 
            searching = False
            #print("optimal")
    
    splitValue = truncate(state['currentValue'],6)
    state['currentValue'] = splitValue[0]
    state['residual'] = splitValue[1]
    return state

def newtonMethod(startingPoint, iteractionCap = 10000, epsolon = 0.00000001, S = 1.15, beta = .1, sigma = .1):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': funcValue(startingPoint[0],startingPoint[1]), 'residual': 0}
    searching = True
    stepDiff = S
    while(searching):
        state['iterations'] += 1
        point = state['currentPoint']
        startingValue = state['currentValue']
        gradientValue = gradValue(point[0],point[1])
        hessian = hessianValue(point[0],point[1])
        inverted = np.linalg.inv(hessian)
        descendValue = np.dot(inverted, gradientValue)
        switches = [1,1]
        if point[0] > 1: switches[0] = -1
        if point[1] > 1: switches[1] = -1
        descendDirection = direction(np.sqrt(descendValue[0]**2)*switches[0],np.sqrt(descendValue[1]**2)*switches[1])
        #print(point, "       ",descendDirection)
        state, stepDiff = stepSize(state, descendDirection,stepDiff * 10, beta, sigma)
        #print(point) - #pra ver o ponto se aproximar do otimo
        if state['iterations'] > iteractionCap: 
            searching = False 
            print('iteration limit')
        valueDiff = startingValue - state['currentValue']
        #print(valueDiff,stepDiff)
        if ( valueDiff < epsolon and stepDiff < epsolon ):
            #print("=====")
            #print(valueDiff,stepDiff) 
            searching = False
            #print("optimal")
    
    splitValue = truncate(state['currentValue'],6)
    state['currentValue'] = splitValue[0]
    state['residual'] = splitValue[1]
    return state

def bfgsMethod(startingPoint, iteractionCap = 10000, epsolon = 0.00000001, S = 3, beta = .5, sigma = .5):
    state = {'startingPoint': startingPoint,'lastPoint':startingPoint, 'currentGradient': [0,0],'currentInverseHessian':np.matrix([[1,0],[0,1]]), 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': funcValue(startingPoint[0],startingPoint[1]), 'residual': 0}
    searching = True
    stepDiff = S
    while(searching):
        state['iterations'] += 1
        lastPoint = state['lastPoint']
        point = state['currentPoint']
        startingValue = state['currentValue']
        gradientValue = gradValue(point[0],point[1])
        lastGradient =state['currentGradient']
        state['currentGradient'] = gradientValue
        q = np.matrix([gradientValue[0] - lastGradient[0], gradientValue[1] - lastGradient[1]])
        p = np.matrix([point[0] - lastPoint[0], point[1] - lastPoint[1]])
        qt, pt = q.transpose(), p.transpose()
        H = state['currentInverseHessian']
        nextH = H + (1 + ((qt*H*q)[0][0])/(pt*q)[0][0])*(p*pt)[0][0]/(pt*q)[0][0] - (p*qt*H + H*q*pt)[0][0]/(pt*q)[0][0]
        state['currentInverseHessian'] = nextH
        descendValue = np.dot(nextH, gradientValue)
        
        
        
        #switches = [1,1]
        #if point[0] > 1: switches[0] = -1
        #if point[1] > 1: switches[1] = -1
        descendDirection = direction(np.sqrt(descendValue[0]**2)*switches[0],np.sqrt(descendValue[1]**2)*switches[1])
        #print(point, "       ",descendDirection)
        state['lastPoint'] = state['currentPoint']
        state, stepDiff = stepSize(state, descendDirection,stepDiff * 10, beta, sigma)
        


        #print(point) - #pra ver o ponto se aproximar do otimo
        if state['iterations'] > iteractionCap: 
            searching = False 
            print('iteration limit')
        valueDiff = startingValue - state['currentValue']
        #print(valueDiff,stepDiff)
        if ( valueDiff < epsolon and stepDiff < epsolon ):
            #print("=====")
            #print(valueDiff,stepDiff) 
            searching = False
            #print("optimal")
    
    splitValue = truncate(state['currentValue'],6)
    state['currentValue'] = splitValue[0]
    state['residual'] = splitValue[1]
    return state

def mockMethod(startingPoint, iteractionCap = 10000, epsolon = .000001, S = 1.3, beta = .4, sigma = .1):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': funcValue(startingPoint[0],startingPoint[1]), 'residual': 0}
    searching = True
    stepDiff = S
    while(searching):
        state['iterations'] += 1
        point = state['currentPoint']
        startingValue = state['currentValue']
        descendValue = [1,1]
        if point[0]>1: descendValue[0] = -1
        if point[1]>1: descendValue[1] = -1        
        descendDirection = direction(descendValue[0],descendValue[1])
        state, stepDiff = stepSize(state, descendDirection,stepDiff * 10, beta, sigma)
        #print(point) - #pra ver o ponto se aproximar do otimo
        if state['iterations'] > iteractionCap: 
            searching = False 
            #print('iteration limit')
        valueDiff = startingValue - state['currentValue']
        if ( valueDiff < epsolon and stepDiff < epsolon ): 
            searching = False
            #print("optimal")
    
    splitValue = truncate(state['currentValue'],6)
    state['currentValue'] = splitValue[0]
    state['residual'] = splitValue[1]
    return state

def mockMethod2(startingPoint, iteractionCap = 10000, epsolon = .000001, S = 1.3, beta = .4, sigma = .1):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': funcValue(startingPoint[0],startingPoint[1]), 'residual': 0}
    searching = True
    stepDiff = S
    while(searching):
        state['iterations'] += 1
        startingValue = state['currentValue']
        west, east, north, south = [-1,0], [1,0], [0,1], [0,-1]
        westStep, eastStep, northStep, southStep = stepSize(state, west,stepDiff * 10, beta, sigma), stepSize(state, east,stepDiff * 10, beta, sigma),stepSize(state, north,stepDiff * 10, beta, sigma), stepSize(state, south,stepDiff * 10, beta, sigma)
        larger = ({},0)
        for step in [westStep, eastStep, northStep, southStep]:
            if step[1] > larger[1]: larger = step
       
        
        state, stepDiff = larger[0],larger[1]
        #print(point) - #pra ver o ponto se aproximar do otimo
        if state['iterations'] > iteractionCap: 
            searching = False 
            #print('iteration limit')
        valueDiff = startingValue - state['currentValue']
        if ( valueDiff < epsolon and stepDiff < epsolon ): 
            searching = False
            #print("optimal")
    
    splitValue = truncate(state['currentValue'],6)
    state['currentValue'] = splitValue[0]
    state['residual'] = splitValue[1]
    return state

def mockMethod3(startingPoint, iteractionCap = 10000, epsolon = .000001, S = 1.3, beta = .4, sigma = .1):
    state = {'startingPoint': startingPoint, 'iterations': 0, 'stepSizeCalls': 0, 'currentPoint': startingPoint, 'currentValue': funcValue(startingPoint[0],startingPoint[1]), 'residual': 0}
    searching = True
    stepDiff = S
    while(searching):
        state['iterations'] += 1
        point = state['currentPoint']
        startingValue = state['currentValue']
        descendValue = [1 - point[0],1 - point[1]]
        
        
        state, stepDiff = stepSize(state,direction(descendValue[0], descendValue[1]),stepDiff * 10, beta, sigma)
        #print(point) - #pra ver o ponto se aproximar do otimo
        if state['iterations'] > iteractionCap: 
            searching = False 
            #print('iteration limit')
        valueDiff = startingValue - state['currentValue']
        if ( valueDiff < epsolon and stepDiff < epsolon ): 
            searching = False
            #print("optimal")
    
    splitValue = truncate(state['currentValue'],6)
    state['currentValue'] = splitValue[0]
    state['residual'] = splitValue[1]
    return state


def simulate(startingPointList, method, header):
    finalList = []
    for point in startingPointList:
        lineOutput = method(point)
        finalList += [[lineOutput['startingPoint'][0],lineOutput['startingPoint'][1],lineOutput['iterations'],lineOutput['stepSizeCalls'],lineOutput['currentPoint'][0],lineOutput['currentPoint'][1],lineOutput['currentValue'],lineOutput['residual']]]
    finalTable = tabulate(finalList, headers=["X°(x1)","X°(x2)", "#Iteracoes", "#Cham.Armijo", "X*(x1)","X*(x2)", "f(X*)", "Erro de Aproximacao"],floatfmt=[".4f",".4f","","",".8f",".8f",".6f"],tablefmt="fancy_outline")
    print('\n',header,'\n')
    print(finalTable)