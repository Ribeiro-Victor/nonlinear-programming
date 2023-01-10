from optmengine import *

def initiateGen(population):
    generated = []
    for i in range(population):
        newS, newBeta, newSigma = (random()/random()) + 1, random(), random()
        generated += [{"S":newS, 'beta':newBeta,'sigma':newSigma, 'fitness':""}]
    return generated

def recombinate(parents):
    childs = []
    for index1 in range(len(parents)):
        for index2 in range(index1 + 1,len(parents)):
            pS1, pS2, pb1, pb2, ps1, ps2 = parents[index1]['S'], parents[index2]['S'], parents[index1]['beta'], parents[index2]['beta'], parents[index1]['sigma'], parents[index2]['sigma']
            childS, childBeta, childSigma = pS1*random() + pS2*random() + 1, (pb1 + pb2)/2, (ps1 + ps2)/2
            childs += [{'S': childS, 'beta': childBeta, 'sigma':childSigma, 'fitness':""}]
    return childs



def algoGenSearch(point, method, elite = 4, randomized = 2, generations = 300):
    population = (elite + randomized)*(elite + randomized + 1)
    currentPopulation = initiateGen(population)
    searching = True
    topOnes = []
    while(searching):
        generations -= 1
        for individual in currentPopulation:
            
            evaluation = method(point,S = individual['S'], beta = individual['beta'], sigma = individual['sigma'])
            individual['fitness'] = evaluation['stepSizeCalls']
            #alguma forma de inserir individual nos TopOnes
            pushing = True
            for i in range(len(topOnes)):
                if topOnes[i]['fitness'] > individual['fitness'] and evaluation['currentValue'] < .00000001 and pushing: 
                    topOnes.insert(i, individual)
                    pushing = False
            if pushing: topOnes += [individual]
        #print(topOnes[0]['fitness'], topOnes[elite-1]['fitness'])
        topOnes = topOnes[0:(elite)]
        #print(topOnes)
        if (generations < 1): 
            print(generations)
            searching = False
        newOnes = initiateGen(randomized)
        childs = recombinate(topOnes + newOnes)
        currentPopulation = newOnes + childs
        #print(generations)    
    return(topOnes[0])    

def fullSearch(pointList, methodList, elite = 4, randomized = 2, generations = 300):
    for method in methodList:
        for point in pointList:
            print(algoGenSearch(point, method, elite, randomized, generations))

points = [getCANTO(getPAREDE1(),getPAREDE2(),getPLANALTO())]
methods = [gradientMethod, newtonMethod, quasiNewtonMethod,mockMethod2]

fullSearch(points,methods)