# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:55:28 2021

@author: Shahrukh
"""

import numpy as np
from scipy.spatial import distance_matrix
import random
import matplotlib.pyplot as plt

def cost_matrix(n_cities):  #costmatrix_for_cities
    coordinates=np.random.randint(1,30,size=(n_cities,2))
    
    costmatrix=distance_matrix(coordinates,coordinates,2)
    return costmatrix

def create_population(size,n_cities):
    population=[]
    for i in range(0,size):
        population.append(create_new_member(n_cities))      
    return population

def create_new_member(n_cities):
    member=np.zeros((n_cities+1),dtype=int)
    member[0]=np.random.randint(1,n_cities)
    member[n_cities]=member[0]
    excludeset={member[0]}
    d= set(i for i in range(1,n_cities+1) if i not in  excludeset)
    member[1:(n_cities)]=random.sample(d,(n_cities-1) )   
    return member

def measure_fitness(member,cm):
    cost=0;
    l=len(member)
    for i in range(0,(l-1)):
        cost=cost+cm[member[i]-1][member[i+1]-1]
    fitness=(1/cost)*100 *len(cm)
    return fitness

def scorepopulation(population,cm):
    l=len(population)
    population_fitness=[]
    for i in range(0,l):
        population_fitness.append((population[i],measure_fitness(population[i],cm)))
    return population_fitness


def selecting_among_population(population_score,elite_size):
    sorted_population = sorted(population_score, key=lambda x: x[1],reverse=True)
    return sorted_population[0:elite_size]

def matingpool(selected_population):
    mating_pool = [a[0] for a in selected_population]
    return mating_pool

def breedpopulation(mating_pool,npop):
    length=len(mating_pool)
        
    children=[]
    for i in range(0,npop):
        r=random.randint(0,(length-2))
        children.append(crossover(mating_pool[r],mating_pool[r+1]))
        
        
    return children
def crossover(a,b):
    '''a=route1
    b=route2
    return child
    '''
    #child[0]=rand(a[0],b[0])
    #child=[]
    l=len(a)
    child=np.zeros(l,dtype=int)
    child[0]=random.choice([a[0],b[0]])
    child[l-1]=child[0]
    d=[]
    
    for i in range(1,(l-1)):
        exclude=set(child[0:i])
        p=set([a[i],b[i]])
        #p=set.union(q,p)    
        p=p-set.intersection(p, exclude)
        if(len(p)==0):
            q=set(range(1,n_cities+1))
            q=q-set.intersection(q, exclude)
            child[i]=random.sample(q,1 )[0] 
        else:   
            child[i]=random.sample(p,1 )[0] 
    return child

def mutatepopulation(breededpopulation):
    l=len(breededpopulation[0])
    for i in range(0,len(breededpopulation)):
        if (random.uniform(0, 1)>=0.2):
            r=random.randint(1,(l-2))

            s=random.randint(1,(l-2))
        
      
            temp = breededpopulation[i][r]
            breededpopulation[i][r] = breededpopulation[i][s]
            breededpopulation[i][s]= temp
    mutated_population=breededpopulation
    return mutated_population

def genetic_algo_TSP(n_cities,generations,elite_size,popsize):
    cm=cost_matrix(n_cities)
    population=create_population(popsize,n_cities)
    #population_score=scorepopulation(population,cm)
    #population_score_avg=sum([x[1] for x in population_score])/20;
    
    population_avg_fitness_score=[]
    for i in range(0,generations):
        
        
    #elite_size=20
        population_score=scorepopulation(population,cm)
        selected_population=selecting_among_population(population_score,elite_size)
        mating_pool=matingpool(selected_population)
        n=len(population)
        breededpopulation=breedpopulation(mating_pool,len(population))
        mutated_population=mutatepopulation(breededpopulation)
        mutated_population_score=scorepopulation(mutated_population,cm)
        mutated_population_score_avg=sum([x[1] for x in mutated_population_score])/len(mutated_population)
        print('Generation No',i, ": population_score_avg",mutated_population_score_avg)
        population_avg_fitness_score.append(mutated_population_score_avg)
        population=mutated_population
    population_score=scorepopulation(population,cm)
    #selected_population=selecting_among_population(population_score,elite_size)

    return population_score,population_avg_fitness_score


n_cities=5
generations=20000
elite_size=5
popsize=25
scored_pop,avg_fitness_score=genetic_algo_TSP(n_cities,generations,elite_size,popsize)
selected_route=selecting_among_population(scored_pop,1)#selecting_best
print(selected_route)
plt.plot(avg_fitness_score)




