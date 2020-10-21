import numpy as np
from random import randint
import random
import matplotlib.pyplot as plt
import copy

from perffcn import *

class GA:
    def __init__(self, pop_size, genereation_size, crossover_p, mutation_p, Kp_lb, Kp_ub, Ki_lb, Ki_ub, Kd_lb, Kd_ub, 
                alpha=0.4):
        self.pop_size = pop_size
        self.genereation_size = genereation_size
        self.crossover_p = crossover_p
        self.mutation_p = mutation_p
        self.Kp_lb = Kp_lb
        self.Kp_ub = Kp_ub
        self.Ki_lb = Ki_lb
        self.Ki_ub = Ki_ub
        self.Kd_lb = Kd_lb
        self.Kd_ub = Kd_ub
        self.alpha = alpha
        self.gene_mutation_p = mutation_p
        self.initialize_random_pop()
        self.best_per_generation = []
        self.pre_best = float('-inf')

    def run(self):
        for i in range(self.genereation_size):
            print(i)
            self.pop_fitness_evaluation()
            self.generate_offspring()
            self.generate_next_population()

    def initialize_random_pop(self):
        Kp = (self.Kp_ub - self.Kp_lb) * np.random.rand(self.pop_size) + self.Kd_lb
        Ki = (self.Ki_ub - self.Ki_lb) * np.random.rand(self.pop_size) + self.Ki_lb
        Kd = (self.Kd_ub - self.Kd_lb) * np.random.rand(self.pop_size) + self.Kd_lb
        self.population = np.array([Kp, Ki, Kd]).transpose()
        self.population = np.around(self.population, decimals=2)
        self.population = self.population.tolist()
    
    def pop_fitness_evaluation(self):
        best = 0
        second_best = 0
        for i in range(self.pop_size):
            ISE, t_r, t_s, M_p = q2_perfFNC(self.population[i][0], self.population[i][1], self.population[i][2])
            fitness = 1.0/(ISE+t_r+ t_s+ M_p)
            if len(self.population[i]) == 3:
                self.population[i].append(fitness)
            else:
                self.population[i][3] = fitness

            if self.population[i][3] > best:
                best = self.population[i][3]
                self.best_parent = self.population[i]
            elif self.population[i][3] > second_best:
                self.second_best_parent = self.population[i]

        if best < self.pre_best:
            print("wrong!")
            print("old survived1:{}".format(self.population[-2]))
            print("old value1:{}".format(self.population[-2][-1]))
            print("old survived2:{}".format(self.population[-1]))
            print("old value2:{}".format(self.population[-1][-1]))
        self.best_per_generation.append(best)
        self.pre_best = best

    def bound_Kp(self,x):
        if round(x,2) > self.Kp_ub:
            return self.Kp_ub
        elif round(x,2) < self.Kp_lb:
            return self.Kp_lb
        else:
            return round(x,2)
    
    def bound_Ki(self,x):
        if round(x,2) > self.Ki_ub:
            return self.Ki_ub
        elif round(x,2) < self.Ki_lb:
            return self.Ki_lb
        else:
            return round(x,2)

    def bound_Kd(self,x):
        if round(x,2) > self.Kd_ub:
            return self.Kd_ub
        elif round(x,2) < self.Kd_lb:
            return self.Kd_lb
        else:
            return round(x,2)

    def crossover(self, p1, p2):
        # whole arithmetric
        c1 = [0, 0, 0]
        c2 = [0, 0, 0]

        c1[0] = self.bound_Kp(self.alpha*p1[0] + (1-self.alpha)*p2[0])
        c2[0] = self.bound_Kp(self.alpha*p2[0] + (1-self.alpha)*p1[0])
        c1[1] = self.bound_Ki(self.alpha*p1[1] + (1-self.alpha)*p2[1])
        c2[1] = self.bound_Ki(self.alpha*p2[1] + (1-self.alpha)*p1[1])
        c1[2] = self.bound_Kd(self.alpha*p1[2] + (1-self.alpha)*p2[2])
        c2[2] = self.bound_Kd(self.alpha*p2[2] + (1-self.alpha)*p1[2])
        return c1[:3], c2[:3]

    def generate_offspring(self):
        self.offspring = []
        # crossover
        for i in range(int(self.pop_size/2)-1):
            p1 = self.parent_selection()
            p2 = self.parent_selection()
            rand = random.uniform(0,1)
            if rand <= self.crossover_p:
                c1, c2 = self.crossover(p1, p2)
                self.offspring.append(c1)
                self.offspring.append(c2)
            else:
                self.offspring.append(copy.deepcopy(p1))
                self.offspring.append(copy.deepcopy(p2))
        # mutation
        for i in range(self.pop_size-2):
            rand = random.uniform(0,1)
            if rand <= self.mutation_p:
                self.offspring[i] = self.mutation(self.offspring[i])
                

    def generate_next_population(self):
        s1, s2 = self.survival_selection()
        self.offspring.append(copy.deepcopy(s1))
        self.offspring.append(copy.deepcopy(s2))
        self.population = copy.deepcopy(self.offspring)
    
    def mutation(self,child):
        # add standard normal noice
        for i in range(3):
            rand = random.uniform(0,1)
            if rand <= self.gene_mutation_p:
                if i == 0:
                    child[i] = self.bound_Kp(child[i] + np.random.normal(scale=0.5))
                elif i == 1:
                    child[i] = self.bound_Ki(child[i] + np.random.normal(scale=0.5))
                elif i == 2:
                    child[i] = self.bound_Kd(child[i] + np.random.normal(scale=0.5))
        return child[:3]


    def survival_selection(self):
        self.population.sort(key=lambda x:x[3], reverse=True)
        return self.population[0], self.population[1]

    def parent_selection(self):
        # normalization
        # self.pop_fitness = self.pop_fitness/self.pop_fitness.sum()
        tot_sum = 0
        for i in range(self.pop_size):
            tot_sum += self.population[i][3]
        # tot_sum = sum(self.pop_fitness)
        t = random.uniform(0, tot_sum)
        par_sum = 0
        for i in range(self.pop_size):
            par_sum += self.population[i][3]
            if par_sum >= t:
                return self.population[i]
        print("no return")

if __name__ == "__main__":
    '''
    1-c
    '''
    ga = GA(50, 150, 0.6, 0.25, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    ga.run()
    x = [i+1 for i in range(150)]
    plt.plot(x, ga.best_per_generation)
    plt.title("GA plot")
    plt.xlabel("Genration")
    plt.ylabel("Fitness of best solution")
    plt.show()
    '''
    1-d-i
    '''
    # ga300 = GA(50, 300, 0.6, 0.25, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    # ga300.run()

    # ga200 = GA(50, 200, 0.6, 0.25, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    # ga200.run()
    # ga100 = GA(50, 100, 0.6, 0.25, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    # ga100.run()

    # x = [i+1 for i in range(300)]
    # plt.figure(1)
    # plt.plot(x, ga300.best_per_generation)
    # plt.title("GA with 300 generations plot")
    # plt.xlabel("Genration")
    # plt.ylabel("Fitness of best solution")
    # plt.show()

    # plt.figure(2)
    # x = [i+1 for i in range(200)]
    # plt.plot(x, ga200.best_per_generation)
    # plt.title("GA with 200 generations plot")
    # plt.xlabel("Genration")
    # plt.ylabel("Fitness of best solution")
    # plt.show()

    # plt.figure(3)
    # x = [i+1 for i in range(100)]
    # plt.plot(x, ga100.best_per_generation)
    # plt.title("GA with 100 generations plot")
    # plt.xlabel("Genration")
    # plt.ylabel("Fitness of best solution")
    # plt.show()


    '''
    1-d-ii
    '''
    # ga30 = GA(30, 150, 0.6, 0.25, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    # ga30.run()

    # ga60 = GA(60, 150, 0.6, 0.25, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    # ga60.run()

    # ga90 = GA(90, 150, 0.6, 0.25, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    # ga90.run()

    # x = [i+1 for i in range(150)]
    # plt.plot(x, ga30.best_per_generation)
    # plt.plot(x, ga60.best_per_generation)
    # plt.plot(x, ga90.best_per_generation)
    # plt.title("GA with different population size plot")
    # plt.xlabel("Genration")
    # plt.ylabel("Fitness of best solution")
    # plt.legend(["30 population size","60 population size","90 population size"],loc="best")
    # plt.show()

    '''
    1-d-iii
    '''
    # ga1 = GA(50, 150, 0.6, 0.25, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    # ga1.run()
    # ga2 = GA(50, 150, 0.7, 0.25, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    # ga2.run()
    # ga3 = GA(50, 150, 0.8, 0.25, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    # ga3.run()

    # ga4 = GA(50, 150, 0.6, 0.1, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    # ga4.run()
    # ga5 = GA(50, 150, 0.6, 0.2, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    # ga5.run()
    # ga6 = GA(50, 150, 0.6, 0.3, 2, 18, 1.05, 9.42, 0.26, 2.37, 0.4)
    # ga6.run()

    # x = [i+1 for i in range(150)]
    # plt.figure(1)
    # plt.plot(x, ga1.best_per_generation)
    # plt.plot(x, ga2.best_per_generation)
    # plt.plot(x, ga3.best_per_generation)
    # plt.title("GA with different crossover probability plot")
    # plt.xlabel("Genration")
    # plt.ylabel("Fitness of best solution")
    # plt.legend(["crossover probability=0.6","crossover probability=0.7","crossover probability=0.8"],loc="best")
    # plt.show()

    # plt.figure(2)
    # plt.plot(x, ga4.best_per_generation)
    # plt.plot(x, ga5.best_per_generation)
    # plt.plot(x, ga6.best_per_generation)
    # plt.title("GA with different mutation probability plot")
    # plt.xlabel("Genration")
    # plt.ylabel("Fitness of best solution")
    # plt.legend(["mutation probability=0.1","mutation probability=0.2","mutation probability=0.3"],loc="best")
    # plt.show()

