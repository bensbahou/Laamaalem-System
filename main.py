# system optimization based on genetic algorithm

# import libraries

import numpy as np

# define the coefficient matrix of two dimensions
c1 = np.array([[0.2, 0.5, 0.5, 0.5], [0.5, 0.2, 0.5, 0.5], [
              0.5, 0.5, 0.2, 0.5], [0.5, 0.5, 0.5, 0.2], ])
c2 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], ])
c3 = np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], ])

# define F0 function


def F0(x, c):
    # x: input matrix of two dimensions
    # c: coefficient matrix of two dimensions
    # sum of the product of the input matrix and the coefficient matrix

    sum = np.sum(np.multiply(x, c))
    return sum

# define F01 function that is the same as F0 function with fixed coefficient matrix


def F01(x):
    return F0(x, c1)

# define F02 function that is the same as F0 function with fixed input matrix


def F02(x):
    return F0(x, c2)

# define F03 function that is the same as F0 function with fixed input matrix and coefficient matrix


def F03(x):
    return F0(x, c3)


# define some examples of input matrix
x1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], ])
x2 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], ])
x3 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], ])


# define a function that generate random input matrix with some constraints


def generate_input_matrix(n, m, min, max):
    # n: number of rows
    # m: number of columns
    # min: minimum value of the input matrix
    # max: maximum value of the input matrix

    # generate a random matrix with the constraints : sum of all elements in the matrix is 1
    x = np.random.uniform(min, max, (n, m))
    x = x / np.sum(x)
    return x


# define a function that generate a random genome


def generate_genome(n, m, min, max):
    matrix = generate_input_matrix(n, m, min, max)
    # convert the matrix to a vector
    vector = matrix.reshape(n * m)
    return vector

# define a function that generate a random population


def generate_population(n, m, min, max, population_size):
    # n: number of rows
    # m: number of columns
    # min: minimum value of the input matrix
    # max: maximum value of the input matrix
    # population_size: number of individuals in the population

    # generate a random population
    population = np.zeros((population_size, n * m))
    for i in range(population_size):
        population[i] = generate_genome(n, m, min, max)
    return population


# test the function generate_population
population = generate_population(4, 4, 0, 1, 5)
# print and format the population matrix to with 2 decimal places
#print(np.around(population, 3))

# define a function that calculate the fitness of each individual in the population


def calculate_fitness1(population):
    # population: a matrix of n individuals and m genes

    # calculate the fitness of each individual in the population
    fitness = np.zeros((len(population), 1))
    for i in range(len(population)):
        # convert the vector to a matrix

        fitness[i] = F01(population[i].reshape(4, 4))
    return fitness


# define a function that select the best individuals in the population


def select_best_individuals(population, fitness, number_of_best_individuals):
    # population: a matrix of n individuals and m genes
    # fitness: a vector of n individuals
    # number_of_best_individuals: number of best individuals to select

    # sort the fitness in descending order
    sorted_fitness = np.sort(fitness, axis=0)[::-1]
    # select the best individuals in the population
    best_individuals = np.zeros(
        (number_of_best_individuals, len(population[0])))
    for i in range(number_of_best_individuals):
        # find the index of the best individual
        index = np.where(fitness == sorted_fitness[i])
        # select the best individual
        best_individuals[i] = population[index[0][0]]
    return best_individuals


# define a function that generate a new population based on the best individuals


def generate_new_population(best_individuals, number_of_best_individuals, population_size):
    # best_individuals: a matrix of n individuals and m genes
    # number_of_best_individuals: number of best individuals to select
    # population_size: number of individuals in the population

    # generate a new population based on the best individuals
    new_population = np.zeros((population_size, len(best_individuals[0])))
    for i in range(number_of_best_individuals):
        new_population[i] = best_individuals[i]
    for i in range(number_of_best_individuals, population_size):
        # select two random individuals
        individual1 = best_individuals[np.random.randint(
            0, number_of_best_individuals)]
        individual2 = best_individuals[np.random.randint(
            0, number_of_best_individuals)]
        # generate a new individual by combining random genes from the two selected individuals
        new_individual = np.zeros((len(individual1)))
        for j in range(len(individual1)):
            if np.random.rand() < 0.5:
                new_individual[j] = individual1[j]
            else:
                new_individual[j] = individual2[j]
        # normalize the new individual
        new_individual = new_individual / np.sum(new_individual)
        new_population[i] = new_individual
    return new_population

# define a function that mutate the population


def mutate_population(population, mutation_rate):
    # population: a matrix of n individuals and m genes
    # mutation_rate: probability of mutation

    # mutate the population
    for i in range(len(population)):
        for j in range(len(population[i])):
            if np.random.rand() < mutation_rate:
                population[i][j] = np.random.rand()
    # normalize the population
    for i in range(len(population)):
        population[i] = population[i] / np.sum(population[i])
    return population

# define a function that run the genetic algorithm


def run_genetic_algorithm(n, m, min, max, population_size, number_of_best_individuals, mutation_rate, number_of_generations):
    # n: number of rows
    # m: number of columns
    # min: minimum value of the input matrix
    # max: maximum value of the input matrix
    # population_size: number of individuals in the population
    # number_of_best_individuals: number of best individuals to select
    # mutation_rate: probability of mutation
    # number_of_generations: number of generations

    # generate the initial population
    population = generate_population(n, m, min, max, population_size)
    # calculate the fitness of the initial population
    fitness = calculate_fitness1(population)
    # run the genetic algorithm
    for i in range(number_of_generations):
        # select the best individuals in the population
        best_individuals = select_best_individuals(
            population, fitness, number_of_best_individuals)
        # generate a new population based on the best individuals
        population = generate_new_population(
            best_individuals, number_of_best_individuals, population_size)
        # mutate the population
        population = mutate_population(population, mutation_rate)
        # calculate the fitness of the new population
        fitness = calculate_fitness1(population)
    return population, fitness


# define a function that run the genetic algorithm and return the best individual and its fitness

def run_genetic_algorithm2(n, m, min, max, population_size, number_of_best_individuals, mutation_rate, number_of_generations):
    # n: number of rows
    # m: number of columns
    # min: minimum value of the input matrix
    # max: maximum value of the input matrix
    # population_size: number of individuals in the population
    # number_of_best_individuals: number of best individuals to select
    # mutation_rate: probability of mutation
    # number_of_generations: number of generations

    # run the genetic algorithm
    population, fitness = run_genetic_algorithm(
        n, m, min, max, population_size, number_of_best_individuals, mutation_rate, number_of_generations)
    # select the best individual in the population
    best_individual = select_best_individuals(
        population, fitness, 1)
    # calculate the fitness of the best individual
    best_fitness = calculate_fitness1(best_individual)
    return best_individual, best_fitness


# test the function run_genetic_algorithm2
best_individual, best_fitness = run_genetic_algorithm2(
    4, 4, 0, 1, 10, 4, 0.05, 20)
print(np.around(best_individual, 3))
print(np.around(best_fitness, 3))
