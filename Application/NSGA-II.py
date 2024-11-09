import pickle
import numpy as np
import random
import pandas as pd
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
random.seed(42)
np.random.seed(42)

def load_model(filepath):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model


# Evaluation of constrains
def evalWithConstraints(individual, model):
    # Johansen's model
    fire_resistance_time = model.predict([individual])[0]
    f_h = 32.8 * (1 - 0.01 * individual[5])
    M_y = 240 * individual[5]**(2.6)
    F1 = (f_h * individual[5] * individual[1]) * individual[6]
    F2 = (f_h * individual[5] * individual[1] * ((2 + 4 * M_y / (f_h * individual[1]**2 * individual[5]))**(0.5) - 1)) * individual[6]
    F3 = (2 * (M_y * f_h * individual[5])**(0.5)) * individual[6]
    LC = min(F1, F2, F3)

    # Constraints check
    if individual[0] - 2 * individual[4] >= individual[2]:
        return 1000, 1000, 1000
    if individual[0] <= individual[4] * 2:
        return 1000, 1000, 1000
    if individual[2] <= individual[0] / 2:
        return 1000, 1000, 1000  # Width of steel plate and timber panel
    # if min(F1, F2, F3) <= 180000:
    #     return 1000, 1000, 1000  # Bearing capacity

    # Objective function: minimize self weight and maximize fire resistance time and load carrying capacity
    weight = 4 * 10**(-7) * 300 * individual[0] * individual[1] * 2 + \
             8 * 10**(-6) * individual[2] * individual[3] * 300 + \
             8 * 10**(-6) * 0.25 * 3.14 * individual[5]**2 * (2 * individual[1] + individual[3]) * individual[6]

    return -fire_resistance_time, weight, -LC


# Initialize model
model = load_model("best_xgb_model.pkl")  # Use XGBoost

# Create type
creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0, -1.0))  # Fitness Min, minmize
creator.create("Individual", list, fitness=creator.FitnessMin)
# Initialize tools
toolbox = base.Toolbox()

# Generate properties
toolbox.register("attr_float_0", random.uniform, 140, 400)
toolbox.register("attr_float_1", random.uniform, 38, 236)
toolbox.register("attr_float_2", random.uniform, 87, 214)
toolbox.register("attr_float_3", random.uniform, 6, 15)
toolbox.register("attr_float_4", random.uniform, 20, 130)
toolbox.register("attr_float_5", random.uniform, 6, 20)
toolbox.register("attr_float_6", random.uniform, 1, 8)
# toolbox.register("attr_float_7", random.uniform, 0, 800)   # Bolt type

def create_individual():
    individual = [
        toolbox.attr_float_0(),  # timber width
        toolbox.attr_float_1(),  # timber thickness
        toolbox.attr_float_2(),  # steel plate width
        toolbox.attr_float_3(),  # steel plate thickness
        toolbox.attr_float_4(),  # edge distance
        toolbox.attr_float_5(),  # bolt diameter
        toolbox.attr_float_6(),  # bolt number
        1,  # bolt type
        30,  # load ratio
    ]
    return creator.Individual(individual)


def custom_mutate(individual, mu, sigma, indpb):
    # mutate 1
    if random.random() < indpb:
        individual[0] += random.gauss(mu, sigma)
        individual[0] = min(max(individual[0], 140), 400)

    # mutate 2
    if random.random() < indpb:
        individual[1] += random.gauss(mu, sigma)
        individual[1] = min(max(individual[1], 38), 236)

    # mutate 3
    if random.random() < indpb:
        individual[2] += random.gauss(mu, sigma)
        individual[2] = min(max(individual[2], 87), 214)

    # mutate 4
    if random.random() < indpb:
        individual[3] += random.gauss(mu, sigma)
        individual[3] = min(max(individual[3], 6), 15)

    # mutate 5
    if random.random() < indpb:
        individual[4] += random.gauss(mu, sigma)
        individual[4] = min(max(individual[4], 20), 130)

    # mutate 6
    if random.random() < indpb:
        individual[5] += random.gauss(mu, sigma)
        individual[5] = min(max(individual[5], 6), 20)
    #
    # mutate 6
    if random.random() < indpb:
        individual[6] += random.gauss(mu, sigma)
        individual[6] = min(max(individual[6], 1), 8)

    return individual,



# Initialize the structure
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register toolbox
toolbox.register("evaluate", evalWithConstraints, model=model)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selNSGA2) #NSGA-II


def main():
    # generate population
    population = toolbox.population(n=500)

    # save the best individual
    hall_of_fame = tools.ParetoFront()

    # record
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    #selection procedure
    algorithms.eaMuPlusLambda(
        population, toolbox, mu=1500, lambda_=3000, cxpb=0.6, mutpb=0.1, ngen=30,
        stats=stats, halloffame=hall_of_fame, verbose=True
    )

    fire_resistances = []
    weights = []
    LC=[]

    for ind in hall_of_fame:
        fire_resistances.append(-ind.fitness.values[0])  # Maximum FRR
        weights.append(ind.fitness.values[1])  # Minimize self weight
        LC.append(ind.fitness.values[2])  # Minimize self weight

    # Plot pareto front
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(weights, fire_resistances, LC, color='b', s=10)
    ax.set_xlabel("Structure Weight")
    ax.set_ylabel("Fire Resistance (min)")
    ax.set_zlabel("Load-carrying Capacity")
    plt.title("3D Pareto Front for Fire Resistance, Weight, and Load-carrying Capacity")
    plt.show()

    # output as csv
    pareto_front_data = []
    for ind in hall_of_fame:
        ind_data = list(ind)
        ind_data.extend(ind.fitness.values)
        pareto_front_data.append(ind_data)

    columns = ["Wood Width", "Wood Thickness", "Iron Width", "Iron Thickness", "Edge Distance", "Bolt Diameter",
               "Bolt Number", "Bolt Type", "Load Ratio", "Weight", "Fire Resistance Time", "Load carrying capacity"]
    pareto_front_df = pd.DataFrame(pareto_front_data, columns=columns)

    pareto_front_df.to_csv("pareto_front.csv", index=False)
    print("Pareto front saved to pareto_front.csv")


    return hall_of_fame


if __name__ == "__main__":
    main()
