import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd

# Laad de dataset vanuit een txt-bestand (zorg ervoor dat 'jouw_dataset.txt' naar het juiste pad wijst)
data = pd.read_csv('diabetes.txt', sep='\t')

# Scheid de kenmerken (X) en het doel (Y)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# Normaliseer de gegevens
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Deel de dataset op in een trainingsset en een testset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Definieer evaluatiefunctie voor regressie
def evaluate_regression_mlp(hyperparameters):
    reg = MLPRegressor(**hyperparameters)
    reg.max_iter = 2000  # Verhoog het maximum aantal iteraties
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    return mse

# Genereer willekeurige hyperparameters als individuen in de populatie
def generate_random_hyperparameters():
    hyperparameters = {
        'hidden_layer_sizes': (random.randint(5, 50),),
        'activation': random.choice(['logistic', 'tanh', 'relu']),
        'alpha': 10.0 ** -np.random.uniform(1, 6)
    }
    return hyperparameters

# Genetisch algoritme en de rest van de code blijven hetzelfde

# Voer het genetisch algoritme uit om de beste hyperparameters te vinden
population_size = 10
num_generations = 20
mutation_rate = 0.1

best_hyperparameters = None
best_mse = float('inf')

for generation in range(num_generations):
    population = [generate_random_hyperparameters() for _ in range(population_size)]

    for hyperparameters in population:
        mse = evaluate_regression_mlp(hyperparameters)
        
        if mse < best_mse:
            best_mse = mse
            best_hyperparameters = hyperparameters
    
    # Mutatie: Voer mutatie uit op sommige individuen
    for i in range(int(population_size * mutation_rate)):
        random_individual = random.choice(population)
        mutated_individual = generate_random_hyperparameters()
        population.remove(random_individual)
        population.append(mutated_individual)

print("Best Hyperparameters:", best_hyperparameters)
print("Best Mean Squared Error:", best_mse)
