from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import random
import pandas as pd

data = pd.read_csv('diabetes.txt', sep='\t')

selected_columns = ['SEX', 'AGE', 'BMI', 'Y']
data = data[selected_columns]

X = data[['SEX', 'AGE', 'BMI']]
Y = data['Y']

# Normaliseren van de data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dataset opdelen in een train en test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def NN(function,layers):
    return MLPClassifier(activation=function, hidden_layer_sizes=layers,random_state=1, max_iter=500).fit(X_train,Y_train)

def fitness(function,layers):
    network = NN(function,layers)
    accuracy = network.score(X_test, Y_test)
    return accuracy


def crossover():
    1


def mutation(funcion,layers):
    gens = [function,layers]
    gen = random.choice(gens)
    while gen == function:
        function = random.choice(functions)
    
    while gen == layers:
        layers = random.randrange(1,100)
    print("The new features are: ['", function,"' '",layers,"']")
    accuracy = fitness(function,layers)
    return accuracy

number = 5

total = []

for i in range(number):
    score = True

    while score == True:
        network = []
        functions = ['identity', 'logistic', 'tanh', 'relu']
        function = random.choice(functions)

        layers = random.randrange(1,100)

        accuracyNN = fitness(function,layers)

        print(accuracyNN)

        if accuracyNN != 1.0: #We don't want the parent to be optimal yet!
            network.append(accuracyNN)
            network.append(function)
            network.append(layers)

            total.append(network)
            score = False

total.sort(reverse=True)

# print(total)

print('There are',number,'parants, there accuracys scores are: \n')