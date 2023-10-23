from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import random
import pandas as pd
import numpy as np

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

def NN(function,layers,learning):
    return MLPClassifier(activation = function, hidden_layer_sizes = layers, learning_rate = learning, random_state = 1, max_iter = 500).fit(X_train, Y_train)

def fitness(function,layers,learning):
    network = NN(function,layers,learning)

    accuracy = network.score(X_test, Y_test)
    return accuracy

def crossover(parent1, parent2):
    child1 = np.append(parent1[1:2], parent2[2:])
    child2 = np.append(parent2[1:2], parent1[2:])
    return child1, child2

def mutation(function,layers,learning):
    gens = [function,layers,learning]
    gen = random.choice(gens)
    while gen == function:
        function = random.choice(functions)
    
    while gen == layers:
        layers = random.randrange(1,100)

    while gen == learning:
        learning = random.choice(learning_rates)
        
    print("The new features are: ['", function,"' '",layers,"' '",learning,"']")
    accuracy = fitness(function,layers,learning)
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

        learning_rates = ['constant','invscaling','adaptive']
        learning = random.choice(learning_rates)

        accuracyNN = fitness(function,layers,learning)

        # print(accuracyNN)

        if accuracyNN != 1.0: #We don't want the parent to be optimal yet!
            network.append(accuracyNN)
            network.append(function)
            network.append(layers)
            network.append(learning)

            total.append(network)
            score = False

total.sort(reverse=True)

# print(total)

print('There are',number,'parants, there accuracys scores are: \n')

for j in range(0,number):
    print(total[j][0])

print('First we will make a child out of the two best parents!')

#Creating childs 
j = 1
loop=True
while loop == True:
    if j <= 4:
        features_childs = crossover(total[0],total[j])

        i = 1
        for child in features_childs:
            print('\nfeatures child',i,': ',child)
            accuracy = fitness(str(child[0]),int(child[1]),str(child[2]))
            print('The accuracy score with this features: ',accuracy)
            print('The features of the parents were:', total[0][1:4],'and',total[1][1:4])
            print('The accuracy score of the parents were: ',total[0][0],' and ', total[1][0])

            if accuracy == 1.0:
                print('This child with the features',child,'performs optimatly!','\n')
                loop = False
                break
            elif accuracy > total[0][0]:
                print('The accuracy score of the child is better than that from both parents! But still not optimal..','\n')
            elif accuracy == total[0][0]:
                print('The accuracy score of the child is equal to the accuracy score from at least one parent, the best parent for sure!','\n')
            else:
                print('The accuracy score of child',i," isn't higher than or, equal to, the accuracy score of the best parent, so it gets mutated!",'\n')
                new_accuracy = mutation(str(child[0]),int(child[1]),str(child[2]))
                if new_accuracy == 1.0:
                    print('This child with the features performs optimatly! The accuracy score is ',new_accuracy,'\n')
                    loop = False
                    break
                elif new_accuracy > total[0][0]:
                    print('The new accuracy score of the child is ',new_accuracy,', this accuracy score is better than the accuracy score from both parents! But still not optimal..','\n')
                elif new_accuracy == total[0][0]:
                    print('The new accuracy score of the child is ',new_accuracy,', this accuracy score is equal to the accuracy score from at least one parent, the best parent for sure!','\n')
                else:
                    print('The new accuracy score of the child is ',new_accuracy,', unfortunatly, this accuracy score is still not better than the accuracy score from both parents or at least the best parent..','\n')
            i += 1
        parent = j + 1
        print('\nWith a crossover (and possibly mutations) between the 1 and the',parent,"the childs weren't optimal, we wil look at the next parent!")
        j += 1
    else:
        print('\nAll parents have done crossover to create, and some childs also had a mutation, but unfortunatlly is there no optimal child found..')
        loop = False