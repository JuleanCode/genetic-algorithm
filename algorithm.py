from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import random
import pandas as pd
import numpy as np

data = pd.read_csv('diabetes.txt', sep='\t')

columns = ['SEX', 'AGE', 'BMI','BP','S1','S2','S3','S4','S5','S6', 'Y']
data = data[columns]

X = data[['SEX', 'Y','BMI','BP','S1','S2','S3','S4','S5','S6',]]
Y = data['AGE']

# Normaliseren van de data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dataset opdelen in een train en test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def NN(function,layers,learning):
    return MLPClassifier(activation = function, hidden_layer_sizes = layers, learning_rate = learning, random_state = 1, max_iter = 5000).fit(X_train, Y_train)

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
        learning = random.choice(learning_rate)
        
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

        learning_rate = ['constant','invscaling','adaptive']
        learning = random.choice(learning_rate)

        accuracyNN = fitness(function,layers,learning)
       
        if accuracyNN != 1.0: #We don't want the parent to be optimal yet!
            network.append(accuracyNN)
            network.append(function)
            network.append(layers)
            network.append(learning)

            total.append(network)
            score = False

total.sort(reverse=True)

print('The number if parents:',number,', accuracy: \n')

for j in range(0,number):
    print(total[j][0])

#Creating childs 
j = 1
loop=True
while loop == True:
    if j <= 4:
        features_childs = crossover(total[0],total[j])

        i = 1
        for child in features_childs:
            print('\nfeatures child:',i,': ',child)
            accuracy = fitness(str(child[0]),int(child[1]),str(child[2]))
            print('score: ',accuracy)
            print('features parents:', total[0][1:4],'and',total[1][1:4])
            print('score: ',total[0][0],' and ', total[1][0])

            if accuracy == 1.0:
                print('child: ',child,'is optimal','\n')
                loop = False
                break
            elif accuracy > total[0][0]:
                print('score  child > parents','\n')
            elif accuracy == total[0][0]:
                print('score child == to that of one parent','\n')
            else:
                print('The score of child',i," < the score of the best parent",'\n')
                new = mutation(str(child[0]),int(child[1]),str(child[2]))
                if new == 1.0:
                    print('This child performs optimal, the score is ',new,'\n')
                    loop = False
                    break
                elif new > total[0][0]:
                    print('score child is ',new,', score > parents','\n')
                elif new == total[0][0]:
                    print('score child is ',new,', soore == parents','\n')
                else:
                    print('score child is ',new,', score < parents','\n')
            i += 1
        parent = j + 1
        print('\n crossover is 1st and the',parent,"the childs weren't optimal")
        j += 1
    else:
        print('\n no optimal child')
        loop = False