from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import random
import pandas as pd
import numpy as np

data = pd.read_csv('diabetes.txt', sep='\t')

selected_columns = ['SEX', 'AGE', 'BMI','BP','S1','S2','S3','S4','S5','S6', 'Y']
data = data[selected_columns]

X = data[['SEX', 'AGE','BMI','BP','S1','S2','S3','S4','S5','S6',]]
Y = data['Y']

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

print('The number if parents:',number,', there accuracys of them is: \n')

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
            print('\nfeatures of the child:',i,': ',child)
            accuracy = fitness(str(child[0]),int(child[1]),str(child[2]))
            print('The accuracy: ',accuracy)
            print('The features of the parents:', total[0][1:4],'and',total[1][1:4])
            print('The accuracy score: ',total[0][0],' and ', total[1][0])

            if accuracy == 1.0:
                print('This child: ',child,'is optimal','\n')
                loop = False
                break
            elif accuracy > total[0][0]:
                print('The score of the child is better than the parents','\n')
            elif accuracy == total[0][0]:
                print('The score of the child is equal to that of one parent','\n')
            else:
                print('The score of child',i," isn't higher than or, equal to, the score of the best parent",'\n')
                new_accuracy = mutation(str(child[0]),int(child[1]),str(child[2]))
                if new_accuracy == 1.0:
                    print('This child performs optimal, the score is ',new_accuracy,'\n')
                    loop = False
                    break
                elif new_accuracy > total[0][0]:
                    print('The new score of the child is ',new_accuracy,', tscore is beter than the parents','\n')
                elif new_accuracy == total[0][0]:
                    print('The new score of the child is ',new_accuracy,', soore is the same as the parents','\n')
                else:
                    print('The new score of the child is ',new_accuracy,', score not beter than the parents','\n')
            i += 1
        parent = j + 1
        print('\n crossover between the 1st and the',parent,"the childs weren't optimal")
        j += 1
    else:
        print('\n no optimal child found..')
        loop = False