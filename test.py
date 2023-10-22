from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import random
import pandas as pd



# Schakel de GPU uit
tf.config.set_visible_devices([], 'GPU')

# Forceer CPU-gebruik voor alle bewerkingen (code werkt sneller op CPU)
tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.set_visible_devices([], 'GPU')


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

#MLPC methode
MLPC = MLPClassifier(hidden_layer_sizes=100, activation ='relu',solver='adam', batch_size=32, max_iter= 100, verbose=0).fit(X_train, Y_train)
print(MLPC.predict_proba(X_test))


def evaluate_regression_nn(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train, epochs=100, verbose=0, batch_size=32)
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    return mse

def generate_random_nn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dropout(0.2),  # Voeg een dropout-laag toe met dropout van 20%
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# TODO!!!       aanpassen zodat die door de gebruiker opgegeven kan worden. MLPClassifier gebruiken??
population_size = 10
num_generations = 20
mutation_rate = 0.1

best_model = None
best_mse = float('inf')

for generation in range(num_generations):
    population = [generate_random_nn_model() for _ in range(population_size)]

    for model in population:
        mse = evaluate_regression_nn(model, X_train, Y_train, X_test, Y_test)
        
        if mse < best_mse:
            best_mse = mse
            best_model = model
    
    for i in range(int(population_size * mutation_rate)):
        random_model = random.choice(population)
        mutated_model = generate_random_nn_model()
        population.remove(random_model)
        population.append(mutated_model)

print("Best Model:", best_model.summary())
print("Best Mean Squared Error:", best_mse)
