import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras.models import Sequential #No idea why this does not read from tensorflow
# from tensorflow.keras.layers import Dense #No idea why this does not read from tensorflow
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
# from tensorflow.python.keras.optimizers import Adam
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# tf.autograph.set_verbosity(0)
from sklearn.model_selection import train_test_split


#Data processing
file_path = "data/raw/coffee.csv"
df = pd.read_csv(file_path)

print(f"df info: {df.info()}")
print(f"df shape: {df.shape}")
print(f"df describe: {df.describe()}")

X = df.iloc[:,:-1].values # Capture all rows, all columns, except the last one (Y)
Y = df.iloc[:,-1].values # Capture all rows of the last column

#Separate the data based on classes, for plotting:
X_class_0 = X[Y == 0]
X_class_1 = X[Y == 1]

plt.scatter(X_class_0[:,0], X_class_0[:,1], marker = 'o', color='blue', label = 'Y=0')
plt.scatter(X_class_1[:,0], X_class_1[:,1], marker = 'X', color='red', label = 'Y=0')
plt.xlabel("Temperature (Celcius)")
plt.ylabel("Time (minutes)")
plt.title("Good (o) vs. bad (x) coffee roasting w.r.t temperature and time")
# plt.show()

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X).numpy()
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

# train_size = int(0.8 * Xn.shape[0])
# X_train = Xn.take(0.8)
# X_test  = Xn.skip(0.8)
# y_train = Y.take(0.8)
# y_test  = Y.skip(0.8)

test_size = 0.2  # Example test size, adjust as needed
X_train, X_test, y_train, y_test = train_test_split(Xn, Y, test_size=test_size, random_state=79456)


#Dataset is too small, copy it 1000 times for a large dataset for fewer epochs towards convergence
print(X_train.shape, y_train.shape)  
Xt = np.tile(X_train,(1000,1))
Yt = np.tile(y_train,(1000,1)).flatten()
print(Xt.shape, Yt.shape)   
# Set the neural network
tf.random.set_seed(9271)
model = Sequential(
    [
        # tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', input_shape=(2,), name = 'L1'),
        Dense(1, activation='sigmoid',                   name = 'L2')
     ]
)

model.summary()


W1, b1 = model.get_layer("L1").get_weights()
W2, b2 = model.get_layer("L2").get_weights()

print(f"W1 shape: {W1.shape}, W1: {W1}, b1 shape: {b1.shape}, b1: {b1}")
print(f"W2 shape: {W2.shape}, W2: {W2}, b2 shape: {b2.shape}, b2: {b2}")


model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy'] 
)

model.fit(
    Xt,Yt,            
    epochs=10,
)


W1, b1 = model.get_layer("L1").get_weights()
W2, b2 = model.get_layer("L2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

### Testing Phase
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# # Evaluate the performance of the model
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# logloss = log_loss(y_test, y_pred_prob)

# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
# print(f"Log Loss: {logloss}")