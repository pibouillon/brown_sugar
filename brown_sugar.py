#Import svm model
from sklearn import svm
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, precision_recall_fscore_support
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import random



################################# 
#
#########MODEL TRAINIING#########
#
########### RAW DATA ############
#
#################################


#loading acoustic data
with open('all_data.csv', 'r') as f:
    reader = csv.reader(f, delimiter = ';')
    data = list(reader)
data_array = np.array(data)
data_array = np.array(data, dtype=float)
#318 samples, 4097 frequencies
data_array.shape

#30 samples exhibited Flesh Browning Disorder
np.count_nonzero((data_array[:,1] == 'non'))
       
#split data into traint and test sets
X = data_array[1:1591,4:4101]
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
y = data_array[1:1591,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y , random_state=35)

np.count_nonzero((y_train== 'BN'))
np.count_nonzero((y_test== 'BN'))


def from_letters_to_factor(data):
    # Define conditions for replacement
    condition_bn = (data == "BN")
    condition_non = (data == "non")
    # Replace values based on conditions
    data[condition_bn] = 1
    data[condition_non] = 0

from_letters_to_factor(y_train)
from_letters_to_factor(y_test)
from_letters_to_factor(y)
#Support Vector Machine classification 
#clf = DecisionTreeClassifier(random_state=0)
#clf =  AdaBoostClassifier(algorithm="SAMME")
#clf = KNeighborsClassifier(n_neighbors=2)
clf = svm.SVC(class_weight='balanced')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred)
precision_recall_fscore_support(y_test,y_pred,average = 'weighted')




from sklearn.model_selection import cross_val_score
clf.score(X_test, y_test)
scores = cross_val_score(clf, X, y, cv=25)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


################################# 
#
##########cGAN TRAINIING#########
#
########DATA AUGMENTATION########
#
#################################

# Convert data to pandas DataFrame
real_data = pd.DataFrame(X)
from_letters_to_factor(y)

# One hot encode labels
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_labels = one_hot_encoder.fit_transform(np.array(y).reshape(-1, 1))

# Constants
NOISE_DIM = 100
NUM_CLASSES = 2
NUM_FEATURES = 4097
BATCH_SIZE = 64
TRAINING_STEPS = 50000

# Generator
def create_generator():
    noise_input = Input(shape=(NOISE_DIM,))
    class_input = Input(shape=(NUM_CLASSES,))
    merged_input = Concatenate()([noise_input, class_input])
    hidden = Dense(128, activation='relu')(merged_input)
    output = Dense(NUM_FEATURES, activation='linear')(hidden)
    model = Model(inputs=[noise_input, class_input], outputs=output)
    return model

# Discriminator
def create_discriminator():
    data_input = Input(shape=(NUM_FEATURES,))
    class_input = Input(shape=(NUM_CLASSES,))
    merged_input = Concatenate()([data_input, class_input])
    hidden = Dense(128, activation='relu')(merged_input)
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=[data_input, class_input], outputs=output)
    return model

# cGAN
def create_cgan(generator, discriminator):
    noise_input = Input(shape=(NOISE_DIM,))
    class_input = Input(shape=(NUM_CLASSES,))
    generated_data = generator([noise_input, class_input])
    validity = discriminator([generated_data, class_input])
    model = Model(inputs=[noise_input, class_input], outputs=validity)
    return model

# Create and compile the Discriminator
discriminator = create_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam())

# Create the Generator
generator = create_generator()

# Create the GAN
gan = create_cgan(generator, discriminator)

# Ensure that only the generator is trained
discriminator.trainable = False

gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Follow loss functions
d_loss = [] 
g_loss = [] 

# Train GAN
for step in range(TRAINING_STEPS):
    # Select a random batch of real data with labels
    idx = np.random.randint(0, real_data.shape[0], BATCH_SIZE)
    real_batch = real_data.iloc[idx].values
    labels_batch = one_hot_labels[idx]

    # Generate a batch of new data
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    generated_batch = generator.predict([noise, labels_batch])

    # Train the discriminator
    real_loss = discriminator.train_on_batch([real_batch, labels_batch], np.ones((BATCH_SIZE, 1)))
    fake_loss = discriminator.train_on_batch([generated_batch, labels_batch], np.zeros((BATCH_SIZE, 1)))
    discriminator_loss = 0.5 * np.add(real_loss, fake_loss)

    # Train the generator
    generator_loss = gan.train_on_batch([noise, labels_batch], np.ones((BATCH_SIZE, 1)))
    

    if step % 500 == 0:
        print(f"Step: {step}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")
        d_loss.append(discriminator_loss)
        print(d_loss)
        g_loss.append(generator_loss)
        print(g_loss)


#plotting loss functions
plt.plot(g_loss)
plt.plot(d_loss)
plt.show()

# Generate instances for a given class
def generate_data(generator, data_class, num_instances):
    one_hot_class = one_hot_encoder.transform(np.array([[data_class]]))
    noise = np.random.normal(0, 1, (num_instances, NOISE_DIM))
    generated_data = generator.predict([noise, np.repeat(one_hot_class, num_instances, axis=0)])
    return pd.DataFrame(generated_data)



# Generate 50 instances for each class
synthetic_data_class_0 = generate_data(generator, 0, 1000)
synthetic_data_class_1 = generate_data(generator, 1, 1000)

# Combine all synthetic data into a single DataFrame and apply inverse transform to bring it back to original scale
synthetic_data = pd.concat([synthetic_data_class_0, synthetic_data_class_1], ignore_index=True)

# Create corresponding class labels
synthetic_labels = [0]*1000 + [1]*1000 

# Add labels to the synthetic data
synthetic_data['class'] = synthetic_labels

# Save synthetic data as a CSV file
#synthetic_data.to_csv('synthetic_sound_data.txt')
#synthetic_data.shape
#synthetic_data = np.array(synthetic_data)


################################# 
#
#########MODEL TRAINIING#########
#
#########SYNTHETIC DATA##########
#
#################################

#loading synthetic data
with open('synthetic_sound_data.txt', 'r') as f:
    reader = csv.reader(f, delimiter = ',')
    synth_data = list(reader)
synth_data_array = np.array(synth_data)
synth_data_array = np.array(synth_data, dtype=float)
#2000 samples, 4097 frequencies
synth_data_array.shape

X_synth = synth_data_array[1:,1:4098]
y_synth = synth_data_array[1:,4098]

X_train, X_test, y_train, y_test = train_test_split(X_synth, y_synth, test_size=0.2, stratify=y_synth , random_state=35)

np.count_nonzero((y_test== '1'))
np.count_nonzero((y_test== '1'))
#Support Vector Machine classification 
#clf = DecisionTreeClassifier(random_state=0)
#clf =  AdaBoostClassifier(algorithm="SAMME")
#clf = KNeighborsClassifier(n_neighbors=2)
clf = svm.SVC(class_weight='balanced')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred)
precision_recall_fscore_support(y_test,y_pred,average = 'weighted')


################################# 
#
#########MODEL TRAINIING#########
#
###########MIXED DATA############
#
#################################

# making new array based on conditions
# Specify the column and condition
column_index = 4098  # Index of the column you want to use for the condition
# Extract the specified column for comparison
column_to_compare = synth_data_array[:, column_index]
# Create a boolean mask based on the condition
condition = column_to_compare == '1'
# Use the boolean mask to select rows that meet the condition
synth_data_array = synth_data_array[condition]
X_bn = synth_data_array[:,1:4098]
y_bn = synth_data_array[:,4098]
#select 850 brown synthetic signals
X_bn, X_osef, y_bn, y_osef = train_test_split(X_bn, y_bn, test_size=0.15, stratify=y_bn , random_state=35)

X_tot = np.concatenate((X, X_bn), axis = 0)
y_tot = np.concatenate((y, y_bn), axis = 0)

X_train, X_test, y_train, y_test = train_test_split(X_tot, y_tot, test_size=0.2, stratify=y_tot , random_state=35)
np.count_nonzero((y_train== '1'))
np.count_nonzero((y_test== '1'))

#Support Vector Machine classification 
#clf = DecisionTreeClassifier(random_state=0)
#clf =  AdaBoostClassifier(algorithm="SAMME")
clf = KNeighborsClassifier(n_neighbors=2)
#clf = svm.SVC(class_weight='balanced')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
precision_recall_fscore_support(y_test,y_pred,average = 'weighted')

################################# 
#
#########SCORE PARAMETERS########
#
###########ITERATIONS############
#
#################################

#Datasets :
#Raw data = X, y
#Normalized data = X_norm, y 
#Synthetic data = X_synth, y_synth
#Mixed data = X_tot, y_tot

from sklearn.model_selection import cross_val_score, cross_validate
clf = DecisionTreeClassifier(random_state=0)
#clf =  AdaBoostClassifier(algorithm="SAMME")
#clf = KNeighborsClassifier(n_neighbors=2)
#clf = svm.SVC(class_weight='balanced')
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
scores = cross_validate(clf, X_tot,y_tot , scoring = ('accuracy','f1_weighted'), cv=sss)
sorted(scores.keys())

np.mean(scores['test_accuracy'])
np.mean(scores['test_f1_weighted'])
np.std(scores['test_accuracy'])
np.std(scores['test_f1_weighted'])