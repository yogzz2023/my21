import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import itertools

r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        # print("Initialized filter state:")
        # print("Sf:", self.Sf)
        # print("pf:", self.pf)

    def predict_step(self, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sf = np.dot(Phi, self.Sf)
        self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q
        self.Meas_Time = current_time
        # print("Predicted filter state:")
        # print("Sf:", self.Sf)
        # print("pf:", self.pf)

    def update_step(self, measurements):
        # JPDA update step
        num_meas = len(measurements)
        num_hypotheses = 2 ** num_meas
        likelihoods = np.zeros(num_hypotheses)
        hypotheses = []

        for h in range(num_hypotheses):
            hypothesis = []
            for m in range(num_meas):
                if h & (1 << m):
                    hypothesis.append(m)
            hypotheses.append(hypothesis)
            likelihood = 0
            for m in hypothesis:
                Z = np.array(measurements[m][:3]).reshape(-1, 1)
                Inn = Z - np.dot(self.H, self.Sf)  # Innovation
                S = np.dot(self.H, np.dot(self.pf, self.H.T)) + self.R
                try:
                    L = -0.5 * np.dot(Inn.T, np.dot(np.linalg.inv(S), Inn)) - 0.5 * np.log(np.linalg.det(2 * np.pi * S))
                except np.linalg.LinAlgError:
                    L = -np.inf
                likelihood += L
            likelihoods[h] = likelihood

        max_likelihood = np.max(likelihoods)
        normalized_likelihoods = np.exp(likelihoods - max_likelihood)
        hypothesis_probs = normalized_likelihoods / np.sum(normalized_likelihoods)
        weights = np.zeros((num_meas, 1))

        for m in range(num_meas):
            weight = 0
            for h, hypothesis in enumerate(hypotheses):
                if m in hypothesis:
                    weight += hypothesis_probs[h]
            weights[m] = weight

        for m in range(num_meas):
            Z = np.array(measurements[m][:3]).reshape(-1, 1)
            Inn = Z - np.dot(self.H, self.Sf)  # Innovation
            S = np.dot(self.H, np.dot(self.pf, self.H.T)) + self.R
            K = np.dot(np.dot(self.pf, self.H.T), np.linalg.inv(S))
            self.Sf += weights[m] * np.dot(K, Inn)
            self.pf = np.dot(np.eye(6) - np.dot(K, self.H), self.pf)
            #print(f"Measurement {m+1} - Weight: {weights[m]}")

        # print("Updated filter state:")
        # print("Sf:", self.Sf)
        # print("pf:", self.pf)

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

# Function to convert Cartesian coordinates to spherical coordinates
def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z/np.sqrt(x**2 + y**2))*180/3.14
    az = math.atan(y/x)

    if x > 0.0:
        az = 3.14/2 - az
    else:
        az = 3*3.14/2 - az
    
    az = az * 180 / 3.14
    
    if az < 0.0:
        az = (az + 360.0)
        
    if az > 360:
        az = (az - 360)
        
    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i]/np.sqrt(x[i]**2 + y[i]**2))*180/3.14)
        az.append(math.atan(y[i]/x[i]))

        if x[i] > 0.0:
            az[i] = 3.14/2 - az[i]
        else:
            az[i] = 3*3.14/2 - az[i]
        
        az[i] = az[i] * 180 / 3.14
        
        if az[i] < 0.0:
            az[i] = (az[i] + 360.0)
            
        if az[i] > 360:
            az[i] = (az[i] - 360)
            
    return r, az, el

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[10])  #
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            print("Cartesian coordinates (x, y, z):", x, y, z)
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            print("Spherical coordinates (r, az, el):", r, az, el)
            measurements.append((r, az, el, mt))
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'data_test.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

csv_file_predicted = "data_test.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['FT', 'FX', 'FY', 'FZ']].values

A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)

# number = 1000

# result = np.divide(A[0], number)

# Lists to store the data for plotting
time_list = []
r_list = []
az_list = []
el_list = []

# Iterate through measurements
for i, (r, az, el, mt) in enumerate(measurements):
    if i == 0:
        # Initialize filter state with the first measurement
        kalman_filter.initialize_filter_state(r, az, el, 0, 0, 0, mt)
    elif i == 1:
        # Initialize filter state with the second measurement and compute velocity
        prev_r, prev_az, prev_el = measurements[i - 1][:3]
        dt = mt - measurements[i - 1][3]
        vx = (r - prev_r) / dt
        vy = (az - prev_az) / dt
        vz = (el - prev_el) / dt
        kalman_filter.initialize_filter_state(r, az, el, vx, vy, vz, mt)
    else:
        # Predict step
        kalman_filter.predict_step(mt)

        # Perform JPDAF for associating measurements
        print("\nClustering measurements...")
        # Cluster determination
        clusters = []
        # For now, assume all measurements are in one cluster
        clusters.append([i for i in range(len(measurements))])

        print("Clusters:", clusters)

        print("\nGenerating hypotheses...")
        # Hypothesis generation
        hypotheses = []
        num_targets = 2  # Assuming there are two targets

        # Generate all possible combinations of associations between measurements and targets
        for cluster in clusters:
            for target_indices in itertools.combinations(cluster, num_targets):
                hypotheses.append(list(target_indices))

        print("Hypotheses:", hypotheses)

        # Perform JPDA update step
        possible_measurements = [measurements[idx] for idx in range(i - 2, i + 1)]
        kalman_filter.update_step(possible_measurements)

        # Append data for plotting
        time_list.append(mt)
        r_list.append(kalman_filter.Sf[0][0])
        az_list.append(kalman_filter.Sf[1][0])
        el_list.append(kalman_filter.Sf[2][0])

# Plot range (r) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, r_list, label='filtered range (code)', color='green', marker='o')
plt.scatter(filtered_values_csv[:, 0], A[0], label='filtered range (track id 57)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Range (r)', color='black')
plt.title('Range vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# Plot azimuth (az) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, az_list, label='filtered azimuth (code)', color='green', marker='o')
plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 57)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Azimuth (az)', color='black')
plt.title('Azimuth vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# Plot elevation (el) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, el_list, label='filtered elevation (code)', color='green', marker='o')
plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 57)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Elevation (el)', color='black')
plt.title('Elevation vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()
