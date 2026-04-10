import os 
import sqlite3
import json
import numpy as np

combined_db_path = os.path.join(os.getcwd(), "data/pretraining_rollouts/50000_transitions/combined_transitions.db")


conn = sqlite3.connect(combined_db_path)
cursor = conn.cursor()

query = '''SELECT * From PretrainingData'''
cursor.execute(query)

rows = cursor.fetchall()

# print(len(rows))

# print(type(json.loads(rows[0][0])["trajectory-1"]))
# print(len(json.loads(rows[0][0])["trajectory-1"]))
# print(len(json.loads(rows[0][0])["trajectory-1"][0]))

state_actions = []
for row in rows:
    trajectory = json.loads(row[0])["trajectory-1"]
    # print(len(trajectory))
    # break
    for step in trajectory:
        state = step[0]
        action = step[2]
        pair = state + action
        state_actions.append(pair)
        # print(state)
        # print(action)
        # break
    # break

print(len(state_actions))
state_action_arr = np.array(state_actions)
print(state_action_arr.shape)
# print(np.max(state_action_arr, axis=0))

mean = np.mean(state_action_arr, axis=0, dtype=np.float32)
std = np.std(state_action_arr, axis=0, dtype=np.float32)



import matplotlib.pyplot as plt

# Data for first line
x = [i for i in range(29)]
y1 = state_action_arr[2, 96:]

# Data for second line
y2 = state_action_arr[2, 9:38]

# Plotting both lines
plt.plot(x, y1, label='action at')
plt.plot(x, y2, label='state t+1')

# Customizing the graph
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('diff in at and st+1')
plt.legend() # Displays labels to identify lines
plt.show()

np.set_printoptions(suppress=True, precision=3)
print("Mean values of state")
print(mean[:3])
print(mean[3:6])
print(mean[6:9])
print(mean[9:38])
print(mean[38:67])
print(mean[67:96])
print("\n\n\n")

print("Std values of state")
print(std[:3])
print(std[3:6])
print(std[6:9])
print(std[9:38])
print(std[38:67])
print(std[67:96])
print("\n\n\n")
# print(std)

print("Mean values of action")
print(mean[96:])
print("\n\n\n")

print("Std values of action")
print(std[96:])
print("\n\n\n")


with open("config.json", "r") as file:
    config = json.load(file)
    file.close()


hardcoded_st_at_mean = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
  ] + [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
  ]


hardcoded_st_at_std = [
    0.95,
    0.85,
    0.65,
    1.2,
    1.05,
    1.75,
    0.18,
    0.18,
    0.17,
    0.85,
    0.767954,
    0.85,
    0.65275474,
    0.3071794,
    0.12,
    0.85,
    0.767954,
    0.85,
    0.65275474,
    0.3071794,
    0.12,
    0.85,
    0.2288,
    0.2288,
    0.85,
    0.844734,
    0.85,
    0.6911519999999999,
    0.85,
    0.7103492,
    0.7103492,
    0.85,
    0.844734,
    0.85,
    0.6911519999999999,
    0.85,
    0.7103492,
    0.7103492,
    2.1,
    1.35,
    1.35,
    2.1,
    2.1,
    1.35,
    2.1,
    1.35,
    1.35,
    2.1,
    2.1,
    1.35,
    0.95,
    0.95,
    0.95,
    1.55,
    1.55,
    1.55,
    1.55,
    1.85,
    1.85,
    1.85,
    1.55,
    1.55,
    1.55,
    1.55,
    1.85,
    1.85,
    1.85,
    11.0,
    11.0,
    11.0,
    11.0,
    11.0,
    11.0,
    11.0,
    11.0,
    11.0,
    11.0,
    38.72,
    38.72,
    38.72,
    38.72,
    38.72,
    61.160000000000004,
    61.160000000000004,
    61.160000000000004,
    61.160000000000004,
    3.0,
    3.0,
    3.0,
    3.0,
    22.0,
    22.0,
    22.0,
    22.0,
    22.0,
    22.0
  ] + [
    0.46,
    0.46,
    0.46,
    0.46,
    0.46,
    0.38,
    0.46,
    0.46,
    0.46,
    0.46,
    0.46,
    0.38,
    0.46,
    0.46,
    0.46,
    0.46,
    0.46,
    0.46,
    0.46,
    0.44,
    0.32,
    0.32,
    0.46,
    0.46,
    0.46,
    0.46,
    0.44,
    0.32,
    0.32
  ]

# config["mean_state_action"] = mean.tolist()
# config["std_state_action"] = std.tolist()
config["mean_state_action"] = hardcoded_st_at_mean
config["std_state_action"] = hardcoded_st_at_std

with open("config.json", "w") as file:
    json.dump(config, file, indent=4)
    file.close()



