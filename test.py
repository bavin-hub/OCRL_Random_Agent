import os
import sqlite3
from typing import List


# all_transitions_path = "/home/bavin/personal_git_repos/OCRL_Random_Agent/"
all_transitions_path = os.path.join(os.getcwd(), "data/pretraining_rollouts/50000_transitions/combined_transitions.db")
all_transition_dbs = os.listdir(all_transitions_path)

print(len(all_transition_dbs))


def get_all_rows_per_db(db_path: str):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = '''SELECT * From PretrainingData'''
    cursor.execute(query)

    rows = cursor.fetchall()
    return rows

# def save_mergerd_transitions(save_path: str, merged_rows: List, 
#                              db_name: str = "combined_transitions.db"):
#     save_path = os.path.join(save_path, db_name)
#     conn = sqlite3.connect(save_path)
#     cursor = conn.cursor()

#     create_table_query = f"""
#         CREATE TABLE IF NOT EXISTS PretrainingData (
#            trajectories TEXT
#         )
#         """
#     cursor.execute(create_table_query)


#     cursor.executemany("INSERT INTO PretrainingData (trajectories) VALUES (?)", merged_rows)
#     conn.commit()
#     print("Saved all trajectories into one db")




# combined_transitions = []
# for file in all_transition_dbs:
#     db_path = os.path.join(all_transitions_path, file)
#     # print(db_path)
#     all_rows_single_db = get_all_rows_per_db(db_path)
    
#     combined_transitions += all_rows_single_db



# save_mergerd_transitions(all_transitions_path,
#                          combined_transitions)

# print(len(combined_transitions)) 
 
# test saved combined trajectories
combined_path = os.path.join(all_transitions_path, 'combined_transitions.db')
total_rows = get_all_rows_per_db(combined_path)
print(len(total_rows))
# print(total_rows[1])