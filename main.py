from runners.agent import Agent
import json, argparse, time, os, sqlite3
from typing import List


# define arguments to parse (should also put the allowed values in the help)
parser = argparse.ArgumentParser(description='arguments to train/inference world model and policy')
parser.add_argument('--model_type', default='wm_gru', help='pass in the model type')
parser.add_argument('--run_mode', default='train', help='pass in the run mode')
parser.add_argument('--model_dir_name', default='', help="pass in the model dir name")
parser.add_argument("--model_name", default="", help="pass in the model name")
parser.add_argument("--load_ckpts_dir", default="", help="pass in the ckpts dir name")
parser.add_argument("--ckpt_name", default="", help="pass in the ckpt name")
parser.add_argument(
    '--db_dir_name',
    default='pretraining_rollouts',
    help='Subfolder under data/ containing rollout .db files (e.g. 1000000_transitions)',
)



def combine_trajectories(transitions_path: str):

    def get_all_rows_per_db(db_path: str):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = '''SELECT * From PretrainingData'''
        cursor.execute(query)

        rows = cursor.fetchall()
        return rows
    
    def save_mergerd_transitions(save_path: str, merged_rows: List, 
                             db_name: str = "combined_transitions.db"):
        save_path = os.path.join(save_path, db_name)
        conn = sqlite3.connect(save_path)
        cursor = conn.cursor()

        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS PretrainingData (
            trajectories TEXT
            )
            """
        cursor.execute(create_table_query)


        cursor.executemany("INSERT INTO PretrainingData (trajectories) VALUES (?)", merged_rows)
        conn.commit()
        print("Saved all trajectories into one db")


    all_transitions_path = os.path.join(os.getcwd(), f"data/{transitions_path}")
    all_transition_dbs = os.listdir(all_transitions_path)
    combined_transitions = []
    for file in all_transition_dbs:
        db_path = os.path.join(all_transitions_path, file)
        # print(db_path)
        all_rows_single_db = get_all_rows_per_db(db_path)
        
        combined_transitions += all_rows_single_db


    save_mergerd_transitions(all_transitions_path,
                         combined_transitions)



def run(args):

    try:
        with open('./config.json', 'r') as file:
            config = json.load(file)
            print('config file loaded successfully')
        file.close()
    except:
        raise FileNotFoundError("'config.json' file not found in the current dir")
    
    # populate ckpt in the config
    if args.run_mode == "train":
        if args.load_ckpts_dir != "" and args.ckpt_name == "":
            raise ValueError("ckpt name is not passed")
        if args.load_ckpts_dir == "" and args.ckpt_name != "":
            raise ValueError("ckpts dir is not passes")
        
        if args.load_ckpts_dir != "" and args.ckpt_name != "":
            config["use_ckpt"] = True
            config["ckpt_dir"] = args.load_ckpts_dir
            config["ckpt_name"] = args.ckpt_name

        else:
            config["use_ckpt"] = False
    
    # check if model dir name is passed
    if args.run_mode == "eval": 
        if args.model_dir_name == "" or args.model_name == "":
            raise ValueError("model dir/name is empty")

    config['model_dir_name'] = args.model_dir_name
    config["model_name"] = args.model_name

    # Accept: "pretraining_rollouts", "data/pretraining_rollouts", or absolute path.
    if os.path.isabs(args.db_dir_name):
        db_base = args.db_dir_name
    elif args.db_dir_name.startswith('data' + os.sep) or args.db_dir_name == 'data':
        db_base = os.path.join(os.getcwd(), args.db_dir_name)
    else:
        db_base = os.path.join(os.getcwd(), 'data', args.db_dir_name)
    if not os.path.isdir(db_base):
        raise FileNotFoundError(f"Database directory not found: {db_base}")
    db_paths = sorted(
        os.path.join(db_base, f) for f in os.listdir(db_base) if f.endswith('.db')
    )
    
    print()

    if not db_paths:
        raise FileNotFoundError(f"No .db files in {db_base}")
    
    # create combined trajectories
    combine_trajectories(args.db_dir_name)
    print("Created combined transitions db")

    config['db_base_dir'] = os.path.abspath(db_base)
    config['db_paths'] = [os.path.abspath(p) for p in db_paths]
    config['db_path'] = config['db_paths'][0]
    config['combined_db_path'] = os.path.join(os.getcwd(), f"data/{args.db_dir_name}/combined_transitions.db")
    config["run_mode"] = args.run_mode
    agent = Agent(config)


    agent(args.run_mode, args.model_type)




if __name__=='__main__':
    args = parser.parse_args()
    run(args)



# check into raise types
