from runners.agent import Agent
import json, argparse, time, os


# define arguments to parse (should also put the allowed values in the help)
parser = argparse.ArgumentParser(description='arguments to train/inference world model and policy')
parser.add_argument('--model_type', default='wm_gru', help='pass in the model type')
parser.add_argument('--run_mode', default='train', help='pass in the run mode')
parser.add_argument('--model_dir_name', default='', help="pass in the model dir name")
parser.add_argument("--model_name", default="", help="pass in the model name")
parser.add_argument(
    '--db_dir_name',
    default='pretraining_rollouts',
    help='Subfolder under data/ containing rollout .db files (e.g. 1000000_transitions)',
)



def run(args):

    try:
        with open('./config.json', 'r') as file:
            config = json.load(file)
            print('config file loaded successfully')
        file.close()
    except:
        raise FileNotFoundError("'config.json' file not found in the current dir")
    
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
    if not db_paths:
        raise FileNotFoundError(f"No .db files in {db_base}")
    config['db_base_dir'] = os.path.abspath(db_base)
    config['db_paths'] = [os.path.abspath(p) for p in db_paths]
    config['db_path'] = config['db_paths'][0]
    agent = Agent(config)


    agent(args.run_mode, args.model_type)




if __name__=='__main__':
    args = parser.parse_args()
    run(args)



# check into raise types
