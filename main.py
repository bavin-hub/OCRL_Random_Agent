from runners.agent import Agent
import json, argparse, time, os


# define arguments to parse (should also put the allowed values in the help)
parser = argparse.ArgumentParser(description='arguments to train/inference world model and policy')
parser.add_argument('--model_type', default='wm_gru', help='pass in the model type')
parser.add_argument('--run_mode', default='train', help='pass in the run mode')
parser.add_argument('--model_dir_name', default='', help="pass in the model dir name")
parser.add_argument("--model_name", default="", help="pass in the model name")



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
    
    

    db_dir = os.path.join(os.getcwd(), 'data/pretraining_rollouts')
    all_dbs = os.listdir(db_dir)
    db_path = os.path.join(db_dir, all_dbs[0])
    config['db_path'] = db_path
    agent = Agent(config)


    agent(args.run_mode, args.model_type)




if __name__=='__main__':
    args = parser.parse_args()
    run(args)



# check into raise types
