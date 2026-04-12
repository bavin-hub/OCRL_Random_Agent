from runners.agent import Agent
import json, argparse, time, os, sqlite3, random
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
    default='',
    help='Subfolder under data/ containing rollout .db files (legacy single-dir mode)',
)
parser.add_argument(
    '--train_dirs',
    nargs='+',
    default=[],
    help=(
        'One or more directories containing training .db files. '
        'Each entry is a run timestamp dir, e.g. '
        'data/pretraining_rollouts/25000_transitions/2026-04-08_14-32-20'
    ),
)
parser.add_argument(
    '--test_dirs',
    nargs='+',
    default=[],
    help='One or more directories containing test/eval .db files.',
)
parser.add_argument(
    '--split',
    type=float,
    default=1.0,
    help=(
        'Train fraction for an automatic 80/20-style split of all files found '
        'in --train_dirs (e.g. 0.8). Ignored when --test_dirs is provided.'
    ),
)
parser.add_argument(
    '--split_seed',
    type=int,
    default=42,
    help='Random seed used for the train/test shuffle-split.',
)
parser.add_argument('--wandb_project', default='', help='W&B project name. Empty = no logging.')
parser.add_argument('--wandb_run_name', default='', help='W&B run name (optional).')
parser.add_argument('--wandb_entity', default='', help='W&B entity/team (optional).')


def combine_trajectories(transitions_path: str):

    def get_all_rows_per_db(db_path: str):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''SELECT * FROM PretrainingData''')
        rows = cursor.fetchall()
        conn.close()
        return rows

    def save_merged_transitions(save_path: str, merged_rows: List,
                                db_name: str = "combined_transitions.db"):
        save_path = os.path.join(save_path, db_name)
        conn = sqlite3.connect(save_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS PretrainingData (trajectories TEXT)
        """)
        cursor.executemany("INSERT INTO PretrainingData (trajectories) VALUES (?)", merged_rows)
        conn.commit()
        conn.close()
        print("Saved all trajectories into one db")

    all_transitions_path = os.path.join(os.getcwd(), f"data/{transitions_path}")
    combined_transitions = []
    for fname in os.listdir(all_transitions_path):
        # Only process .db files; skip combined output and any directories.
        if not fname.endswith(".db") or fname == "combined_transitions.db":
            continue
        db_path = os.path.join(all_transitions_path, fname)
        combined_transitions += get_all_rows_per_db(db_path)

    save_merged_transitions(all_transitions_path, combined_transitions)



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

    def _resolve_dir(d: str) -> str:
        """Resolve a directory path relative to cwd or data/."""
        if os.path.isabs(d):
            return d
        if os.path.isdir(d):
            return os.path.abspath(d)
        candidate = os.path.join(os.getcwd(), d)
        if os.path.isdir(candidate):
            return candidate
        candidate2 = os.path.join(os.getcwd(), 'data', d)
        if os.path.isdir(candidate2):
            return candidate2
        raise FileNotFoundError(f"Directory not found: {d}")

    def _dbs_from_dirs(dirs):
        paths = []
        for d in dirs:
            resolved = _resolve_dir(d)
            found = sorted(
                os.path.join(resolved, f)
                for f in os.listdir(resolved)
                if f.endswith('.db') and f != 'combined_transitions.db'
            )
            if not found:
                raise FileNotFoundError(f"No .db files in {resolved}")
            paths.extend(found)
        return paths

    # --train_dirs / --test_dirs take priority over legacy --db_dir_name.
    if args.train_dirs:
        all_dbs = _dbs_from_dirs(args.train_dirs)

        if args.test_dirs:
            # Explicit test dirs — use as-is.
            train_db_paths = all_dbs
            test_db_paths  = _dbs_from_dirs(args.test_dirs)
        elif 0.0 < args.split < 1.0:
            # Auto split all collected .db files by the given fraction.
            rng = random.Random(args.split_seed)
            shuffled = list(all_dbs)
            rng.shuffle(shuffled)
            n_train = max(1, round(len(shuffled) * args.split))
            train_db_paths = shuffled[:n_train]
            test_db_paths  = shuffled[n_train:]
            print(f"Auto split (seed={args.split_seed}): "
                  f"{len(train_db_paths)} train / {len(test_db_paths)} test")
        else:
            train_db_paths = all_dbs
            test_db_paths  = []

        db_paths = train_db_paths  # backward compat alias
        db_base  = os.path.dirname(train_db_paths[0])
    else:
        if not args.db_dir_name:
            raise ValueError("Pass --train_dirs or --db_dir_name")
        db_base = _resolve_dir(args.db_dir_name)
        db_paths = sorted(
            os.path.join(db_base, f) for f in os.listdir(db_base)
            if f.endswith('.db') and f != 'combined_transitions.db'
        )
        if not db_paths:
            raise FileNotFoundError(f"No .db files in {db_base}")
        train_db_paths = db_paths
        test_db_paths  = []

    print(f"Train .db files : {len(train_db_paths)}")
    print(f"Test  .db files : {len(test_db_paths)}")

    config['db_base_dir']     = os.path.abspath(db_base)
    config['db_paths']        = [os.path.abspath(p) for p in train_db_paths]
    config['db_path']         = config['db_paths'][0]
    config['test_db_paths']   = [os.path.abspath(p) for p in test_db_paths]
    config['combined_db_path'] = os.path.join(db_base, 'combined_transitions.db')
    config["run_mode"] = args.run_mode
    config["wandb_project"]  = (args.wandb_project or "").strip()
    config["wandb_run_name"] = (args.wandb_run_name or "").strip()
    config["wandb_entity"]   = (args.wandb_entity or "").strip()

    # Combined DB only needed for state-based world model (wm_gru), not vision.
    if args.model_type not in ("wm_vision_rssm", "wm_vision"):
        if not (args.db_dir_name or "").strip():
            raise ValueError(
                "State-based training requires --db_dir_name (subfolder under data/ with rollout .db). "
                "For vision use --model_type wm_vision and --train_dirs ..."
            )
        combine_trajectories(args.db_dir_name)
        print("Created combined transitions db")

    agent = Agent(config)
    agent(args.run_mode, args.model_type)




if __name__=='__main__':
    args = parser.parse_args()
    run(args)



# check into raise types
