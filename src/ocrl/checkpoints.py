"""Model checkpoint persistence and run directories."""
import os
from pathlib import Path
from datetime import datetime
import torch

def get_model_name(model_type):
    string_date_time = "_".join(str(datetime.now()).split(".")[0].split(" "))
    model_name = model_type + "_" + string_date_time
    return model_name

def create_runs_dir(model_dir_name):
    models_dir = os.path.join(os.path.join(os.getcwd(), f'logs/saved_models/{model_dir_name}'))
    if not os.path.isdir(models_dir):
        print('Creating runs directory')
        os.makedirs(models_dir)

    plots_dir = os.path.join(os.path.join(os.getcwd(), f'logs/plots/{model_dir_name}'))
    if not os.path.isdir(plots_dir):
        print('Creating plots directory')
        os.makedirs(plots_dir)

    ckpts_dir = os.path.join(os.path.join(os.getcwd(), f"logs/ckpts/{model_dir_name}"))
    if not os.path.isdir(ckpts_dir):
        print("Creating ckpts dir")
        os.makedirs(ckpts_dir)

    return models_dir, plots_dir, ckpts_dir

def SaveModel(model_obj, model_name, model_dir_name):
    save_dir, _, _ = create_runs_dir(model_dir_name)
    model_dir = os.path.join(save_dir, model_name)
    torch.save(model_obj.state_dict(), model_dir)
    print("\n#########")
    print("Model saved")
    print("#########\n")

def load_ppo_policy_from_checkpoint(
    algo,
    checkpoint_path,
    *,
    load_optimizer: bool = True,
    strict: bool = True,
):

    p = Path(checkpoint_path)
    if not p.is_file():
        p = Path(os.getcwd()) / checkpoint_path
    if not p.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")

    loaded = torch.load(str(p), map_location="cpu", weights_only=False)
    load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": load_optimizer,
        "iteration": False,
        "rnd": True,
    }
    algo.load(loaded, load_cfg, strict=strict)
    return loaded.get("iter")

def SaveCkpt(model_obj, model_name, model_dir_name, current_epoch, current_loss):
    _, _, ckpt_dir = create_runs_dir(model_dir_name)
    ckpt_dir = os.path.join(ckpt_dir, model_name)

    checkpoint = {"model_state_dict": model_obj.state_dict(),
                  "optimizer_state_dict": model_obj.optimizer.state_dict(),
                  "epoch":current_epoch,
                  "loss":current_loss}
    torch.save(checkpoint, ckpt_dir)
    print("\n\n#########")
    print("Checkpoint saved")
    print("#########\n")

def LoadCkpt(model_obj, model_name, model_dir_name):
    direct = Path(model_dir_name) / model_name
    ckpt_dir = direct if direct.is_file() else Path('logs/ckpts') / model_dir_name / model_name
    checkpoints = torch.load(ckpt_dir, map_location=next(model_obj.parameters()).device, weights_only=True)

    model_obj.load_state_dict(checkpoints["model_state_dict"])
    model_obj.optimizer.load_state_dict(checkpoints["optimizer_state_dict"])
    model_obj.train()
    print("\n\n#########")
    print("Checkpoint loaded")
    print("#########\n")
    return model_obj

def count_parameters(model):
    trainable_params = 0
    non_trainable_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_params += parameter.numel()
        else:
            non_trainable_params += parameter.numel()
            
    print(f"Total Trainable Params: {trainable_params}")
    print(f"Total Non-Trainable Params: {non_trainable_params}")
    print(f"Total Params: {trainable_params + non_trainable_params}")
    return trainable_params, non_trainable_params
