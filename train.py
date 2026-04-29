import os
import torch
import torch.nn.functional as F
from wm import TransWM


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_synthetic_batches(batch_size, state_dim, action_dim, device, **kwargs):
    while True:
        st_at = torch.randn(batch_size, state_dim + action_dim, device=device)
        st_next = torch.randn(batch_size, state_dim, device=device)
        yield st_at, st_next


class Trainer:
    def __init__(self, config: dict):
        self.config = config

    def update(self, model_type: str, load_dataset=load_synthetic_batches):
        wt = self.config["world_model_training_params"]
        rp = self.config["robot_params"]
        arch = self.config["world_model_arch_params"]
        S, A = rp["state_dims"], rp["action_dims"]
        device = torch.device(self.config["device"])

        self.data_loader = load_dataset(
            batch_size=wt["batch_size"],
            state_dim=S,
            action_dim=A,
            device=device,
            db_paths=self.config.get("db_paths"),
            db_path=self.config.get("db_path"),
            combined_db_path=self.config.get("combined_db_path"),
            run_mode=self.config.get("run_mode"),
            mean=self.config.get("mean_state_action"),
            std=self.config.get("std_state_action"),
        )

        world_model = TransWM(
            state_dim=S,
            action_dim=A,
            embed_dim=arch["embed_dim"],
            mlp_dim=arch["mlp_dim"],
            num_heads=arch.get("num_heads", 4),
            num_layers=arch.get("num_layers", 1),
            lr=arch.get("lr", 1e-3),
            weight_decay=arch.get("weight_decay", 0.0),
            device=str(device),
            std_range=tuple(arch.get("std_range", (0.01, 0.06))),
            dropout=arch.get("dropout", 0.0),
        )
        world_model.train()
        print("World model (TransWM) instantiated")
        print("Total params:", count_parameters(world_model))

        save_dir = os.path.join(self.config.get("model_dir", "checkpoints"), model_type)
        os.makedirs(save_dir, exist_ok=True)

        if self.config.get("use_ckpt"):
            ckpt_path = os.path.join(self.config["ckpt_dir"], self.config["ckpt_name"])
            ckpt = torch.load(ckpt_path, map_location=device)
            world_model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                world_model.optimizer.load_state_dict(ckpt["optimizer"])

        training_loss = []

        for epoch in range(1, wt["epochs"] + 1):
            print(f"start of epoch {epoch}")
            epoch_loss = 0.0
            steps = self.config.get("steps_per_epoch", 50)

            for step, batch in enumerate(self.data_loader):
                if step >= steps:
                    break

                st_at, st_next = batch
                st_at = st_at.to(device)
                st_next = st_next.to(device)

                st = st_at[:, :S]
                at = st_at[:, S : S + A]

                mu, _std = world_model(
                    st, at, predict=True, sample=True, x_prev=st
                )
                batch_loss = F.mse_loss(mu, st_next)

                world_model.optimizer.zero_grad()
                batch_loss.backward()
                world_model.optimizer.step()

                epoch_loss += batch_loss.item()
                training_loss.append(batch_loss.item())

            print(f"end of epoch {epoch} (avg loss {epoch_loss / max(steps, 1):.6f})\n")

            if epoch % self.config.get("model_save_freq", 10) == 0:
                path = os.path.join(save_dir, f"{model_type}-epoch_{epoch}.pth")
                torch.save(world_model.state_dict(), path)
                print("saved", path)

            if epoch % self.config.get("ckpt_save_freq", 10) == 0:
                path = os.path.join(save_dir, f"{model_type}-ckpt-epoch_{epoch}.pth")
                torch.save(
                    {
                        "model": world_model.state_dict(),
                        "optimizer": world_model.optimizer.state_dict(),
                        "epoch": epoch,
                        "epoch_loss": epoch_loss,
                    },
                    path,
                )
                print("saved ckpt", path)


if __name__ == "__main__":
    cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "robot_params": {"state_dims": 12, "action_dims": 4},
        "world_model_training_params": {
            "batch_size": 32,
            "epochs": 2,
        },
        "world_model_arch_params": {
            "embed_dim": 64,
            "mlp_dim": 128,
            "num_heads": 4,
            "num_layers": 1,
            "lr": 1e-3,
        },
        "model_save_freq": 1,
        "ckpt_save_freq": 1,
        "model_dir": "checkpoints",
        "use_ckpt": False,
        "steps_per_epoch": 20,
    }
    Trainer(cfg).update("transwm_demo")
