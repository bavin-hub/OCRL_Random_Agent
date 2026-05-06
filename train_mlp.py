import os
import torch
from wm import MlpWM


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class MlpTrainer:
    def __init__(self, config: dict):
        self.config = config

    def update(self, model_type: str, load_dataset):
        wt = self.config["world_model_training_params"]
        rp = self.config["robot_params"]
        arch = self.config["world_model_arch_params_mlp"]
        S, A = rp["state_dims"], rp["action_dims"]
        device = torch.device(self.config["device"])

        N = wt["N"]
        decay = wt.get("forecast_decay", 1.0)

        self.data_loader = load_dataset(
            batch_size=wt["batch_size"],
            db_paths=self.config.get("db_paths"),
            db_path=self.config.get("db_path"),
            combined_db_path=self.config.get("combined_db_path"),
            run_mode=self.config.get("run_mode"),
            mean=self.config.get("mean_state_action"),
            std=self.config.get("std_state_action"),
            M=1,
            N=N,
        )

        world_model = MlpWM(
            state_dim=S,
            action_dim=A,
            hidden_dim=arch["hidden_dim"],
            num_layers=arch["num_layers"],
            mlp_head_dim=arch["mlp_head_dim"],
            with_uncertainty=arch.get("with_uncertainty", False),
            lr=arch.get("lr", wt.get("learning_rate", 1e-3)),
            weight_decay=wt.get("weight_decay", 0.0),
            device=str(device),
        )
        world_model.train()
        print("World model (MlpWM) instantiated")
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
            n_steps = 0

            for step, batch in enumerate(self.data_loader):
                x = batch.to(device)  # (B, N+1, S+A)

                s_pred = x[:, 0, :S]  # real first state
                batch_loss = 0.0
                alpha = 1.0

                for t in range(N):
                    a_t = x[:, t, S:S + A]
                    mu = world_model(s_pred, a_t, x_prev=s_pred)
                    target = x[:, t + 1, :S]
                    loss_t = world_model.mse_loss(mu, target)
                    batch_loss = batch_loss + alpha * loss_t
                    alpha *= decay
                    s_pred = mu

                batch_loss = batch_loss / N

                world_model.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=1.0)
                world_model.optimizer.step()

                epoch_loss += batch_loss.item()
                training_loss.append(batch_loss.item())
                n_steps += 1

            print(f"end of epoch {epoch} (avg loss {epoch_loss / max(n_steps, 1):.6f})\n")

            # if epoch % self.config.get("model_save_freq", 10) == 0:
            #     path = os.path.join(save_dir, f"{model_type}-epoch_{epoch}.pth")
            #     torch.save(world_model.state_dict(), path)
            #     print("saved", path)

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
