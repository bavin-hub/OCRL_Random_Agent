import json, time
from tqdm.notebook import trange, tqdm
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from utils import SaveModel, CreateWorlModelInstance, get_model_name,\
                  count_parameters, SaveCkpt, LoadCkpt
import time
from data.preprocessor import load_vision_dataset


def _wandb_init(config: dict, extra_tags: dict | None = None):
    """Start a W&B run if --wandb_project was provided. Returns the wandb module or None."""
    project = config.get("wandb_project", "")
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed, skipping. pip install wandb")
        return None
    kw = {"project": project, "config": extra_tags or {}}
    entity = config.get("wandb_entity", "")
    if entity:
        kw["entity"] = entity
    name = config.get("wandb_run_name", "")
    if name:
        kw["name"] = name
    wandb.init(**kw)
    return wandb


def _augment_batch(rgb_t, depth_t, rgb_t1, depth_t1):
    """Random horizontal flip + brightness/contrast jitter on (B, T, C, H, W) batches.
    Same augmentation is applied to rgb_t and rgb_t1 so frame pairs stay consistent.
    """
    B = rgb_t.shape[0]
    device = rgb_t.device

    flip = (torch.rand(B, device=device) < 0.5).float().view(B, 1, 1, 1, 1)
    rgb_t   = rgb_t   * (1 - flip) + torch.flip(rgb_t,   dims=[-1]) * flip
    depth_t = depth_t * (1 - flip) + torch.flip(depth_t, dims=[-1]) * flip
    rgb_t1  = rgb_t1  * (1 - flip) + torch.flip(rgb_t1,  dims=[-1]) * flip
    depth_t1 = depth_t1 * (1 - flip) + torch.flip(depth_t1, dims=[-1]) * flip

    brightness = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(0.8, 1.2)
    contrast   = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(-0.1, 0.1)
    rgb_t  = torch.clamp(rgb_t  * brightness + contrast, 0.0, 1.0)
    rgb_t1 = torch.clamp(rgb_t1 * brightness + contrast, 0.0, 1.0)

    return rgb_t, depth_t, rgb_t1, depth_t1

# only state-action pair
class Trainer:

    def __init__(self, config: dict):
        self.config = config
        # self.data_loader = load_dataset(db_path=self.config["db_path"])


    def get_model_params(self, model):
        num_model_params = 0
        for param in model.parameters():
            num_model_params += param.flatten().shape[0]
        
        print('Total params in the world model : ', num_model_params)


    def update(self, model_type, load_dataset):
        if model_type == "wm_vision":
            self.update_vision(model_type)
            return
        
        # get data loader obj
        wt = self.config['world_model_training_params']
        device = self.config["device"] if torch.cuda.is_available() else "cpu"
        db_paths = self.config.get('db_paths') or [self.config['db_path']]
        self.data_loader = load_dataset(
            db_paths=db_paths,
            batch_size=wt['batch_size'],
            M=wt['M'],
            N=wt['N'],
            combined_db_path=self.config.get("combined_db_path"),
            run_mode=self.config.get("run_mode")
        )

        # create model instance
        M, N = self.config['world_model_training_params']['M'], self.config['world_model_training_params']['N']
        decay = self.config['world_model_training_params']['forecast_decay']
        state_dims = self.config['robot_params']['state_dims']
        action_dims = self.config['robot_params']['action_dims']
        world_model = CreateWorlModelInstance(self.config, model_type=model_type)
        world_model.train()
        print('World Model instantiated')
        # self.get_model_params(world_model)
        count_parameters(world_model)
        training_loss = []

        # model dir 
        model_dir_name = get_model_name(model_type)
        print('this is the model name : ', model_dir_name)


        # load checkpoints
        if self.config["use_ckpt"]:
            world_model = LoadCkpt(world_model,
                                   self.config["ckpt_name"],
                                   self.config["ckpt_dir"]) 



        ######################### Training Starts #########################

        # Iterate over epochs
        for epoch in range(1, self.config['world_model_training_params']['epochs']+1):
            print(f'start of epoch {epoch}')
            # Iterate over batches
            epoch_loss = 0.0
            for step, batch_st_ct_at in enumerate(self.data_loader):
                ht = torch.zeros((self.config['world_model_arch_params']['num_gru_layers'], 
                                  batch_st_ct_at.shape[0], 
                                  self.config['world_model_arch_params']['gru_hidden_dim'])).to(device)
                
                x = batch_st_ct_at.to(device) # x -> (bs, M+N, s_dim+a_dim)
                seq_len = x.shape[1]
                batch_loss = 0
                alpha = 1.0


                # Iterate over RNN timestamps
                for t in range(seq_len-1):
                    loss_t = 0
                    if t < M-1:
                        ht = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), 
                                                 ht, predict=False) # torch.unsqueeze(x[:, t, :], dim=1) => (bs, s_dim+a_dim) -> (bs, 1, s_dim+a_dim)
                    else:
                        if t == M-1:
                            x_prev = torch.unsqueeze(x[:, t, :96], dim=1)
                            # std_logits, state_mean
                            st_next_pred, ht, std_logits, state_mean = world_model.forward(torch.unsqueeze(x[:, t, :], dim=1), ht, predict=True, x_prev=x_prev)
                        else:
                            st_next_pred, ht, std_logits, state_mean = world_model.forward(torch.cat((st_next_pred, torch.unsqueeze(x[:, t, -action_dims:], dim=1)), dim=2), 
                                                                                 ht, predict=True, x_prev=st_next_pred)
                        target = torch.unsqueeze(x[:, t+1, :state_dims], dim=1)
                        # loss_t = world_model.nll_loss(dist, target)
                        # print(st_next_pred.shape)
                        # print(target.shape)
                        # print("\n")
                        # loss_t = world_model.mse_loss(st_pred=torch.squeeze(st_next_pred, dim=1),
                        #                               st_true=torch.squeeze(target, dim=1))
                        loss_t = world_model.gnll_loss(state_mean=state_mean,
                                                       state_std=std_logits,
                                                       state_target=target)
                        batch_loss += alpha * loss_t
                        alpha *= decay
                
                batch_loss /= N

                # optimize
                world_model.optimizer.zero_grad()
                batch_loss.backward()
                world_model.optimizer.step()

                epoch_loss += batch_loss
                training_loss.append(batch_loss.item())
                # print(f'Loss at step {step} : {batch_loss.item()}')


            print(f'end of epoch {epoch}\n\n')
            
            # save model
            if epoch % self.config["model_save_freq"] == 0:
                model_name = f'{model_type}-epoch_{epoch}.pth'
                SaveModel(world_model, model_name, model_dir_name)

            # save checkpoint
            if epoch % self.config["ckpt_save_freq"] == 0:
                model_name = f"{model_type}-ckpt-epoch_{epoch}.pth"
                SaveCkpt(world_model, model_name, model_dir_name, epoch, epoch_loss)


        
        ######################### Training Ends #########################

    def update_vision(self, model_type: str):
        vt = self.config["vision_world_model_training_params"]
        device = self.config["device"] if torch.cuda.is_available() else "cpu"
        db_paths = self.config.get("db_paths") or [self.config["db_path"]]

        wb = _wandb_init(self.config, extra_tags=vt)

        vp = self.config["vision_world_model_arch_params"]
        target_size = (vp["image_height"], vp["image_width"])

        self.data_loader = load_vision_dataset(
            db_paths=db_paths,
            batch_size=vt["batch_size"],
            shuffle=True,
            drop_last=True,
            traj_cache_size=vt.get("traj_cache_size", 16),
            seq_len=vt.get("seq_len", 16),
            target_size=target_size,
        )

        test_db_paths = self.config.get("test_db_paths", [])
        self.test_loader = None
        if test_db_paths:
            self.test_loader = load_vision_dataset(
                db_paths=test_db_paths,
                batch_size=vt.get("eval_batch_size", 32),
                shuffle=False,
                drop_last=False,
                traj_cache_size=vt.get("traj_cache_size", 16),
                seq_len=vt.get("seq_len", 16),
                target_size=target_size,
            )
            print(f"Test dataloader: {len(self.test_loader)} batches")

        world_model = CreateWorlModelInstance(self.config, model_type=model_type)
        world_model.train()
        print("Vision World Model instantiated")
        count_parameters(world_model)

        model_dir_name = get_model_name(model_type)
        print("this is the model name : ", model_dir_name)

        epochs        = vt["epochs"]
        depth_scale   = vt.get("depth_scale", 50.0)
        rgb_weight    = vt.get("rgb_loss_weight", 1.0)
        depth_weight  = vt.get("depth_loss_weight", 1.0)
        augment       = vt.get("augment", True)
        accum_steps   = max(1, vt.get("grad_accum_steps", 1))

        # Cosine LR schedule with linear warmup
        # Scheduler steps once per optimizer step (every accum_steps micro-batches).
        total_steps  = epochs * (len(self.data_loader) // accum_steps)
        warmup_steps = min(vt.get("warmup_steps", 1000), total_steps // 10)
        warmup_sched = LinearLR(world_model.optimizer, start_factor=0.01, end_factor=1.0,
                                total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(world_model.optimizer,
                                         T_max=max(1, total_steps - warmup_steps),
                                         eta_min=vt.get("lr_min", vt["learning_rate"] * 0.01))
        scheduler = SequentialLR(world_model.optimizer,
                                 schedulers=[warmup_sched, cosine_sched],
                                 milestones=[warmup_steps])

        print(f"Gradient accumulation: {accum_steps} micro-batches → "
              f"effective batch = {vt['batch_size'] * accum_steps}")

        for epoch in range(1, epochs + 1):
            epoch_loss = epoch_rgb = epoch_depth = 0.0
            micro_count = 0
            print(f"start of epoch {epoch}")

            world_model.optimizer.zero_grad()
            n_batches = len(self.data_loader)
            t_epoch_start = time.time()

            for batch_idx, batch in enumerate(self.data_loader):
                rgb_t    = batch["rgb_t"].to(device).float()
                depth_t  = batch["depth_t"].to(device).float() / depth_scale
                action_t = batch["action_t"].to(device).float()
                prop_t   = batch["prop_t"].to(device).float()
                rgb_t1   = batch["rgb_t1"].to(device).float()
                depth_t1 = batch["depth_t1"].to(device).float() / depth_scale

                if augment:
                    rgb_t, depth_t, rgb_t1, depth_t1 = _augment_batch(
                        rgb_t, depth_t, rgb_t1, depth_t1
                    )

                outputs = world_model(
                    rgb_t=rgb_t, depth_t=depth_t,
                    action_t=action_t, prop_t=prop_t,
                )
                losses = world_model.compute_loss(
                    outputs=outputs, rgb_t1=rgb_t1, depth_t1=depth_t1,
                    rgb_weight=rgb_weight, depth_weight=depth_weight,
                )
                micro_loss = losses["total_loss"] / accum_steps
                micro_loss.backward()

                batch_rgb   = float(losses["rgb_loss"])
                batch_depth = float(losses["depth_loss"])

                micro_count += 1
                epoch_loss  += batch_rgb + batch_depth
                epoch_rgb   += batch_rgb
                epoch_depth += batch_depth

                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    elapsed = time.time() - t_epoch_start
                    eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"  batch {batch_idx+1}/{n_batches}  "
                          f"loss={avg_loss:.4f}  "
                          f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                          flush=True)

                if micro_count % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in world_model.parameters() if p.requires_grad],
                        vt.get("grad_clip_norm", 10.0),
                    )
                    world_model.optimizer.step()
                    scheduler.step()
                    world_model.optimizer.zero_grad()

            # Flush remaining gradients if epoch didn't end on an accum boundary.
            if micro_count % accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in world_model.parameters() if p.requires_grad],
                    vt.get("grad_clip_norm", 10.0),
                )
                world_model.optimizer.step()
                scheduler.step()
                world_model.optimizer.zero_grad()

            n = len(self.data_loader)
            train_metrics = {
                "train/total": epoch_loss / n,
                "train/rgb": epoch_rgb / n,
                "train/depth": epoch_depth / n,
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
            }
            print(
                f"epoch {epoch} train | "
                f"total={train_metrics['train/total']:.6f} "
                f"rgb={train_metrics['train/rgb']:.6f} "
                f"depth={train_metrics['train/depth']:.6f}"
            )
            if wb:
                wb.log(train_metrics, step=epoch)

            # Per-epoch test evaluation.
            if self.test_loader is not None:
                world_model.eval()
                t_loss = t_rgb = t_depth = 0.0
                max_eval = vt.get("max_eval_batches", len(self.test_loader))
                with torch.no_grad():
                    for eval_idx, batch in enumerate(self.test_loader):
                        if eval_idx >= max_eval:
                            break
                        rgb_t    = batch["rgb_t"].to(device).float()
                        depth_t  = batch["depth_t"].to(device).float() / depth_scale
                        action_t = batch["action_t"].to(device).float()
                        prop_t   = batch["prop_t"].to(device).float()
                        rgb_t1   = batch["rgb_t1"].to(device).float()
                        depth_t1 = batch["depth_t1"].to(device).float() / depth_scale

                        outputs = world_model(
                            rgb_t=rgb_t, depth_t=depth_t,
                            action_t=action_t, prop_t=prop_t,
                        )
                        losses = world_model.compute_loss(
                            outputs=outputs, rgb_t1=rgb_t1, depth_t1=depth_t1,
                            rgb_weight=rgb_weight, depth_weight=depth_weight,
                        )
                        t_loss  += float(losses["total_loss"])
                        t_rgb   += float(losses["rgb_loss"])
                        t_depth += float(losses["depth_loss"])

                nt = min(max_eval, len(self.test_loader))
                test_metrics = {
                    "test/total": t_loss / nt,
                    "test/rgb": t_rgb / nt,
                    "test/depth": t_depth / nt,
                }
                print(
                    f"epoch {epoch} test  | "
                    f"total={test_metrics['test/total']:.6f} "
                    f"rgb={test_metrics['test/rgb']:.6f} "
                    f"depth={test_metrics['test/depth']:.6f}"
                )
                if wb:
                    wb.log(test_metrics, step=epoch)
                world_model.train()

            print(f"end of epoch {epoch}\n\n")

            if epoch % self.config["model_save_freq"] == 0:
                model_name = f"{model_type}-epoch_{epoch}.pth"
                SaveModel(world_model, model_name, model_dir_name)

            if epoch % self.config["ckpt_save_freq"] == 0:
                ckpt_name = f"{model_type}-ckpt-epoch_{epoch}.pth"
                SaveCkpt(world_model, ckpt_name, model_dir_name, epoch, epoch_loss)

        if wb:
            wb.finish()

