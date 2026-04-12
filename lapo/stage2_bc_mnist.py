#%%
import sys
# 1. Override sys.argv to simulate the command-line inputs.
sys.argv = [
    "stage2_bc.py", 
    # "env_name=bigfish", 
    "env_name=moving_mnist", 
    "exp_name=mnist_run",
]

import config
import data_loader
import doy
import paths
import torch
import torch.nn.functional as F
import utils
from doy import loop
import numpy as np

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_videos

sns.set_style("whitegrid")

## Fix the torch and numpy seeds
torch.manual_seed(42)
np.random.seed(42)

TRAIN = False

#%%

print(paths.get_models_path(config.get().exp_name))

state_dicts = torch.load(paths.get_models_path(config.get().exp_name), weights_only=False)
cfg = config.get(base_cfg=state_dicts["cfg"], reload_keys=["stage2", "stage3"])
cfg.stage_exp_name = doy.random_proquint(1)
doy.print("[bold green]Running LAPO stage 2 (latent behavior cloning) with config:")
config.print_cfg(cfg)

if state_dicts["step"] != cfg.stage1.steps:
    doy.log(
        f"[bold red]Warning: using IDM/WM from incomplete training run {state_dicts['step']}/{cfg.stage1.steps} steps"
    )

idm, wm = utils.create_dynamics_models(cfg.model, state_dicts=state_dicts)
idm.eval()

policy = utils.create_policy(cfg.model, cfg.model.la_dim)
opt, lr_sched = doy.LRScheduler.make(
    policy=(
        doy.PiecewiseLinearSchedule(
            [0, 50, cfg.stage2.steps + 1], [0.01 * cfg.stage2.lr, cfg.stage2.lr, 0]
        ),
        [policy],
    ),
)



#%%

# train_data, test_data = data_loader.load(cfg.env_name)
# train_iter = train_data.get_iter(cfg.stage2.bs)
# test_iter = test_data.get_iter(128)


from utils import MovingMNIST_LAPO_Stager

# --- Point this to your actual MovingMNIST .npy file path! ---
DATA_PATH = "/home/gb21553/Projects/Video-WARP/data/MovingMNIST/mnist_test_seq.npy" 

train_data = MovingMNIST_LAPO_Stager(DATA_PATH, is_test=False)
test_data = MovingMNIST_LAPO_Stager(DATA_PATH, is_test=True)

# Generate the infinite iterators used in the training loop
train_iter = train_data.get_iter(cfg.stage1.bs)
test_iter = test_data.get_iter(128)

print("MovingMNIST custom dataloaders ready!")




#%%

# _, eval_metrics = utils.eval_latent_repr(train_data, idm)
# doy.log(f"Decoder metrics sanity check: {eval_metrics}")

run, logger = config.wandb_init("lapo_stage2", config.get_wandb_cfg(cfg))


if TRAIN:
        
    for step in loop(
        cfg.stage2.steps + 1, desc="[green bold](stage-2) Training latent policy via BC"
    ):
        lr_sched.step(step)

        policy.train()
        batch = next(train_iter)
        idm.label(batch)

        preds = policy(batch["obs"][:, -2])  # the -2 selects last the pre-transition ob
        loss = F.mse_loss(preds, batch["la"])

        opt.zero_grad()
        loss.backward()
        opt.step()

        logger(
            step=step,
            loss=loss,
            **lr_sched.get_state(),
        )

        if step % 200 == 0:
            policy.eval()
            test_batch = next(test_iter)
            idm.label(test_batch)
            test_loss = F.mse_loss(policy(test_batch["obs"][:, -2]), test_batch["la"])
            logger(step=step, test_loss=test_loss)

    torch.save(
        dict(policy=doy.state_dict_orig(policy), cfg=cfg, logger=logger),
        paths.get_latent_policy_path(cfg.exp_name),
    )


else:
    # Load the trained policy for evaluation and visualization
    state_dict = torch.load(paths.get_latent_policy_path(cfg.exp_name), weights_only=False) 
    policy.load_state_dict(state_dict["policy"])
    policy.eval()
    print("Loaded trained latent policy for evaluation and visualization!")



















#%% Cell 1: Stage 2 Policy Evaluation

# 1. Fetch Training Curves from W&B
api = wandb.Api()
entity = wandb.run.entity if wandb.run else "your_wandb_username" 
project = "lapo_stage2"
run_id = run.id if run else "your_run_id"

try:
    wandb_run = api.run(f"{entity}/{project}/{run_id}")
    history = wandb_run.history()
    # print(history)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.lineplot(data=history, x="_step", y="loss", ax=axes[0], color="blue")
    axes[0].set_title("Policy Behavior Cloning Loss (Train)")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_yscale("log")
    
    if "test_loss" in history.columns:
        sns.lineplot(data=history.dropna(subset=["test_loss"]), x="_step", y="test_loss", ax=axes[1], color="red", marker="o")
        axes[1].set_title("Policy Generalization Loss (Test)")
        axes[1].set_ylabel("MSE Loss")
        axes[1].set_yscale("log")
    
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Could not fetch W&B data: {e}")

# 2. Visualize Latent Action Codebook Usage (Policy vs IDM)
idm.eval()
policy.eval()

test_batch = next(test_iter)
with torch.no_grad():
    # Get True Latent Actions from IDM
    idm.label(test_batch)
    true_la = test_batch["la"]
    _, _, _, true_inds = idm.vq(true_la)
    
    # Get Predicted Latent Actions from Policy
    pred_la = policy(test_batch["obs"][:, -2])
    _, _, _, pred_inds = idm.vq(pred_la)

# Flatten indices for the first codebook to compare distributions
true_inds_flat = true_inds[0].cpu().numpy().flatten()
pred_inds_flat = pred_inds[0].cpu().numpy().flatten()

plt.figure(figsize=(10, 4))
plt.hist([true_inds_flat, pred_inds_flat], bins=cfg.model.vq.num_embs, 
         label=['IDM (Expert)', 'Policy (Agent)'], color=['skyblue', 'salmon'])
plt.title("Codebook 0 Usage: Did the Policy learn the expert's behavior?")
plt.xlabel("Latent Action Index")
plt.ylabel("Frequency")
plt.legend()
plt.show()









#%% Cell 2: 10-Frame Context + 10-Frame Autoregressive Forecast

idm.eval()
policy.eval()
wm.eval()

class MovingMNIST_Rollout_Stager:
    def __init__(self, data_path):
        print("Loading full 20-frame sequences from Test Set...")
        raw_data = np.load(data_path)
        # Transpose to (Sequences, Time, H, W) and take the test set
        data = np.transpose(raw_data, (1, 0, 2, 3))[8000:]
        # Add channel dim: (Num_Seqs, 20, 1, 64, 64)
        self.data = np.expand_dims(data, axis=2) 
        
    def get_random_sequence(self, device=config.DEVICE):
        idx = np.random.randint(len(self.data))
        # Fetch 1 sequence, normalize to [-0.5, 0.5], add Batch dimension
        seq_np = self.data[idx : idx+1].astype(np.float32) / 255.0 - 0.5
        return torch.from_numpy(seq_np).to(device)
    
    def get_specific_sequence(self, idx, device=config.DEVICE):
        seq_np = self.data[idx : idx+1].astype(np.float32) / 255.0 - 0.5
        return torch.from_numpy(seq_np).to(device)

rollout_stager = MovingMNIST_Rollout_Stager(DATA_PATH)

# We use the rollout stager we defined previously to get a full 20-frame sequence
gt_seq_tensor = rollout_stager.get_random_sequence() # Shape: (1, 20, 1, 64, 64)

# 1. Warmup: Fill the buffer with the first 10 Ground Truth frames
pred_frames = [gt_seq_tensor[:, i] for i in range(10)]

print("Forecasting frames 10 to 19 using the Latent Policy...")
with torch.no_grad():
    # 2. Forecast: We are at t=9, predicting t=10, up to t=19
    for t in range(9, 19):
        # The World Model always needs the last 2 frames for momentum context
        wm_context = torch.stack([pred_frames[-2], pred_frames[-1]], dim=1)
        
        # The Policy only looks at the CURRENT frame to decide what to do
        curr_frame = pred_frames[-1]
        
        # Generate the action and quantize it!
        la_continuous = policy(curr_frame)
        la_quantized, _, _, _ = idm.vq(la_continuous)
        
        # World Model predicts the next frame
        next_frame = wm(wm_context, la_quantized)
        pred_frames.append(next_frame)

# Format for plotting
pred_seq_tensor = torch.stack(pred_frames, dim=1)
gt_video = np.clip(gt_seq_tensor[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5, 0, 1)
pred_video = np.clip(pred_seq_tensor[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5, 0, 1)

plot_videos(
    video=pred_video, ref_video=gt_video, plot_ref=True, 
    forecast_start=11, # Bolds the 10th frame!
    save_name="wandb/lapo_10_frame_forecast.png", 
    show_borders=True, cmap='gray'
)



#%% Cell 3: Fully Autonomous "Dream" (2-Frame Bootstrap)

policy.eval()
wm.eval()

gt_seq_tensor = rollout_stager.get_random_sequence() 

# Bootstrap with exactly 2 frames (the bare minimum for velocity calculation)
pred_frames = [gt_seq_tensor[:, 0], gt_seq_tensor[:, 1]]

print("Dreaming 18 frames autonomously...")
with torch.no_grad():
    for t in range(1, 19):
        wm_context = torch.stack([pred_frames[-2], pred_frames[-1]], dim=1)
        
        # Policy chooses action based on its own hallucinated frame!
        curr_frame = pred_frames[-1]
        la_continuous = policy(curr_frame)
        la_quantized = idm.vq(la_continuous)[0]
        
        next_frame = wm(wm_context, la_quantized)
        pred_frames.append(next_frame)

pred_seq_tensor = torch.stack(pred_frames, dim=1)
gt_video = np.clip(gt_seq_tensor[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5, 0, 1)
pred_video = np.clip(pred_seq_tensor[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5, 0, 1)

plot_videos(
    video=pred_video, ref_video=gt_video, plot_ref=True, 
    forecast_start=2, 
    save_name="wandb/lapo_fully_autonomous.png", 
    show_borders=True, cmap='gray'
)



#%% Cell 4: The Alien Injection Experiment

policy.eval()
wm.eval()

# # Sequence A: The base reality
# seq_A = rollout_stager.get_random_sequence()
# # Sequence B: The Alien reality
# seq_B = rollout_stager.get_random_sequence()


## Get specific sequences
idxA, idxB = 46, 1026
seq_A = rollout_stager.get_specific_sequence(idxA)
seq_B = rollout_stager.get_specific_sequence(idxB)

pred_frames = [seq_A[:, 0], seq_A[:, 1]]

print("Running Alien Injection Experiment...")
with torch.no_grad():
    for t in range(1, 19):
        
        # true_pred_frame = torch.clone(pred_frames[-1])

        # --- THE INJECTION ---
        if t == 10:
            print("Injecting alien frame at t=10!")
            pred_frames[-1] = seq_B[:, 10] # Overwrite current frame with Sequence B
        # ---------------------

        wm_context = torch.stack([pred_frames[-2], pred_frames[-1]], dim=1)
        
        la_continuous = policy(pred_frames[-1])
        # la_continuous = policy(true_pred_frame)
        la_quantized = idm.vq(la_continuous)[0]
        
        next_frame = wm(wm_context, la_quantized)
        pred_frames.append(next_frame)

pred_seq_tensor = torch.stack(pred_frames, dim=1)

# We use seq_A as the reference just to see how far off the rails it goes
gt_video = np.clip(seq_A[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5, 0, 1)
pred_video = np.clip(pred_seq_tensor[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5, 0, 1)

plot_videos(
    video=pred_video, ref_video=gt_video, plot_ref=True, 
    forecast_start=11, 
    save_name="wandb/lapo_alien_injection.png", 
    show_borders=True, cmap='gray'
)










#%% Cell 5: State Corruption & Morphing (PyTorch Adaptation)
import os

policy.eval()
wm.eval()
idm.eval()

corrupt_seq_id = 54
test_seq_id = 57

print(f"\nGenerating morphing visualization for test sequence ID: {test_seq_id} corrupted by sequence ID: {corrupt_seq_id}")

# 1. Fetch sequences
seq_clean = rollout_stager.get_specific_sequence(test_seq_id)
seq_corrupt = rollout_stager.get_specific_sequence(corrupt_seq_id)

# In this architecture, state == recent frames. 
# We grab the first frame of the corrupt sequence to act as our injection state.
corrupt_frame = seq_corrupt[:, 0].clone()

# Let's visualize the corrupt frame briefly
plt.imshow(np.clip(corrupt_frame[0].cpu().numpy().transpose(1, 2, 0) + 0.5, 0, 1), cmap='gray')
plt.title(f"Corrupting Frame (from Sequence {corrupt_seq_id})")
plt.axis('off')
plt.show()

# 2. Define the Inference Rollout 
def inference_rollout_morph(seq_ref, corrupt_frame_tensor=None, corrupt_step=-1):
    # Bootstrap with exactly 2 frames
    pred_frames = [seq_ref[:, 0], seq_ref[:, 1]]
    
    with torch.no_grad():
        for t in range(1, 19): 
            # --- STATE CORRUPTION INJECTION ---
            # If we hit the corrupt step, override the most recent frame in the buffer
            if t == corrupt_step and corrupt_frame_tensor is not None:
                pred_frames[-1] = corrupt_frame_tensor.clone()
            # ----------------------------------

            wm_context = torch.stack([pred_frames[-2], pred_frames[-1]], dim=1)
            la_continuous = policy(pred_frames[-1])
            
            # Quantize the action (vq returns a tuple: quantized, loss, perplexity, encodings)
            vq_out = idm.vq(la_continuous)
            la_quantized = vq_out[0] if isinstance(vq_out, tuple) else vq_out
            
            next_frame = wm(wm_context, la_quantized)
            pred_frames.append(next_frame)

    return torch.stack(pred_frames, dim=1)

# 3. Generate both clean and corrupted rollouts (corrupting at t=5)
pred_seq_clean = inference_rollout_morph(seq_clean, corrupt_frame_tensor=None)
pred_seq_corrupt = inference_rollout_morph(seq_clean, corrupt_frame_tensor=corrupt_frame, corrupt_step=5)

# 4. Format for plotting and saving (convert to numpy, shift from [-0.5, 0.5] to [0, 1])
video_clean = np.clip(pred_seq_clean[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5, 0, 1)
video_corrupt = np.clip(pred_seq_corrupt[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5, 0, 1)
video_gt = np.clip(seq_clean[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5, 0, 1)
frame_corrupt_np = np.clip(corrupt_frame[0].cpu().numpy().transpose(1, 2, 0) + 0.5, 0, 1)

os.makedirs("wandb", exist_ok=True)
os.makedirs("artefacts", exist_ok=True)

# 5. Plot corrupted vs clean prediction
plot_videos(
    video=video_corrupt, 
    ref_video=video_clean, # Display the clean run as the reference row
    plot_ref=True,
    forecast_start=2, # Marks where the autoregressive loop begins
    save_name="wandb/lapo_corrupted_morph.png", 
    show_borders=True, 
    cmap='gray',
    save_video=True,
)

# 6. Save all the frames, videos, etc., into an npz array for later use
np.savez(
    "artefacts/lapo_corrupt.npz", 
    corrupt_pred_video=video_corrupt,
    clean_pred_video=video_clean,
    ref_video=video_gt,
    corrupt_frame_ref=frame_corrupt_np
)
print("Artifacts successfully saved to artefacts/vwarp_corrupt.npz")

# %%

