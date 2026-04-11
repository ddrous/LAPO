#%%

import sys
# 1. Override sys.argv to simulate the command-line inputs.
sys.argv = [
    "stage1_idm.py", 
    # "env_name=bigfish", 
    "env_name=moving_mnist", 
    "exp_name=mnist_run",
]

import config
import data_loader
import doy
import paths
import torch
import numpy as np
from tensordict import TensorDict
import utils
from doy import loop


## Fix the torch and numpy seeds
torch.manual_seed(42)
np.random.seed(42)


#%%
cfg = config.get()
doy.print("[bold green]Running LAPO stage 1 (IDM/FDM training) with config:")
config.print_cfg(cfg)

run, logger = config.wandb_init("lapo_stage1", config.get_wandb_cfg(cfg))

idm, wm = utils.create_dynamics_models(cfg.model)


if cfg.env_name != "moving_mnist":
    train_data, test_data = data_loader.load(cfg.env_name)
    train_iter = train_data.get_iter(cfg.stage1.bs)
    test_iter = test_data.get_iter(128)

else:
    from utils import MovingMNIST_LAPO_Stager

    # --- Point this to your actual MovingMNIST .npy file path! ---
    DATA_PATH = "/home/gb21553/Projects/Video-WARP/data/MovingMNIST/mnist_test_seq.npy" 

    train_data = MovingMNIST_LAPO_Stager(DATA_PATH, is_test=False)
    test_data = MovingMNIST_LAPO_Stager(DATA_PATH, is_test=True)

    # Generate the infinite iterators used in the training loop
    train_iter = train_data.get_iter(cfg.stage1.bs)
    test_iter = test_data.get_iter(128)

    print("MovingMNIST custom dataloaders ready!")







opt, lr_sched = doy.LRScheduler.make(
    all=(
        doy.PiecewiseLinearSchedule(
            [0, 50, cfg.stage1.steps + 1],
            [0.1 * cfg.stage1.lr, cfg.stage1.lr, 0.01 * cfg.stage1.lr],
        ),
        [wm, idm],
    ),
)

#%%
def train_step():
    idm.train()
    wm.train()

    lr_sched.step(step)

    batch = next(train_iter)

    vq_loss, vq_perp = idm.label(batch)
    wm_loss = wm.label(batch)
    loss = wm_loss + vq_loss

    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([*idm.parameters(), *wm.parameters()], 2)
    opt.step()

    logger(
        step,
        wm_loss=wm_loss,
        global_step=step * cfg.stage1.bs,
        vq_perp=vq_perp,
        vq_loss=vq_loss,
        grad_norm=grad_norm,
        **lr_sched.get_state(),
    )


def test_step():
    idm.eval()  # disables idm.vq ema update
    wm.eval()

    # evaluate IDM + FDM generalization on (action-free) test data
    batch = next(test_iter)
    idm.label(batch)
    wm_loss = wm.label(batch)

    # train latent -> true action decoder and evaluate its predictiveness
    # _, eval_metrics = utils.eval_latent_repr(train_data, idm)

    # logger(step, wm_loss_test=wm_loss, global_step=step * cfg.stage1.bs, **eval_metrics)
    logger(step, wm_loss_test=wm_loss, global_step=step * cfg.stage1.bs)


#%%
for step in loop(cfg.stage1.steps + 1, desc="[green bold](stage-1) Training IDM + FDM"):
    train_step()

    if step % 500 == 0:
        test_step()

    if step > 0 and (step % 5000 == 0 or step == cfg.stage1.steps):

        torch.save(
            dict(
                **doy.get_state_dicts(wm=wm, idm=idm, opt=opt),
                step=step,
                cfg=cfg,
                logger=logger,
            ),
            paths.get_models_path(cfg.exp_name),
        )

        # print(f"Saving model to:\n{paths.get_models_path(cfg.exp_name).absolute()}")



#%%





















## Visualise the training data
import matplotlib.pyplot as plt
## White background for better visibility
import seaborn as sns
sns.set_style("whitegrid")

batch = next(train_iter)
obs = batch["obs"].cpu().numpy() + 0.5  # un-normalize the observations for visualization
print("Observation shape:", obs.shape)      ## (B, L, C, H, W)

fig, axes = plt.subplots(1, 3, figsize=(15, 3))
for i in range(3):
    axes[i].imshow(obs[0, i].transpose(1, 2, 0))  # Show the last observation in the sequence
    axes[i].set_title(f"Frame {i}")
    axes[i].axis("off")
# plt.suptitle("Sample Observations from Training Data")
plt.show()



#%% Cell 1: Parameter Counting
import torchinfo

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Inverse Dynamics Model (IDM) Parameters: {count_parameters(idm):,}")
print(f"Forward Dynamics Model (WM) Parameters:  {count_parameters(wm):,}")
print(f"Total Stage 1 Parameters:                {count_parameters(idm) + count_parameters(wm):,}")

# Optional: Print a detailed summary of the World Model architecture
# We pass dummy inputs matching (obs_seq, latent_action)
dummy_obs = torch.zeros(1, cfg.stage1.bs, 3, 64, 64).to(config.DEVICE)
dummy_la = torch.zeros(1, cfg.model.la_dim).to(config.DEVICE)
# print(torchinfo.summary(wm, input_data=(dummy_obs, dummy_la), depth=2))


#%% Cell 2: Reconstructions (Next Frame Prediction)
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import plot_videos

idm.eval()
wm.eval()

# Grab a batch from the test set
test_batch = next(test_iter)

with torch.no_grad():
    # .label() populates test_batch with 'la_q' (latent action) and 'wm_pred' (predicted frame)
    idm.label(test_batch)
    wm.label(test_batch)

# We will plot the sequence for the first sample in the batch
sample_idx = 0

# Extract the full ground truth sequence: (T, C, H, W) -> (T, H, W, C)
# test_batch["obs"] is shape (B, T, C, H, W)
gt_seq = test_batch["obs"][sample_idx].cpu().numpy()
gt_seq = np.transpose(gt_seq, (0, 2, 3, 1))

# The predicted frame from the World Model is just the final step
# test_batch["wm_pred"] is shape (B, C, H, W)
pred_last_frame = test_batch["wm_pred"][sample_idx].cpu().numpy()
pred_last_frame = np.transpose(pred_last_frame, (1, 2, 0))

# Construct the predicted sequence by copying the context frames 
# and replacing the last frame with the prediction
pred_seq = gt_seq.copy()
pred_seq[-1] = pred_last_frame

# Un-normalize from [-0.5, 0.5] to [0.0, 1.0] for plotting
gt_video = np.clip(gt_seq + 0.5, 0.0, 1.0)
pred_video = np.clip(pred_seq + 0.5, 0.0, 1.0)

print(f"GT Video Shape: {gt_video.shape} | Pred Video Shape: {pred_video.shape}")

# Call your custom plotting function!
plot_videos(
    video=pred_video, 
    ref_video=gt_video, 
    plot_ref=True, 
    forecast_start=1, # Bolds the title for the 3rd frame to indicate it's generated
    save_name="wandb/lapo_step_reconstruction.png", # Optional: saves the plot as a PNG
    show_borders=True,
    cmap='gray' if gt_video.shape[-1] == 1 else 'coolwarm' # Handles MovingMNIST vs Procgen
)

#%% Cell 3: Latent Action / Codebook Usage
import seaborn as sns

# test_batch["la_qinds"] contains the indices of the codebook vectors chosen by the IDM
# Shape: (num_codebooks, Batch_Size, num_discrete_latents, 1)
q_inds = test_batch["la_qinds"].cpu().numpy()

num_codebooks = q_inds.shape[0]
fig, axes = plt.subplots(1, num_codebooks, figsize=(6 * num_codebooks, 4))

# If there is only 1 codebook, wrap axes in a list for consistent indexing
if num_codebooks == 1:
    axes = [axes]

for i in range(num_codebooks):
    # Flatten the indices used for this specific codebook
    inds_flat = q_inds[i].flatten()
    
    sns.histplot(inds_flat, bins=cfg.model.vq.num_embs, ax=axes[i], discrete=True, color="skyblue")
    axes[i].set_title(f"Codebook {i} Usage Frequency")
    axes[i].set_xlabel("Latent Action Index")
    axes[i].set_ylabel("Count")
    axes[i].set_xlim(0, cfg.model.vq.num_embs - 1)

plt.tight_layout()
plt.show()


#%% Cell 4: PCA of Latent Action Embeddings
from sklearn.decomposition import PCA

# Extract the actual learned vectors from the EMA quantizer
# Shape: (num_codebooks, num_embs, emb_dim)
embeddings = idm.vq.embedding.detach().cpu().numpy()

print("Found embeddings with shape:", embeddings.shape)  # Should be (num_codebooks, num_embs, emb_dim)

num_codebooks = embeddings.shape[0]
fig, axes = plt.subplots(1, num_codebooks, figsize=(6 * num_codebooks, 5))

if num_codebooks == 1:
    axes = [axes]

for i in range(num_codebooks):
    emb_matrix = embeddings[i] # Shape: (num_embs, emb_dim)
    
    # Use PCA to reduce the emb_dim down to 2 dimensions for plotting
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb_matrix)
    
    axes[i].scatter(emb_2d[:, 0], emb_2d[:, 1], c=np.arange(len(emb_2d)), cmap='viridis', s=50)
    
    # Annotate the points with their codebook index
    for j, (x, y) in enumerate(emb_2d):
        axes[i].text(x + 0.05, y + 0.05, str(j), fontsize=8, alpha=0.7)
        
    axes[i].set_title(f"Codebook {i} Embeddings (PCA)")
    axes[i].set_xlabel("Principal Component 1")
    axes[i].set_ylabel("Principal Component 2")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#%% Cell 5: Training Curves (Fetched from W&B)
import wandb
import pandas as pd

# We use the wandb API to fetch the run data we just logged
api = wandb.Api()

# Construct the run path: entity/project/run_id
# Note: You may need to replace 'your_wandb_username' with your actual W&B username or team name
entity = wandb.run.entity if wandb.run else "your_wandb_username" 
project = "lapo_stage1"
run_id = run.id if run else "your_run_id"

try:
    wandb_run = api.run(f"{entity}/{project}/{run_id}")
    history = wandb_run.history()
    
    # Plotting WM Loss and VQ Loss side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.lineplot(data=history, x="global_step", y="wm_loss", ax=axes[0], color="orange")
    axes[0].set_title("World Model (FDM) Reconstruction Loss")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_yscale("log")
    
    sns.lineplot(data=history, x="global_step", y="vq_loss", ax=axes[1], color="purple")
    axes[1].set_title("Vector Quantization (Commitment) Loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Could not fetch data from W&B. Are you logged in? Error:", e)
    print("Alternatively, you can just view the interactive charts at:", run.url if run else "W&B Dashboard")





#%% Cell: 20-Frame Autoregressive Rollout Test

# ---------------------------------------------------------
# 1. Simplified 20-Frame DataLoader
# ---------------------------------------------------------
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

DATA_PATH = "/home/gb21553/Projects/Video-WARP/data/MovingMNIST/mnist_test_seq.npy" 
rollout_stager = MovingMNIST_Rollout_Stager(DATA_PATH)

# ---------------------------------------------------------
# 2. Autoregressive Generation Loop
# ---------------------------------------------------------
idm.eval()
wm.eval()

# Get a full 20-frame Ground Truth sequence (Shape: 1, 20, 1, 64, 64)
gt_seq_tensor = rollout_stager.get_random_sequence()

# Initialize our predicted video buffer with the first two Ground Truth frames (Context)
pred_frames = [gt_seq_tensor[:, 0], gt_seq_tensor[:, 1]] # List of (1, 1, 64, 64)

print("Running autoregressive generation...")
with torch.no_grad():
    # Loop from t=1 to t=18 (to predict frames t=2 to t=19)
    for t in range(1, 19):
        # A. Use IDM to extract the True Latent Action for this specific timestep
        gt_context = gt_seq_tensor[:, t-1 : t+2] # 3-frame chunk
        action_td, _, _ = idm(gt_context)
        la_q = action_td["la_q"] # The quantized latent action
        
        # B. Predict the NEXT frame using the World Model
        # CRITICAL: We feed the WM our *own past predictions*, not the ground truth!
        wm_in = torch.stack(pred_frames[-2:], dim=1) # The last 2 predicted frames
        pred_next = wm(wm_in, la_q) # Generates frame t+1
        
        # C. Append the newly generated frame to our buffer
        pred_frames.append(pred_next)

# ---------------------------------------------------------
# 3. Format and Visualize
# ---------------------------------------------------------
# Stack the list of predicted frames into a single tensor (1, 20, 1, 64, 64)
pred_seq_tensor = torch.stack(pred_frames, dim=1)

# Move to CPU, change to (Time, Height, Width, Channels), and un-normalize
gt_video = gt_seq_tensor[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5
pred_video = pred_seq_tensor[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5

# Clip strictly to [0.0, 1.0] to prevent matplotlib warnings
gt_video = np.clip(gt_video, 0.0, 1.0)
pred_video = np.clip(pred_video, 0.0, 1.0)

print(f"Generated Video Shape: {pred_video.shape}")

# Plot using your custom function!
plot_videos(
    video=pred_video, 
    ref_video=gt_video, 
    plot_ref=True, 
    forecast_start=3, # Bolds the title at frame 3 to show where true generation begins
    save_name="wandb/lapo_autoregressive_rollout.png", 
    show_borders=True,
    cmap='gray'
)