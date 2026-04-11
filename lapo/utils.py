import config
import data_loader
import doy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import IDM, Policy, WorldModel
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader


def obs_to_img(obs: Tensor) -> Tensor:
    return ((obs.permute(1, 2, 0) + 0.5) * 255).to(torch.uint8).numpy(force=True)


def create_decoder(in_dim, out_dim, device=config.DEVICE, hidden_sizes=(128, 128)):
    decoder = []
    in_size = h = in_dim
    for h in hidden_sizes:
        decoder.extend([nn.Linear(in_size, h), nn.ReLU()])
        in_size = h
    decoder.append(nn.Linear(h, out_dim))
    return nn.Sequential(*decoder).to(device)


def create_dynamics_models(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:

    obs_depth = 1       ## TODO: Change back to 3
    # print(model_cfg)

    idm_in_depth = obs_depth * (2 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])

    return idm, wm


def create_policy(
    model_cfg: config.ModelConfig,
    action_dim: int,
    # policy_in_depth: int = 3,     ##TODO: Change back to 3 when using RGB obs
    policy_in_depth: int = 1,
    state_dict: dict | None = None,
    strict_loading: bool = True,
):
    policy = Policy(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
    ).to(config.DEVICE)

    if state_dict is not None:
        policy.load_state_dict(state_dict, strict=strict_loading)

    return policy


def eval_latent_repr(labeled_data: data_loader.DataStager, idm: IDM):
    batch = labeled_data.td_unfolded[:131072]
    actions = idm.label_chunked(batch).select("ta", "la").to(config.DEVICE)
    return train_decoder(data=actions)


def train_decoder(
    data: TensorDict,  # tensordict with keys "la", "ta"
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(data["la"].shape[-1], TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)

    train_data, test_data = data[: len(data) // 2], data[len(data) // 2 :]

    dataloader = DataLoader(
        train_data,  # type: ignore
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    step = 0
    for i in range(epochs):
        for batch in dataloader:
            pred_ta = decoder(batch["la"])
            ta = batch["ta"][:, -2]
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            if step % 10 == 0:
                with torch.no_grad():
                    test_pred_ta = decoder(test_data["la"])
                    test_ta = test_data["ta"][:, -2]

                    logger(
                        step=i,
                        test_loss=F.cross_entropy(test_pred_ta, test_ta),
                        test_acc=(test_pred_ta.argmax(-1) == test_ta).float().mean(),
                    )
            step += 1

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=logger["test_acc"][-1],
        test_loss=logger["test_loss"][-1],
    )

    return decoder, metrics










from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFont, ImageDraw









class MovingMNIST_LAPO_Stager:
    def __init__(self, data_path, is_test=False, seq_len=3):
        print(f"Loading MovingMNIST {'Test' if is_test else 'Train'} data...")
        
        # 1. Load raw array: default shape is (Time, Sequences, H, W) -> (20, 10000, 64, 64)
        raw_data = np.load(data_path)
        
        # 2. Transpose to (Sequences, Time, H, W) -> (10000, 20, 64, 64)
        data = np.transpose(raw_data, (1, 0, 2, 3))
        
        # 3. Train/Test split (8000 / 2000)
        if is_test:
            data = data[8000:]
        else:
            data = data[:8000]
            
        B, T, H, W = data.shape
        
        # 4. LAPO requires sequences of exactly length 3 (t-1, t, t+1)
        # We slice the 20-frame videos into overlapping 3-frame chunks
        chunks = []
        for i in range(T - seq_len + 1):
            chunks.append(data[:, i : i + seq_len])
        
        # 5. Combine into massive array. Shape: (Num_Chunks, 3, 64, 64)
        self.data = np.concatenate(chunks, axis=0) 
        
        # 6. Add the Channel dimension required by CNNs (C=1 for grayscale)
        # Shape becomes: (Num_Chunks, 3, 1, 64, 64)
        self.data = np.expand_dims(self.data, axis=2)
        
    def get_iter(self, batch_size, device=config.DEVICE):
        dataset_size = len(self.data)
        indices = np.arange(dataset_size)
        
        while True:
            np.random.shuffle(indices)
            for i in range(0, dataset_size, batch_size):
                batch_idx = indices[i : i + batch_size]
                if len(batch_idx) < batch_size:
                    continue # Drop the last incomplete batch
                    
                # 7. Fetch data and normalize to [-0.5, 0.5] as expected by LAPO models
                batch_np = self.data[batch_idx].astype(np.float32) / 255.0 - 0.5
                
                # 8. LAPO expects the batch to be wrapped in a TensorDict
                yield TensorDict({
                    "obs": torch.from_numpy(batch_np).to(device)
                }, batch_size=[batch_size], device=device)







def plot_videos(video, ref_video=None, plot_ref=True, show_titles=True, show_labels=True, forecast_start=None, 
                vmin=None, vmax=None, save_name=None, 
                wspace=0.05, hspace=0.02, forecast_gap=0.2, 
                save_video=False, video_gap=5, show_borders=False, corner_radius=5,
                no_rescale=True, cmap='coolwarm', row_height="auto", gif_scale=4):
    """
    Plots a camera-ready rollout of ground truth and predicted video frames, 
    and saves a high-res, properly scaled GIF.
    """
    with plt.rc_context({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 18,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    }):
        
        nb_frames = video.shape[0]
        C = video.shape[-1]
        
        if plot_ref and ref_video is None:
            raise ValueError("ref_video must be provided if plot_ref is True.")

        rescale = False
        if plot_ref and ref_video[..., :C].min() < -0.5:
            rescale = True
            ref_video = (ref_video + 1.0) / 2.0
        elif not plot_ref and video.min() < -0.5:
            rescale = True
        
        if no_rescale:
            rescale = False

        nrows = 2 if plot_ref else 1
        has_gap = forecast_start is not None and 1 < forecast_start <= nb_frames
        ncols = nb_frames + 1 if has_gap else nb_frames
        
        width_ratios = [1.0] * ncols
        spacer_col = -1
        if has_gap:
            spacer_col = forecast_start - 1
            width_ratios[spacer_col] = forecast_gap

        # Base width calculation
        fig_width = (nb_frames + (forecast_gap if has_gap else 0.0)) * 1.5
        
        if row_height == "auto":
            H, W = video.shape[1:3]
            aspect = H / W
            calculated_row_height = aspect * 1.2 if show_titles else aspect * 1.5
            title_buffer = 0.5 if show_titles else 0.1
            fig_height = (nrows * calculated_row_height) + title_buffer
        else:
            fig_height = nrows * float(row_height)
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(nrows, ncols, wspace=wspace, hspace=hspace, width_ratios=width_ratios)
        axes = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axes[r, c] = fig.add_subplot(gs[r, c])

        # Establish Global Min/Max
        if vmin is None or vmax is None:
            if plot_ref:
                global_min = ref_video.min()
                global_max = ref_video.max()
            else:
                global_min = video.min()
                global_max = video.max()

            if vmin is None: vmin = global_min
            if vmax is None: vmax = global_max

        imshow_kwargs = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax}

        frame_idx = 0
        for c in range(ncols):
            if c == spacer_col:
                for r in range(nrows):
                    axes[r, c].axis('off')
                continue
                
            pred_frame = video[frame_idx]
            if rescale: pred_frame = (pred_frame + 1.0) / 2.0
            
            # Fix: Only clip if RGB. Let imshow handle scalar arrays natively.
            if pred_frame.shape[-1] in [3, 4]:
                pred_frame = np.clip(pred_frame, 0.0, 1.0)
            elif pred_frame.shape[-1] == 1:
                pred_frame = pred_frame[..., 0]

            if plot_ref:
                ref_idx = min(frame_idx, ref_video.shape[0] - 1)
                ref_frame = ref_video[ref_idx]
                if rescale: ref_frame = (ref_frame + 1.0) / 2.0
                
                if ref_frame.shape[-1] in [3, 4]:
                    ref_frame = np.clip(ref_frame, 0.0, 1.0)
                elif ref_frame.shape[-1] == 1:
                    ref_frame = ref_frame[..., 0]

            if plot_ref:
                im_ref  = axes[0, c].imshow(ref_frame,  **imshow_kwargs)
                im_pred = axes[1, c].imshow(pred_frame, **imshow_kwargs)
                target_axes = [(axes[0, c], im_ref, ref_frame), (axes[1, c], im_pred, pred_frame)]
            else:
                im_pred = axes[0, c].imshow(pred_frame, **imshow_kwargs)
                target_axes = [(axes[0, c], im_pred, pred_frame)]

            for ax, im_obj, frame_data in target_axes:
                ax.set_xticks([])
                ax.set_yticks([])
                
                h, w = frame_data.shape[:2]
                for spine in ax.spines.values():
                    spine.set_visible(False)
                    
                if show_borders:
                    rect = patches.FancyBboxPatch(
                        (-0.5, -0.5), w, h,
                        boxstyle=f"round,pad=0,rounding_size={corner_radius}", 
                        linewidth=1.2, edgecolor='black', facecolor='none',
                        transform=ax.transData
                    )
                    ax.add_patch(rect)
                    im_obj.set_clip_path(rect)

            if show_titles:
                top_ax = axes[0, c]
                title_str = f"$t={frame_idx + 1}$" if (frame_idx == 0 or (frame_idx + 1 == forecast_start)) else str(frame_idx + 1)
                font_weight = 'bold' if (has_gap and frame_idx + 1 == forecast_start) else 'normal'
                top_ax.set_title(title_str, pad=8, fontsize=18, fontweight=font_weight)

            frame_idx += 1

        if show_labels:
            if plot_ref:
                axes[0, 0].set_ylabel("GT",   rotation=0, labelpad=25, ha='right', va='center', fontsize=28, fontweight='bold')
            axes[-1, 0].set_ylabel("Pred", rotation=0, labelpad=25, ha='right', va='center', fontsize=28, fontweight='bold')

        if save_name:
            plt.savefig(save_name, dpi=100, bbox_inches='tight', facecolor='white', transparent=False)
        else:
            plt.draw()

        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            plt.show()
            
        plt.close(fig)

        # ---------------------------------------------------------
        # GIF Generation
        # ---------------------------------------------------------
        if save_video and save_name is not None:
            
            # Scale fonts up based on gif_scale
            try:
                font = ImageFont.truetype("arial.ttf", 14 * gif_scale)
            except IOError:
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14 * gif_scale)
                except IOError:
                    font = ImageFont.load_default()

            def process_pil_image(img_array, radius=corner_radius, apply_frame=show_borders):
                h, w = img_array.shape[:2]
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                
                # Resize the underlying image using NEAREST to maintain sharp grid pixels
                new_w, new_h = w * gif_scale, h * gif_scale
                img = img.resize((new_w, new_h), Image.NEAREST)
                
                if not apply_frame: return img
                
                scaled_radius = radius * gif_scale
                mask = Image.new("L", (new_w, new_h), 0)
                draw = ImageDraw.Draw(mask)
                draw.rounded_rectangle((0, 0, new_w, new_h), radius=scaled_radius, fill=255)
                
                rounded_img = Image.new("RGB", (new_w, new_h), "white")
                rounded_img.paste(img, (0, 0), mask=mask)
                
                draw_border = ImageDraw.Draw(rounded_img)
                draw_border.rounded_rectangle((0, 0, new_w-1, new_h-1), radius=scaled_radius, outline="black", width=max(1, gif_scale//2))
                return rounded_img

            def apply_cmap_to_frame(frame, v_min, v_max):
                # Fix: If the frame is already RGB/RGBA, do NOT apply a colormap
                if frame.ndim == 3 and frame.shape[-1] in [3, 4]:
                    # Clip to [0, 1] to be safe, then return the RGB channels
                    return np.clip(frame[..., :3], 0.0, 1.0)
                    
                # If it's a single channel, squeeze it for the colormap
                if frame.ndim == 3 and frame.shape[-1] == 1: 
                    frame = frame[..., 0]
                    
                norm = plt.Normalize(vmin=v_min, vmax=v_max)
                colormap = plt.get_cmap(cmap)
                return colormap(norm(frame))[..., :3]

            gif_frames = []
            scaled_gap = video_gap * gif_scale
            header_height = 20 * gif_scale

            for t in range(nb_frames):
                p_f = video[t]
                if rescale: p_f = (p_f + 1.0) / 2.0
                p_f = apply_cmap_to_frame(p_f, vmin, vmax)

                if plot_ref:
                    r_idx = min(t, ref_video.shape[0] - 1)
                    r_f = ref_video[r_idx]
                    if rescale: r_f = (r_f + 1.0) / 2.0
                    r_f = apply_cmap_to_frame(r_f, vmin, vmax)
                    
                    img_ref  = process_pil_image(r_f)
                    img_pred = process_pil_image(p_f)
                    
                    combined_w = img_ref.width + scaled_gap + img_pred.width
                    combined_h = max(img_ref.height, img_pred.height)
                    combined_frame = Image.new('RGB', (combined_w, combined_h), 'white')
                    combined_frame.paste(img_ref,  (0, 0))
                    combined_frame.paste(img_pred, (img_ref.width + scaled_gap, 0))
                else:
                    combined_frame = process_pil_image(p_f)

                final_img = Image.new('RGB', (combined_frame.width, combined_frame.height + header_height), color='white')
                final_img.paste(combined_frame, (0, header_height))
                
                draw = ImageDraw.Draw(final_img)
                
                if plot_ref:
                    gt_w = draw.textlength("GT", font=font) if hasattr(draw, 'textlength') else 20 * gif_scale
                    pred_w = draw.textlength("Pred", font=font) if hasattr(draw, 'textlength') else 30 * gif_scale
                    
                    draw.text(((img_ref.width - gt_w) // 2, 2 * gif_scale), "GT", font=font, fill="black")
                    draw.text((img_ref.width + scaled_gap + (img_pred.width - pred_w) // 2, 2 * gif_scale), "Pred", font=font, fill="black")
                else:
                    pred_w = draw.textlength("Pred", font=font) if hasattr(draw, 'textlength') else 30 * gif_scale
                    draw.text(((combined_frame.width - pred_w) // 2, 2 * gif_scale), "Pred", font=font, fill="black")
                
                gif_frames.append(final_img)
            
            gif_path = Path(save_name).with_suffix('.gif')
            gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=150, loop=0)
            print(f"Saved rollout animation to {gif_path}")

            try:
                from IPython.display import Image as IPyImage, display
                display(IPyImage(filename=str(gif_path)))
            except ImportError:
                pass