import torch
import torch.nn as nn
import os
import contextlib

from PU.PGW_Net.models.SSIM_method import SSIM


# -------------------------- Basic Utility Functions --------------------------
def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def calculate_psnr(pred: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    if pred.dtype not in (torch.float32, torch.float16):
        pred = pred.float()
    if gt.dtype not in (torch.float32, torch.float16):
        gt = gt.float()

    assert pred.shape == gt.shape, f"PSNR Calculation: Shape mismatch! pred={pred.shape}, gt={gt.shape}"
    pred = torch.clamp(pred, 0.0 - 1e-5, data_range + 1e-5)
    gt = torch.clamp(gt, 0.0 - 1e-5, data_range + 1e-5)

    mse = torch.mean((pred - gt) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return torch.mean(psnr)


def clip_grads(model, max_norm: float = 5.0):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def freeze_batch_norm_layers(model, freeze_affine: bool = True):
    bn_classes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    frozen_count = 0

    for module in model.modules():
        if isinstance(module, bn_classes):
            module.eval()
            if freeze_affine:
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.requires_grad = False
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.requires_grad = False
            frozen_count += 1

    print(f"❄️ Frozen {frozen_count} BatchNorm layers (affine params {'frozen' if freeze_affine else 'reserved'})")


def warmup_lr_scheduler(optimizer, epoch: int, warmup_epochs: int, init_lr: float):
    if epoch < warmup_epochs and warmup_epochs > 0:
        lr = 1e-6 + (init_lr - 1e-6) * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    return None


def load_unet_model(
        checkpoint_path: str,
        model_class,
        in_channels: int = 3,
        num_classes: int = 3,
        device: torch.device = None
) -> tuple[nn.Module, dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Loading model on {device.type}: {os.path.basename(checkpoint_path)}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}") from e

    state_dict = checkpoint['model_state_dict']
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            cleaned_key = key[len("_orig_mod."):]
            print(f"⚠️ Compiled model prefix detected, auto converted: {key} → {cleaned_key}")
        elif key.startswith("module."):
            cleaned_key = key[len("module."):]
            print(f"⚠️ Multi-GPU training prefix detected, auto converted: {key} → {cleaned_key}")
        else:
            cleaned_key = key
        cleaned_state_dict[cleaned_key] = value

    try:
        model = model_class(
            in_channels=in_channels,
            num_classes=num_classes,
        ).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to build U_net model: {str(e)}") from e

    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
        print("✅ Model parameters strictly matched, loaded successfully!")
    except RuntimeError as e:
        print(f"❌ Strict mode loading failed: {str(e)}")
        print("⚠️ Attempting non-strict mode loading (verify inference results)...")
        model.load_state_dict(cleaned_state_dict, strict=False)
        print("✅ Non-strict mode loading completed")

    model.eval()
    print(f"🔚 Model loading completed, current mode: eval (Device: {device.type})")
    return model, checkpoint


# -------------------------- Core Training Function --------------------------
def train_model(
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        criterion,
        optimizer,
        scheduler,
        scaler,
        writer,
        denormalize_fn,
        total_epochs: int = 800,
        warmup_epochs: int = 50,
        frozen_bn_epochs: int = 200,
        init_lr: float = 1e-4,
        in_channels: int = 3,
        use_normalize: bool = True,
        force_init_lr: bool = False,
        device: torch.device = torch.device("cuda"),
        save_dir: str = "PWU_32_weights",
        base_model_name: str = "unet_model",
        train_config: dict = None,
        grad_clip_max_norm: float = 5.0,
        loss_clip_value: float = 1e4,
        print_per_epoch: bool = True,
        use_test: bool = True,
        use_freeze_bn_affine: bool = True,
):
    train_config = train_config or {}
    ACCUMULATE_GRAD = train_config.get("ACCUMULATE_GRAD", 8)
    MAX_VAL_BATCHES = train_config.get("MAX_VAL_BATCHES", 10)
    MAX_TEST_BATCHES = train_config.get("MAX_TEST_BATCHES", 10)
    SAVE_INTERVAL = train_config.get("SAVE_INTERVAL", 50)
    MIXED_PREC = train_config.get("MIXED_PRECISION", False)

    best_val_loss = float('inf')
    best_val_avg_ssim = -1.0
    best_val_avg_psnr = -1.0
    global_best_sample_ssim = 0.0
    best_val_sample_psnr = -1.0
    global_best_sample_epoch = 0
    best_sample_psnr_info = {"haze": None, "gt": None, "output": None, "epoch": 0, "psnr": -1.0}
    current_lr = get_current_lr(optimizer)

    ssim_calculator = SSIM(window_size=11, size_average=True)
    ssim_per_sample = SSIM(window_size=11, size_average=False)

    total_batches = len(train_dataloader)
    total_updates_per_epoch = (total_batches + ACCUMULATE_GRAD - 1) // ACCUMULATE_GRAD
    batch_size = train_dataloader.batch_size
    effective_batch_size = batch_size * ACCUMULATE_GRAD
    is_compiled = hasattr(model, '_orig_mod')
    frozen_bn_start_epoch = total_epochs - frozen_bn_epochs
    print(f"\n📅 Training Phase Division:")
    print(f"  - Warmup Phase: 0~{warmup_epochs - 1} epoch (Total {warmup_epochs} epochs)")
    print(f"  - Normal Training: {warmup_epochs}~{frozen_bn_start_epoch - 1} epoch")
    print(f"  - FrozenBN Phase: {frozen_bn_start_epoch}~{total_epochs - 1} epoch (Total {frozen_bn_epochs} epochs)")
    if force_init_lr:
        print(f"🔒 Learning Rate Info: Initial LR={init_lr:.8f} (Forced Initialization), dynamically adjusted by scheduler")
    else:
        print(f"🔒 Learning Rate Info: Initial LR={current_lr:.8f}, dynamically adjusted by scheduler")

    # -------------------------- Main Training Loop --------------------------
    for epoch in range(total_epochs):
        current_epoch = epoch + 1
        epoch_header = f"\n=== Epoch {current_epoch:04d}/{total_epochs:04d} ==="
        print(epoch_header + "=" * (50 - len(epoch_header)))

        if hasattr(criterion, 'trigger_epoch_start_print'):
            criterion.trigger_epoch_start_print()

        warmup_lr = warmup_lr_scheduler(optimizer, epoch, warmup_epochs, init_lr)
        if warmup_lr is not None:
            current_lr = warmup_lr
            print(f"🔥 Warmup Phase: Current LR={current_lr:.8f} ({current_epoch}/{warmup_epochs})")
        else:
            current_lr = get_current_lr(optimizer)

        is_frozen_bn_phase = current_epoch >= frozen_bn_start_epoch
        if is_frozen_bn_phase:
            freeze_batch_norm_layers(model, freeze_affine=use_freeze_bn_affine)
        else:
            model.train()

        # -------------------------- Training Phase --------------------------
        model.train()
        total_train_loss = 0.0
        total_train_ssim = 0.0
        total_train_psnr = 0.0
        update_step = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (haze_imgs, gt_imgs) in enumerate(train_dataloader):
            haze_imgs = haze_imgs.to(device, non_blocking=True)
            gt_imgs = gt_imgs.to(device, non_blocking=True)
            current_batch_size = haze_imgs.size(0)

            with (torch.autocast(device_type=device.type, dtype=torch.float16)
                  if MIXED_PREC else contextlib.nullcontext()):
                outputs = model(haze_imgs)
                if use_normalize:
                    outputs = torch.clamp(outputs, -1.0, 1.0)
                    gt_imgs_clamped = torch.clamp(gt_imgs, -1.0, 1.0)
                else:
                    outputs = torch.clamp(outputs, 0.0, 1.0)
                    gt_imgs_clamped = torch.clamp(gt_imgs, 0.0, 1.0)
                loss = criterion(outputs.float(), gt_imgs_clamped.float()) / ACCUMULATE_GRAD
                if loss_clip_value > 0:
                    loss = torch.clamp(loss, -loss_clip_value, loss_clip_value)

            outputs_01 = denormalize_fn(outputs.float())
            gt_01 = denormalize_fn(gt_imgs_clamped.float())
            outputs_01 = torch.clamp(outputs_01, 0.0, 1.0)
            gt_01 = torch.clamp(gt_01, 0.0, 1.0)
            ssim_batch = ssim_calculator(outputs_01, gt_01)
            psnr_batch = calculate_psnr(outputs_01, gt_01)

            if MIXED_PREC:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_train_loss += (loss.item() * ACCUMULATE_GRAD) * current_batch_size
            total_train_ssim += ssim_batch.item() * current_batch_size
            total_train_psnr += psnr_batch.item() * current_batch_size

            if (batch_idx + 1) % ACCUMULATE_GRAD == 0 or (batch_idx + 1) == total_batches:
                grad_norm = 0.0
                if MIXED_PREC:
                    if grad_clip_max_norm > 0:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grads(model, grad_clip_max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip_max_norm > 0:
                        grad_norm = clip_grads(model, grad_clip_max_norm)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                torch.cuda.empty_cache()

                global_step = (epoch) * total_updates_per_epoch + update_step
                writer.add_scalar("Train/Update_Loss", loss.item() * ACCUMULATE_GRAD, global_step)
                writer.add_scalar("Train/Update_SSIM", ssim_batch.item(), global_step)
                writer.add_scalar("Train/Update_PSNR", psnr_batch.item(), global_step)
                writer.add_scalar("Train/Learning_Rate", current_lr, global_step)
                if grad_norm > 0:
                    writer.add_scalar("Train/Update_Grad_Norm", grad_norm, global_step)

                if update_step % 20 == 0:
                    print(f"  🚀 Update {update_step:04d}/{total_updates_per_epoch:04d}：")
                    print(f"     - Loss={loss.item() * ACCUMULATE_GRAD:.6f} | SSIM={ssim_batch.item():.6f} | PSNR={psnr_batch.item():.2f}dB")
                    print(f"     - LR={current_lr:.8f} | Grad Norm={grad_norm:.4f}")
                    print(f"     - Output Range：min={outputs.min():.4f}, max={outputs.max():.4f}")

        train_dataset_size = len(train_dataloader.dataset)
        avg_train_loss = total_train_loss / train_dataset_size
        avg_train_ssim = total_train_ssim / train_dataset_size
        avg_train_psnr = total_train_psnr / train_dataset_size

        writer.add_scalar("Train/Epoch_Avg_Loss", avg_train_loss, current_epoch)
        writer.add_scalar("Train/Epoch_Avg_SSIM", avg_train_ssim, current_epoch)
        writer.add_scalar("Train/Epoch_Avg_PSNR", avg_train_psnr, current_epoch)

        if print_per_epoch:
            print(f"📊 Train Set Summary：")
            print(f"  - Avg Loss={avg_train_loss:.6f} | Avg SSIM={avg_train_ssim:.6f} | Avg PSNR={avg_train_psnr:.2f}dB")
            print(f"  - LR={current_lr:.8f}")

        # -------------------------- Validation Phase --------------------------
        model.eval()
        total_val_loss = 0.0
        total_val_ssim = 0.0
        total_val_psnr = 0.0
        val_total_samples = 0
        epoch_best_sample_ssim = 0.0
        epoch_best_sample_psnr = -1.0
        epoch_best_sample_info = {
            "ssim_haze": None, "ssim_gt": None, "ssim_output": None,
            "psnr_haze": None, "psnr_gt": None, "psnr_output": None
        }
        val_vis_haze = None
        val_vis_gt = None
        val_vis_output = None

        with torch.no_grad():
            for batch_idx, (haze_imgs, gt_imgs) in enumerate(val_dataloader):
                if MAX_VAL_BATCHES is not None and batch_idx >= MAX_VAL_BATCHES:
                    break

                haze_imgs = haze_imgs.to(device, non_blocking=True)
                gt_imgs = gt_imgs.to(device, non_blocking=True)
                current_batch_size = haze_imgs.size(0)
                val_total_samples += current_batch_size

                with (torch.autocast(device_type=device.type, dtype=torch.float16)
                      if MIXED_PREC else torch.no_grad()):
                    outputs = model(haze_imgs)
                    if use_normalize:
                        outputs = torch.clamp(outputs, -1.0, 1.0)
                        gt_imgs_clamped = torch.clamp(gt_imgs, -1.0, 1.0)
                    else:
                        outputs = torch.clamp(outputs, 0.0, 1.0)
                        gt_imgs_clamped = torch.clamp(gt_imgs, 0.0, 1.0)
                    loss = criterion(outputs.float(), gt_imgs_clamped.float())

                outputs_01 = denormalize_fn(outputs.float())
                gt_01 = denormalize_fn(gt_imgs_clamped.float())
                outputs_01 = torch.clamp(outputs_01, 0.0, 1.0)
                gt_01 = torch.clamp(gt_01, 0.0, 1.0)
                ssim_sample = ssim_per_sample(outputs_01, gt_01)
                psnr_sample = calculate_psnr(outputs_01, gt_01)
                avg_batch_ssim = ssim_sample.mean().item()
                avg_batch_psnr = psnr_sample.mean().item()

                if ssim_sample.max().item() > epoch_best_sample_ssim:
                    epoch_best_sample_ssim = ssim_sample.max().item()
                    best_ssim_idx = ssim_sample.argmax().item()
                    epoch_best_sample_info["ssim_haze"] = haze_imgs[best_ssim_idx].detach().clone()
                    epoch_best_sample_info["ssim_gt"] = gt_imgs_clamped[best_ssim_idx].detach().clone()
                    epoch_best_sample_info["ssim_output"] = outputs[best_ssim_idx].detach().clone()

                if psnr_sample.max().item() > epoch_best_sample_psnr:
                    epoch_best_sample_psnr = psnr_sample.max().item()
                    best_psnr_idx = psnr_sample.argmax().item()
                    epoch_best_sample_info["psnr_haze"] = haze_imgs[best_psnr_idx].detach().clone()
                    epoch_best_sample_info["psnr_gt"] = gt_imgs_clamped[best_psnr_idx].detach().clone()
                    epoch_best_sample_info["psnr_output"] = outputs[best_psnr_idx].detach().clone()

                total_val_loss += loss.item() * current_batch_size
                total_val_ssim += avg_batch_ssim * current_batch_size
                total_val_psnr += avg_batch_psnr * current_batch_size

                if batch_idx == 0:
                    val_vis_haze = haze_imgs.detach().clone()
                    val_vis_gt = gt_imgs_clamped.detach().clone()
                    val_vis_output = outputs.detach().clone()

                del haze_imgs, gt_imgs, outputs
                torch.cuda.empty_cache()

        if val_total_samples == 0:
            print(f"⚠️ No valid samples in validation set!")
            avg_val_loss = float('inf')
            avg_val_ssim = -1.0
            avg_val_psnr = -1.0
        else:
            avg_val_loss = total_val_loss / val_total_samples
            avg_val_ssim = total_val_ssim / val_total_samples
            avg_val_psnr = total_val_psnr / val_total_samples

            writer.add_scalar("Val/Epoch_Avg_Loss", avg_val_loss, current_epoch)
            writer.add_scalar("Val/Epoch_Avg_SSIM", avg_val_ssim, current_epoch)
            writer.add_scalar("Val/Epoch_Avg_PSNR", avg_val_psnr, current_epoch)
            writer.add_scalar("Val/Epoch_Best_Sample_SSIM", epoch_best_sample_ssim, current_epoch)
            writer.add_scalar("Val/Epoch_Best_Sample_PSNR", epoch_best_sample_psnr, current_epoch)

            print(f"📊 Validation Set Summary：")
            print(f"  - Avg Loss={avg_val_loss:.6f} | Avg SSIM={avg_val_ssim:.6f} | Avg PSNR={avg_val_psnr:.2f}dB")
            print(f"  - Best Sample SSIM={epoch_best_sample_ssim:.6f} | Best Sample PSNR={epoch_best_sample_psnr:.2f}dB")

        # -------------------------- Model Saving & Visualization --------------------------
        if val_total_samples > 0:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_loss_path = os.path.join(save_dir, f"{base_model_name}_best_loss.pth")
                torch.save({
                    'model_state_dict': model._orig_mod.state_dict() if is_compiled else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': current_epoch,
                    'best_loss': best_val_loss,
                    'current_lr': current_lr,
                    'use_normalize': use_normalize,
                    'is_frozen_bn_phase': is_frozen_bn_phase,
                    'train_config': train_config
                }, best_loss_path)
                print(f"🏆 Best Loss Model Updated: {best_val_loss:.6f} (Epoch {current_epoch})")

            if avg_val_ssim > best_val_avg_ssim:
                best_val_avg_ssim = avg_val_ssim
                best_ssim_path = os.path.join(save_dir, f"{base_model_name}_best_avg_ssim.pth")
                torch.save({
                    'model_state_dict': model._orig_mod.state_dict() if is_compiled else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': current_epoch,
                    'best_avg_ssim': best_val_avg_ssim,
                    'current_lr': current_lr,
                    'use_normalize': use_normalize,
                    'is_frozen_bn_phase': is_frozen_bn_phase,
                    'train_config': train_config
                }, best_ssim_path)
                print(f"🏆 Best Avg SSIM Model Updated: {best_val_avg_ssim:.6f} (Epoch {current_epoch})")

            if avg_val_psnr > best_val_avg_psnr:
                best_val_avg_psnr = avg_val_psnr
                best_psnr_path = os.path.join(save_dir, f"{base_model_name}_best_avg_psnr.pth")
                torch.save({
                    'model_state_dict': model._orig_mod.state_dict() if is_compiled else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': current_epoch,
                    'best_avg_psnr': best_val_avg_psnr,
                    'current_lr': current_lr,
                    'use_normalize': use_normalize,
                    'is_frozen_bn_phase': is_frozen_bn_phase,
                    'train_config': train_config
                }, best_psnr_path)
                print(f"🏆 Best Avg PSNR Model Updated: {best_val_avg_psnr:.2f}dB (Epoch {current_epoch})")

            if epoch_best_sample_ssim > global_best_sample_ssim:
                global_best_sample_ssim = epoch_best_sample_ssim
                global_best_sample_epoch = current_epoch
                print(f"🏆 Global Best Single Sample SSIM Updated: {global_best_sample_ssim:.6f} (Epoch {current_epoch})")

            if epoch_best_sample_psnr > best_val_sample_psnr:
                best_val_sample_psnr = epoch_best_sample_psnr
                best_sample_psnr_info.update({
                    "haze": epoch_best_sample_info["psnr_haze"],
                    "gt": epoch_best_sample_info["psnr_gt"],
                    "output": epoch_best_sample_info["psnr_output"],
                    "epoch": current_epoch,
                    "psnr": best_val_sample_psnr
                })
                print(f"🏆 Global Best Single Sample PSNR Updated: {best_val_sample_psnr:.2f}dB (Epoch {current_epoch})")

        if current_epoch % 5 == 0 and val_vis_haze is not None:
            print(f"\n🖼️ Epoch {current_epoch} Visualization Processing...")
            val_vis_haze_uint8 = (denormalize_fn(val_vis_haze.float()) * 255).clamp(0, 255).to(torch.uint8)
            val_vis_gt_uint8 = (denormalize_fn(val_vis_gt.float()) * 255).clamp(0, 255).to(torch.uint8)
            val_vis_output_uint8 = (denormalize_fn(val_vis_output.float()) * 255).clamp(0, 255).to(torch.uint8)
            num_vis = min(4, val_vis_haze_uint8.size(0))

            writer.add_images(f"Val/Epoch_{current_epoch}/1_Haze", val_vis_haze_uint8[:num_vis], current_epoch, dataformats="NCHW")
            writer.add_images(f"Val/Epoch_{current_epoch}/2_GT", val_vis_gt_uint8[:num_vis], current_epoch, dataformats="NCHW")
            writer.add_images(f"Val/Epoch_{current_epoch}/3_Prediction", val_vis_output_uint8[:num_vis], current_epoch, dataformats="NCHW")
            print(f"✅ Visualization Completed")

        if use_test and test_dataloader is not None and current_epoch % 5 == 0:
            print(f"\n📝 Epoch {current_epoch} Test Set Evaluation...")
            model.eval()
            total_test_ssim = 0.0
            total_test_psnr = 0.0
            test_total_samples = 0

            with torch.no_grad():
                for batch_idx, (haze_imgs, gt_imgs) in enumerate(test_dataloader):
                    if MAX_TEST_BATCHES is not None and batch_idx >= MAX_TEST_BATCHES:
                        break

                    haze_imgs = haze_imgs.to(device, non_blocking=True)
                    gt_imgs = gt_imgs.to(device, non_blocking=True)
                    current_batch_size = haze_imgs.size(0)
                    test_total_samples += current_batch_size

                    with (torch.autocast(device_type=device.type, dtype=torch.float16)
                          if MIXED_PREC else torch.no_grad()):
                        outputs = model(haze_imgs)
                        if use_normalize:
                            outputs = torch.clamp(outputs, -1.0, 1.0)
                        else:
                            outputs = torch.clamp(outputs, 0.0, 1.0)

                    outputs_01 = denormalize_fn(outputs.float()).clamp(0.0, 1.0)
                    gt_01 = denormalize_fn(gt_imgs.float()).clamp(0.0, 1.0)
                    test_ssim = ssim_calculator(outputs_01, gt_01).item()
                    test_psnr = calculate_psnr(outputs_01, gt_01).item()

                    total_test_ssim += test_ssim * current_batch_size
                    total_test_psnr += test_psnr * current_batch_size
                    del haze_imgs, gt_imgs, outputs

            if test_total_samples > 0:
                avg_test_ssim = total_test_ssim / test_total_samples
                avg_test_psnr = total_test_psnr / test_total_samples
                writer.add_scalar("Test/Epoch_Avg_SSIM", avg_test_ssim, current_epoch)
                writer.add_scalar("Test/Epoch_Avg_PSNR", avg_test_psnr, current_epoch)
                print(f"📊 Test Set Summary: Avg SSIM={avg_test_ssim:.6f} | Avg PSNR={avg_test_psnr:.2f}dB")

        if current_epoch >= warmup_epochs and val_total_samples > 0:
            scheduler.step(avg_val_loss)
            current_lr = get_current_lr(optimizer)

        if current_epoch % SAVE_INTERVAL == 0 and current_epoch != 0 and val_total_samples > 0:
            middle_model_path = os.path.join(save_dir, f"{base_model_name}_epoch_{current_epoch:04d}.pth")
            torch.save({
                'model_state_dict': model._orig_mod.state_dict() if is_compiled else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': current_epoch,
                'current_lr': current_lr,
                'val_metrics': {
                    'avg_loss': avg_val_loss,
                    'avg_ssim': avg_val_ssim,
                    'avg_psnr': avg_val_psnr
                }
            }, middle_model_path)
            print(f"📌 Intermediate Model Saved: {middle_model_path}")

    # -------------------------- Training Completion --------------------------
    final_model_path = os.path.join(save_dir, f"{base_model_name}_final.pth")
    torch.save({
        'model_state_dict': model._orig_mod.state_dict() if is_compiled else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_epochs': total_epochs,
        'best_metrics': {
            'val_loss': best_val_loss,
            'val_avg_ssim': best_val_avg_ssim,
            'val_avg_psnr': best_val_avg_psnr,
            'best_sample_ssim': global_best_sample_ssim,
            'best_sample_psnr': best_val_sample_psnr
        }
    }, final_model_path)

    print(f"\n" + "=" * 80)
    print(f"🎉 Training Completed (Total Epochs: {total_epochs})")
    print(f"📊 Best Metrics Summary:")
    print(f"  - Best Avg Loss: {best_val_loss:.6f}")
    print(f"  - Best Avg SSIM: {best_val_avg_ssim:.6f}")
    print(f"  - Best Avg PSNR: {best_val_avg_psnr:.2f}dB")
    print(f"  - Best Single Sample SSIM: {global_best_sample_ssim:.6f} (Epoch {global_best_sample_epoch})")
    print(f"  - Best Single Sample PSNR: {best_val_sample_psnr:.2f}dB (Epoch {best_sample_psnr_info['epoch']})")
    print(f"💾 Final Model Path: {final_model_path}")
    print("=" * 80)

    writer.close()
    print(f"✅ Training Process Fully Completed!")