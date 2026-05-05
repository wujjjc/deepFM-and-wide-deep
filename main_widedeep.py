import os
import subprocess

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score

from readdata import *
from test import extract_best_auc
from net import *


# ============================================================================
#                            DDP 初始化 / 清理
# ============================================================================

def init_distributed():
    if 'LOCAL_RANK' not in os.environ:
        return get_device_single(), 1, 0, False

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device(f'cuda:{local_rank}')

    if dist.get_rank() == 0:
        gpus = scan_gpus()
        if gpus:
            free_map = {gid: mem for gid, mem in gpus}
            busy_ranks = [
                i for i in range(world_size)
                if free_map.get(i, 0) < 10000
            ]
            if busy_ranks:
                print(f"  !!! 警告: 以下 GPU 显存占用较高，可能导致 OOM 或训练变慢:")
                for i in busy_ranks:
                    print(f"      GPU {i}: 空闲 {free_map.get(i, 0)} MB")
                suggest_free_gpus()
            else:
                print(f"[DDP] 启动 {world_size} 卡训练，设备: {device}")

    return device, world_size, local_rank, True


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
#                         GPU 状态扫描 & 空闲选择
# ============================================================================

def scan_gpus():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free',
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            gpu_id = int(parts[0].strip())
            free_mem = int(parts[1].strip())
            gpus.append((gpu_id, free_mem))
        return gpus
    except Exception:
        return []


def suggest_free_gpus(min_free_mb=10000):
    gpus = scan_gpus()
    if not gpus:
        return ''

    free_ids = [gid for gid, mem in gpus if mem >= min_free_mb]

    print("GPU 状态扫描:")
    for gid, mem in gpus:
        status = "空闲" if mem >= min_free_mb else "占用"
        print(f"  GPU {gid}: 空闲 {mem:>6d} MB  [{status}]")

    if free_ids:
        visible = ','.join(str(i) for i in free_ids)
        print(f"\n  推荐 DDP 启动命令（仅使用空闲 GPU {free_ids}）:")
        print(f"  CUDA_VISIBLE_DEVICES={visible} torchrun --nproc_per_node={len(free_ids)} main_widedeep.py")
        return visible
    else:
        print("  !!! 警告: 没有空闲 GPU，训练可能失败或极慢")
        return ''


# ============================================================================
#                            单卡设备选择
# ============================================================================

def get_device_single():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    gpus = scan_gpus()
    if gpus:
        best_gpu, _ = max(gpus, key=lambda x: x[1])
        print(f"GPU 空闲显存 (MB): {dict(gpus)}, 选择 cuda:{best_gpu}")
        return torch.device(f'cuda:{best_gpu}')
    return torch.device('cuda')


# ============================================================================
#                          辅助：判断是否为主进程
# ============================================================================

def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# ============================================================================
#                              数据集与 DataLoader
# ============================================================================

def build_dataloaders(train_ds, val_ds, batch_size, world_size, rank):
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            collate_fn=collect_fn,
            num_workers=4,
            pin_memory=True,
        )

        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collect_fn,
        )
        return train_loader, val_loader, train_sampler

    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collect_fn,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collect_fn,
        )
        return train_loader, val_loader, None


# ============================================================================
#                               训练主循环
# ============================================================================

def train(model, device, train_loader, val_loader, train_sampler,
          is_ddp, best_auc, config):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    no_improve_count = 0

    for e in range(config['num_epochs']):
        # ======================== 训练 ========================
        model.train()

        if train_sampler is not None:
            train_sampler.set_epoch(e)

        total_loss = 0
        for batch in train_loader:
            cat = batch['cat'].to(device, non_blocking=True)
            num = batch['num'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)
            x_val = batch['x_val'].to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(x_val, num, cat)
            loss = criterion(output.squeeze(1), label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if is_ddp:
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / dist.get_world_size()

        if is_main_process():
            lr_now = optimizer.param_groups[0]['lr']
            print(f'Epoch {e+1:3d}/{config["num_epochs"]} | '
                  f'Loss: {avg_loss:.4f} | LR: {lr_now:.2e}')
            with open('loss_widedeep.txt', 'a') as f:
                f.write(f'Epoch {e+1}/{config["num_epochs"]}, '
                        f'Loss: {avg_loss:.4f}\n')

        # ======================== 验证 ========================
        if (e + 1) % config['eval_every'] == 0:
            model.eval()
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for batch in val_loader:
                    cat = batch['cat'].to(device, non_blocking=True)
                    num = batch['num'].to(device, non_blocking=True)
                    label = batch['label'].to(device, non_blocking=True)
                    x_val = batch['x_val'].to(device, non_blocking=True)
                    output = model(x_val, num, cat, 'eval')
                    all_labels.extend(label.cpu().numpy())
                    all_preds.extend(output.squeeze(1).cpu().numpy())

            if is_ddp:
                world_size = dist.get_world_size()
                local_labels = torch.tensor(all_labels, device=device)
                local_preds = torch.tensor(all_preds, device=device)

                local_size = torch.tensor([len(all_labels)], device=device)
                sizes = [torch.zeros(1, dtype=torch.long, device=device)
                         for _ in range(world_size)]
                dist.all_gather(sizes, local_size)
                max_size = max(int(s.item()) for s in sizes)

                pad_labels = torch.zeros(max_size, device=device)
                pad_preds = torch.zeros(max_size, device=device)
                pad_labels[:len(all_labels)] = local_labels
                pad_preds[:len(all_preds)] = local_preds

                gathered_labels = [torch.zeros(max_size, device=device)
                                   for _ in range(world_size)]
                gathered_preds = [torch.zeros(max_size, device=device)
                                  for _ in range(world_size)]
                dist.all_gather(gathered_labels, pad_labels)
                dist.all_gather(gathered_preds, pad_preds)

                if dist.get_rank() == 0:
                    all_labels = torch.cat(
                        [g[:int(s)] for g, s in zip(gathered_labels, sizes)]
                    ).cpu().tolist()
                    all_preds = torch.cat(
                        [g[:int(s)] for g, s in zip(gathered_preds, sizes)]
                    ).cpu().tolist()

            if is_main_process():
                auc_score = roc_auc_score(all_labels, all_preds)
                scheduler.step(auc_score)

                if auc_score > best_auc:
                    best_auc = auc_score
                    no_improve_count = 0

                    state = model.module.state_dict() if is_ddp \
                            else model.state_dict()
                    torch.save(state, 'best_widedeep.pth')

                    with open('best_auc_widedeep.txt', 'w') as f:
                        f.write(f'Epoch {e+1}/{config["num_epochs"]}, '
                                f'Best AUC: {best_auc:.4f}\n')
                    print(f'  >>> New best AUC: {best_auc:.4f}')
                else:
                    no_improve_count += 1

                with open('auc_widedeep.txt', 'a') as f:
                    f.write(f'Epoch {e+1}/{config["num_epochs"]}, '
                            f'AUC: {auc_score:.4f}  Best: {best_auc:.4f}\n')

                if no_improve_count >= config['early_stop_patience']:
                    print(f'Early stopping at epoch {e+1}, '
                          f'best AUC: {best_auc:.4f}')
                    return best_auc

    return best_auc


# ============================================================================
#                               主入口
# ============================================================================

def main():
    """
    启动方式:
      # 单卡（自动选最空闲 GPU）
      python main_widedeep.py

      # 多卡 DDP，N 为使用的 GPU 数量
      torchrun --nproc_per_node=N main_widedeep.py
    """
    # ===================== 超参数 =====================
    config = {
        'emb_dim': 32,
        'batch_size': 2048,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'num_epochs': 200,
        'eval_every': 5,
        'early_stop_patience': 10,
    }

    # ===================== DDP 初始化 =====================
    device, world_size, rank, is_ddp = init_distributed()

    # ===================== 读取数据 =====================
    path = os.path.join(os.getcwd(), 'data/dac/train.txt')
    if is_main_process():
        print(f"数据路径: {path}")

    train_cat_list, train_label_list, cat_vocab_sizes, val_cat_list, \
        val_label_list, train_num_list, val_num_list, \
        train_x_val, val_x_val = read_data(path)

    train_ds = deepFMDataset(train_cat_list, train_num_list,
                             train_label_list, train_x_val)
    val_ds = deepFMDataset(val_cat_list, val_num_list,
                           val_label_list, val_x_val)

    # ===================== 构建 DataLoader =====================
    train_loader, val_loader, train_sampler = build_dataloaders(
        train_ds, val_ds, config['batch_size'], world_size, rank,
    )

    # ===================== 构建 WideDeep 模型 =====================
    model = WideDeep(
        num_kinds=13,
        cat_kinds=cat_vocab_sizes,
        kinds=39,
        embedding_dim=config['emb_dim'],
        hidden_size=[512, 256, 128, 64, 32],
        dropout=0.2,
    ).to(device)

    # 加载已有 checkpoint
    best_auc = 0.0
    if os.path.exists('best_auc_widedeep.txt'):
        best_auc = extract_best_auc('best_auc_widedeep.txt')
    if os.path.exists('best_widedeep.pth'):
        if is_main_process():
            print("加载已有模型参数...")
        model.load_state_dict(
            torch.load('best_widedeep.pth', map_location=device)
        )

    # ===================== DDP 包装模型 =====================
    if is_ddp:
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
        )

    # ===================== 开始训练 =====================
    best_auc = train(
        model, device, train_loader, val_loader, train_sampler,
        is_ddp, best_auc, config,
    )

    # ===================== 收尾 =====================
    if is_main_process():
        print(f'Training finished. Best AUC: {best_auc:.4f}')

    if is_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
