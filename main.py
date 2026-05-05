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
    """
    初始化 NCCL 分布式通信，返回 (device, world_size, local_rank, is_ddp)

    torchrun 会在启动进程时自动设置以下环境变量：
      LOCAL_RANK   — 当前进程在本节点的 GPU 编号（从 0 开始）
      RANK         — 全局进程编号（多机时跨节点唯一）
      WORLD_SIZE   — 总进程数（= 总 GPU 数，单机为 GPU 数，多机为所有机器 GPU 之和）
      MASTER_ADDR  — 主节点 IP（单机默认为 localhost）
      MASTER_PORT  — 主节点端口（单机默认为随机端口）

    Returns:
        device:      torch.device，当前进程绑定的 GPU
        world_size:  总进程数（GPU 总数），单卡时为 1
        local_rank:  当前进程的本地 GPU 编号，单卡时为 0
        is_ddp:      是否为 DDP 模式
    """
    if 'LOCAL_RANK' not in os.environ:
        # 非 DDP 模式（直接 python main.py 启动）：沿用 nvidia-smi 自动选最空闲 GPU
        return get_device_single(), 1, 0, False

    # ---- DDP 模式（由 torchrun 启动）----
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # 将当前进程显式绑定到 local_rank 号 GPU，后续 model.cuda() 和 tensor.cuda()
    # 都会默认使用这张卡，无需手动指定 device
    torch.cuda.set_device(local_rank)

    # 初始化 NCCL 通信组
    # backend='nccl'：专为 NVIDIA GPU 优化的集合通信后端，支持 AllReduce / AllGather 等
    # init_method 默认 'env://'：从环境变量 MASTER_ADDR / MASTER_PORT 自动读取
    dist.init_process_group(backend='nccl')

    device = torch.device(f'cuda:{local_rank}')

    # rank=0 扫描并打印各 GPU 空闲状态，警告是否跑在忙碌的卡上
    if dist.get_rank() == 0:
        gpus = scan_gpus()
        if gpus:
            free_map = {gid: mem for gid, mem in gpus}
            free_mem = free_map.get(local_rank * world_size + 0, 0)
            # 当前使用的 GPU（0 到 world_size-1）是否忙碌
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
    """销毁 NCCL 通信组，释放 GPU 间通信资源"""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
#                         GPU 状态扫描 & 空闲选择
# ============================================================================

def scan_gpus():
    """
    通过 nvidia-smi 获取所有 GPU 的空闲显存（MB），失败返回空列表。
    返回: [(gpu_id, free_mem_mb), ...]  按 GPU 编号升序排列
    """
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
    """
    扫描并打印 GPU 空闲情况，返回推荐的 CUDA_VISIBLE_DEVICES 字符串。

    单卡模式直接靠 get_device_single() 自动处理；
    这个函数主要给 DDP 模式提供参考——告知用户哪些卡空闲，
    建议在 torchrun 前设置 CUDA_VISIBLE_DEVICES。
    """
    gpus = scan_gpus()
    if not gpus:
        return ''

    free_ids = [gid for gid, mem in gpus if mem >= min_free_mb]

    # 打印全量 GPU 状态
    print("GPU 状态扫描:")
    for gid, mem in gpus:
        status = "空闲" if mem >= min_free_mb else "占用"
        print(f"  GPU {gid}: 空闲 {mem:>6d} MB  [{status}]")

    if free_ids:
        visible = ','.join(str(i) for i in free_ids)
        print(f"\n  推荐 DDP 启动命令（仅使用空闲 GPU {free_ids}）:")
        print(f"  CUDA_VISIBLE_DEVICES={visible} torchrun --nproc_per_node={len(free_ids)} main.py")
        return visible
    else:
        print("  !!! 警告: 没有空闲 GPU，训练可能失败或极慢")
        return ''


# ============================================================================
#                            单卡设备选择
# ============================================================================

def get_device_single():
    """
    单卡模式：通过 nvidia-smi 查询每块 GPU 的空闲显存，
    选择显存最空闲的 GPU；无 GPU 则回退到 CPU。
    """
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
    """
    DDP 模式下仅 rank=0 返回 True；
    非 DDP 模式（dist 未初始化）始终返回 True。
    用于控制「只让一个进程做打印 / 写文件 / 保存模型」。
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# ============================================================================
#                              数据集与 DataLoader
# ============================================================================

def build_dataloaders(train_ds, val_ds, batch_size, world_size, rank):
    """
    构建训练 & 验证 DataLoader。

    —— DDP 模式（world_size > 1）——
    训练集：
      DistributedSampler 将数据集按 rank 均分为 world_size 份，每张卡只看到
      1/world_size 的数据。不同 rank 看到的数据无重叠。
      每个 epoch 必须在训练开始前调用 sampler.set_epoch(epoch) 来随机打乱数据，
      否则各 epoch 的每张卡看到的 shuffle 顺序完全相同，失去了数据多样性的意义。
      DataLoader 的 shuffle 参数必须设为 False，否则会和 sampler 冲突。
    验证集：
      同样用 DistributedSampler 均分（shuffle=False），每卡分别推理后再通过
      all_gather 汇集到 rank=0 计算完整 AUC。

    —— 单卡模式（world_size == 1）——
    直接使用默认 DataLoader，训练集 shuffle=True 打乱。
    """
    if world_size > 1:
        # -------- DDP 训练 DataLoader --------
        # num_replicas=world_size: 总卡数
        # rank=rank:              当前卡编号（0 ~ world_size-1）
        # shuffle=True:           由 DistributedSampler 控制 shuffle
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,   # 用了 sampler 就必须 shuffle=False
            shuffle=False,
            collate_fn=collect_fn,
            num_workers=4,           # 子进程数，预加载数据到内存
            pin_memory=True,         # 锁页内存，加速 CPU→GPU 数据传输
        )

        # -------- DDP 验证 DataLoader --------
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,           # 验证集不打乱
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
        # -------- 单卡 DataLoader --------
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,            # 单卡直接 shuffle
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
    """
    统一的训练循环（DDP / 单卡通用）。

    参数:
        model:         DeepFM 模型（单卡模式）或 DDP 包装后的模型（多卡模式）
        device:        当前进程使用的 cuda device
        train_loader:  训练 DataLoader
        val_loader:    验证 DataLoader
        train_sampler: DistributedSampler 实例（单卡为 None）
        is_ddp:        是否为 DDP 模式
        best_auc:      从 best_auc.txt 恢复的最佳 AUC
        config:        超参数字典

    DDP 核心机制说明:
      ———————————————————————————————————————————————————————————————————
      1. sampler.set_epoch(epoch)
         每个 epoch 开始前必须调用。DistributedSampler 内部用 epoch 做随机种子，
         保证不同 epoch 各卡看到的数据 shuffle 顺序不同。如果漏掉这行，每个 epoch
         每张卡看到的数据顺序和上一轮完全一样，等价于数据只用了一遍就过拟合。

      2. all_reduce 聚合全局 loss
         每张卡计算的是自己那 1/world_size 数据的局部 loss，不能直接代表全局 loss。
         需要 all_reduce(SUM) 把各卡 loss 求和，再除以 world_size 得到全局均值。
         公式: global_loss = (loss_0 + loss_1 + ... + loss_{N-1}) / N

      3. model.module
         DDP 将原始模型包装在 model.module 里。所以：
         - 保存参数: model.module.state_dict() 而不是 model.state_dict()
         - 访问内部层: model.module.fm 而不是 model.fm
         如果保存了 DDP wrapper 的 state_dict，它带的 "module." 前缀会导致
         torch.load 报 key mismatch。

      4. 只在 rank=0 执行 I/O
         打印、写文件、保存 checkpoint 都加 is_main_process() 判断。
         否则 world_size 个进程同时写同一个文件会互相覆盖、数据损坏。
      ———————————————————————————————————————————————————————————————————

      5. find_unused_parameters=True（在 main() 中设置）
         DeepFM 的 forward 在 mode='train' 时不调用 sigmoid 层。DDP 默认要求
         所有参数都必须参与 forward，否则 backward 会因梯度同步不匹配而报错：
         "Expected to have finished reduction in the prior iteration..."
         设置为 True 后，DDP 会跳过未使用参数的梯度同步。
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',         # 监控 AUC，越大越好
        factor=0.5,         # 每次降低到当前 lr 的一半
        patience=3,         # 连续 3 次 eval 无提升则降 lr
        min_lr=1e-6,        # lr 下限
    )

    no_improve_count = 0

    for e in range(config['num_epochs']):
        # ======================== 训练 ========================
        model.train()

        # 【关键】DDP: 每个 epoch 前设置 sampler 的随机种子
        if train_sampler is not None:
            train_sampler.set_epoch(e)

        total_loss = 0
        for batch in train_loader:
            # non_blocking=True: 异步 CPU→GPU 拷贝，与 GPU 计算并行，节省时间
            cat = batch['cat'].to(device, non_blocking=True)
            num = batch['num'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)
            x_val = batch['x_val'].to(device, non_blocking=True)

            optimizer.zero_grad()
            # mode='train'（默认），返回 logits 供 BCEWithLogitsLoss
            output = model(x_val, num, cat)
            loss = criterion(output.squeeze(1), label)
            loss.backward()

            # 梯度裁剪：限制所有参数梯度的 L2 范数 ≤ max_norm，防止梯度爆炸
            # DDP 下各卡独立做梯度裁剪，不影响同步正确性
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 【关键】DDP: all_reduce 聚合各卡 loss → 全局 loss
        if is_ddp:
            # 仅用于日志：汇总各卡的 avg_loss，不影响梯度或参数更新
            loss_tensor = torch.tensor([avg_loss], device=device)
            # ReduceOp.SUM: 将所有 rank 的 loss_tensor 相加，结果写回 loss_tensor
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / dist.get_world_size()

        # 只在 rank=0 输出和写日志
        if is_main_process():
            lr_now = optimizer.param_groups[0]['lr']
            print(f'Epoch {e+1:3d}/{config["num_epochs"]} | '
                  f'Loss: {avg_loss:.4f} | LR: {lr_now:.2e}')
            with open('loss.txt', 'a') as f:
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
                    # mode='eval' → 输出 sigmoid 概率 [0, 1]
                    output = model(x_val, num, cat, 'eval')
                    all_labels.extend(label.cpu().numpy())
                    all_preds.extend(output.squeeze(1).cpu().numpy())

            # 【关键】DDP: all_gather 将各卡的验证结果汇集到 rank=0
            if is_ddp:
                world_size = dist.get_world_size()
                local_labels = torch.tensor(all_labels, device=device)
                local_preds = torch.tensor(all_preds, device=device)

                # 各卡的样本数可能不等（数据集大小不一定能整除 world_size）
                # 先收集各卡的样本数，确定最大样本数，再 padding 统一长度
                local_size = torch.tensor([len(all_labels)], device=device)
                sizes = [torch.zeros(1, dtype=torch.long, device=device)
                         for _ in range(world_size)]
                dist.all_gather(sizes, local_size)  # 每个 rank 都拿到所有 rank 的 sizes
                max_size = max(int(s.item()) for s in sizes)

                # padding 到统一长度
                pad_labels = torch.zeros(max_size, device=device)
                pad_preds = torch.zeros(max_size, device=device)
                pad_labels[:len(all_labels)] = local_labels
                pad_preds[:len(all_preds)] = local_preds

                # all_gather: 每个 rank 将自己的数据广播，其他 rank 收集
                gathered_labels = [torch.zeros(max_size, device=device)
                                   for _ in range(world_size)]
                gathered_preds = [torch.zeros(max_size, device=device)
                                  for _ in range(world_size)]
                dist.all_gather(gathered_labels, pad_labels)
                dist.all_gather(gathered_preds, pad_preds)

                # 拼接时根据各卡实际样本数去掉 padding
                if dist.get_rank() == 0:
                    all_labels = torch.cat(
                        [g[:int(s)] for g, s in zip(gathered_labels, sizes)]
                    ).cpu().tolist()
                    all_preds = torch.cat(
                        [g[:int(s)] for g, s in zip(gathered_preds, sizes)]
                    ).cpu().tolist()

            # 只在 rank=0 计算 AUC、调度 lr、保存 checkpoint
            if is_main_process():
                auc_score = roc_auc_score(all_labels, all_preds)
                scheduler.step(auc_score)

                if auc_score > best_auc:
                    best_auc = auc_score
                    no_improve_count = 0

                    # 【关键】DDP: 保存 model.module.state_dict() 而非 model.state_dict()
                    state = model.module.state_dict() if is_ddp \
                            else model.state_dict()
                    torch.save(state, 'best_deepfm.pth')

                    with open('best_auc.txt', 'w') as f:
                        f.write(f'Epoch {e+1}/{config["num_epochs"]}, '
                                f'Best AUC: {best_auc:.4f}\n')
                    print(f'  >>> New best AUC: {best_auc:.4f}')
                else:
                    no_improve_count += 1

                with open('auc.txt', 'a') as f:
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
      python main.py

      # 多卡 DDP，N 为使用的 GPU 数量
      torchrun --nproc_per_node=N main.py

    torchrun 会自动设置 LOCAL_RANK / WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT。
    无需手动配置任何环境变量。
    """
    # ===================== 超参数 =====================
    config = {
        'emb_dim': 32,
        'batch_size': 2048,          # 每张卡的有效 batch size
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'num_epochs': 200,
        'eval_every': 5,
        'early_stop_patience': 10,   # 连续 10 次 eval（= 50 epochs）无提升则 early stop
    }

    # ===================== DDP 初始化 =====================
    device, world_size, rank, is_ddp = init_distributed()

    # ===================== 读取数据 =====================
    # 每个进程独立读取数据文件（数据在共享磁盘上）。
    # 5M 行数据各进程独立读约 10-20 秒，可接受。
    # 如果数据量极大（>1 亿行），可改为仅 rank=0 读取再 broadcast 到其他 rank。
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

    # ===================== 构建模型 =====================
    model = DeepFM(
        num_kinds=13,
        cat_kinds=cat_vocab_sizes,
        kinds=39,
        embedding_dim=config['emb_dim'],
        hidden_size=[512, 256, 128, 64, 32],
        dropout=0.2,
    ).to(device)

    # 加载已有 checkpoint（DDP 下所有 rank 必须加载同一份参数，
    # 否则各卡模型初始权重不同，all_reduce 梯度同步会对不齐，导致无法收敛）
    best_auc = extract_best_auc()
    if os.path.exists('best_deepfm.pth'):
        if is_main_process():
            print("加载已有模型参数...")
        model.load_state_dict(
            torch.load('best_deepfm.pth', map_location=device)
        )

    # ===================== DDP 包装模型 =====================
    if is_ddp:
        # find_unused_parameters=True:
        #   训练时 mode='train' 不调用 sigmoid → 该层参数"未参与 forward"。
        #   设为 True 后 DDP 会检测并跳过未用参数的梯度同步，避免报错。
        model = DDP(
            model,
            device_ids=[rank],                # 当前进程使用的 GPU 编号
            output_device=rank,               # 输出 tensor 放在哪张卡
            find_unused_parameters=True,      # 允许部分参数不参与 forward
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