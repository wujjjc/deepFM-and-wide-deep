def extract_best_auc(file_path='best_auc.txt'):
    """
    从 best_auc.txt 中提取最佳 AUC 值
    """
    try:
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            # 假设格式: "Epoch 5/20, Best AUC: 0.8765"
            if 'Best AUC:' in line:
                # 分割并提取浮点数
                auc_str = line.split('Best AUC:')[-1].strip()
                best_auc = float(auc_str)
                return best_auc
            else:
                print("文件格式不正确，未找到 'Best AUC:'")
                return 0
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")
        return 0