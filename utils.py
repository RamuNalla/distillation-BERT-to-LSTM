import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    Standard Knowledge Distillation Loss
    student_logits: raw output from LSTM
    teacher_logits: raw output from BERT (loaded from your .pkl file)
    labels: ground truth (0 or 1)
    T: Temperature (softens the probability distribution)
    alpha: weight for soft targets (0.5 means 50/50 balance)
    """
    # 1. Soft Target Loss (KL Divergence)
    # We apply Temperature scaling to both sets of logits
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    student_soft = F.log_softmax(student_logits / T, dim=1)
    
    # KLDivLoss expects log-probabilities for the student
    loss_soft = F.kl_div(student_soft, soft_targets, reduction='batchmean') * (T**2)
    
    # 2. Hard Target Loss (Standard Cross Entropy)
    loss_hard = F.cross_entropy(student_logits, labels)
    
    # 3. Combined weighted loss
    return alpha * loss_soft + (1 - alpha) * loss_hard