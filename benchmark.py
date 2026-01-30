import torch
import time
import os
from model_arch import BiLSTMStudent
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup paths and device
DEVICE = torch.device("cpu") # Benchmarking for CPU production
TEACHER_PATH = "./models/teacher/final_teacher"
STUDENT_PATH = "./models/student/student_lstm.pth"

def count_parameters(model):
    """Returns total and trainable parameter count."""
    return sum(p.numel() for p in model.parameters())

def measure_metrics(model, sample_input, iterations=100):
    """Measures average latency and throughput."""
    model.eval()
    # Warm-up (standard practice to avoid initial cold-start lag)
    with torch.no_grad():
        for _ in range(10):
            _ = model(**sample_input) if isinstance(model, torch.nn.Module) else model(sample_input)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            if hasattr(model, "config"): # Teacher (HuggingFace)
                _ = model(**sample_input)
            else: # Student (PyTorch)
                _ = model(sample_input['input_ids'])
    end_time = time.perf_counter()
    
    avg_latency = (end_time - start_time) / iterations * 1000 # in ms
    throughput = iterations / (end_time - start_time) # samples/sec
    return avg_latency, throughput

def run_benchmark():
    print("--- 🚀 Initializing Expert Benchmark ---")
    
    # 1. Load Models
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_PATH).to(DEVICE)
    
    student = BiLSTMStudent(vocab_size=30522, embed_dim=128, hidden_dim=256, output_dim=2)
    student.load_state_dict(torch.load(STUDENT_PATH, map_location=DEVICE))
    student.to(DEVICE)

    # 2. Prepare Sample Input (Standard Sentiment Review)
    text = "The acting was phenomenal and the plot was incredibly engaging."
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=128).to(DEVICE)

    # 3. Parameter and Size Metrics
    t_params = count_parameters(teacher)
    s_params = count_parameters(student)
    compression_ratio = t_params / s_params
    
    # 4. Latency & Throughput Metrics
    t_latency, t_thr = measure_metrics(teacher, inputs)
    s_latency, s_thr = measure_metrics(student, inputs)
    speedup = t_latency / s_latency

    # --- Print Professional Report ---
    print(f"\n{'Metric':<25} | {'Teacher (BERT)':<15} | {'Student (LSTM)':<15}")
    print("-" * 65)
    print(f"{'Total Parameters':<25} | {t_params/1e6:>13.2f}M | {s_params/1e6:>13.2f}M")
    print(f"{'Model Size (Approx)':<25} | {t_params*4/1e6:>13.2f}MB | {s_params*4/1e6:>13.2f}MB")
    print(f"{'Avg CPU Latency (ms)':<25} | {t_latency:>13.2f}ms | {s_latency:>13.2f}ms")
    print(f"{'Throughput (req/sec)':<25} | {t_thr:>13.2f} | {s_thr:>13.2f}")
    
    print(f"\n--- 📈 EXPERT ANALYSIS ---")
    print(f"✅ Compression Ratio: {compression_ratio:.1f}x smaller")
    print(f"✅ Speedup Factor: {speedup:.1f}x faster on CPU")

if __name__ == "__main__":
    run_benchmark()