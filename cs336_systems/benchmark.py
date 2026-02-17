from cs336_basics.model import BasicsTransformerLM
import timeit
import argparse
import torch
from tqdm import tqdm

ROPE_THETA = 10000.0
BATCH_SIZE = 4
VOCAB_SIZE = 10000
DEVICE = 'cuda'

def args_parser():
    parser = argparse.ArgumentParser(description='Benchmarking the transformer language model')
    parser.add_argument('--d_model', type=int, default=768, help='Dimension of the model')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--context_length', type=int, default=128, help='Sequence length for benchmarking')
    parser.add_argument('--d_ff', type=int, default=3072, help='Dimension of the feedforward network')
    parser.add_argument('--warmup_iters', type=int, default=5, help='Number of warmup iterations')
    parser.add_argument('--test_iters', type=int, default=10, help='Number of test iterations for averaging')
    parser.add_argument('--test_only_forward', action='store_true', help='Only test the forward pass')
    return parser.parse_args()

def benchmark_basics_transformer_end2end(args):
    model = BasicsTransformerLM(
        vocab_size = VOCAB_SIZE,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = ROPE_THETA
    )
    # random input for benchmarking
    # shape: (batch_size, context_length)
    model.to(DEVICE)
    input_data = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, args.context_length))
    times = []
    timer = timeit.default_timer
    for _ in range(args.warmup_iters):
        start_time = timer()
        with torch.no_grad():
            model(input_data)
        end_time = timer()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f'Average inference time over {args.warmup_iters} warm up iterations: {avg_time:.6f} seconds')
    times = []
    foward_times = []
    for _ in tqdm(range(args.test_iters), desc='Testing'):
        start_time = timer()
        model(input_data)
        torch.cuda.synchronize()
        forward_end_time = timer()
        foward_times.append(forward_end_time - start_time)
        if not args.test_only_forward:
            output = model(input_data)
            loss = output.mean()  
            loss.backward()
        torch.cuda.synchronize()
        end_time = timer()
        times.append(end_time - start_time)
    avg_time = sum(times[args.warmup_iters:]) / args.test_iters
    print(f'Average time over {args.test_iters} iterations: {avg_time:.6f} seconds on {"forward pass only" if args.test_only_forward else "forward and backward pass"}')
    backward_times = [t - f for t, f in zip(times, foward_times)]
    avg_forward_time = sum(foward_times[args.warmup_iters:]) / args.test_iters
    avg_backward_time = sum(backward_times[args.warmup_iters:]) / args.test_iters
    print(f'Average forward time: {avg_forward_time:.6f} seconds')
    if not args.test_only_forward:
        print(f'Average backward time: {avg_backward_time:.6f} seconds')
    std_forward_time = (sum((t - avg_forward_time) ** 2 for t in foward_times[args.warmup_iters:]) / args.test_iters) ** 0.5
    std_backward_time = (sum((t - avg_backward_time) ** 2 for t in backward_times[args.warmup_iters:]) / args.test_iters) ** 0.5
    print(f'Standard deviation of forward time: {std_forward_time:.6f} seconds')
    if not args.test_only_forward:
        print(f'Standard deviation of backward time: {std_backward_time:.6f} seconds')

if __name__ == '__main__':
    args = args_parser()
    benchmark_basics_transformer_end2end(args)