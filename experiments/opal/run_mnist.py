"""Small local OPAL experiment using the upstream MNIST MLP recipe.

Training loads only the training split. Evaluate saved weights in another process.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms


def model():
    return nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))


def load(data_dir, train):
    ds = datasets.MNIST(data_dir, train=train, download=True, transform=transforms.ToTensor())
    x = torch.stack([ds[i][0] for i in range(len(ds))]).reshape(len(ds), -1)
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    return (x - 0.1307) / 0.3081, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['plain', 'keep', 'ollama', 'evaluate', 'download'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'data')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--decisions', type=int, default=3)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--model', default='gemma4:e2b-qat-google')
    p.add_argument('--think', action='store_true')
    p.add_argument('--reasoning-level', choices=['low', 'medium', 'high', 'max'])
    p.add_argument('--max-output-tokens', type=int, default=4096)
    p.add_argument('--model-timeout', type=int, default=180)
    p.add_argument('--temperature', type=float, default=0.2)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.mode == 'download':
        for train in (True, False):
            datasets.MNIST(args.data_dir, train=train, download=True)
        print('MNIST downloaded', flush=True)
        return
    if args.mode == 'evaluate':
        net = model()
        net.load_state_dict(torch.load(args.out / 'model.pt', weights_only=True))
        net.eval()
        x, y = load(args.data_dir, False)
        with torch.no_grad():
            logits = torch.cat([net(x[i:i+2000]) for i in range(0, len(x), 2000)])
            result = {'test_accuracy': (logits.argmax(1) == y).float().mean().item(),
                      'test_loss': F.cross_entropy(logits, y).item(), 'test_examples': len(y),
                      'separate_evaluation_process': True}
        (args.out / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return

    import opal
    from opal.results import opal_commit
    x, y = load(args.data_dir, True)
    net = model()
    batch_size = 128
    steps_per_epoch = len(x) // batch_size
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    if args.mode != 'plain':
        backend = 'keep'
        if args.mode == 'ollama':
            from ollama_backend import OllamaBackend
            backend = OllamaBackend(record_root=args.out / 'ollama_calls', model=args.model,
                                    think=args.reasoning_level or args.think, num_predict=args.max_output_tokens,
                                    timeout=args.model_timeout, temperature=args.temperature)
        opt = opal.wrap(opt, net, scheduler=sched, total_steps=total_steps,
                        goal='3-layer MLP on MNIST; judged by test accuracy after the last update',
                        backend=backend, decisions=args.decisions, schedule=[(1.0, args.decisions)], every=50,
                        max_decisions=args.decisions,
                        blocking=True, summarize=False, run_dir=args.out / 'opal', share_code=False,
                        epochs=args.epochs, train_size=len(x), batch_size=batch_size,
                        same_order_each_epoch=False,
                        task_facts='CPU pilot of upstream mnist_mlp.py recipe. Test split is unavailable during training. '
                                   'Seed and minibatch order match the plain and keep controls. No development set feedback.')
    g = torch.Generator().manual_seed(args.seed)
    t0 = time.perf_counter()
    step = 0
    losses = []
    with (args.out / 'training.jsonl').open('w') as log:
        for epoch in range(args.epochs):
            perm = torch.randperm(len(x), generator=g)
            for i in range(steps_per_epoch):
                idx = perm[i*batch_size:(i+1)*batch_size]
                loss = F.cross_entropy(net(x[idx]), y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(loss=loss) if args.mode != 'plain' else opt.step()
                sched.step()
                step += 1
                val = float(loss.detach())
                losses.append(val)
                log.write(json.dumps({'step': step, 'epoch': epoch+1, 'loss': val}) + '\n')
            log.flush()
            print(f'epoch {epoch+1}: training loss {val:.6f}, elapsed {time.perf_counter()-t0:.1f}s', flush=True)
    if args.mode != 'plain':
        opt.close()
    elapsed = time.perf_counter() - t0
    torch.save(net.state_dict(), args.out / 'model.pt')
    h = hashlib.sha256()
    for v in net.parameters():
        h.update(v.detach().contiguous().view(torch.uint8).numpy().tobytes())
    result = dict(mode=args.mode, seed=args.seed, epochs=args.epochs, steps=step, batch_size=batch_size,
                  threads=args.threads, training_seconds=elapsed, final_training_loss=losses[-1],
                  parameter_sha256=h.hexdigest(), opal_commit=opal_commit(), torch_version=torch.__version__,
                  decisions_requested=args.decisions if args.mode != 'plain' else 0,
                  decision_schedule=[(1.0, args.decisions)] if args.mode != 'plain' else None,
                  agent_model=args.model if args.mode == 'ollama' else None,
                  agent_thinking=(args.reasoning_level or args.think) if args.mode == 'ollama' else None,
                  agent_max_output_tokens=args.max_output_tokens if args.mode == 'ollama' else None,
                  agent_temperature=args.temperature if args.mode == 'ollama' else None,
                  agent_wait_seconds=getattr(opt, 'agent_wait_seconds', 0.0))
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
