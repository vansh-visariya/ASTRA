# Async Federated Learning with Robust Aggregation

A modular, reproducible, research-grade PyTorch framework for asynchronous federated learning with Byzantine-resilient aggregation, differential privacy, and HuggingFace PEFT support.

## Features

- **Asynchronous Server**: True async event-driven server with immediate update processing
- **Robust Aggregation**: Coordinate-wise median, trimmed mean, and hybrid pipeline
- **Byzantine Resilience**: Dynamic trust scoring with soft quarantine
- **Privacy**: DP-SGD (client/server modes) and simulated secure aggregation
- **Compression**: Top-k sparsification for communication efficiency
- **HF/PEFT**: HuggingFace model support with LoRA adapter federation
- **Non-IID Data**: Dirichlet splitting with class imbalance
- **Malicious Simulation**: Label flip, noise, sign flip, scale, backdoor attacks

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo (5 clients, 200 steps)
python main.py --config config.yaml --demo

# Run full experiment
python main.py --config config.yaml --experiment-id my_experiment
```

## Project Structure

```
federated/
  server.py          # Async server event loop
  client.py          # Client local training
  aggregator.py      # Aggregation strategies
  robust_aggregation.py  # Trimmed mean, median, hybrid
  trust_manager.py   # Dynamic trust scoring
  privacy.py         # DP + secure aggregation
  compression.py     # Top-k sparsification
  model_zoo.py       # CNN models
  hf_models.py       # HuggingFace PEFT support
  data_splitter.py  # Dirichlet/non-IID splits
  malicious_simulator.py  # Attack injection
  utils/
    logging_utils.py
    metrics.py
    plotting.py
    seed.py
  experiments/
    run_experiment.py
    experiments_spec.yaml
  tests/
    test_aggregator.py
    test_trust_manager.py
    test_privacy.py
    test_compression.py
    test_reproducibility.py
```

## Key Files to Review First

1. **main.py** - Entry point and CLI
2. **federated/server.py** - Async server implementation
3. **federated/robust_aggregation.py** - Core Byzantine-resilient algorithms
4. **federated/experiments/run_experiment.py** - Experiment runner

## Experiments

Run experiments from specification:

```bash
# Run all experiments
python federated/experiments/run_experiment.py

# Run specific experiment
python federated/experiments/run_experiment.py --experiment-id async_hybrid_trust_dp
```

## Testing

```bash
# Run all tests
pytest federated/tests/ -v

# Run specific test
pytest federated/tests/test_aggregator.py -v
```

## Docker

```bash
# Build image
docker build -t async-fl:latest .

# Run container
docker run --gpus all -v $(pwd):/app async-fl:latest
```

## Configuration

See `config.yaml` for all configuration options:

- Dataset settings (MNIST, CIFAR-10)
- Model settings (CNN, HF transformer)
- PEFT settings (LoRA rank, target modules)
- Server parameters (optimizer, learning rate, async lambda)
- Robust aggregation method
- Trust manager parameters
- DP settings
- Communication compression

## References

- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Xie et al., "Asynchronous Federated Optimization"
- Yin et al., "Byzantine-Robust Distributed Learning"
- Abadi et al., "Deep Learning with Differential Privacy"
- Hu et al., "LoRA: Low-Rank Adaptation"

## License

MIT License
