# Benchmarking

Benchmarks compare TorchDCM against Biogeme, Apollo, and R estimators with
aligned data, specification, starting values, and metrics.

Core metrics:

- parameter-estimation time;
- covariance/Hessian time;
- log likelihood;
- beta, standard error, t-value, and covariance differences;
- probability differences;
- WTP and elasticity differences where defined.

Remote execution remains the source of truth:

```bash
cd /home/baichuan-mo/torchdcm
.venv/bin/python validation/benchmarks/run_estimator_benchmark_suite.py --profile full
```
