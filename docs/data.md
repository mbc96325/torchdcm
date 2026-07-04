# Data Hub

TorchDCM separates estimator validation data into three layers:

1. `datasets/small`: small public datasets that can live in GitHub.
2. `datasets/large`: metadata and Google Drive links for processed large
   datasets.
3. `validation/datasets`: reproducible raw download/export machinery used for
   benchmark development.

Large survey data only enters the public release after a preprocessing recipe
defines choice sets, attributes, availability, IDs, and weights.
