# Edge-IIoTset audit checklist

1. Prefer the selected **DNN CSV** for the first baseline.
2. Record the exact filename, row count, column count, and SHA-256.
3. Identify the binary label and multiclass attack label.
4. Check whether attack labels leak through obvious fields or filenames.
5. Do not include raw labels, attack names, or post-event fields as input features.
6. Fit imputation, encoders, and scaling on training data only.
7. Keep a held-out global test set that is never allocated to clients.
8. Create both IID and non-IID client partitions.
9. Preserve order only when order is authentic and documented.
10. Report class imbalance and use macro-F1, not accuracy alone.
