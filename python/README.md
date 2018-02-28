### How to run

```bash
bash install.sh
python3 feed-forward-iris-softmax-cross-entropy.py
python3 rnn-generator.py
```

### results feed-forward
```text
epoch 30, accuracy 0.750000, cost 0.328933
epoch 31, accuracy 0.783333, cost 0.322089
epoch 32, accuracy 0.808333, cost 0.315966
epoch 33, accuracy 0.825000, cost 0.310315
epoch 34, accuracy 0.841667, cost 0.305078
epoch 35, accuracy 0.841667, cost 0.300091
epoch 36, accuracy 0.850000, cost 0.295296
epoch 37, accuracy 0.866667, cost 0.290590
epoch 38, accuracy 0.875000, cost 0.285924
epoch 39, accuracy 0.891667, cost 0.281233
epoch 40, accuracy 0.908333, cost 0.276485
epoch 41, accuracy 0.925000, cost 0.271640
epoch 42, accuracy 0.933333, cost 0.266691
epoch 43, accuracy 0.941667, cost 0.261614
epoch 44, accuracy 0.950000, cost 0.256435
epoch 45, accuracy 0.958333, cost 0.251126
epoch 46, accuracy 0.958333, cost 0.245783
epoch 47, accuracy 0.958333, cost 0.240328
epoch 48, accuracy 0.933333, cost 0.235533
epoch 49, accuracy 0.958333, cost 0.234271
```

### results recurrent
```text
epoch 50, loss 4.175321, accuracy 0.078125
epoch 100, loss 3.883469, accuracy 0.158854
epoch 150, loss 3.507442, accuracy 0.214844
epoch 200, loss 3.388112, accuracy 0.273438
epoch 250, loss 3.327744, accuracy 0.281250
epoch 300, loss 3.042421, accuracy 0.315104
epoch 350, loss 2.942040, accuracy 0.334635
epoch 400, loss 3.002462, accuracy 0.294271
epoch 450, loss 2.517064, accuracy 0.432292
epoch 500, loss 2.653959, accuracy 0.401042
epoch 550, loss 2.719194, accuracy 0.356771
epoch 600, loss 2.386390, accuracy 0.444010
epoch 650, loss 2.535589, accuracy 0.406250
epoch 700, loss 2.578567, accuracy 0.399740
epoch 750, loss 2.221211, accuracy 0.505208
epoch 800, loss 2.231152, accuracy 0.490885
epoch 850, loss 2.168349, accuracy 0.476562
epoch 900, loss 2.021851, accuracy 0.532552
epoch 950, loss 2.022622, accuracy 0.522135
epoch 1000, loss 2.160469, accuracy 0.489583
```
