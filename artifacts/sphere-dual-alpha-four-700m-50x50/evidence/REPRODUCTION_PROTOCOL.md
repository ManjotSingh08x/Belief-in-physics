# Reproduce the saved visualizations

Run the commands from the packet root, using an existing Python environment with NumPy and Plotly's included JavaScript. These commands restore already measured results and export saved probabilities; they perform no training or neural inference.

```sh
python3 reproduction/restore_large_files.py --root .
python3 -m zipfile -e analysis/MODEL_NAME.zip analysis/MODEL_NAME
python3 reproduction/sphere_dual_joint3d.py --analysis analysis/MODEL_NAME --output joint3d --plotly-js reproduction/plotly-4.1.1.min.js
```

Repeat extraction for each of the four model names listed in analysis/summary.json. Reassembly validates part hashes, byte counts and original SHA256. The ZIP member hashes are in each analysis/MODEL_NAME/EXPORT_VERIFIED.json; probes.csv numerically matches analysis.json.

The training archive restores the exact source revision92bb2afa0fd61e86a0402cbffb2a50907ba0822f. The analysis script is independently pinned to SHA256b5da1dd21570954a191f998123e5bfded59e7077f6b1b5d4f16d67080db76d06. Re-running neural training or CPU analysis is unnecessary to view these outputs and requires a separate bounded compute decision.
