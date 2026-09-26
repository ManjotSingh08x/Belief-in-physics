# Vast.ai Budget Monitor & Automated Results Sync Guide

This document is the official operational guide for **[`scripts/vast_cost_monitor.py`](file:///home/vedansh/projects/ML/Belief-in-physics/scripts/vast_cost_monitor.py)**, a lightweight, zero-dependency safety daemon and supervisor that tracks cloud GPU spending in real time, automatically uploads results to **Google Drive** or your **local machine**, and **terminates/destroys the Vast.ai instance** when training completes or if spending crosses a predefined budget (e.g. `$5.00`).

---

## 1. Why This Service Exists & The Mandatory Pre-Deletion Hook

Cloud GPU instances (such as an RTX 5090 on Vast.ai at ~$0.85–$1.20/hr) can lead to unexpected bills if:
1. Training finishes in the middle of the night, leaving the instance idling for hours.
2. A training run diverges, hangs, or experiences an infinite loop.
3. Network connection to your remote terminal drops.

`vast_cost_monitor.py` acts as an **automated cost guard and zero-data-loss data exporter**:
- **Real-Time Telemetry**: Queries the official Vast.ai REST API every 30 seconds to track exact dollar spend.
- **Budget Circuit Breaker**: If spend exceeds `--max-spend` (default: `$5.00`), it triggers emergency data backup and destroys/stops the instance.
- **Mandatory Pre-Deletion Invariant**:
  > [!IMPORTANT]
  > **Zero Data Loss Guarantee**: In **any case** where the instance is to be destroyed (whether due to budget limit reached, `--cmd` completion with `--destroy-on-cmd-finish`, or an error), the service **FIRST** flushes its execution log (`vast_monitor.log`) and packages all current results (`double_pendulum_results.zip` or dynamically bundled partial snapshots), uploads them to **Google Drive via `rclone`**, and **ONLY THEN** issues the `DELETE` API call to terminate the instance.
- **Dual Log Capture (`TeeLogger`)**: Simultaneously pipes all terminal output and supervised training logs to both the terminal and `vast_monitor.log`.
- **Zero External Dependencies**: Implemented entirely with Python's built-in standard library (`urllib`, `subprocess`, `zipfile`, `threading`, `time`).

---

## 2. Prerequisites & API Key Configuration

### Finding Your Vast.ai API Key
1. Log in to [cloud.vast.ai](https://cloud.vast.ai).
2. Go to **Account** $\to$ **API Key** $\to$ Click **Copy**.
3. Set it as an environment variable in your terminal:
   ```bash
   export VAST_API_KEY="your_api_key_here"
   ```
*(Alternatively, save it to `~/.vast_api_key` or pass it via `--api-key <KEY>`).*

---

## 3. Architecture & Operational Lifecycle

```
+-----------------------------------------------------------------------------------+
|                            VAST.AI REMOTE INSTANCE                                |
|                                                                                   |
|   +------------------------------------+    +----------------------------------+  |
|   |         Training Process           |    |       vast_cost_monitor.py       |  |
|   | (scripts/production_train.py)     |    |                                  |  |
|   |  - 270 models on RTX 5090          |    | - Dual TeeLogger: vast_monitor.log|
|   |  - Outputs double_pendulum_results |    | - Queries spend every 30s        |  |
|   +-----------------+------------------+    | - Monitors budget limit ($5.00)  |  |
|                     |                       +-----------------+----------------+  |
+---------------------|-----------------------------------------|-------------------+
                      |                                         |
                      | 1. Training Finishes                    | 2. Termination Triggered
                      |    or Budget Exceeded                   |    (Spend >= $5.00 / Done)
                      +-----------------------------------------+
                                        |
                                        v
             +-----------------------------------------------------+
             |           MANDATORY PRE-DELETION HOOK               |
             |                                                     |
             |  1. Flush vast_monitor.log buffer to disk           |
             |  2. Package current results into zip archive        |
             |  3. rclone copy vast_monitor.log -> gdrive:.../logs |
             |  4. rclone copy results.zip -> gdrive:.../          |
             +--------------------------+--------------------------+
                                        |
                                        | 3. Only after upload completes
                                        v
                     +-------------------------------------+
                     |        Vast.ai Cloud Console        |
                     |       (REST API DELETE CALL)        |
                     |  - Destroys instance immediately    |
                     |  - Billing halts permanently        |
                     +-------------------------------------+
```

---

## 4. Method 1: Google Drive Sync via `rclone` (Recommended for Cloud Runs)

Using Google Drive via `rclone` is the **gold standard for cloud GPU training** because you can close your laptop and turn it off. When training finishes on Vast.ai (or if the $5.00 budget is hit), the instance automatically uploads logs and results directly to Google Drive and self-destructs.

### Step 1: One-Time Google Drive Setup (On your local machine)
1. Install `rclone`:
   ```bash
   sudo apt install rclone   # Ubuntu / Debian
   # or: brew install rclone # macOS
   ```
2. Run configuration:
   ```bash
   rclone config
   ```
   - Type `n` for **New remote**.
   - Name: `gdrive`
   - Storage: type `drive` (Google Drive).
   - Leave `client_id` and `client_secret` empty (press Enter).
   - Scope: `1` (Full access).
   - Edit advanced config: `n`.
   - Use web browser to authenticate: `y`. A browser tab opens; log in to your Google account and grant permissions.
   - Keep this remote: `y` $\to$ `q` to quit.

### Step 2: Copy `rclone.conf` to Vast.ai
Transfer your `rclone.conf` to the Vast.ai instance over SSH:
```bash
scp -P <SSH_PORT> ~/.config/rclone/rclone.conf root@<SSH_HOST>:~/.config/rclone/rclone.conf
```
*(On Vast.ai, install rclone if needed: `curl https://rclone.org/install.sh | bash`)*.

### Step 3: Launch Training with Budget Monitor & Auto-Upload
Run this command inside the Vast.ai instance:
```bash
python scripts/vast_cost_monitor.py \
  --max-spend 5.0 \
  --cmd "python scripts/production_train.py --platform rtx5090" \
  --rclone-dest "gdrive:belief_results" \
  --destroy-on-cmd-finish
```

**Workflow Lifecycle:**
1. Spawns and supervises the 270-model production queue on the RTX 5090.
2. In parallel, checks spend every 30 seconds against the `$5.00` budget.
3. Streams training output to both the console and `vast_monitor.log`.
4. When training finishes (or if the budget cap is reached):
   - **Pre-Deletion Hook executes**:
     - Bundles `experiments/double_pendulum_results.zip` (or current partial snapshot).
     - Uploads `vast_monitor.log` $\to$ `gdrive:belief_results/logs/vast_monitor.log`.
     - Uploads the results archive $\to$ `gdrive:belief_results/`.
   - **Destruction executes**:
     - Submits `DELETE /instances/{id}/` to Vast.ai only after cloud transfer succeeds.

---

## 5. Method 2: Direct Local Sync via `rsync` (Supervised from Laptop)

If you prefer having the results downloaded directly onto your local laptop/workstation, run `vast_cost_monitor.py` **on your local machine**.

Because your laptop can connect directly into Vast.ai over SSH, the monitor script reads `ssh_host` and `ssh_port` from the Vast API, runs `rsync` to pull the results folder down, and then destroys the remote instance.

```bash
# Run on your local laptop:
export VAST_API_KEY="your_api_key_here"

python scripts/vast_cost_monitor.py \
  --max-spend 5.0 \
  --sync-local-dir ~/Downloads/double_pendulum_results/ \
  --remote-path /workspace/experiments/double_pendulum_results.zip \
  --destroy-on-cmd-finish
```

---

## 6. Standalone Daemon Mode (Watchdog)

If you already launched a script or are working interactively in Jupyter, you can run the monitor in the background purely as a spending watchdog:

```bash
# Terminate instance if account spend crosses $5.00 (uploads logs & results first!):
python scripts/vast_cost_monitor.py --max-spend 5.0 &

# Or pause/stop compute instead of destroying:
python scripts/vast_cost_monitor.py --max-spend 5.0 --action stop &
```

---

## 7. Command-Line Options Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--max-spend` | float | `5.0` | Maximum spend limit in USD before triggering termination |
| `--action` | string | `destroy` | Action when budget is reached: `destroy` (terminates contract) or `stop` (pauses compute) |
| `--budget-type` | string | `lifetime` | `lifetime` (since instance creation) or `session` (since monitor started) |
| `--api-key` | string | `$VAST_API_KEY` | Vast.ai API key |
| `--instance-id` | int | `None` | Specific instance ID to track (auto-detects active instances if omitted) |
| `--check-interval` | int | `30` | Interval in seconds between API balance queries |
| `--cmd` | string | `None` | Shell command to execute and supervise |
| `--destroy-on-cmd-finish` | flag | `False` | Automatically destroy instance when `--cmd` and sync finish cleanly |
| `--rclone-dest` | string | `gdrive:belief_results` | Google Drive destination via rclone (can also set `$VAST_GDRIVE_DEST`) |
| `--log-file` | string | `vast_monitor.log` | Path to tee service output logs (saved and uploaded before deletion) |
| `--results-path` | string | `./experiments` | Directory or zip file containing results to package and upload |
| `--sync-local-dir` | string | `None` | Local folder path to pull remote results into via `rsync`/`scp` |
| `--remote-path` | string | `./experiments/double_pendulum_results.zip` | Remote file or directory to pull when using `--sync-local-dir` |
| `--ssh-key-path` | string | `None` | Path to SSH private key if needed for `rsync`/`scp` authentication |
| `--skip-upload` | flag | `False` | Bypass mandatory pre-deletion upload (for testing only) |
| `--webhook-url` | string | `None` | Optional Discord or Slack webhook URL to receive budget alerts |
| `--dry-run` | flag | `False` | Simulates budget checks and alert triggers without sending DELETE/PUT calls |

---

## 8. Sample Telemetry Output

```text
============================================================================
 VAST.AI BUDGET SAFETY & AUTOMATED GDRIVE RESULTS EXPORTER
============================================================================
Spending Limit:       $5.00 USD
Action on Limit:      DESTROY instance
Budget Metric:        LIFETIME (From instance launch)
Check Interval:       Every 30 seconds
Google Drive Remote:  gdrive:belief_results
Service Log File:     /workspace/vast_monitor.log
Results Directory:    /workspace/experiments
Supervised Command:   python scripts/production_train.py --platform rtx5090
Dry-Run Mode:         False
============================================================================

[16:40:10] [ID: 1482910] RTX 5090 ($0.85/hr) | Elapsed: 1h 45m | Spend: $  1.49 / $5.00 ( 29.8%) | Remaining: ~$3.51 (~4.1h)
[16:40:40] [ID: 1482910] RTX 5090 ($0.85/hr) | Elapsed: 1h 45m | Spend: $  1.50 / $5.00 ( 30.0%) | Remaining: ~$3.50 (~4.1h)
...
[Supervisor] Supervised command completed with exit code: 0
[Supervisor] --destroy-on-cmd-finish is active. Initiating pre-deletion upload and DESTROY...

############################################################################
# INITIATING DELETION SEQUENCE FOR INSTANCE 1482910
############################################################################

============================================================================
 MANDATORY PRE-DELETION HOOK: UPLOADING LOGS & CURRENT RESULTS TO GDRIVE
============================================================================
Rclone Destination: gdrive:belief_results
Service Log File:   /workspace/vast_monitor.log
Results Base Path:  /workspace/experiments

[Artifact Sync] Uploading service execution log (184,320 bytes)...
  $ rclone copy "/workspace/vast_monitor.log" "gdrive:belief_results/logs/" --stats-one-line
[Artifact Sync] -> Service log successfully uploaded to gdrive:belief_results/logs/vast_monitor.log

[Artifact Sync] Uploading results archive (42.50 MB)...
  $ rclone copy "/workspace/experiments/double_pendulum_results.zip" "gdrive:belief_results/" --stats-one-line
[Artifact Sync] -> Results successfully uploaded to gdrive:belief_results/double_pendulum_results.zip
============================================================================
[Artifact Sync] Pre-deletion Google Drive upload PASSED successfully!
============================================================================

!!! EXECUTING API DELETE ON INSTANCE 1482910 !!!
Vast.ai API Response: {'success': True}
[Success] Instance 1482910 destroyed cleanly after all data was secured.
```
