# TDRL_MARL Handoff Memory

## Project Scope

- Repo has been simplified to keep only the multi-agent path:
  - `MADDPG/train.py`: baseline MADDPG
  - `MADDPG/train_tdrl.py`: MADDPG + TdRL
  - `reward_model/reward_model_ma_tdrl.py`: multi-agent TdRL reward model
  - `tester/spread.py`: behavioral tests for `simple_spread_v3`
- Single-agent SAC/PPO/DMControl/SB3 code was removed.

## Important Code Changes Already Made

- Fixed environment creation bug in `MADDPG/common/pettingzoo_wrapper.py`:
  - `mod.parallel_env(...)` -> `module.parallel_env(...)`
- Fixed nested save-dir creation in `MADDPG/maddpg/maddpg.py`:
  - use `os.makedirs(..., exist_ok=True)` so date-based nested paths work
- Added configurable training/perf knobs in `MADDPG/runner.py` and both configs:
  - `train_interval`
  - `updates_per_train`
  - `loss_log_interval`
  - `explore_log_interval`
  - `eval_save_interval`
  - `eval_plot_interval`
- Fixed TensorBoard loss logging bug:
  - loss is now logged by `update_counter`, not raw `time_step`
  - otherwise loss could disappear when `train_interval` and `loss_log_interval` were misaligned
- Reduced unnecessary I/O in TdRL runner:
  - no more `np.save(...)` every step
  - eval saves/plots now respect configured intervals

## TdRL / Tester Work Done

### Initial Diagnosis

- TdRL underperformed baseline badly.
- Return network often had high `return_acc`, but final task performance was still poor.
- This suggested that the test labels themselves might be misaligned with the real task objective.

### `simple_spread_v3` Reward Alignment Findings

- True environment reward is mixed:
  - global term: negative sum of nearest-agent distances to landmarks
  - local term: collision penalty
  - implemented in PettingZoo `simple_spread_v3` with `local_ratio=0.5`
- Original tester had a serious mismatch:
  - collision threshold used `0.038`
  - true environment collision threshold is `0.30` because agent size is `0.15`

### Tester Modifications Already Applied in `tester/spread.py`

- Collision threshold aligned to environment:
  - `_collision_dist = 0.30`
- Coverage distance aligned better with benchmark notion:
  - `_cover_dist = 0.10`
- Added `ind_global_reward_proxy`
  - initially as accumulated distance proxy
  - later changed to per-timestep mean to reduce scale problems
- Relaxed PFs:
  - `pf_all_covered` no longer requires near-whole-trajectory coverage
  - uses terminal window logic
  - later relaxed further
  - `pf_no_collision` changed from strict zero-collision over full segment to low collision rate over terminal window

### Extra Diagnostics Added in `MADDPG/train_tdrl.py`

- New TdRL TensorBoard logs now include:
  - `tdrl/pf_0`, `tdrl/pf_1`
  - `tdrl/pf_count_0`, `tdrl/pf_count_1`
  - `tdrl/ind_std_i`
  - `tdrl/ind_min_i`
  - `tdrl/ind_max_i`

## Key Experimental Results So Far

### Baseline MADDPG Best Reference

- Baseline run:
  - `model/MADDPG/2026-04-07/18-49-14/simple_spread_v3/returns.pkl.npy`
  - `model/MADDPG/2026-04-07/18-49-14/simple_spread_v3/tb_logs/events.out.tfevents.1775558955.asus.174173.0`
- Summary:
  - best eval return about `-6.27`
  - final eval return about `-9.13`
  - last-10 eval mean about `-10.16`

### Older TdRL Runs

- `model_tdrl/2026-04-09/11-16-56/...`
  - weaker TdRL run
- `model_tdrl/2026-04-09/16-44-55/...`
  - best old TdRL run before tester changes
  - best eval return about `-13.09`
  - final eval return about `-17.65`
  - last-10 eval mean about `-20.13`

### New Tester First Attempt

- `model_tdrl/2026-04-09/19-50-54/...`
- Internal reward/critic stability improved a lot
- But final performance got worse overall:
  - best eval return about `-13.15`
  - final eval return about `-21.39`
  - last-10 eval mean about `-23.88`
- Diagnosis:
  - `ind_global_reward_proxy` scale improved later
  - but `pf_all_covered` was still too sparse and nearly dead

### Latest Run After More Tester + Config Changes

- Latest run:
  - `model_tdrl/2026-04-10/10-09-43/simple_spread_v3/returns.pkl.npy`
  - `model_tdrl/2026-04-10/10-09-43/simple_spread_v3/tb_logs/events.out.tfevents.1775786984.asus.287287.0`
  - config:
    - `outputs/2026-04-10/10-09-43/.hydra/config.yaml`
- Summary:
  - best eval return about `-11.60`
  - final eval return about `-19.88`
  - last-10 eval mean about `-21.02`
  - last-50 eval mean about `-19.76`
- Interpretation:
  - better than `2026-04-09/19-50-54`
  - still not clearly better than `2026-04-09/16-44-55`
  - still far below baseline

## Most Important Current Diagnosis

- Reward/critic stability is much better now.
  - `reward_loss_agent0` and critic loss dropped a lot in later experiments
- But final task performance is still not competitive.
- The main remaining problem appears to be **test signal structure**, not exploding losses.

### Strongest Current Suspicion

- `pf_all_covered` is still too sparse / too harsh.
- In the latest run:
  - `tdrl/pf_0` stays almost constant around `0.025`
  - `tdrl/pf_count_0` stays around `5.0`
- This means the first PF test is nearly dead and contributes almost no useful ranking signal.
- Since PF tests have high priority in preference generation, this is likely hurting return learning.

### Secondary Suspicion

- Return preference learning remains weaker than desired:
  - latest `tdrl/return_acc` ends around `0.94`
  - latest `tdrl/return_loss` still much higher than the stronger old TdRL runs

## Important Caveat for Comparing Runs

- Latest run changed more than the tester:
  - `lr_actor = 0.0002`
  - `lr_critic = 0.0001`
  - `train_interval = 25`
- Therefore, latest improvements/regressions cannot be attributed purely to tester changes.
- Future comparisons should avoid changing tester and training schedule at the same time.

## Recommended Next Step

- Most recommended next experiment:
  - remove or heavily weaken `pf_all_covered`
  - keep collision PF
  - rely more on continuous indicative metrics, especially `ind_global_reward_proxy`
- If doing a clean ablation:
  - keep optimizer and training schedule fixed
  - only change tester logic

## Useful Files To Reopen First In A New Chat

- `MADDPG/conf/config_tdrl.yaml`
- `MADDPG/train_tdrl.py`
- `MADDPG/runner.py`
- `tester/spread.py`
- `reward_model/reward_model_ma_tdrl.py`
- `memory.md`

## Suggested Restart Prompt

Use something like this in a new conversation:

> Continue from `memory.md`. The current main suspicion is that `pf_all_covered` in `tester/spread.py` is still too sparse and harms TdRL preference learning on `simple_spread_v3`. Please read `memory.md`, `tester/spread.py`, `MADDPG/train_tdrl.py`, and `MADDPG/conf/config_tdrl.yaml`, then help me make the next tester ablation.
