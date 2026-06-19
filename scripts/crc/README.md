# Running the tactile forecaster on ND CRC — full walkthrough

Scheduler: **Univa Grid Engine (UGE)**, `qsub`/`qrsh`/`qstat`. GPU: `-q gpu -l gpu_card=1`
(4-day limit; nodes ~32 cores / 4 GPUs). Docs:
[connecting](https://docs.crc.nd.edu/new_user/connecting_to_crc.html) ·
[GPU](https://docs.crc.nd.edu/resources/gpu.html) ·
[conda](https://docs.crc.nd.edu/popular_modules/conda.html).

> **Virtual environment?** Yes — use a **conda env** (step 2), not a bare `python -m venv`.
> CRC serves Python via modules; conda isolates Python + CUDA PyTorch cleanly.

## 0. Connect (SSH)
You need a CRC account and your ND **NetID**. From **off campus**, first connect to the campus
VPN, *or* hop through the bastion host.
```bash
# on campus / VPN:
ssh -Y netid@crcfe01.crc.nd.edu          # front-end (crcfe02 also works)
# off campus without VPN:
ssh -Y netid@bastion.crc.nd.edu          # then: ssh crcfe01
```
(Optional) passwordless login from your machine:
```bash
ssh-keygen -t ed25519                      # if you don't have a key
ssh-copy-id netid@crcfe01.crc.nd.edu
```
Conda requires bash — check with `echo $0`; if not bash, run `bash`.

## 1. Get the code + data onto CRC
**Code** — preferred: clone your fork (after the work is committed/pushed):
```bash
cd ~
git clone https://github.com/Jiayi459/TouchAnything.git
cd TouchAnything
```
*Or* push from local, *or* rsync the working tree from your machine:
```bash
# (run on your LOCAL machine, repo root)
rsync -avz --exclude '.git' --exclude '.venv' --exclude 'datasets' \
      ./ netid@crcfe01.crc.nd.edu:~/TouchAnything/
```
**Data** — the tactile subset is gitignored, so transfer it separately (small, no MP4s).
Put large data on scratch and symlink, to avoid home-quota limits:
```bash
# on CRC:
mkdir -p /scratch365/$USER/egotouch
ln -s /scratch365/$USER/egotouch ~/TouchAnything/datasets   # datasets -> scratch
# on your LOCAL machine:
rsync -avz datasets/grasp_hold_lift_tactile/ \
      netid@crcfe01.crc.nd.edu:/scratch365/NETID/egotouch/grasp_hold_lift_tactile/
# (for pretraining, also rsync the full EgoTouch npz tree)
```

## 2. One-time conda environment (the "virtual environment")
```bash
cd ~/TouchAnything
bash scripts/crc/setup_crc_env.sh
```
This runs `module load conda; conda init; source ~/.bashrc; module unload conda`, creates env
**`tactile`** (Python 3.10 + data deps), and installs **CUDA PyTorch 2.5.1 (cu124)**.
`torch.cuda.is_available()` is `False` on the front-end — verify on a GPU node (step 3).

## 3. Smoke test on a GPU
Interactive GPU shell (quickest):
```bash
qrsh -q gpu -l gpu_card=1                 # waits for a GPU node, drops you into a shell
conda activate tactile
cd ~/TouchAnything
python scripts/crc/smoke_test.py          # prints "SMOKE TEST PASSED"
exit
```

## 4. Train (the experiment)
Edit your **netid** in `scripts/crc/train_gpu.job`, then `mkdir -p logs`.
```bash
# (a) Pretrain on ALL trajectories (needs full EgoTouch npz), once per model:
qsub -v CONFIG=configs/tactile/convgru.yaml,SCOPE=full scripts/crc/train_gpu.job
#   (or run interactively with --pretrain --out runs/convgru_pretrain)

# (b) Fine-tune + Leave-Trajectory-Out CV (5 folds) on the 82 grasp clips:
for f in 0 1 2 3 4; do
  qsub -v CONFIG=configs/tactile/convgru.yaml,FOLD=$f,SCOPE=grasp,\
PRETRAINED=runs/convgru_pretrain/best.pt scripts/crc/train_gpu.job
done

# (c) Leave-One-Task-Out: add PROTOCOL=loto, FOLD=0..7
```
Monitor: `qstat -u $USER` (state `qw`=queued, `r`=running); kill with `qdel JOBID`. Logs land in
`logs/`. Models: `convgru` (primary), `convlstm` (baseline), `simvp` (headline CNN).

## 5. Evaluate / collect results
```bash
python -m src.tactile_forecast.eval --ckpt runs/convgru_grasp_lto_f0/best.pt --fold 0
# pull results back to your machine (LOCAL):
rsync -avz netid@crcfe01.crc.nd.edu:~/TouchAnything/runs/ ./runs/
```
Each run writes `best.pt`, `train_log.csv`, `test_metrics.csv`, `summary.json`. Headline =
**mean skill vs persistence** (must be > 0).

## Notes
- The repo's top-level `environment.yaml` (full DINOv2/xformers stack) is **not** needed here —
  `environment_tactile_cuda.yaml` is the lean env.
- Bump the torch/cu124 pin in `setup_crc_env.sh` if the cluster driver needs a newer CUDA.
- Support: crcsupport@nd.edu.
