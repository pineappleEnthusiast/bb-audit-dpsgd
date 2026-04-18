#!/usr/bin/env python3
"""
Interactive command builder for parallel_audit_model.py.

Walks through every argument group, forces explicit input for critical
parameters, and emits a ready-to-use command for a TACC idev session
or a .slurm batch file.

Usage (on a login node or in an idev session):
    python3 craft_command.py

Modes:
    1 = write a .slurm batch file
    2 = full audit in idev session (multi-node, multi-GPU) — srun + torchrun
    3 = single model run in idev session (single GPU) — torchrun --standalone
"""

import sys
import textwrap

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
DIM    = "\033[2m"


def _c(text, *codes):
    return "".join(codes) + text + RESET


def section(title):
    print()
    print(_c(f"  {title}  ", BOLD, CYAN))
    print(_c("  " + "─" * (len(title) + 2), DIM))


def ask(
    name: str,
    description: str = "",
    default=None,
    required: bool = False,
    choices: list = None,
    is_flag: bool = False,
) -> str:
    """
    Prompt the user for a value.

    - required=True : user must type something (Enter loops back).
    - default       : shown in brackets; Enter accepts it.
    - is_flag       : y/N prompt, returns 'true' or None.
    - choices       : validated against this list.
    """
    tag = _c("[REQUIRED]", BOLD, RED) if required else _c(f"[{default}]", DIM)
    desc_str = f"  {_c(description, DIM)}\n" if description else ""
    if is_flag:
        yn = "Y/n" if default else "y/N"
        prompt = f"{desc_str}  {_c(name, BOLD)} [{yn}]: "
        while True:
            answer = input(prompt).strip().lower()
            if answer in ("", "y", "n", "yes", "no"):
                break
            print(f"  Enter y or n.")
        if answer == "":
            return "true" if default else None
        return "true" if answer in ("y", "yes") else None

    choice_hint = f"  {_c('choices: ' + ', '.join(choices), DIM)}\n" if choices else ""
    prompt = f"{desc_str}{choice_hint}  {_c(name, BOLD)} {tag}: "
    while True:
        answer = input(prompt).strip()
        if answer == "" and required:
            print(f"  {_c('This argument is required — please enter a value.', RED)}")
            continue
        if answer == "" and default is not None:
            return str(default)
        if answer == "" and not required:
            return None
        if choices and answer not in choices:
            print(f"  {_c('Invalid choice. Options: ' + ', '.join(choices), RED)}")
            continue
        return answer


def confirm(message: str, default: bool = True) -> bool:
    yn = "Y/n" if default else "y/N"
    answer = input(f"\n  {_c(message, BOLD)} [{yn}]: ").strip().lower()
    if answer == "":
        return default
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def _print_command_box(cmd: str) -> None:
    print()
    print(_c("  ┌─── Copy-paste command ───────────────────────────────────┐", BOLD, YELLOW))
    print()
    for line in cmd.splitlines():
        print("  " + line)
    print()
    print(_c("  └──────────────────────────────────────────────────────────┘", BOLD, YELLOW))
    print()


# ---------------------------------------------------------------------------
# Command rendering
# ---------------------------------------------------------------------------

def _build_args_str(script_args: dict) -> str:
    parts = []
    for flag, value in script_args.items():
        if value is None:
            continue
        if value == "true":
            parts.append(f"--{flag}")
        else:
            parts.append(f"--{flag} {value}")
    indent = " " * 4
    return (" \\\n" + indent * 3).join(parts)


def render_srun_command(n_nodes: int, master_port: int, script_args: dict) -> str:
    """Build the srun / torchrun command string for an idev session."""
    args_str = _build_args_str(script_args)
    cmd = textwrap.dedent(f"""\
        srun --ntasks={n_nodes} --nodes={n_nodes} bash -c '
          MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
          MASTER_PORT={master_port};
          torchrun \\
            --nnodes={n_nodes} \\
            --nproc_per_node=1 \\
            --rdzv_backend=c10d \\
            --rdzv_endpoint=${{MASTER_ADDR}}:${{MASTER_PORT}} \\
            parallel_audit_model.py \\
            {args_str}'
    """)
    return cmd.rstrip()


def render_single_gpu_command(script_args: dict) -> str:
    """Build a standalone torchrun command for a single-GPU idev session."""
    args_str = _build_args_str(script_args)
    indent = " " * 2
    cmd = textwrap.dedent(f"""\
        torchrun \\
          --standalone \\
          --nproc_per_node=1 \\
          parallel_audit_model.py \\
          {args_str}
    """)
    return cmd.rstrip()


def render_slurm_file(n_nodes: int, master_port: int, script_args: dict, sbatch_opts: dict) -> str:
    """Build a full #SBATCH script matching spec.md Pattern 2."""
    job_name   = sbatch_opts["job_name"]
    time_limit = sbatch_opts["time_limit"]
    account    = sbatch_opts["account"]

    args_str = _build_args_str(script_args)

    script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH -J {job_name}
        #SBATCH -o {job_name}.o%j
        #SBATCH -e {job_name}.e%j
        #SBATCH -p gh
        #SBATCH -N {n_nodes}
        #SBATCH -n {n_nodes}
        #SBATCH --ntasks-per-node=1
        #SBATCH -t {time_limit}
        #SBATCH -A {account}

        module load cuda/12.4

        set -e
        cd $SCRATCH
        eval "$(conda shell.bash hook)"
        conda activate bb_audit_dpsgd
        cd bb-audit-dpsgd

        srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
          MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
          MASTER_PORT={master_port};
          RANK=$SLURM_PROCID;
          torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \\
            --rdzv_backend=c10d \\
            --rdzv_endpoint=${{MASTER_ADDR}}:${{MASTER_PORT}} \\
            parallel_audit_model.py \\
            {args_str}'
    """)
    return script.rstrip()


# ---------------------------------------------------------------------------
# Dataset-specific defaults
# ---------------------------------------------------------------------------

DATASET_DEFAULTS = {
    "mnist":    {"lr": "3",  "batch_size": "4000",  "block_size": "4000",  "model_name": "cnn"},
    "cifar10":  {"lr": "3",  "batch_size": "3125",  "block_size": "3125",  "model_name": "cnn"},
    "purchase": {"lr": "10", "batch_size": "12143", "block_size": "12143", "model_name": "mlp"},
}


# ---------------------------------------------------------------------------
# Wizard sections 2–8 (shared across all modes)
# ---------------------------------------------------------------------------

def collect_wizard_sections(mode: int) -> tuple:
    """
    Run wizard sections 2–8 and return (args, data_name, epsilon, use_defense).
    mode adjusts defaults for n_reps and fit_world_only.
    """
    args = {}

    # ------------------------------------------------------------------
    # 2. Data and model
    # ------------------------------------------------------------------
    section("2 · Data and model")

    data_name = ask("data_name",
                    choices=["mnist", "cifar10", "cifar100", "purchase"],
                    required=True)
    args["data_name"] = data_name

    ds_defaults = DATASET_DEFAULTS.get(data_name, {})

    model_default = ds_defaults.get("model_name", None)
    model_name = ask("model_name",
                     choices=["lr", "cnn", "wideresnet", "mlp", "lstm"],
                     default=model_default,
                     required=(model_default is None))
    args["model_name"] = model_name

    n_df = ask("n_df",
               description="|D| — dataset size (0 = full dataset)",
               default="0")
    if n_df and n_df != "0":
        args["n_df"] = n_df

    # ------------------------------------------------------------------
    # 3. Training hyperparameters
    # ------------------------------------------------------------------
    section("3 · Training")

    n_reps_default = "2" if mode == 3 else None
    args["n_reps"] = ask("n_reps",
                         description="Shadow models to train (split across nodes)",
                         default=n_reps_default,
                         required=(mode != 3))

    args["n_epochs"] = ask("n_epochs", default="100")

    lr_default = ds_defaults.get("lr", None)
    args["lr"] = ask("lr",
                     description="Learning rate",
                     default=lr_default,
                     required=(lr_default is None))

    bs_default = ds_defaults.get("batch_size", None)
    args["batch_size"] = ask("batch_size",
                             description="Training batch size",
                             default=bs_default,
                             required=(bs_default is None))

    blk_default = ds_defaults.get("block_size", None)
    args["block_size"] = ask("block_size",
                             description="Sub-block size for per-sample gradient computation (≤ batch_size)",
                             default=blk_default,
                             required=(blk_default is None))

    args["aug_mult"] = ask("aug_mult",
                           description="Augmentation multiplicity (1 = no extra augmentation)",
                           default="1")

    args["sampling"] = ask("sampling",
                           description="Minibatch sampling — poisson matches DP analysis",
                           default="poisson",
                           choices=["poisson", "shuffle"])

    # ------------------------------------------------------------------
    # 4. Privacy budget
    # ------------------------------------------------------------------
    section("4 · Privacy")

    epsilon = ask("epsilon",
                  description="DP ε budget (leave blank for non-private training)",
                  default=None)
    if epsilon:
        args["epsilon"] = epsilon

    args["delta"] = ask("delta", default="1e-5")
    args["max_grad_norm"] = ask("max_grad_norm",
                                description="Per-sample gradient clipping norm",
                                default="1")

    # ------------------------------------------------------------------
    # 5. Canary / target sample
    # ------------------------------------------------------------------
    section("5 · Canary")

    target_type = ask("target_type",
                      description="How to craft the canary sample",
                      default="blank",
                      choices=["blank", "mislabeled", "clipbkd", "fgsm",
                               "badnets", "gradient_space_canary"])
    args["target_type"] = target_type

    if target_type == "blank":
        blank_alpha = ask("blank_alpha",
                          description="0 = all-zeros, 1 = label-9 image",
                          default="0.0")
        if blank_alpha and blank_alpha != "0.0":
            args["blank_alpha"] = blank_alpha

    elif target_type == "mislabeled":
        args["mislabeled_target_class"] = ask("mislabeled_target_class", default="1")

    elif target_type == "gradient_space_canary":
        canary_pt = ask("gradient_space_canary_pt",
                        description="Path to pre-crafted gradient canary .pt file",
                        required=True)
        args["gradient_space_canary_pt"] = canary_pt

    # ------------------------------------------------------------------
    # 6. Audit configuration
    # ------------------------------------------------------------------
    section("6 · Audit configuration")

    args["seed"] = ask("seed", default="0")

    fixed_init = ask("fixed_init",
                     description="Fix model init across all reps (path to weights, or blank for random fixed init)",
                     is_flag=False,
                     default=None)
    if fixed_init:
        args["fixed_init"] = fixed_init
    elif ask("fixed_init (use fixed random init, no path)", is_flag=True, default=False):
        args["fixed_init"] = "true"

    if ask("holdout_audit", description="Hold out half the reps for threshold selection", is_flag=True, default=True):
        args["holdout_audit"] = "true"

    if ask("store_all_losses", description="Save per-sample training losses for every rep (large!)", is_flag=True, default=False):
        args["store_all_losses"] = "true"

    fw_default = "in" if mode == 3 else None
    args["fit_world_only"] = ask("fit_world_only",
                                  description="Train only one world (skip to run both in/out)",
                                  default=fw_default,
                                  choices=["in", "out"])

    # ------------------------------------------------------------------
    # 7. Defense
    # ------------------------------------------------------------------
    section("7 · Defense")

    use_defense = ask("defense", description="Enable gradient-norm filtering defense", is_flag=True, default=False)
    if use_defense:
        args["defense"] = "true"
        args["defense_k"] = ask("defense_k",
                                 description="Samples dropped per class per filter epoch",
                                 default="5")
        args["defense_score_fn"] = ask("defense_score_fn",
                                        description="Scoring function used to rank samples",
                                        default="grad_norm",
                                        choices=["grad_norm", "grad_norm_percentile",
                                                 "grad_dir_volatility", "rand_proj_var",
                                                 "inv_confidence", "prediction_margin",
                                                 "pred_entropy", "loss", "loss_momentum",
                                                 "loss_volatility", "grad_norm_x_loss"])
        args["defense_score_norm"] = ask("defense_score_norm",
                                          default="linf",
                                          choices=["linf", "l2", "l1"])
        if ask("defense_apply_ascent",
               description="Apply gradient ascent instead of dropping flagged samples",
               is_flag=True, default=False):
            args["defense_apply_ascent"] = "true"

        args["defense_filter_every"] = ask("defense_filter_every",
                                            description="Apply defense every N epochs",
                                            default="1")

    # ------------------------------------------------------------------
    # 8. Output directory
    # ------------------------------------------------------------------
    section("8 · Output")

    eps_str = f"eps{epsilon}" if epsilon else "non_private"
    defense_str = "_defense" if use_defense else "_no_defense"
    sampling_str = "" if args.get("sampling") == "poisson" else "_shuffle"
    aug_str = f"_augmult{args.get('aug_mult', 1)}" if args.get("aug_mult", "1") != "1" else ""
    suggested_out = f"{data_name}_{model_name}_{eps_str}{defense_str}{aug_str}{sampling_str}"

    out = ask("out",
              description="Output directory (results will be written here)",
              default=suggested_out,
              required=False)
    args["out"] = out or suggested_out

    return args, data_name, epsilon, use_defense


# ---------------------------------------------------------------------------
# SLURM directives (mode 1 only)
# ---------------------------------------------------------------------------

def collect_slurm_directives(args: dict) -> tuple:
    """Prompt for SBATCH fields. Returns (sbatch_opts dict, slurm_filename str)."""
    section("9 · SLURM job directives")

    job_name = ask("job_name",
                   description="SLURM job name (-J)",
                   default=args.get("out", "audit_job"))

    time_limit = ask("time_limit",
                     description="Wall time limit (HH:MM:SS)",
                     default="04:00:00")

    account = ask("account",
                  description="TACC allocation account (-A)",
                  default="ASC25081",
                  choices=["ASC25081", "ASC25102"])

    slurm_filename = ask("slurm_filename",
                         description="Output .slurm filename",
                         default=f"{job_name}.slurm")

    sbatch_opts = {
        "job_name":   job_name,
        "time_limit": time_limit,
        "account":    account,
    }
    return sbatch_opts, slurm_filename


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------

def main():
    print()
    print(_c("  ╔══════════════════════════════════════════════╗", BOLD, CYAN))
    print(_c("  ║    parallel_audit_model.py command builder  ║", BOLD, CYAN))
    print(_c("  ╚══════════════════════════════════════════════╝", BOLD, CYAN))
    print()
    print("  For each argument: press Enter to accept the default shown in [brackets].")
    print("  Arguments marked " + _c("[REQUIRED]", RED, BOLD) + " must be set explicitly.")

    # ------------------------------------------------------------------
    # 0. Mode selection
    # ------------------------------------------------------------------
    section("0 · Mode")

    mode = int(ask(
        "mode",
        description=(
            "1 = write a .slurm batch file\n"
            "  2 = full audit in idev session (multi-node, multi-GPU)\n"
            "  3 = single model run in idev session (single GPU, --fit_world_only in, --n_reps 2)"
        ),
        choices=["1", "2", "3"],
        required=True,
    ))

    # ------------------------------------------------------------------
    # 1. TACC distributed setup (skipped for mode 3)
    # ------------------------------------------------------------------
    if mode in (1, 2):
        section("1 · TACC / idev session")

        n_nodes = ask("n_nodes",
                      description="Number of GPU nodes allocated in your idev session",
                      required=True)
        n_nodes = int(n_nodes)

        master_port = ask("master_port",
                          description="Base torchrun rendezvous port",
                          default=29500)
        master_port = int(master_port)
    else:
        n_nodes = 1
        master_port = None

    # ------------------------------------------------------------------
    # 2–8. Shared wizard sections
    # ------------------------------------------------------------------
    args, data_name, epsilon, use_defense = collect_wizard_sections(mode)

    # ------------------------------------------------------------------
    # 9. SLURM directives (mode 1 only)
    # ------------------------------------------------------------------
    sbatch_opts = None
    slurm_filename = None
    if mode == 1:
        sbatch_opts, slurm_filename = collect_slurm_directives(args)

    # ------------------------------------------------------------------
    # Review and confirm
    # ------------------------------------------------------------------
    print()
    print(_c("  ╔══════════ Review ══════════╗", BOLD, GREEN))
    for k, v in args.items():
        if v is None:
            continue
        flag_str = f"--{k}" if v == "true" else f"--{k} {v}"
        print(f"  {_c(flag_str, BOLD)}")
    print(_c("  ╚════════════════════════════╝", BOLD, GREEN))

    if not confirm("Generate command?", default=True):
        print("  Aborted.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Render and output
    # ------------------------------------------------------------------
    if mode == 2:
        cmd = render_srun_command(n_nodes, master_port, args)
        _print_command_box(cmd)
        save = ask("save_to",
                   description="Save command to a file (leave blank to skip)",
                   default=None)
        if save:
            with open(save, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(cmd + "\n")
            print(f"  {_c('Written to ' + save, GREEN)}")

    elif mode == 3:
        cmd = render_single_gpu_command(args)
        _print_command_box(cmd)
        save = ask("save_to",
                   description="Save command to a file (leave blank to skip)",
                   default=None)
        if save:
            with open(save, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(cmd + "\n")
            print(f"  {_c('Written to ' + save, GREEN)}")

    elif mode == 1:
        slurm_script = render_slurm_file(n_nodes, master_port, args, sbatch_opts)
        _print_command_box(slurm_script)
        with open(slurm_filename, "w") as f:
            f.write(slurm_script + "\n")
        print(f"  {_c('Written to ' + slurm_filename, GREEN)}")
        print(f"  Submit with: {_c('sbatch ' + slurm_filename, BOLD)}")


if __name__ == "__main__":
    main()
