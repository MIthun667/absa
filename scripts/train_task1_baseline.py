from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from dimabsa.experiment_data import (
    TaskRecord,
    load_task_records,
)
from dimabsa.metrics import evaluate_task1
from dimabsa.training_views import (
    build_relation_expanded_view,
    build_unambiguous_view,
    build_drop_ambiguous_keep_duplicates_view,
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def current_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
            )
            .strip()
        )
    except Exception:
        return None


# ---------------------------------------------------------------------
# Flat examples
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FlatTask1Example:
    record_id: str
    text: str
    aspect: str
    valence: float
    arousal: float
    language: str
    domain: str
    split: str
    ambiguous_source_group: bool = False
    source_opinion: str | None = None
    source_category: str | None = None


def flatten_train_records(
    records: list[TaskRecord],
    view: str,
) -> list[FlatTask1Example]:

    if view == "relation_expanded":
        projected = build_relation_expanded_view(records)

    elif view == "unambiguous":
        projected = build_unambiguous_view(records)

    elif view == "drop_ambiguous_keep_duplicates":
        projected = build_drop_ambiguous_keep_duplicates_view(records)

    else:
        raise ValueError(f"Unknown training view: {view}")

    result: list[FlatTask1Example] = []

    for example in projected:
        target = example.target

        result.append(
            FlatTask1Example(
                record_id=example.record_id,
                text=example.text,
                aspect=target.aspect,
                valence=target.valence,
                arousal=target.arousal,
                language=example.language,
                domain=example.domain,
                split="train",
                ambiguous_source_group=example.ambiguous_source_group,
                source_opinion=target.source_opinion,
                source_category=target.source_category,
            )
        )

    return result


def flatten_dev_records(
    records: list[TaskRecord],
) -> list[FlatTask1Example]:

    result: list[FlatTask1Example] = []

    for record in records:
        seen: set[str] = set()

        for target in record.targets:
            key = target.aspect.casefold()

            if key in seen:
                raise ValueError(
                    f"Duplicate Task-1 dev aspect: "
                    f"{record.record_id} / {target.aspect}"
                )

            seen.add(key)

            result.append(
                FlatTask1Example(
                    record_id=record.record_id,
                    text=record.text,
                    aspect=target.aspect,
                    valence=target.valence,
                    arousal=target.arousal,
                    language=record.language,
                    domain=record.domain,
                    split="dev",
                )
            )

    return result


# ---------------------------------------------------------------------
# Dataset / collator
# ---------------------------------------------------------------------


class Task1Dataset(Dataset):
    def __init__(
        self,
        examples: list[FlatTask1Example],
    ) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self,
        index: int,
    ) -> FlatTask1Example:
        return self.examples[index]


class Task1Collator:
    def __init__(
        self,
        tokenizer,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self,
        batch: list[FlatTask1Example],
    ) -> dict[str, Any]:

        texts = [x.text for x in batch]
        aspects = [x.aspect for x in batch]

        encoded = self.tokenizer(
            texts,
            aspects,
            padding=True,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = torch.tensor(
            [
                [x.valence, x.arousal]
                for x in batch
            ],
            dtype=torch.float32,
        )

        return {
            "inputs": encoded,
            "labels": labels,
            "examples": batch,
        }


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


class Task1Regressor(nn.Module):
    """
    Simple aspect-conditioned XLM-R regression baseline.

    Input:
        sentence + aspect as a sequence pair

    Representation:
        <s> / first-token hidden state

    Output:
        two bounded values:
            Valence in [1, 9]
            Arousal in [1, 9]
    """

    def __init__(
        self,
        model_name: str,
        dropout: float,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 2)

    def forward(
        self,
        **inputs,
    ) -> torch.Tensor:

        outputs = self.encoder(**inputs)

        cls = outputs.last_hidden_state[:, 0]

        logits = self.regressor(
            self.dropout(cls)
        )

        # Keep predictions inside the official [1, 9] VA space.
        predictions = 1.0 + 8.0 * torch.sigmoid(logits)

        return predictions


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def to_device(
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:

    return {
        key: value.to(device, non_blocking=True)
        for key, value in inputs.items()
    }


def amp_context(
    use_amp: bool,
    amp_dtype: torch.dtype | None,
):

    if not use_amp:
        return nullcontext()

    return torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
    )


def safe_number(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, dict):
        return {
            k: safe_number(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            safe_number(v)
            for v in value
        ]

    return value


def save_json(
    path: Path,
    value: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            safe_number(value),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def format_va(
    valence: float,
    arousal: float,
) -> str:

    return f"{valence:.8f}#{arousal:.8f}"


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    dev_records_by_domain: dict[str, list[TaskRecord]],
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
]:

    model.eval()

    pred_map: dict[
        str,
        dict[str, list[dict[str, str]]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    diagnostics: dict[
        str,
        dict[str, list[float]],
    ] = defaultdict(
        lambda: {
            "gold_v": [],
            "gold_a": [],
            "pred_v": [],
            "pred_a": [],
        }
    )

    for batch in loader:

        inputs = to_device(
            batch["inputs"],
            device,
        )

        with amp_context(
            use_amp,
            amp_dtype,
        ):
            predictions = model(**inputs)

        predictions = (
            predictions
            .float()
            .cpu()
            .tolist()
        )

        for example, prediction in zip(
            batch["examples"],
            predictions,
        ):

            pv, pa = prediction

            pred_map[
                example.domain
            ][
                example.record_id
            ].append(
                {
                    "Aspect": example.aspect,
                    "VA": format_va(pv, pa),
                }
            )

            d = diagnostics[
                example.domain
            ]

            d["gold_v"].append(
                example.valence
            )
            d["gold_a"].append(
                example.arousal
            )
            d["pred_v"].append(pv)
            d["pred_a"].append(pa)

    metrics: dict[str, Any] = {}
    prediction_files: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for domain, records in (
        dev_records_by_domain.items()
    ):

        gold_records: list[
            dict[str, Any]
        ] = []

        pred_records: list[
            dict[str, Any]
        ] = []

        for record in records:

            gold_items = [
                {
                    "Aspect": target.aspect,
                    "VA": format_va(
                        target.valence,
                        target.arousal,
                    ),
                }
                for target in record.targets
            ]

            gold_records.append(
                {
                    "ID": record.record_id,
                    "Aspect_VA": gold_items,
                }
            )

            predictions = pred_map[
                domain
            ].get(
                record.record_id,
                [],
            )

            pred_records.append(
                {
                    "ID": record.record_id,
                    "Aspect_VA": predictions,
                }
            )

        official = evaluate_task1(
            gold_records,
            pred_records,
            normalize_rmse=False,
        )

        diag = diagnostics[domain]

        n = len(diag["gold_v"])

        rmse_v = math.sqrt(
            sum(
                (p - g) ** 2
                for p, g in zip(
                    diag["pred_v"],
                    diag["gold_v"],
                )
            )
            / n
        )

        rmse_a = math.sqrt(
            sum(
                (p - g) ** 2
                for p, g in zip(
                    diag["pred_a"],
                    diag["gold_a"],
                )
            )
            / n
        )

        mae_v = (
            sum(
                abs(p - g)
                for p, g in zip(
                    diag["pred_v"],
                    diag["gold_v"],
                )
            )
            / n
        )

        mae_a = (
            sum(
                abs(p - g)
                for p, g in zip(
                    diag["pred_a"],
                    diag["gold_a"],
                )
            )
            / n
        )

        domain_metrics = {
            **official,
            "RMSE_V": rmse_v,
            "RMSE_A": rmse_a,
            "MAE_V": mae_v,
            "MAE_A": mae_a,
        }

        metrics[domain] = domain_metrics
        prediction_files[domain] = pred_records

    macro_rmse = sum(
        metrics[d]["RMSE_VA"]
        for d in metrics
    ) / len(metrics)

    macro_rmse_v = sum(
        metrics[d]["RMSE_V"]
        for d in metrics
    ) / len(metrics)

    macro_rmse_a = sum(
        metrics[d]["RMSE_A"]
        for d in metrics
    ) / len(metrics)

    metrics["macro"] = {
        "RMSE_VA": macro_rmse,
        "RMSE_V": macro_rmse_v,
        "RMSE_A": macro_rmse_a,
    }

    return metrics, prediction_files


def save_predictions(
    output_dir: Path,
    predictions: dict[
        str,
        list[dict[str, Any]],
    ],
) -> None:

    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for domain, records in (
        predictions.items()
    ):

        path = (
            pred_dir
            / f"eng_{domain}_dev_task1.jsonl"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        default="data/raw/dimabsa",
    )

    parser.add_argument(
        "--language",
        default="eng",
    )

    parser.add_argument(
        "--domains",
        nargs="+",
        default=[
            "laptop",
            "restaurant",
        ],
    )

    parser.add_argument(
        "--view",
        choices=[
            "relation_expanded",
            "unambiguous",
            "drop_ambiguous_keep_duplicates"
        ],
        required=True,
    )

    parser.add_argument(
        "--model-name",
        default="FacebookAI/xlm-roberta-base",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output-root",
        default="outputs/phase2/task1",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    set_seed(args.seed)

    raw_root = Path(
        args.raw_root
    ).resolve()

    output_dir = (
        Path(args.output_root)
        / args.view
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    train_examples: list[
        FlatTask1Example
    ] = []

    dev_examples: list[
        FlatTask1Example
    ] = []

    dev_records_by_domain: dict[
        str,
        list[TaskRecord],
    ] = {}

    train_domain_counts = {}

    for domain in args.domains:

        train_records = load_task_records(
            raw_root,
            task=1,
            language=args.language,
            domain=domain,
            split="train",
        )

        domain_train = flatten_train_records(
            train_records,
            args.view,
        )

        train_examples.extend(
            domain_train
        )

        train_domain_counts[
            domain
        ] = len(domain_train)

        dev_records = load_task_records(
            raw_root,
            task=1,
            language=args.language,
            domain=domain,
            split="dev",
        )

        dev_records_by_domain[
            domain
        ] = dev_records

        dev_examples.extend(
            flatten_dev_records(
                dev_records
            )
        )

    if not train_examples:
        raise RuntimeError(
            "No training examples loaded"
        )

    if not dev_examples:
        raise RuntimeError(
            "No development examples loaded"
        )

    ambiguous_examples = sum(
        example.ambiguous_source_group
        for example in train_examples
    )

    print()
    print("=" * 72)
    print("TASK 1 BASELINE")
    print("=" * 72)
    print(
        f"view             : {args.view}"
    )
    print(
        f"model            : {args.model_name}"
    )
    print(
        f"seed             : {args.seed}"
    )
    print(
        f"train examples   : {len(train_examples)}"
    )
    print(
        f"dev targets      : {len(dev_examples)}"
    )
    print(
        f"ambiguous train  : {ambiguous_examples}"
    )
    print(
        f"domain counts    : {train_domain_counts}"
    )
    print("=" * 72)
    print()

    # --------------------------------------------------------------
    # Tokenizer / loaders
    # --------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )

    collator = Task1Collator(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        Task1Dataset(train_examples),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            args.num_workers > 0
        ),
        generator=generator,
    )

    dev_loader = DataLoader(
        Task1Dataset(dev_examples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    # --------------------------------------------------------------
    # Device / AMP
    # --------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
        and not args.no_amp
    )

    amp_dtype: torch.dtype | None = None

    if use_amp:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

    print(
        f"device           : {device}"
    )
    print(
        f"AMP              : {use_amp}"
    )
    print(
        f"AMP dtype        : {amp_dtype}"
    )

    if device.type == "cuda":
        print(
            f"GPU              : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = Task1Regressor(
        model_name=args.model_name,
        dropout=args.dropout,
    ).to(device)

    no_decay_terms = (
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
    )

    decay_params = []
    no_decay_params = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if any(
            term in name
            for term in no_decay_terms
        ):
            no_decay_params.append(
                parameter
            )
        else:
            decay_params.append(
                parameter
            )

    optimizer = AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": args.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ],
        lr=args.learning_rate,
    )

    warmup_steps = int(
        args.max_steps
        * args.warmup_ratio
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=args.max_steps,
        )
    )

    use_fp16_scaler = (
        use_amp
        and amp_dtype == torch.float16
    )

    scaler = None

    if use_fp16_scaler:
        scaler = torch.amp.GradScaler(
            "cuda"
        )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    run_config = {
        **vars(args),
        "git_commit": current_git_commit(),
        "torch_version": torch.__version__,
        "transformers_version": (
            transformers.__version__
        ),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "amp": use_amp,
        "amp_dtype": str(amp_dtype),
        "train_examples": len(
            train_examples
        ),
        "dev_examples": len(
            dev_examples
        ),
        "ambiguous_training_examples": (
            ambiguous_examples
        ),
        "train_domain_counts": (
            train_domain_counts
        ),
    }

    save_json(
        output_dir / "run_config.json",
        run_config,
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    history = []

    best_macro_rmse = float("inf")
    best_step = None

    checkpoint_path = (
        output_dir
        / "best_checkpoint.pt"
    )

    train_iterator = iter(
        train_loader
    )

    running_loss = 0.0
    running_count = 0

    start_time = time.time()

    for step in range(
        1,
        args.max_steps + 1,
    ):

        model.train()

        try:
            batch = next(
                train_iterator
            )

        except StopIteration:
            train_iterator = iter(
                train_loader
            )
            batch = next(
                train_iterator
            )

        inputs = to_device(
            batch["inputs"],
            device,
        )

        labels = (
            batch["labels"]
            .to(
                device,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with amp_context(
            use_amp,
            amp_dtype,
        ):

            predictions = model(
                **inputs
            )

            # Equivalent objective to the official joint VA RMSE
            # up to monotonic scaling.
            loss = F.mse_loss(
                predictions,
                labels,
                reduction="mean",
            )

        if scaler is not None:

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )

            optimizer.step()

        scheduler.step()

        running_loss += float(
            loss.detach().cpu()
        )
        running_count += 1

        if (
            step % args.log_every == 0
        ):

            mean_loss = (
                running_loss
                / running_count
            )

            current_lr = (
                scheduler
                .get_last_lr()[0]
            )

            print(
                f"step={step:4d} "
                f"loss={mean_loss:.6f} "
                f"lr={current_lr:.3e}"
            )

            running_loss = 0.0
            running_count = 0

        should_evaluate = (
            step % args.eval_every == 0
            or step == args.max_steps
        )

        if not should_evaluate:
            continue

        metrics, predictions_json = (
            evaluate(
                model=model,
                loader=dev_loader,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                dev_records_by_domain=(
                    dev_records_by_domain
                ),
            )
        )

        macro_rmse = (
            metrics["macro"][
                "RMSE_VA"
            ]
        )

        history_entry = {
            "step": step,
            "metrics": metrics,
        }

        history.append(
            history_entry
        )

        save_json(
            output_dir
            / "history.json",
            history,
        )

        print()
        print(
            f"[DEV @ step {step}]"
        )

        for domain in args.domains:

            m = metrics[domain]

            print(
                f"{domain:10s} "
                f"RMSE_VA={m['RMSE_VA']:.6f} "
                f"RMSE_V={m['RMSE_V']:.6f} "
                f"RMSE_A={m['RMSE_A']:.6f} "
                f"PCC_V={m['PCC_V']:.4f} "
                f"PCC_A={m['PCC_A']:.4f}"
            )

        print(
            f"macro      "
            f"RMSE_VA={macro_rmse:.6f}"
        )
        print()

        if macro_rmse < best_macro_rmse:

            best_macro_rmse = (
                macro_rmse
            )

            best_step = step

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "step": step,
                    "macro_rmse": (
                        macro_rmse
                    ),
                    "view": args.view,
                    "seed": args.seed,
                    "model_name": (
                        args.model_name
                    ),
                },
                checkpoint_path,
            )

            save_json(
                output_dir
                / "best_metrics.json",
                {
                    "step": step,
                    "metrics": metrics,
                },
            )

            save_predictions(
                output_dir,
                predictions_json,
            )

            print(
                f"NEW BEST: "
                f"step={step}, "
                f"macro RMSE="
                f"{macro_rmse:.6f}"
            )
            print()

    # --------------------------------------------------------------
    # Reload best checkpoint and verify
    # --------------------------------------------------------------

    if not checkpoint_path.exists():
        raise RuntimeError(
            "No checkpoint was saved"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    final_metrics, final_predictions = (
        evaluate(
            model=model,
            loader=dev_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            dev_records_by_domain=(
                dev_records_by_domain
            ),
        )
    )

    elapsed_seconds = (
        time.time() - start_time
    )

    final_report = {
        "view": args.view,
        "seed": args.seed,
        "best_step": best_step,
        "best_macro_rmse": (
            best_macro_rmse
        ),
        "metrics": final_metrics,
        "elapsed_seconds": (
            elapsed_seconds
        ),
    }

    save_json(
        output_dir
        / "final_report.json",
        final_report,
    )

    save_predictions(
        output_dir,
        final_predictions,
    )

    print()
    print("=" * 72)
    print("FINAL BEST RESULT")
    print("=" * 72)
    print(
        f"view      : {args.view}"
    )
    print(
        f"seed      : {args.seed}"
    )
    print(
        f"best step : {best_step}"
    )
    print(
        f"macro RMSE: "
        f"{best_macro_rmse:.6f}"
    )

    for domain in args.domains:
        m = final_metrics[domain]

        print(
            f"{domain:10s} "
            f"RMSE_VA={m['RMSE_VA']:.6f} "
            f"RMSE_V={m['RMSE_V']:.6f} "
            f"RMSE_A={m['RMSE_A']:.6f}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
