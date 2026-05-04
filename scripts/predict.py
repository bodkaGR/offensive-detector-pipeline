from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import Settings
from src.inference.predictor import OffensiveTextPredictor


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

def format(result: dict) -> str:
    icon = "OFFENSIVE" if result["is_offensive"] else "CLEAN   "
    return (
        f"\n{'─' * 56}\n  {'[!]' if result['is_offensive'] else '[ok]'}  "
        f"{icon}  (confidence: {result['confidence']:.1%})\n{'─' * 56}\n"
        f"  Text        : {result['text'][:70]}\n"
        f"  P(offensive): {result['p_offensive']:.4f}\n"
        f"  P(clean)    : {result['p_clean']:.4f}\n"
    )

def main(args: argparse.Namespace) -> None:
    cfg = Settings()
    setup_logging()

    predictor = OffensiveTextPredictor.from_checkpoint(
        model_path=cfg.paths.model_checkpoint,
        tokenizer_path=cfg.paths.tokenizer_dir,
        inference_cfg=cfg.inference,
        model_cfg=cfg.model,
    )

    if args.text:
        for result in predictor.predict(args.text):
            print(format(result))
    elif args.input_csv:
        predictor.predict_csv(args.input_csv, args.output_csv)
    else:
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",       nargs="+", type=str)
    parser.add_argument("--input_csv",  type=str)
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    main(parser.parse_args())