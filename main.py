#!/usr/bin/env python3
"""BRIDGE — Indian Sign Language recognition.

Two commands:

    python main.py train      train the transformer on the INCLUDE dataset
    python main.py run        run the model live (trains first if untrained)

The model code itself is a straight port of isl-transformer-openhand.ipynb.
"""

import argparse
import sys

from src import config


def cmd_train(args):
    from src.train import train

    model, class_names, val_loader, best_val, history = train(
        epochs=args.epochs,
        rebuild_data=args.rebuild_data,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
    )

    if not args.no_eval:
        from src.evaluate import evaluate

        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        evaluate(model=model, val_loader=val_loader, class_names=class_names,
                 plots=not args.no_plots, show=False if args.no_show else None, history=history)

    return 0


def cmd_run(args):
    # The whole point of `run`: use the trained model, or train one first.
    if not config.MODEL_BUNDLE.exists():
        print(f"No trained model at {config.MODEL_BUNDLE}.")
        if args.no_train:
            print("Passed --no-train, so stopping. Run `python main.py train` first.")
            return 1
        print("Training one now — this takes a while.\n")
        from src.train import train

        train(epochs=args.epochs)
        print("\nTraining finished. Starting live recognition.\n")
    else:
        print(f"Using trained model: {config.MODEL_BUNDLE}")

    if args.video:
        from src.infer import predict_video

        predict_video(args.video, topk=args.topk, swap_hands=args.swap_hands)
        return 0

    from src.infer import run_realtime

    run_realtime(
        source=args.source,
        topk=args.topk,
        threshold=args.threshold,
        swap_hands=args.swap_hands,
        mirror=not args.no_mirror,
        idle_reset=args.idle_reset,
        stability=args.stability,
    )
    return 0


def cmd_eval(args):
    from src.evaluate import evaluate

    if not config.MODEL_BUNDLE.exists():
        print(f"No trained model at {config.MODEL_BUNDLE}. Run `python main.py train` first.")
        return 1
    evaluate(plots=not args.no_plots, show=False if args.no_show else None)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="bridge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    # --- train ---
    t = sub.add_parser("train", help="train the sign transformer on the INCLUDE dataset")
    t.add_argument("--epochs", type=int, default=config.EPOCHS)
    t.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    t.add_argument("--lr", type=float, default=config.LR)
    t.add_argument("--rebuild-data", action="store_true",
                   help="re-encode the .pkl dataset instead of using cached X.npy/y.npy")
    t.add_argument("--resume", action="store_true", help="warm-start from the best checkpoint")
    t.add_argument("--no-eval", action="store_true", help="skip evaluation after training")
    t.add_argument("--no-plots", action="store_true", help="skip writing plots to results/")
    t.add_argument("--no-show", action="store_true",
                   help="save charts to results/ without opening them on screen")
    t.set_defaults(func=cmd_train)

    # --- run ---
    r = sub.add_parser("run", help="run the trained model (trains first if none exists)")
    r.add_argument("--source", default="0",
                   help="camera index (0) or stream URL, e.g. http://10.0.0.5:8080/video")
    r.add_argument("--video", help="classify a video file instead of a live camera")
    r.add_argument("--topk", type=int, default=3, help="how many predictions to show")
    r.add_argument("--threshold", type=float, default=0.35,
                   help="minimum confidence for a window to count toward a commit")
    r.add_argument("--stability", type=int, default=5,
                   help="consecutive windows that must agree before a sign is committed "
                        "(lower = faster but twitchier)")
    r.add_argument("--idle-reset", type=int, default=8,
                   help="frames with no hands visible that end a sign and clear the window")
    r.add_argument("--swap-hands", action="store_true",
                   help="swap left/right hand blocks if predictions look mirrored")
    r.add_argument("--no-mirror", action="store_true", help="don't mirror the preview window")
    r.add_argument("--no-train", action="store_true",
                   help="fail instead of training when no model is found")
    r.add_argument("--epochs", type=int, default=config.EPOCHS,
                   help="epochs to use if training is triggered")
    r.set_defaults(func=cmd_run)

    # --- eval ---
    e = sub.add_parser("eval", help="score the trained model on the validation split")
    e.add_argument("--no-plots", action="store_true", help="skip writing plots to results/")
    e.add_argument("--no-show", action="store_true",
                   help="save charts to results/ without opening them on screen")
    e.set_defaults(func=cmd_eval)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
