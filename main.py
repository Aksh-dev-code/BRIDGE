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
        split=args.split,
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
        margin=args.margin,
        swap_hands=args.swap_hands,
        mirror=not args.no_mirror,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        end_silence=args.end_silence,
        debug=args.debug,
    )
    return 0


def cmd_eval(args):
    from src.evaluate import evaluate

    if not config.MODEL_BUNDLE.exists():
        print(f"No trained model at {config.MODEL_BUNDLE}. Run `python main.py train` first.")
        return 1
    evaluate(plots=not args.no_plots, show=False if args.no_show else None)
    return 0


def cmd_diagnose(args):
    from src.diagnose import diagnose

    diagnose(n=args.samples, seconds=args.seconds, source=args.source,
             skip_live=args.no_camera)
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
    t.add_argument("--split", choices=("auto", "official", "random"), default="auto",
                   help="auto: use INCLUDE's official signer-disjoint split when it's "
                        "available, else random; official: require it; random: force the "
                        "old random stratified split")
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
    r.add_argument("--threshold", type=float, default=0.5,
                   help="minimum smoothed confidence to commit a sign")
    r.add_argument("--margin", type=float, default=0.15,
                   help="how far the top guess must beat the runner-up to commit")
    r.add_argument("--min-duration", type=float, default=0.5,
                   help="shortest gesture that counts as a sign (seconds)")
    r.add_argument("--max-duration", type=float, default=6.0,
                   help="a gesture is force-ended after this long (seconds)")
    r.add_argument("--end-silence", type=float, default=0.5,
                   help="seconds with hands down that end the current sign")
    r.add_argument("--debug", action="store_true",
                   help="show live fps / buffer / hand-coverage numbers on the preview")
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

    # --- diagnose ---
    d = sub.add_parser("diagnose",
                       help="compare training features against live camera features")
    d.add_argument("--samples", type=int, default=40, help="how many .pkl files to inspect")
    d.add_argument("--seconds", type=float, default=6.0, help="seconds of camera to capture")
    d.add_argument("--source", default="0", help="camera index or stream URL")
    d.add_argument("--no-camera", action="store_true",
                   help="dataset stats only, don't open the camera")
    d.set_defaults(func=cmd_diagnose)

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
