import argparse

from src.training.config import load_config


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 50)
    print("TRAINING")
    print("=" * 50)

    print(
        f"Project      : "
        f"{config['project']['name']}"
    )

    print(
        f"Version      : "
        f"{config['project']['version']}"
    )

    print(
        f"Backbone     : "
        f"{config['model']['backbone']}"
    )

    print(
        f"Train Dataset: "
        f"{config['dataset']['train']}"
    )

    print(
        f"Epochs       : "
        f"{config['training']['epochs']}"
    )

    print(
        f"Batch Size   : "
        f"{config['training']['batch_size']}"
    )

    print(
        f"Learning Rate: "
        f"{config['training']['learning_rate']}"
    )


if __name__ == "__main__":
    main()