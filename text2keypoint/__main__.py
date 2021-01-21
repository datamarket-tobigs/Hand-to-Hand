from modeling.helpers import load_config
from train import Train
from test import Test


def main():
    """
    Launch Train or Test depending on the "mode" in config file(Base.yaml).
    """

    # Load config dictionary
    config_path = "./Base.yaml"
    cfg = load_config(config_path)
    
    # Launch Train
    if cfg["mode"] == "Train":
        Train(cfg = cfg)
    # Launch Test
    elif cfg["mode"] == "Test":
        Test(cfg = cfg)

if __name__ == "__main__":
    main()
