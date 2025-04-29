from SnortGame import SnortGame
from PUCTPlayer import PUCTPlayer
from MCTSPlayer import MCTSPlayer
from GameNetwork import GameNetwork
from Trainer import Trainer
from tqdm import tqdm  


### open the terminal and run python main.py ###
### notice that we saved the data, so you can use it again because it required a lot of time ###
def main():
    print("Welcome to Snort Game!")

    try:
        size = int(input("Enter board size (default is 5): ").strip() or 5)
    except ValueError:
        print("Invalid input! Defaulting to size 5.")
        size = 5

    print("\n🛠 Select player type:")
    print("1. MCTS (Monte Carlo Tree Search) - Debug mode")
    print("2. PUCT (Policy Upper Confidence Trees) - Uses a neural network")
    player_choice= input("Enter 1 or 2: ").strip()

    game =SnortGame(size=size)

    if player_choice == "1":
        print("\nRunning with MCTSPlayer (No neural network)")
        try:
            iters = int(input("Enter number of MCTS iterations (default 10000): ").strip() or 10000)
        except ValueError:
            print("Invalid input! Defaulting to 10,000 iterations.")
            iters=10000
        player = MCTSPlayer(iters=iters)

    elif player_choice == "2":
        print("\nRunning with PUCTPlayer (Neural Network)")
        network = GameNetwork(input_size=(size * size + 2), action_size=size * size)

        load_model = input("Load pre-trained model? (y/n): ").strip().lower()
        if load_model == "y":
            model_path = input("Enter model path (default: snort_model.pth): ").strip() or "snort_model.pth"
            try:
                network.load_model(model_path)
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                retry = input("Do you want to train a new model instead? (y/n): ").strip().lower()
                if retry != "y":
                    print("Exiting program.")
                    return

        else:
            print("\nTraining a new neural network...")
            try:
                train_iters =int(input("Enter training iterations (default 1000): ").strip() or 1000)
            except ValueError:
                print("Invalid input! Defaulting to 1000 iterations.")
                train_iters = 1000

            trainer =Trainer(network, game, iters=train_iters, lr=0.001)
            with tqdm(total=train_iters) as pbar:  
                trainer.train(update_callback=lambda: pbar.update(1))
            network.save_model("snort_model.pth")
            print(" New model trained and saved as 'snort_model.pth'.")

        player=PUCTPlayer(network, iters=1000)

    else:
        print("\n Invalid choice! Defaulting to MCTSPlayer.")
        player=MCTSPlayer(iters=1000)

    print("\n Starting Game...")
    game.play(player)

if __name__ == "__main__":
    main()
