import numpy as np
def play_game():
    print("Welcome to the game!")

def main():
    play_game()
    print("Game over!")

if __name__ == "__main__":
    main()
    np.random.seed(42)  # Set a random seed for reproducibility
    print("Random seed set to 42 for reproducibility.")
    