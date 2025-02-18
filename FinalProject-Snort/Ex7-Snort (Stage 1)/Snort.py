import copy

class SnortGame:
    def __init__(self, size=10):
        """
        Initialize a new game.
        :param size: Board size (default is 10x10)
        """
        self.size = size  # Store board size
        self.board = [[' ' for _ in range(size)] for _ in range(size)]  # Initialize empty board
        self.current_player = 'R'  # 'R' for Red player, 'B' for Blue player
        self.blocked_positions = set()  # Set to track blocked positions
        self.initialize_blocked_positions()  # Randomly place blocked positions
        self.game_status = 'ongoing'  # Track game state: 'ongoing', 'R_wins', 'B_wins'

    def initialize_blocked_positions(self):
        """
        Initialize 3 random blocked squares on the board.
        """
        import random
        while len(self.blocked_positions) < 3:  # Ensure exactly 3 blocked squares
            x = random.randint(0, self.size - 1)  # Random row index
            y = random.randint(0, self.size - 1)  # Random column index
            if (x, y) not in self.blocked_positions:  # Avoid duplicates
                self.blocked_positions.add((x, y))  # Add to blocked positions set
                self.board[x][y] = 'X'  # Mark blocked position on the board

    def make_move(self, move):
        """
        Apply a move to the current game state.
        :param move: tuple (x, y) representing the move
        :return: True if the move is valid and applied, else False
        """
        x, y = move  # Extract move coordinates
        if self.is_valid_move(x, y):  # Check if move is valid
            self.board[x][y] = self.current_player  # Mark move on the board
            # Switch turns: 'R' -> 'B', 'B' -> 'R'
            self.current_player = 'B' if self.current_player == 'R' else 'R'
            if not self.has_legal_moves():  # Check if the opponent has any valid moves
                self.game_status = 'R_wins' if self.current_player == 'B' else 'B_wins'  # Determine winner
            return True  # Move successfully applied
        return False  # Invalid move

    def unmake_move(self, move):
        """
        Reverse a move, removing a piece from the board and restoring the previous state.
        :param move: tuple (x, y) representing the move to reverse
        """
        x, y = move  # Extract move coordinates
        if self.board[x][y] in ['R', 'B']:  # Check if a piece was placed here
            self.current_player = self.board[x][y]  # Restore previous player's turn
            self.board[x][y] = ' '  # Remove the piece from the board
            self.game_status = 'ongoing'  # Reset game status to ongoing

    def clone(self):
        """
        Create a deep copy of the current game state.
        :return: A new SnortGame object with the same state
        """
        return copy.deepcopy(self)  # Use deep copy to duplicate the entire game state

    def encode(self):
        """
        Encode the game state as a binary vector.
        :return: List representing the board state, current player's turn, and game status
        """
        # Convert board state to a flat list: 1 for 'R', 2 for 'B', 0 for empty spaces
        flat_board = [1 if cell == 'R' else 2 if cell == 'B' else 0 for row in self.board for cell in row]
        # Encode the current player's turn: 1 for 'R', 2 for 'B'
        player_turn = [1 if self.current_player == 'R' else 2]
        # Encode game status: 0 for 'ongoing', 1 for 'R_wins', 2 for 'B_wins'
        status = [0 if self.game_status == 'ongoing' else 1 if self.game_status == 'R_wins' else 2]
        return flat_board + player_turn + status  # Return full encoded state

    def decode(self, action_index):
        """
        Translate an action index into a move in the game.
        :param action_index: Index of the move
        :return: tuple (x, y) representing the move
        """
        x = action_index // self.size  # Compute row index
        y = action_index % self.size  # Compute column index
        return x, y  # Return move coordinates

    def status(self):
        """
        Return the game result if the game is over, or indicate that the game is ongoing.
        :return: 'ongoing', 'R_wins', or 'B_wins'
        """
        return self.game_status  # Return current game status

    def legal_moves(self):
        """
        Return a list of the legal moves in the current position.
        :return: List of tuples representing legal moves
        """
        moves = []
        for x in range(self.size):  # Iterate through board rows
            for y in range(self.size):  # Iterate through board columns
                if self.is_valid_move(x, y):  # Check if move is valid
                    moves.append((x, y))  # Add to list of legal moves
        return moves  # Return all legal moves

    def is_valid_move(self, x, y):
        """
        Check the validity of a move.
        :param x: Row index
        :param y: Column index
        :return: True if the move is valid, else False
        """
        if not (0 <= x < self.size and 0 <= y < self.size):  # Ensure move is within board bounds
            return False
        if self.board[x][y] != ' ':  # Check if the cell is empty
            return False
        adjacent_positions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # Get adjacent cells
        for (i, j) in adjacent_positions:
            if 0 <= i < self.size and 0 <= j < self.size:
                if self.board[i][j] == ('B' if self.current_player == 'R' else 'R'):  # Check adjacency rules
                    return False
        return True  # Move is valid

    def has_legal_moves(self):
        """
        Check if there are any legal moves left for the current player.
        :return: True if there are legal moves, else False
        """
        return any(self.is_valid_move(x, y) for x in range(self.size) for y in range(self.size))  # Check all board cells

    def display_board(self):
        """
        Display the current board state.
        """
        print("  " + " ".join(str(i) for i in range(self.size)))  # Print column indices
        for idx, row in enumerate(self.board):  # Iterate over board rows
            print(f"{idx} " + " ".join(row))  # Print row index and board row contents
        print()  # Add spacing for readability

    def play(self):
        """
        Manage the game flow, including user input, board display, and declaring the winner at the end.
        """
        print("Welcome to Snort!")  # Print game start message
        self.display_board()  # Show initial board state
        while self.status() == 'ongoing':  # Continue until game is over
            print(f"Current player: {'Red' if self.current_player == 'R' else 'Blue'}")  # Indicate current player
            try:
                x = int(input("Enter row number: "))  # Get row input
                y = int(input("Enter column number: "))  # Get column input
                if self.make_move((x, y)):  # Attempt move
                    self.display_board()  # Show updated board
                else:
                    print("Invalid move. Try again.")  # Handle invalid move
            except ValueError:
                print("Invalid input. Please enter numbers.")  # Handle invalid input
        winner = 'Red' if self.game_status == 'R_wins' else 'Blue'  # Determine winner
        print(f"Game over! {winner} wins!")  # Display winner message

if __name__ == "__main__":
    game = SnortGame()
    game.play()
