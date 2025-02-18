import copy  # Importing the copy module to enable deep copying of game states

class SnortGame:
    """
    Represents the game Snort.
    This class implements the Snort game, managing the board, moves, and game status.

    The game is played on a square grid, and players take turns placing their pieces.
    A move is legal if it does not place a piece adjacent to an opponent’s piece.
    The player who cannot make a legal move loses the game.
    
    Attributes:
    - size (int): The board size (default is 10x10).
    - board (list of lists): The game board represented as a 2D list.
    - current_player (str): Indicates whose turn it is ('R' for Red, 'B' for Blue).
    - blocked_positions (set): Set of positions that are initially blocked.
    - game_status (str): The status of the game ('ongoing', 'R_wins', 'B_wins').
    - _legal_moves_cache (list): Cached list of legal moves for optimization.
    - _has_legal_moves_cache (bool): Cached boolean indicating if legal moves exist.
    """

    def __init__(self, size=10):
        """
        Initializes a new Snort game.

        Parameters:
        - size (int): The size of the board (default is 10x10).
        """
        self.size = size
        self.board = [[' ' for _ in range(size)] for _ in range(size)]  # Initialize an empty board
        self.current_player = 'R'  # Red plays first
        self.blocked_positions = set()  # Positions that are blocked at the start
        self.initialize_blocked_positions()  # Randomly place blocked positions
        self.game_status = 'ongoing'  # The game starts as ongoing

        # Caching mechanisms to improve performance
        self._legal_moves_cache = None
        self._has_legal_moves_cache = None

    def initialize_blocked_positions(self):
        """
        Randomly selects 3 blocked positions on the board and marks them with 'X'.
        """
        import random
        while len(self.blocked_positions) < 3:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if (x, y) not in self.blocked_positions:
                self.blocked_positions.add((x, y))
                self.board[x][y] = 'X'  # Mark the position as blocked

    def make_move(self, move):
        """
        Applies a move to the game state.

        Parameters:
        - move (tuple): The (x, y) position to place the piece.

        Returns:
        - bool: True if the move was successful, False if it was invalid.
        """
        x, y = move
        if self.is_valid_move(x, y):
            self.board[x][y] = self.current_player  # Place the piece
            self.current_player = 'B' if self.current_player == 'R' else 'R'  # Switch turns

            # Reset cached legal moves
            self._legal_moves_cache = None
            self._has_legal_moves_cache = None

            # Check if the next player has any legal moves
            if not self.has_legal_moves():
                self.game_status = 'R_wins' if self.current_player == 'B' else 'B_wins'
            return True
        return False

    def unmake_move(self, move):
        """
        Reverses a move, removing the piece and restoring the previous state.

        Parameters:
        - move (tuple): The (x, y) position of the move to undo.
        """
        x, y = move
        if self.board[x][y] in ['R', 'B']:
            self.current_player = self.board[x][y]  # Restore the player turn
            self.board[x][y] = ' '  # Clear the position
            self.game_status = 'ongoing'  # Reset game status

            # Reset cached legal moves
            self._legal_moves_cache = None
            self._has_legal_moves_cache = None

    def clone(self):
        """
        Creates a deep copy of the current game state.

        Returns:
        - SnortGame: A new SnortGame object with the same state.
        """
        return copy.deepcopy(self)

    def encode(self):
        """
        Encodes the game state as a list of numbers.

        Returns:
        - list: Encoded state representation.
        """
        flat_board = [1 if cell == 'R' else 2 if cell == 'B' else 0 for row in self.board for cell in row]
        player_turn = [1 if self.current_player == 'R' else 2]  # Encode the current player
        status = [0 if self.game_status == 'ongoing' else 1 if self.game_status == 'R_wins' else 2]  # Encode status
        return flat_board + player_turn + status  # Return the encoded state

    def decode(self, action_index):
        """
        Converts an action index into a (x, y) board position.

        Parameters:
        - action_index (int): The index of the action.

        Returns:
        - tuple: (x, y) position on the board.
        """
        x = action_index // self.size
        y = action_index % self.size
        return x, y

    def status(self):
        """
        Returns the current game status.

        Returns:
        - str: 'ongoing', 'R_wins', or 'B_wins'.
        """
        return self.game_status

    def legal_moves(self):
        """
        Returns a list of all legal moves for the current player.
        Uses caching to optimize repeated queries.

        Returns:
        - list: A list of (x, y) tuples representing valid moves.
        """
        if self._legal_moves_cache is None:
            self._legal_moves_cache = [
                (x, y) for x in range(self.size) for y in range(self.size) if self.is_valid_move(x, y)
            ]
        return self._legal_moves_cache

    def is_valid_move(self, x, y):
        """
        Determines if a move is legal.

        Parameters:
        - x (int): Row index.
        - y (int): Column index.

        Returns:
        - bool: True if the move is valid, False otherwise.
        """
        if not (0 <= x < self.size and 0 <= y < self.size):  # Check if within board bounds
            return False
        if (x, y) in self.blocked_positions or self.board[x][y] != ' ':  # Check if occupied or blocked
            return False

        adjacent_positions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # Get adjacent positions
        opponent = 'B' if self.current_player == 'R' else 'R'  # Determine the opponent's piece

        # If an adjacent position contains an opponent's piece, the move is invalid
        return not any(
            0 <= i < self.size and 0 <= j < self.size and self.board[i][j] == opponent
            for i, j in adjacent_positions
        )

    def has_legal_moves(self):
        """
        Checks if the current player has any legal moves.

        Returns:
        - bool: True if legal moves exist, False otherwise.
        """
        if self._has_legal_moves_cache is None:
            self._has_legal_moves_cache = bool(self.legal_moves())
        return self._has_legal_moves_cache

    def display_board(self):
        """
        Displays the current board state in the console.
        """
        print("  " + " ".join(str(i) for i in range(self.size)))
        for idx, row in enumerate(self.board):
            print(f"{idx} " + " ".join(row))
        print()

    def play(self, player):
        """
        Runs a game with a given player.

        Parameters:
        - player: The player object that selects moves.
        """
        print("Welcome to Snort!")
        self.display_board()
        while self.status() == 'ongoing':
            print(f"Current player: {'Red' if self.current_player == 'R' else 'Blue'}")
            move = player.select_move(self)
            if self.make_move(move):
                self.display_board()
            else:
                print("Invalid move. The player attempted an illegal move.")

        winner = 'Red' if self.game_status == 'R_wins' else 'Blue'
        print(f"Game over! {winner} wins!")
