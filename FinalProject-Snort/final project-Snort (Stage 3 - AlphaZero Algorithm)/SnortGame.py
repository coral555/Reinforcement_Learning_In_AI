import random
import copy

class SnortGame:
    ONGOING = "ongoing"
    R_WINS = "R_wins"
    B_WINS = "B_wins"
    
    def __init__(self, size=5):
        self.size = size
        self.board = [[' ' for _ in range(size)] for _ in range(size)]
        self.current_player = 'R'
        self.blocked_positions = set()
        self.initialize_blocked_positions()
        self.game_status = SnortGame.ONGOING
        self._legal_moves_cache = None
        self._has_legal_moves_cache = None

    def initialize_blocked_positions(self):
        while len(self.blocked_positions) < 3:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if (x, y) not in self.blocked_positions:
                self.blocked_positions.add((x, y))
                self.board[x][y] = 'X'

    def clone(self):
        return copy.deepcopy(self)

    def make_move(self, move):
        #print(f"Move received: {move}") 
        x, y = move
        if self.is_valid_move(x, y):
            self.board[x][y] = self.current_player
            self.current_player = 'B' if self.current_player == 'R' else 'R'
            self._legal_moves_cache = None
            self._has_legal_moves_cache = None
            if not self.has_legal_moves():
                self.game_status = SnortGame.R_WINS if self.current_player == 'B' else SnortGame.B_WINS
            return True
        return False

    def unmake_move(self, move):
        x, y = move
        if self.board[x][y] in ['R', 'B']:
            self.current_player = self.board[x][y]
            self.board[x][y] = ' '
            self.game_status = SnortGame.ONGOING
            self._legal_moves_cache = None
            self._has_legal_moves_cache = None

    def encode(self):
        flat_board = [1 if cell == 'R' else 2 if cell == 'B' else 0 for row in self.board for cell in row]
        player_turn = [1 if self.current_player == 'R' else 2]
        status = [0 if self.game_status == SnortGame.ONGOING else 1 if self.game_status == SnortGame.R_WINS else 2]
        return flat_board + player_turn + status

    def status(self):
        return self.game_status

    def legal_moves(self):
        if self._legal_moves_cache is None:
            self._legal_moves_cache = [(x, y) for x in range(self.size) for y in range(self.size) if self.is_valid_move(x, y)]
        return self._legal_moves_cache

    def is_valid_move(self, x, y):
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        if (x, y) in self.blocked_positions or self.board[x][y] != ' ':
            return False
        adjacent_positions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        opponent = 'B' if self.current_player == 'R' else 'R'
        return not any(0 <= i < self.size and 0 <= j < self.size and self.board[i][j] == opponent for i, j in adjacent_positions)

    def has_legal_moves(self):
        if self._has_legal_moves_cache is None:
            self._has_legal_moves_cache = bool(self.legal_moves())
        return self._has_legal_moves_cache

    def display_board(self):
        print("  " + " ".join(str(i) for i in range(self.size)))
        for idx, row in enumerate(self.board):
            print(f"{idx} " + " ".join(row))
        print()

    def play(self, player):
        print("Welcome to Snort!")
        self.display_board()
        
        while self.status() == SnortGame.ONGOING:
            print(f"Current player: {'Red' if self.current_player == 'R' else 'Blue'}")
            move_selection = player.select_move(self) 
            if isinstance(move_selection, tuple) and len(move_selection) == 2:
                move = move_selection[0] if isinstance(move_selection[0], tuple) else move_selection
            else:
                print(f" ERROR: Invalid move received: {move_selection}. Skipping turn.")
                continue

            if self.make_move(move):
                self.display_board()
            else:
                print("Invalid move. The player attempted an illegal move.")

        winner = 'Red' if self.game_status == SnortGame.R_WINS else 'Blue'
        print(f"Game over! {winner} wins!")