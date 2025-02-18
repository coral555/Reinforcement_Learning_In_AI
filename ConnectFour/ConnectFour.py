from MCTSPlayer import MCTSPlayer
class ConnectFour:

    RED = 1
    YELLOW = -1
    EMPTY = 0

    RED_WIN = 1
    YELLOW_WIN = -1
    DRAW = 0
    ONGOING = -17

    def __init__(self):
        self.board = [[self.EMPTY for _ in range(6)] for _ in range(7)]
        self.heights = [0 for _ in range(7)]  
        self.player = self.RED
        self.status = self.ONGOING

    def legal_moves(self):
        return [i for i in range(7) if self.heights[i] < 6]

    def make(self, move):  
        self.board[move][self.heights[move]] = self.player
        self.heights[move] += 1
        if self.winning_move(move):
            self.status = self.player
        elif len(self.legal_moves()) == 0:
            self.status = self.DRAW
        else:
            self.player = self.other(self.player)

    def other(self, player):
        return self.RED if player == self.YELLOW else self.YELLOW

    def unmake(self, move):
        self.heights[move] -= 1
        self.board[move][self.heights[move]] = self.EMPTY
        self.player = self.other(self.player)
        self.status = self.ONGOING

    def clone(self):
        clone = ConnectFour()
        clone.board = [col[:] for col in self.board]  
        clone.heights = self.heights[:]  
        clone.player = self.player
        clone.status = self.status
        return clone

    def winning_move(self, move):
        col = move
        row = self.heights[col] - 1
        player = self.board[col][row]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 0
            x, y = col + dx, row + dy
            while 0 <= x < 7 and 0 <= y < 6 and self.board[x][y] == player:
                count += 1
                x += dx
                y += dy
            x, y = col - dx, row - dy
            while 0 <= x < 7 and 0 <= y < 6 and self.board[x][y] == player:
                count += 1
                x -= dx
                y -= dy
            if count >= 3:
                return True
        return False

    def __str__(self):
        rows = []
        for r in range(5, -1, -1):  
            row = []
            for c in range(7):
                if self.board[c][r] == self.RED:
                    row.append("R")
                elif self.board[c][r] == self.YELLOW:
                    row.append("Y")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

def main():
    game = ConnectFour()
    mcts_player = MCTSPlayer(iters=10000)
    while game.status == game.ONGOING:
        print(game)

        if game.player == ConnectFour.RED:
            move = mcts_player.select_move(game)
        else:
            try:
                move = int(input("Enter a column (0-6): "))
                if move not in game.legal_moves():
                    print("Illegal move. Try again.")
                    continue
            except ValueError:
                print("Invalid input. Enter a number between 0 and 6.")
                continue

        game.make(move)

    print(game)
    if game.status == ConnectFour.RED:
        print("RED wins!")
    elif game.status == ConnectFour.YELLOW:
        print("YELLOW wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()