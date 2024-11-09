class Grid(object):

    def __init__(self):
        self.content = None
        self.filled = False

    def set(self, content):
        if content == 'L':
            # self.content = '\\'
            self.content = 'L'
            self.filled = True
        elif content == 'R':
            # self.content = '/'
            self.content = 'R'
            self.filled = True
        else:
            self.content = content

    def get(self):
        return self.content

    def __str__(self):
        return '|%s|' % self.content


class Board(object):

    def __init__(self, board_str):
        self.dim_x, self.dim_y = self.calc_dim(board_str)
        self.g_count, self.v_count, self.z_count = self.calc_monster_count(board_str)
        self.board = []
        self._init_board()
        self.generate_board(board_str)
        self.north_count, self.east_count, self.south_count, self.west_count = self.calc_board_count(board_str)

    @staticmethod
    def calc_dim(board_str):
        """
        Given a string converts it to dimensions 
        :return: x, y
        """
        dims = []
        for dim in board_str.split(':')[0].split('x'):
            dims.append(int(dim))
        
        return dims[0], dims[1]

    @staticmethod
    def calc_monster_count(board_str):
        board_split = board_str.split(':')[1].split(',')
        ghost_count = int(board_split[0])
        vampire_count = int(board_split[1])
        zombie_count = int(board_split[2])

        return ghost_count, vampire_count, zombie_count

    def calc_board_count(self, board_str):
        board_split = board_str.split(':')[1].split(',')

        board_count_start = 4

        top_row = board_split[board_count_start: board_count_start + self.dim_x]
        right_col = board_split[board_count_start + self.dim_x: board_count_start + self.dim_x + self.dim_y]

        bottom_row = board_split[board_count_start + self.dim_x + self.dim_y: board_count_start + (self.dim_x * 2) + self.dim_y]
        left_col = board_split[board_count_start + (self.dim_x * 2) + self.dim_y: board_count_start + (self.dim_x * 2) + (self.dim_y * 2)]

        return top_row, right_col, bottom_row, left_col

    def _init_board(self):
        for x in range(0, self.dim_x):
            self.board.append([])
            for y in range(0, self.dim_y):
                empty_grid = Grid()
                self.board[x].append(empty_grid)
        

    def generate_board(self, board_str):
        board_split = board_str.split(':')[1].split(',')
        board_layout = board_split[3]

        x = 0
        y = 0


        for space in board_layout:
            if space == "L" or space == "R":
                self.board[x][y].set(space)
                if y < self.dim_y - 1:
                    y += 1
                else:
                    y = 0
                    x += 1
            else:
                for count in range(0, (ord(space) - 96)):
                    self.board[x][y].set(' ')
                    if y < self.dim_y - 1:
                        y += 1
                    else:
                        y = 0
                        x += 1    

    def print_board(self):
        x_str = []
        y_str = []
        for x in range(0, self.dim_x):
            for y in range(0, self.dim_y):
                y_str.append(str(self.board[x][y]))
            print(''.join(y_str))
            y_str = []


class Walker(object):
    def __init__(self):
        self.current_x = 0
        self.current_y = 0

    def walk(self, board, row, col, direction, path=[]):
        path.append((row, col))
        <vul/>print(row, col)</vul>
        if board.board[row][col].get() == 'L':
            direction = self.left_bounce(direction)
        elif board.board[row][col].get() == 'R':
            direction = self.right_bounce(direction)

        if direction == 'north':
            # The top left of the array is considered to be 0. 0
            # as we go down we are incrementing col
            row -= 1
        elif direction == 'east':
            col += 1
        elif direction == 'south':
            row += 1
        elif direction == 'west':
            col -= 1

        if (row >= 0 and row < board.dim_x) and (col >= 0 and col < board.dim_y):
            self.walk(board, row, col, direction)

    def solve(self):
        pass

    @staticmethod
    def right_bounce(direction):
        if direction == 'north':
            return 'east'
        elif direction == 'east':
            return 'north'
        elif direction == 'south':
            return 'west'
        elif direction == 'west':
            return 'south'
        else:
            raise ValueError('%s not a valid direction' % direction) 

    @staticmethod
    def left_bounce(direction):
        if direction == 'north':
            return 'west'
        elif direction == 'west':
            return 'north'
        elif direction == 'south':
            return 'east'
        elif direction == 'east':
            return 'south'
        else:
            raise ValueError('%s not a valid direction' % direction) 


if __name__ == "__main__":
    test_link = "https://www.chiark.greenend.org.uk/~sgtatham/puzzles/js/undead.html#4x4:5,2,4,cRdRLbLbR,2,3,1,3,3,3,1,0,0,1,4,0,0,2,3,1"
    board_txt = test_link.split('#')[-1]
    board = Board(board_txt)
    print('printing board out')
    board.print_board()

    walkman = Walker()
    row = 0
    col = 0
    walkman.walk(board, row, col, 'east')