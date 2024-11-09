import undead
import unittest

class TestWalker(unittest.TestCase):

    def setUp(self):
        test_link = "https://www.chiark.greenend.org.uk/~sgtatham/puzzles/js/undead.html#4x4:5,2,4,cRdRLbLbR,2,3,1,3,3,3,1,0,0,1,4,0,0,2,3,1"
        board_txt = test_link.split('#')[-1]
        <fix/>self.board = undead.Board(board_txt)</fix>

    def test_walker(self):
        <fix/>walkman = undead.Walker()
        row = 0</fix>
        col = 0
        <fix/>actual = walkman.walk(self.board, row, col, 'east')

        expected = [(0, 0), (0, 1), (0, 2), (0, 3)]

        self.assertEqual(actual, expected)</fix>


if __name__ == "__main__":
    unittest.main()