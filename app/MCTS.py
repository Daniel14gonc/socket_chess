import math
from app.utils import move_to_index

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0

    def expand(self, policy):
        for action in self.state.get_legal_moves():
            if str(action) not in self.children:
                new_state = self.state.copy()
                new_state.step(action)
                self.children[str(action)] = MCTSNode(new_state, parent=self, action=action)
                self.children[str(action)].prior = policy[move_to_index(action, self.state.get_board())]

    def select_child(self, c_puct=1.0):
        """
        Selecciona el mejor hijo basándose en el puntaje UCB.
        """
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            ucb_score = child.get_ucb_score(c_puct)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        return best_child

    def get_ucb_score(self, c_puct):
        u = (c_puct * self.prior * math.sqrt(self.parent.visit_count) /
             (1 + self.visit_count))
        q = self.value_sum / self.visit_count if self.visit_count > 0 else 0
        return q + u

    def update(self, value):
        self.visit_count += 1
        self.value_sum += value

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None