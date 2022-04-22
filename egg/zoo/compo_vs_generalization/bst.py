class BinarySearchTree:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def insert(self, value):
        if self.value is None:
            self.value = value
            return
        if value > self.value:
            if self.right is None:
                self.right = BinarySearchTree(value)
            else:
                self.right.insert(value)
        else:
            if self.left is None:
                self.left = BinarySearchTree(value)
            else:
                self.left.insert(value)

    def __contains__(self, value):
        if self.value is None:
            return False
        if value == self.value:
            return True
        else:
            if value > self.value:
                if self.right is not None:
                    return (value in self.right)
                else:
                    return False
            else:
                if self.left is not None:
                    return (value in self.left)
                else:
                    return False

