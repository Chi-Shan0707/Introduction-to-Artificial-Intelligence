from datetime import datetime

class TodoItem:
    def __init__(self, title, priority="medium"):
        self.title = title
        self.priority = priority
        self.done = False
        self.created_at = datetime.now()

    def complete(self):
        self.done = True

    def __str__(self):
        status = "[x]" if self.done else "[ ]"
        return f"{status} [{self.priority}] {self.title}"

class TodoList:
    def __init__(self):
        self.items = []

    def add(self, title, priority="medium"):
        self.items.append(TodoItem(title, priority))

    def show(self):
        for i, item in enumerate(self.items, 1):
            print(f"  {i}. {item}")

if __name__ == "__main__":
    todos = TodoList()
    todos.add("Learn Python decorators", "high")
    todos.add("Buy groceries", "low")
    todos.add("Review pull request", "high")
    todos.add("Clean desk", "low")
    todos.items[0].complete()
    todos.show()
