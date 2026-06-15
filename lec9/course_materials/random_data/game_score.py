import random

class Player:
    def __init__(self, name):
        self.name = name
        self.score = 0
        self.history = []

    def roll(self):
        points = random.randint(1, 6) + random.randint(1, 6)
        self.score += points
        self.history.append(points)
        return points

def play_game(player_names, rounds=5):
    players = [Player(name) for name in player_names]
    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num} ---")
        for p in players:
            roll = p.roll()
            print(f"  {p.name} rolled {roll} (total: {p.score})")
    print("\n=== Final Scores ===")
    for p in sorted(players, key=lambda x: x.score, reverse=True):
        print(f"  {p.name}: {p.score} points")

if __name__ == "__main__":
    play_game(["Alice", "Bob", "Charlie"], rounds=4)
