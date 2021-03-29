#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import abc
import sys
import pickle
from typing import List, Union

import numpy as np

from tqdm import tqdm

# Tic-Tac-Toe Reinforcement Learning with a direct reward/backpropagation system.
# Trying to test out basic reinforcement learning with tic-tac-toe
# NOTED ISSUES:
# After literally one or two moves, the algorithm suddenly becomes stuck and isn't able to
# finish a game, likely getting stuck somewhere in the choosing of a next move.

class Player(object, metaclass=abc.ABCMeta):
   """Base class for all player objects."""

   def __init__(self, player=None, board_state=None, turn=None):
      # Initialize the player state, with the board and the turn.
      self.player = player
      self.board_state = None
      self.turn = None
      self.update_state(board_state, turn)

   def __init_subclass__(cls, **kwargs):
      # Ensure that only the human player and computer player exist.
      if cls.__name__ not in ['Human', 'RandomComputer', 'ReinforcedComputer']:
         raise ValueError("Player subclasses should only be the Human or Computer.")

   def update_state(self, board_state: List[Union[str, None]], turn) -> None:
      """Updates the class state, with the board and whose turn it is."""
      # Track the state of the board (to make future moves).
      self.board_state = board_state
      # Keep track of whose move it currently is.
      self.turn = turn

   @abc.abstractmethod
   def determine_next_move(self):
      """Determine the player's next move."""
      raise NotImplementedError("The `determine_next_move` needs to be implemented.")


class Human(Player):
   """Player class representing a human moving in Tic-Tac-Toe."""

   def __init__(self, player=None, board_state=None, turn=None):
      # Initialize the class with its player state.
      super(Human, self).__init__(
         player=player, board_state=board_state, turn=turn)

   def determine_next_move(self):
      """Determine the player's next move."""
      try:
         return int(input(f"Player \'{self.player}\', your turn: "))
      except ValueError:
         return "invalid move"


class RandomComputer(Player):
   """Computer class representing the computer in Tic-Tac-Toe."""

   def __init__(self, player=None, board_state=None, turn=None):
      super(RandomComputer, self).__init__(
         player=player, board_state=board_state, turn=turn)

   def determine_next_move(self):
      """Determine the computer's next move."""
      # Determine the unplayed indices.
      unplayed_indices = [
         i for i, value in enumerate(self.board_state) if value is None]

      # Return a random choice.
      choice = int(np.random.choice(unplayed_indices, size=1))
      print(f'Player \'{self.player}\' Chooses: {choice}')
      return choice


class ReinforcedComputer(Player):
   def __init__(self, player=None, board_state=None, turn=None,
                exp_rate=0.3, file='policy1.pickle'):
      # Initialize the superclass.
      super(ReinforcedComputer, self).__init__(
         player=player, board_state=board_state, turn=turn)

      # Initialize the learning parameters.
      self.lr = 0.2
      self.exp_rate = exp_rate
      self.gamma_decay = 0.9

      # Initialize the player state dictionaries.
      self.states = []
      self.state_values = {}

      # Create the save file.
      self.file = os.path.join(os.path.dirname(__file__), file)

   def append_state(self, state):
      """Add a state to the list of saved states."""
      self.states.append(self.hash_board(state))

   def reset(self):
      """Resets the holder list of states for a game."""
      self.states = []

   @staticmethod
   def hash_board(board_state):
      """Converts the board into a hashable type for storage."""
      return str(np.array(board_state))

   def feed_reward(self, reward):
      """Backpropagate the reward at the end of a certain game."""
      for state in reversed(self.states):
         # If there is no associated state value for the current state,
         # then set it directly to zero after the current game.
         if self.state_values.get(state) is None:
            self.state_values[state] = 0

         # Afterwards, update the state using the learning parameters.
         self.state_values[state] += self.lr * (
               self.gamma_decay * reward - self.state_values[state])

         # Update the new reward after the current state/state value.
         reward = self.state_values[state]

   def determine_next_move(self):
      """Determine the reinforcement agent's next move."""
      # Get a list of valid positions.
      unplayed_indices = [
         i for i, value in enumerate(self.board_state) if value is None]
      action = None

      # Determine whether to perform exploration and exploitation and then
      # choose the move of the agent based on the result of that choice.
      if np.random.uniform(0, 1) <= self.exp_rate:
         # Perform exploration, e.g. choose a random action.
         action = unplayed_indices[np.random.choice(len(unplayed_indices))]
      else:
         # Otherwise, perform exploitation and try and find the best move.
         max_value = -999
         for move in unplayed_indices:
            # Get the state of the future board on each of the potential moves.
            future_board = self.board_state.copy()
            future_board[move] = self.player

            # Determine the value of the current move, e.g. whether it is already in
            # the current state value list of whether it needs to be discovered.
            value = 0 if self.state_values.get(self.hash_board(future_board)) is None \
               else self.state_values.get(self.hash_board(future_board))

            # Determine the next action of the agent.
            if value >= max_value:
               max_value = value
               action = move

      # Return the chosen action.
      return action

   def save_policy(self):
      """Save the policy with the different game states and their values."""
      with open(f'{self.file}', 'wb') as pickle_file:
         pickle.dump(self.state_values, pickle_file)

   def load_policy(self):
      """Load in a policy with different game states and values."""
      with open(f'{self.file}', 'rb') as pickle_file:
         self.state_values = pickle.load(pickle_file)


class PlayerList(object):
   """A list which contains player objects and can update their states."""

   def __init__(self, players=()):
      # Initialize the players within the class.
      self.players: List[Union[Human, RandomComputer, ReinforcedComputer]] \
         = players if len(players) > 0 else []

   def __len__(self):
      # Return the number of players inside the list.
      return len(self.players)

   def __getitem__(self, indx):
      # Return the index of the player in the list.
      return self.players[indx]

   def append(self, player):
      """Add a new player to the list of players."""
      self.players.append(player)

   def update_states(self, board_state, player):
      """Updates the states of each of the players in the list."""
      if self.players[0].player == player:
         self.players[0].update_state(board_state, turn=True)
         self.players[1].update_state(board_state, turn=False)
      else:
         self.players[0].update_state(board_state, turn=False)
         self.players[1].update_state(board_state, turn=True)

   def clear_computer_states(self):
      """Clears the states of each of the reinforcement computer agents."""
      for player in self.players:
         if isinstance(player, ReinforcedComputer):
            player.reset()

   def update_current_agent_state(self, board):
      """Updates the state of the agent who has currently played."""
      if self.players[0].turn:
         self.players[0].append_state(board)
      else:
         self.players[1].append_state(board)

   def get_next_move(self):
      """Gets the next move from the player whose turn it is."""
      if self.players[0].turn:
         return self.players[0].determine_next_move()
      else:
         return self.players[1].determine_next_move()


class Game(object):
   def __init__(self):
      # Initialize the board.
      self.board: List[Union[None, str]] = [None, None, None, None, None, None, None, None, None]
      # Initialize the first player.
      self.current_player = "X"
      # Initialize the player list.
      self.players: PlayerList = PlayerList()
      # Initialize the final game state tracker.
      self.game_completion_state = None

   def __str__(self):
      # Replace `None` with a blank space.
      board = [item if item is not None else " " for item in self.board]

      # Construct the new game board.
      draw_string = ""
      draw_string += f"{board[0]}|{board[1]}|{board[2]}\n"
      draw_string += "-+-+-\n"
      draw_string += f"{board[3]}|{board[4]}|{board[5]}\n"
      draw_string += "-+-+-\n"
      draw_string += f"{board[6]}|{board[7]}|{board[8]}\n"

      # Return the board.
      return draw_string

   def draw(self) -> None:
      """Draws the current game state into the console."""
      draw_string = str(self)

      # Print out the game board.
      sys.stdout.write(draw_string)
      sys.stdout.flush()

   def print_victory(self):
      """Prints a colorful victory message at the conclusion of a game."""
      sys.stdout.write(
         f"\n\033[92mPlayer \'{self.current_player}\' wins!\033[0m\n")
      sys.stdout.write("================\n")

   def new_player(self) -> None:
      """Switch to a new player after a turn is complete."""
      if self.current_player == "X":
         self.current_player = "O"
      else:
         self.current_player = "X"

   def check_if_winner(self) -> bool:
      """Check whether a player has won the game."""
      # Check whether there is a column Tic-Tac-Toe victory.
      for i in range(0, 2):
         if self.board[i] == self.board[i + 3] == self.board[i + 6] and all(
               (self.board[i], self.board[i + 3], self.board[i + 6])):
            return True

      # Check whether there is a row Tic-Tac-Toe victory.
      for i in range(0, 9, 3):
         if self.board[i] == self.board[i + 1] == self.board[i + 2] and all(
               (self.board[i], self.board[i + 1], self.board[i + 2])):
            return True

      # Check for the different diagonal victories.
      if self.board[0] == self.board[4] == self.board[8] and all(
            (self.board[0], self.board[4], self.board[8])):
         return True
      if self.board[6] == self.board[4] == self.board[2] and all(
            (self.board[6], self.board[4], self.board[2])):
         return True

      # Otherwise, there is no victory.
      return False

   def check_if_draw(self) -> bool:
      """Check whether the game has resulted in a draw."""
      return all(self.board)

   def check_if_valid_move(self, move):
      """Check whether the move has already been played or not."""
      if self.board[move - 1] is not None:
         return False
      return True

   def start_game(self):
      """Choose the starting players and start the game."""
      # If the game has already been initialized (e.g., this is the second or further
      # round), then the game should not be re-initialized, so simply skip this method.
      if len(self.players) == 2:
         # Return a NoneType placeholder value.
         return None

      # Otherwise, initialize the game from user options.
      play_vs_computer = input("Would you like to play against the computer? (y|n) ")
      if play_vs_computer == "y":
         # If the player wants to play against the computer, then determine
         # whether they want to go first or second against the computer.
         first_or_second = input("Would you like to go first or second? (f|s) ")
         if first_or_second == "f":
            # The human player would like to go first, so add them to the list first.
            self.players.append(Human(player="X", board_state=self.board, turn=True))
            self.players.append(RandomComputer(player="O", board_state=self.board, turn=False))
         elif first_or_second == "s":
            # The human player would like to go second, so add them to the list second.
            self.players.append(RandomComputer(player="X", board_state=self.board, turn=True))
            self.players.append(Human(player="O", board_state=self.board, turn=False))
         else:
            # Re-prompt the user if an invalid choice was received.
            print("Invalid choice received.")
            self.start_game()
      elif play_vs_computer == "n":
         # Otherwise, the player wants to play against another human.
         self.players.append(Human(player="X", board_state=self.board, turn=True))
         self.players.append(Human(player="O", board_state=self.board, turn=False))
      else:
         # Re-prompt the user if an invalid choice was received.
         print("Invalid choice received.")
         self.start_game()

   def reset_game(self):
      """Resets the board and players for a new round."""
      self.board = [None, None, None, None, None, None, None, None, None]
      self.current_player = "X"

   def run_one_loop(self):
      """Executes one single game loop."""
      while True:
         # On each turn of a single game, we need to get the current player's move.
         turn = self.players.get_next_move()

         # Then, the turns need to be validated. First, we need to figure out whether the
         # user has or has not entered a valid number corresponding to a square and simply
         # ask for another input rather than crashing the entire game for no reason.
         if turn not in list(range(1, 10)) or turn == "invalid move":
            print("That is an invalid move. Enter a square in the range 1-9.")
            continue

         # We then need to determine whether the move has already been played, either by
         # the current player or by their opponent, and again simply ask for another input.
         if not self.check_if_valid_move(turn):
            print("Another player has already played there. Please enter another move.")
            continue

         # Then, update the board and print out the current board status after the turn.
         self.board[turn - 1] = self.current_player
         self.draw()

         # We then need to check whether a player has won or whether the game has ended up in
         # a draw, and in either of these cases stop the game there and rerun it.
         if self.check_if_winner():
            self.game_completion_state = self.current_player
            self.print_victory()
            break
         elif self.check_if_draw():
            self.game_completion_state = "draw"
            print(f"The game was a draw!")
            break

         # If the game is to continue, then switch the current game state to the next player, and
         # update the states of each of the different players (which will be called upon).
         self.new_player()
         self.players.update_states(self.board, player=self.current_player)

   def play(self):
      """Configures and runs the actual Tic-Tac-Toe game."""
      while True:
         # Start each game by determining who the players are, e.g. is it a human
         # versus another human, or is it human versus computer. If human versus
         # computer, then we need to figure out whether the human wants to go first
         # or second (although going second is an obvious disadvantage).
         self.start_game()

         # Execute a single game loop.
         self.run_one_loop()

         # After the game is complete, prompt the user as to whether they want to play again.
         again = input("Play again? (y|n) ")
         if again.lower() != "y":
            # If they do not want to, then exit the game loop and stop the program.
            break
         else:
            # Otherwise, the game needs to be reset, e.g. the board and the list of players.
            self.reset_game()
            continue


class TrainingGame(Game):
   def __init__(self, epochs):
      # Initialize the game.
      super(TrainingGame, self).__init__()

      # Set the training values.
      self.epochs = epochs

      # Add the game agents.
      self.add_agents()

   def get_board_hash(self):
      """Return the board as a hashable type for the reinforcement agents."""
      return str(np.array(self.board))

   def add_agents(self):
      """Add player agents to the game."""
      self.players.append(
         ReinforcedComputer(player="X", board_state=self.board, turn=True))
      self.players.append(
         ReinforcedComputer(player="O", board_state=self.board, turn=False, file="policy2.pickle"))

   def feed_rewards(self):
      """Feed rewards to the different players."""
      if self.game_completion_state == "X":
         # `X` is the victory, so gets a reward of 1 while `O`
         # is the loser, so gets no reward and has to learn to win.
         self.players[0].feed_reward(1)
         self.players[1].feed_reward(0)
      elif self.game_completion_state == "O":
         # `O` is the victory, so gets a reward of 1 while `X`
         # is the loser, so gets no reward and has to learn to win.
         self.players[0].feed_reward(0)
         self.players[1].feed_reward(1)
      else:
         # The game is a draw, but since going first has its own inherent
         # advantage, `X` gets less reward than `O` as a result.
         self.players[0].feed_reward(0.1)
         self.players[1].feed_reward(0.5)

   def play(self):
      """Execute the actual game for a certain number of training epochs."""
      for epoch in tqdm(range(self.epochs)):
         while True:
            # On each turn of a single game, we need to get the current player's move.
            turn = self.players.get_next_move()

            # Then, the turns need to be validated. First, we need to figure out whether the
            # user has or has not entered a valid number corresponding to a square and simply
            # ask for another input rather than crashing the entire game for no reason.
            if turn not in list(range(1, 10)) or turn == "invalid move":
               continue

            # We then need to determine whether the move has already been played, either by
            # the current player or by their opponent, and again simply ask for another input.
            if not self.check_if_valid_move(turn):
               continue

            # Then, update the board and print out the current board status after the turn.
            self.board[turn - 1] = self.current_player

            # Append the new state to the agent who has just played.
            self.players.update_current_agent_state(self.board)

            # We then need to check whether a player has won or whether the game has ended up in
            # a draw, and in either of these cases stop the game there and rerun it.
            if self.check_if_winner():
               self.game_completion_state = self.current_player
               break
            elif self.check_if_draw():
               self.game_completion_state = "draw"
               break

            # If the game is to continue, then switch the current game state to the next player, and
            # update the states of each of the different players (which will be called upon).
            self.new_player()
            self.players.update_states(self.board, player=self.current_player)

         # Reset the game state for the next one.
         self.reset_game()

