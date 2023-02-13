import discord
import hashlib
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import hashlib
import re
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from discord.ext import commands

bot = commands.Bot(command_prefix='.', intents=discord.Intents.all())


@bot.event
async def on_ready():
  print(f'{bot.user} has connected to Discord!')


@bot.command()
async def mines(ctx, server_hash):
  game_data = extract_info(server_hash)
  if len(game_data) < 64:
    await ctx.send("Not enough game data to make a prediction.")
    return
  grid, mines = format_grid(game_data)
  prediction = predict(grid, mines)
  grid_str = ''
  for row in grid:
    row_str = ''
    for cell in row:
      if cell == -1:
        row_str += '❌'
      else:
        row_str += '✅'
    grid_str += row_str + '\n'

  embed = discord.Embed(
    title="Bloxflip Mines Prediction",
    description=f"Accuracy: {prediction[1]:.2f}%\n\n{grid_str}",
    color=0x00ff00)
  await ctx.send(embed=embed)


def extract_info(server_hash):
  m = hashlib.sha256()
  m.update(server_hash.encode('utf-8'))
  server_hash = m.hexdigest()
  pattern = re.compile(r"(\w{2})(\w{2})(\w{2})")
  game_data = pattern.findall(server_hash)
  return game_data


def format_grid(game_data):
  grid = []
  mines = []
  if len(game_data) != 64:
    return np.array(grid), np.array(mines)

  for i in range(25):
    row = int(game_data[i][0], 16) % 5
    col = int(game_data[i][1], 16) % 5
    is_mine = int(game_data[i][2], 16) % 2 == 1
    grid.append((row, col))
    mines.append(is_mine)
  return np.array(grid), np.array(mines)


def predict(grid, mines):
  X_train, X_test, y_train, y_test = train_test_split(grid,
                                                      mines,
                                                      test_size=0.2)
  model = RandomForestClassifier(n_estimators=100, random_state=0)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred) * 100
  return y_pred, accuracy

@bot.command()
async def minesV2(ctx, serverhash: str):
    grid = [["❌" for i in range(5)] for j in range(5)]
    hash_input = hashlib.sha256(serverhash.encode())
    hex_dig = hash_input.hexdigest()
    num_list = re.findall(r"\d+", hex_dig)
    data = []
    for i in range(0, 25):
        data.append([int(num_list[i]), int(num_list[i+25]), int(num_list[i+50]) % 2])
    X = np.array(data)
    X_train, X_test, y_train, y_test = train_test_split(X[:, :2], X[:, 2], test_size=0.33, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            grid[X_test[i][0]][X_test[i][1]] = "✅"
    prediction = "\n".join([" ".join(row) for row in grid])
    embed = discord.Embed(title="Bloxflip Minefield Prediction", description=prediction, color=0x00ff00)
    embed.set_footer(text=f"Accuracy: {accuracy * 100:.2f}%")
    await ctx.send(embed=embed)

@bot.command(name="minesV4")
async def minesV4(ctx, server_hash: str):
    # Preprocessing of server hash to obtain the game grid
    game_grid = []
    for i in range(0, len(server_hash), 2):
        game_grid.append(int(server_hash[i:i+2], 16))
    game_grid = np.array(game_grid).reshape(5, 5)
    
    # Use the trained model to predict the unknown spots
    prediction = self.model.predict(game_grid.reshape(1, -1))
    prediction = prediction.reshape(5, 5)
    
    # Convert predictions to symbols
    symbols = []
    for row in prediction:
        symbols_row = []
        for spot in row:
            if spot == 1:
                symbols_row.append("✅")
            else:
                symbols_row.append("❌")
        symbols.append(symbols_row)
    
    # Create an accuracy score for the prediction
    accuracy = np.sum(prediction == game_grid) / 25 * 100
    
    # Create the embed for the prediction
    embed = discord.Embed(title="Bloxflip Predictor", color=0x00ff00)
    embed.add_field(name="Prediction", value="\n".join(" ".join(row) for row in symbols), inline=False)
    embed.add_field(name="Accuracy", value=f"{accuracy:.2f}%", inline=False)
    await ctx.send(embed=embed)
    
def train_model(self):
    # Load the training data
    data = pd.read_csv("training_data.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Train the model using a random forest classifier
    self.model = RandomForestClassifier(n_estimators=100, random_state=0)
    self.model.fit(X_train, y_train)
    
    # Evaluate the model on the testing set
    y_pred = self.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}%")

class BloxflipPredictor(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def bloxflip(self, ctx, serverhash: str):
        game_grid = [int(x) for x in re.findall("\d+", hashlib.sha256(serverhash.encode()).hexdigest())]
        game_grid = np.array(game_grid[:25]).reshape(5, 5)

        X = np.array(game_grid)
        y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        grid = ''
        for row in game_grid:
            for cell in row:
                if cell == 1:
                    grid += '✅'
                else:
                    grid += '❌'
            grid += '\n'

        embed = discord.Embed(
            title='Bloxflip Predictor',
            description=f'Serverhash: {serverhash}\n\n{grid}Accuracy: {accuracy:.2f}%',
            color=discord.Color.blue()
        )
        await ctx.send(embed=embed)


def setup(bot):
    bot.add_cog(BloxflipPredictor(bot))

# Start of code

def bloxflip_predictor(serverhash):
    # Hashing the server hash with SHA256
    hashed_serverhash = hashlib.sha256(serverhash.encode('utf-8')).hexdigest()
    # Converting the hashed server hash into a list of integers
    game_grid = [int(i, 16) for i in re.findall(r'.{2}', hashed_serverhash)]
    # Reshaping the list into a 5x5 grid
    game_grid = np.array(game_grid).reshape(5, 5)
    # Predicting the state of the game using the random forest classifier
    prediction = clf.predict(game_grid.reshape(1, -1))
    # Counting the number of right predictions
    right_predictions = np.count_nonzero(prediction == game_grid)
    # Calculating the accuracy of the prediction
    accuracy = right_predictions / 25 * 100
    # Creating the grid of the game as a string
    game_grid_str = ''
    for row in game_grid:
        for value in row:
            if value == 0:
                game_grid_str += '❌ '
            else:
                game_grid_str += '✅ '
        game_grid_str += '\n'
    # Creating the embed to display the prediction
    embed = discord.Embed(title='BloxFlip Predictor',
                          description=f'Prediction: {prediction}\nAccuracy: {accuracy:.2f}%\n\n{game_grid_str}',
                          color=0x00ff00)
    return embed

# Loading the trained classifier
clf = joblib.load('predictor.joblib')
