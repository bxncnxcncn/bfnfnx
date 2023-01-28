import hashlib
import re
import random
from random import sample
import discord
from discord.ext import commands
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

@bot.command()
async def mines(ctx, round_id: str):
    # Hash the round ID
    hashed_id = hashlib.sha256(round_id.encode()).hexdigest()

    # Extract only the numeric characters from the hashed ID
    numbers = re.sub("\D", "", hashed_id)

    # Determine the number of mines based on the length of the hashed ID
    num_mines = len(numbers) // 2

    # Create the grid
    grid = [[int(x) for x in row] for row in [numbers[i:i+2] for i in range(0, len(numbers), 2)]]

    # Add the mines to the grid
    for i in range(num_mines):
        x, y = int(numbers[i*2]), int(numbers[i*2+1])
        grid[x][y] = "❤️"

    # Add the safe spots to the grid
    for i in range(3):
        x, y = random.randint(0, 4), random.randint(0, 4)
        while grid[x][y] != "⚠️":
            grid[x][y] = "✅"

    # Add the random spot to the grid
    x, y = random.randint(0, 4), random.randint(0, 4)
    grid[x][y] = "⭐"

    # Create the embed
    embed = discord.Embed(title="Predicting for round ID: " + round_id, color=0x00ff00)
    for row in grid:
        embed.add_field(name="\u200b", value=" ".join(row), inline=False)

    # Send the embed
    await ctx.reply(embed=embed)

bot.run('MTA2ODgxNDg3MTUyNjkyMDIzMg.G8IpOp.781_e-aaOFIcXRGGC2X3NpaKo_OoewpdcD_MRo')
