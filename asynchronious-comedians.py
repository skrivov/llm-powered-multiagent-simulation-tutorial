import os
import asyncio
import time
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load API key from .env file
# This loads the environment variables from a .env file into the application's environment.
# It allows the secure handling of sensitive information like the API key, which is not hard-coded in the script.
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Agent:
    def __init__(self, name, role):
        """
        Initialize an Agent with a name and role.
        The system_prompt defines the personality and style of the agent based on a famous comedian.

        :param name: The name of the comedian (e.g., "Groucho Marx")
        :param role: The comedic style or notable trait (e.g., "quick wit and one-liner jokes")
        """
        self.name = name
        self.role = role
        self.system_prompt = f"You are {self.name}, a famous comedian known for {self.role}."

    async def act(self, context):
        """
        Asynchronously generate a response from the agent based on the provided context.

        :param context: The context or prompt for the agent to respond to (e.g., "Tell a one-liner joke.")
        :return: The generated response from the agent
        """
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.8,  # Adjusts the randomness of the response
            max_tokens=60,    # Limits the length of the response
            frequency_penalty=0.7  # Penalizes repetition
        )
        return response.choices[0].message.content 

async def simulation_loop(agents, rounds):
    """
    Asynchronously run the simulation loop, allowing each agent to respond to a given context over multiple rounds.

    :param agents: A list of Agent objects representing different comedians.
    :param rounds: The number of rounds the simulation should run.
    """
    context = "Tell a one-liner joke."
    for _ in range(rounds):
        # Create a list of asyncio tasks for all agents to act simultaneously
        tasks = [agent.act(context) for agent in agents]
        completions = await asyncio.gather(*tasks)  # Run all tasks concurrently and gather the results
        for agent, completion in zip(agents, completions):
            # Print the response for each agent after completing their act
            print(f"{agent.name}: {completion.strip()}")

if __name__ == "__main__":
    # Initialize a list of agents, each representing a different comedian
    agents = [
        Agent("Groucho Marx", "his quick wit and one-liner jokes"),
        Agent("George Carlin", "his observational humor and clever wordplay"),
        Agent("Rodney Dangerfield", "his self-deprecating humor and catchphrase 'I don't get no respect'")
    ]
    # Measure the start time of the simulation
    start_time = time.time()
    # Run the asynchronous simulation loop with the list of agents and the number of rounds
    asyncio.run(simulation_loop(agents, 2))
    # Measure the end time of the simulation and calculate the total execution time
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")
