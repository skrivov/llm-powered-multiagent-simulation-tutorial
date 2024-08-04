import os
import time
from openai import OpenAI
from dotenv import load_dotenv


# This function looks for a file named .env in the current directory and loads
# any environment variables contained within it into the program's environment.
# This is useful for securely managing sensitive information like API keys.
# The .env file should contain the API key in the format: OPENAI_API_KEY=your_openai_api_key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Agent:
    def __init__(self, name, role):
        """
        Initialize an Agent with a name and role.
        Each agent is characterized by a unique style of humor based on a famous comedian.

        :param name: The name of the comedian (e.g., "Groucho Marx")
        :param role: The comedic style or notable trait of the comedian (e.g., "quick wit and one-liner jokes")
        """
        self.name = name
        self.role = role
        self.system_prompt = f"You are {self.name}, a famous comedian known for {self.role}."

    def act(self, context):
        """
        Generate a response from the agent based on the given context.

        :param context: The context or prompt for the agent (e.g., "Tell a one-liner joke.")
        :return: The generated response from the agent
        """
        response = client.chat.completions.create(
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

def simulation_loop(agents, rounds):
    """
    Simulate a conversation loop where each agent responds to a given context.

    :param agents: A list of Agent objects
    :param rounds: The number of rounds to run the simulation
    """
    context = "Tell a one-liner joke."
    for _ in range(rounds):
        for agent in agents:
            completion = agent.act(context)
            print(f"{agent.name}: {completion.strip()}")

if __name__ == "__main__":
    # Initialize a list of agents, each representing a famous comedian
    agents = [
        Agent("Groucho Marx", "his quick wit and one-liner jokes"),
        Agent("George Carlin", "his observational humor and clever wordplay"),
        Agent("Rodney Dangerfield", "his self-deprecating humor and catchphrase 'I don't get no respect'")
    ]
    
    # Record the start time of the simulation
    start_time = time.time()
    
    # Run the simulation loop with the defined agents for a specified number of rounds
    simulation_loop(agents, 2)
    
    # Record the end time of the simulation and calculate the total execution time
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
