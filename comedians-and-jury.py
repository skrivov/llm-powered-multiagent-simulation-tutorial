import os
import asyncio
import time
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load API key from the .env file to securely access the OpenAI API
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Agent:
    def __init__(self, name, role):
        """
        Initialize an Agent with a name and role.
        
        :param name: The name of the agent (e.g., "Groucho Marx")
        :param role: The agent's role or notable characteristic (e.g., "his quick wit and one-liner jokes")
        """
        self.name = name
        self.role = role
        self.system_prompt = f"You are {self.name}, a famous {role}."

    async def act(self, context):
        """
        Generate an asynchronous response from the agent based on the given context.
        
        :param context: The context or prompt for the agent (e.g., "Tell a one-liner joke.")
        :return: The generated response from the agent
        """
        # Make an asynchronous call to the OpenAI API to generate a response
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context}
            ]
        )
        # Return the generated response content
        return response.choices[0].message.content 

async def simulation_loop(comedians, jury, rounds):
    """
    Simulate a loop where comedians tell jokes and the jury decides the best joke.
    
    :param comedians: A list of Agent objects representing comedians
    :param jury: An Agent object representing the jury
    :param rounds: The number of rounds to run the simulation
    """
    context = "Tell a one-liner joke."  # The shared context for comedians

    for round_num in range(1, rounds + 1):
        print(f"\nRound {round_num}:\n")
        
        # Comedians generate their jokes asynchronously
        tasks = [comedian.act(context) for comedian in comedians]
        completions = await asyncio.gather(*tasks)
        
        # Store jokes with comedian names
        jokes = {comedian.name: completion.strip() for comedian, completion in zip(comedians, completions)}
        
        # Print each comedian's joke
        for comedian_name, joke in jokes.items():
            print(f"{comedian_name}: {joke}")

        # Prepare context for the jury to judge the jokes
        jury_context = f"Here are the jokes told by the comedians:\n"
        for comedian_name, joke in jokes.items():
            jury_context += f"{comedian_name}: {joke}\n"
        jury_context += "Please decide which joke is the best and explain why it is so."

        # Jury gives its decision asynchronously
        jury_decision = await jury.act(jury_context)
        print(f"\nJury Decision:\n{jury_decision}\n")

if __name__ == "__main__":
    # Initialize comedians as agents with specific humor styles
    comedians = [
        Agent("Groucho Marx", "his quick wit and one-liner jokes"),
        Agent("George Carlin", "his observational humor and clever wordplay"),
        Agent("Rodney Dangerfield", "his self-deprecating humor and catchphrase 'I don't get no respect'")
    ]
    
    # Initialize the jury agent
    jury = Agent("Jury", "critical and fair judge of humor")

    # Record the start time of the simulation
    start_time = time.time()

    # Run the asynchronous simulation loop
    asyncio.run(simulation_loop(comedians, jury, 2))

    # Record the end time and calculate the execution duration
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
