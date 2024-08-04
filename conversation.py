import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file for secure access to the OpenAI API
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Agent:
    def __init__(self, name, role, system_prompt):
        """
        Initialize an Agent with a name, role, and system prompt.
        
        :param name: The name of the agent (e.g., "Ronald Reagan")
        :param role: The agent's role or identity (e.g., "former President")
        :param system_prompt: The system prompt that defines the agent's behavior
                              and responses
        """
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        # Initialize with system prompt
        self.messages = [{"role": "system", "content": system_prompt}]

    def act(self):
        """
        Generate a response from the agent based on its own conversation history.
        
        :return: The generated response from the agent
        """
        # Make the API call to generate a response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=self.messages
        )
        
        # Extract and clean up the response content
        response_content = response.choices[0].message.content.strip()
        
        # Remove the agent's name if it appears at the start of the response
        if response_content.startswith(self.name):
            response_content = response_content[len(self.name):].strip(" :")
        
        # Add the response to the agent's message history
        self.messages.append({"role": "assistant", "content": response_content})
        
        return response_content

def simulation_loop(agents, rounds):
    for round_num in range(1, rounds + 1):
        print(f"\nRound {round_num}:\n")
        
        for agent in agents:
            # Each agent generates a response based on its own conversation history
            response = agent.act()
            
            # Print the agent's response for this round
            print(f"{agent.name}: {response}")
            
            # Share the response with all other agents by appending it to their history
            for other_agent in agents:
                if other_agent != agent:
                    other_agent.messages.append(
                        {"role": "assistant", 
                         "content": f"{agent.name}: {response}"}
                    )

if __name__ == "__main__":
    # Define the system prompts for each president, shaping their identity
    reagan_prompt = (
        "You are Ronald Reagan, the former President known for your wit, humor, "
        "and ability to tell great anecdotes. You're sitting in a restaurant with "
        "Richard Nixon and Jimmy Carter, talking about politics, making light-hearted "
        "jokes about Republicans, Democrats, and each other's jobs during your tenures "
        "as Presidents. Respond with one or two lines as Ronald Reagan without "
        "mentioning your name, maintaining relevance of your response to the conversation."
    )
    nixon_prompt = (
        "You are Richard Nixon, the former President known for your complex personality "
        "and historical impact. You're sitting in a restaurant with Ronald Reagan and "
        "Jimmy Carter, discussing politics and making jokes about Republicans, "
        "Democrats, and each other's jobs during your tenures as Presidents. "
        "Respond with one or two lines as Richard Nixon without mentioning your name, "
        "maintaining relevance of your response to the conversation."
    )
    carter_prompt = (
        "You are Jimmy Carter, the former President known for your diplomatic skills "
        "and humanitarian efforts. You're sitting in a restaurant with Ronald Reagan "
        "and Richard Nixon, having a conversation about politics and making jokes "
        "about Republicans, Democrats, and each other's jobs during your tenures as "
        "Presidents. Respond with one or two lines as Jimmy Carter without mentioning "
        "your name, maintaining relevance of your response to the conversation."
    )

    # Create Agent instances for each president with their respective system prompts
    reagan = Agent("Ronald Reagan", "former President", reagan_prompt)
    nixon = Agent("Richard Nixon", "former President", nixon_prompt)
    carter = Agent("Jimmy Carter", "former President", carter_prompt)

    # Store the agents in a list for easy iteration
    agents = [reagan, nixon, carter]

    # Record the start time of the simulation for performance measurement
    start_time = time.time()
    
    # Run the conversation simulation loop for the specified number of rounds
    simulation_loop(agents, 2)
    
    # Record the end time and calculate the total execution time of the simulation
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
