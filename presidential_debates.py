import os
import asyncio
import random
import time
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load API key from the .env file for secure access to the OpenAI API
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Agent:
    def __init__(self, name, role, system_prompt):
        """
        Initialize an Agent with a name, role, and system prompt.
        
        :param name: The name of the agent (e.g., "Donald Trump")
        :param role: The agent's role or identity (e.g., "former President")
        :param system_prompt: The system prompt that defines the agent's behavior
        """
        self.name = name
        self.role = role
        self.messages = [{"role": "system", "content": system_prompt}]  # Initialize with system prompt
    
    async def act(self, additional_message):
        """
        Generate a response from the agent based on the additional message and 
        update the agent's message history.
        
        :param additional_message: The new message or prompt to which the agent 
                                   will respond
        :return: The generated response from the agent
        """
        # Add the new user message to the agent's message history
        self.messages.append({"role": "user", "content": additional_message})
        
        # Make an asynchronous API call to generate a response
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=self.messages
        )
        
        # Append the generated response to the agent's message history
        self.messages.append({"role": "assistant", 
                              "content": response.choices[0].message.content})
        
        # Return the generated response content
        return response.choices[0].message.content

async def simulation_loop(candidate1, candidate2, moderator, audience, rounds):
    """
    Simulate a debate between two candidates, moderated by a third agent, and 
    observed by an audience of agents.
    
    :param candidate1: The first candidate agent participating in the debate
    :param candidate2: The second candidate agent participating in the debate
    :param moderator: The moderator agent guiding the debate
    :param audience: A list of audience agents reacting to the debate
    :param rounds: The number of rounds to run the simulation
    """
    for round_num in range(1, rounds + 1):
        print(f"\nRound {round_num}:\n")
        
        # The moderator generates a new debate question
        question = await moderator.act("Create a new debate question.")
        print(f"{moderator.name}: {question.strip()}")

        # Randomly choose which candidate answers first
        first_candidate = random.choice([candidate1, candidate2])
        second_candidate = candidate2 if first_candidate == candidate1 else candidate1

        # The first candidate responds to the question
        print(f"\n{moderator.name}: {first_candidate.name}, you are the first to answer.")
        first_response = await first_candidate.act(
            f"{moderator.name}: {question.strip()}\n"
            "Please give a short and crisp answer."
        )
        print(f"{first_candidate.name}: {first_response.strip()}")

        # The second candidate responds, considering the first candidate's answer
        print(f"\n{moderator.name}: {second_candidate.name}, your response.")
        second_response = await second_candidate.act(
            f"{moderator.name}: {question.strip()}\n"
            f"{first_candidate.name} answered: {first_response.strip()}\n"
            "Now it's your turn. Please keep it short and crisp."
        )
        print(f"{second_candidate.name}: {second_response.strip()}")

        # Two rounds of back and forth between the candidates
        for _ in range(2):  
            # The first candidate rebuts the second candidate's response
            print(f"\n{moderator.name}: {first_candidate.name}, your rebuttal.")
            back_and_forth_response_1 = await first_candidate.act(
                f"{second_candidate.name} said: {second_response.strip()}\n"
                "Respond to their points. Please keep it short and crisp."
            )
            print(f"{first_candidate.name}: {back_and_forth_response_1.strip()}")

            # The second candidate rebuts the first candidate's rebuttal
            print(f"\n{moderator.name}: {second_candidate.name}, your rebuttal.")
            back_and_forth_response_2 = await second_candidate.act(
                f"{first_candidate.name} said: {back_and_forth_response_1.strip()}\n"
                "Respond to their points. Please keep it short and crisp."
            )
            print(f"{second_candidate.name}: {back_and_forth_response_2.strip()}")

            # Update the context for the next round of back and forth
            second_response = back_and_forth_response_2

        # Audience reacts to the entire round, including the initial responses 
        # and rebuttals
        audience_context = (
            f"Here are the responses from the candidates:\n"
            f"{first_candidate.name}: {first_response.strip()}\n"
            f"{second_candidate.name}: {second_response.strip()}\n"
            f"{first_candidate.name}: {back_and_forth_response_1.strip()}\n"
            f"{second_candidate.name}: {back_and_forth_response_2.strip()}\n"
            "Who do you think won this round and why?"
        )
        
        # Collect responses from all audience members concurrently
        audience_responses = await asyncio.gather(
            *[aud.act(audience_context) for aud in audience]
        )
        
        # Display audience decisions
        print("\nAudience Decisions:\n")
        for aud, response in zip(audience, audience_responses):
            print(f"{aud.name}: {response.strip()}\n")
        print("\n")

if __name__ == "__main__":
    # Define the first candidate as Donald Trump with his specific characteristics
    candidate1 = Agent(
        "Donald Trump", 
        "former President known for his bold statements and unique rhetoric", 
        "You are Donald Trump, the former President known for your bold statements, "
        "unique rhetoric, and unfiltered personality. Be true to all aspects of your character."
    )
    
    # Define the second candidate as Kamala Harris with her specific characteristics
    candidate2 = Agent(
        "Kamala Harris", 
        "Vice President known for her articulate and sharp responses", 
        "You are Kamala Harris, the Vice President known for your articulate and sharp "
        "responses, compassion, and firm stances. Be true to all aspects of your character."
    )
    
    # Define the moderator as Bret Baier, known for fair and balanced moderation
    moderator = Agent(
        "Bret Baier", 
        "Fox News anchor known for his fair and balanced moderation",
        "You are Bret Baier, a Fox News anchor known for your fair and balanced moderation. "
        "When asked to create a new debate question, simply ask the question without any preamble or introductory phrases."
    )
    
    # Define the audience as three distinct groups with different ideological leanings
    audience = [
        Agent(
            "Liberal Democrats", 
            "a group of progressives advocating for social justice",
            "You are a group of progressives who prioritize social justice, equality, "
            "and inclusive policies. You advocate for a fairer society and believe in "
            "the power of government to address systemic issues. React passionately to "
            "the candidates' responses."
        ),
        Agent(
            "Conservatives", 
            "a group of conservatives who uphold traditional values",
            "You are a group of conservatives who uphold traditional values, personal "
            "responsibility, and a strong adherence to the Constitution. You believe in "
            "limited government, free markets, and the importance of preserving the "
            "nation's foundational principles. React strongly to the candidates' responses."
        ),
        Agent(
            "Independents", 
            "a group of independent voters seeking pragmatic solutions",
            "You are a group of independent voters who value pragmatism, balanced "
            "perspectives, and clear, actionable plans. You are not ideologically bound "
            "and seek practical solutions that work for the majority of people. React "
            "thoughtfully to the candidates' responses."
        )
    ]

    # Measure the start time of the simulation
    start_time = time.time()
    
    # Run the simulation loop asynchronously
    asyncio.run(simulation_loop(candidate1, candidate2, moderator, audience, 2))
    
    # Measure and display the total execution time
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
