#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from weavehacks_flow.crews.poem_crew.poem_crew import PoemCrew


class PoemState(BaseModel):
    sentence_count: int = 1
    poem: str = ""
    action: str = "generate_poem"


class PoemFlow(Flow[PoemState]):

    @start()
    def generate_sentence_count(self):
        self.state.action = "easy"

    @listen(generate_sentence_count)
    def generate_poem(self):
        if self.state.action == "easy":
            self.state.sentence_count = randint(1, 5)
        print("Generating poem")
        result = (
            PoemCrew()
            .crew()
            .kickoff(inputs={"sentence_count": self.state.sentence_count})
        )

        print("Poem generated", result.raw)
        self.state.poem = result.raw

    @listen(generate_poem)
    def save_poem(self):
        print("Saving poem")
        with open("poem.txt", "w") as f:
            f.write(self.state.poem)


def kickoff():
    poem_flow = PoemFlow()
    poem_flow.kickoff()


def plot():
    poem_flow = PoemFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
