# SpaceInstruct

## Description
* This is the language reasoning backend the robotics system, responsible for interpreting natural language commands and translating them into robot trajectory updates.
* This repository implements an LLM-driven reasoning bridge between Unity and the robot control pipeline.
* It receives scene state and user speech input from Unity, reasons using a Large Language Model (LLM), and returns validated trajectory edits or target points in a structured JSON format.
