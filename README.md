# RL Learning

This repo contains the implementations of RL algorithms for use in the OpenAI Gym. REINFORCE and A2C are currently implemented and PPO is in progress.
This repo was my first exposure to PyTorch, so there are a lot of things that can be improved.

## Usage

The repo is configured to use `pipenv`. Running `pipenv shell` should hopefully set up the correct virtual environment.

The main program is `run/run.py`. Running the program should show the help if the right parameters are not provided.

## Future work

- Finish PPO
- Properly implemented A2C and REINFORCE for continuous action spaces (currently a hacky approach for discretizing the action space is used)
- Migrate from `pipenv` to Docker.
- More algorithms
- Better usage of torch's optimizers and batching mechanisms
- Better testing harness, so things "just work" once a new Learner is implemented



