# OpenPipe Reinforcement Learning Experiments

This repository contains a series of reinforcement learning experiments.

Experiment notebooks can be found in the `/experiments` directory, along with library code and data. The repository also contains some helpful scripts for starting dev machines and other utilities.

Experiment results can be viewed on [Weights & Biases](https://wandb.ai/bradhilton/rl-experiments).

To learn more about our research, check out our [blog post](https://openpipe.ai/blog) at [OpenPipe](https://openpipe.ai).

If you want to reproduce our work on deductive reasoning, take a look at our streamlined [training recipe here](https://github.com/openpipe/deductive-reasoning).

If you're interested in training your own models with reinforcement learning or just chatting, feel free to [contact us](https://openpipe.ai/contact) or email Kyle at kyle@openpipe.ai!

## Dependencies

Install project dependencies using `uv`:

```bash
uv sync
```

This will install all dependencies specified in `pyproject.toml` and create a virtual environment at `.venv`.

## Environment Setup

1. Create your environment file by copying the example:

   ```bash
   cp .env.example .env
   ```

2. Edit your `.env` file with your personal details:

   ```bash
   # Git Configuration
   GIT_USER_NAME="Your Name"
   GIT_USER_EMAIL="your.email@example.com"

   # GitHub Authentication
   # Generate a personal access token at https://github.com/settings/tokens
   # Required scopes: repo
   GITHUB_TOKEN="your_github_personal_access_token"
   ```

   To get your GitHub token:

   1. Go to https://github.com/settings/tokens
   2. Click "Generate new token"
   3. Select the `repo` scope
   4. Copy the generated token and paste it into your `.env` file

## Cluster Management

### Launching a Cluster

To launch a cluster with the default configuration:

```bash
./launch-cluster.sh
```

This will:

1. Load your environment variables from `.env`
2. Launch a cluster named "openpipe" using the configuration in `cluster.yaml`
3. Set up the development environment on the cluster

Additional sky launch options can be passed to the script. For example, to launch a cluster with 2 A100 GPUs:

```bash
./launch-cluster.sh --gpus A100:2
```

### Connecting to Your Cluster

To SSH into your running cluster:

```bash
ssh openpipe
```

To use VSCode with the cluster:

1. Press `Cmd/Ctrl + Shift + P`
2. Type `Remote-SSH: Connect Current Window to Host`
3. Select `openpipe` from the list
4. Open the `openpipe` folder to access the repo

### Other Useful Commands

- List all running clusters:

  ```bash
  sky status
  ```

- Stop your cluster (to pause billing):

  ```bash
  sky stop openpipe
  ```

- Start a stopped cluster:

  ```bash
  sky start openpipe
  ```

- Terminate your cluster (to delete all resources):
  ```bash
  sky down openpipe
  ```
