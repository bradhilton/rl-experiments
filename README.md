# openpipe-rl

OpenPipe Reinforcement Learning Experiments

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
2. Launch a cluster named "openpipe-rl" using the configuration in `cluster.yaml`
3. Set up the development environment on the cluster

Additional sky launch options can be passed to the script. For example, to launch a cluster with 2 A100 GPUs:

```bash
./launch-cluster.sh --gpus A100:2
```

### Connecting to Your Cluster

To SSH into your running cluster:

```bash
ssh openpipe-rl
```

To use VSCode with the cluster:

1. Press `Cmd/Ctrl + Shift + P`
2. Type `Remote-SSH: Connect Current Window to Host`
3. Select `openpipe-rl` from the list
4. Open the `openpipe-rl` folder to access the repo

### Other Useful Commands

- List all running clusters:

  ```bash
  sky status
  ```

- Stop your cluster (to pause billing):

  ```bash
  sky stop openpipe-rl
  ```

- Start a stopped cluster:

  ```bash
  sky start openpipe-rl
  ```

- Terminate your cluster (to delete all resources):
  ```bash
  sky down openpipe-rl
  ```
