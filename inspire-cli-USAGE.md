# Inspire CLI Usage Guide

This document provides a comprehensive guide to using the `inspire-cli`, a command-line interface for the Inspire HPC training platform.

## Installation

`inspire-cli` is a uv-tool, has been installed in this environment. You can directly use the `inspire-cli` command, for example:

```bash
inspire-cli --help
```

## Configuration

The `inspire-cli` is configured entirely through environment variables. You can set these variables by:
```bash
source login.sh
```

### Core Configuration

These are the essential variables required for most operations.

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `INSPIRE_USERNAME` | Your Inspire platform username. | (None) |
| `INSPIRE_PASSWORD` | Your Inspire platform password. | (None) |
| `INSPIRE_TARGET_DIR` | The root directory on the shared filesystem for all Bridge operations (code sync, logs, etc.). | (None) |
| `INSP_GITHUB_REPO` | The GitHub repository in `owner/repo` format, used for Bridge operations. | (None) |
| `INSP_GITHUB_TOKEN`| A GitHub Personal Access Token with repository access. You can also use `gh auth token`. | (None) |
| `UV_PYTHON_INSTALL_DIR` | The directory where Python packages and uv environments are installed, we should use this python in Inspire HPC training command. | (None) |


## Commands

### `config`

| Command | Description |
| --- | --- |
| `inspire config check` | Validates your environment variables and API authentication. |

### `job`

| Command | Description |
| --- | --- |
| `inspire job create` | Creates a new training job. |
| `inspire job status <id>` | Checks the status of a job. |
| `inspire job stop <id>` | Stops a running job. |
| `inspire job wait <id>` | Waits for a job to complete. |
| `inspire job list` | Lists recent jobs from the local cache. |
| `inspire job logs <id>` | Views the logs of a job. |

### `resources`

| Command | Description |
| --- | --- |
| `inspire resources list`| Lists the available GPU configurations. |

### `nodes`

| Command | Description |
| --- | --- |
| `inspire nodes list` | Lists the cluster nodes. |

### `sync`

| Command | Description |
| --- | --- |
| `inspire sync` | Syncs your local branch to the Bridge shared filesystem. |

### `bridge`

| Command | Description |
| --- | --- |
| `inspire bridge exec` | Executes a shell command on the Bridge self-hosted runner. |

## Workflows

The `inspire-cli` uses GitHub Actions workflows to perform tasks on the "Bridge" (a self-hosted runner with access to the shared filesystem).

### Remote Log Retrieval

To fetch logs from a machine without direct access to the shared filesystem, you need to set up the log retrieval workflow.

### Code Sync

The `inspire sync` command pushes your local branch to GitHub and triggers a workflow to sync the code to the shared filesystem.

### Bridge Exec

The `inspire bridge exec` command allows you to run shell commands on the Bridge runner.


