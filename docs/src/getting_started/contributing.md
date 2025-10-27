# Contributing

Welcome and thank you for your interest in contributing to Guardrails! We appreciate all contributions, big or small, from bug fixes to new features. Before diving in, let's go through some guidelines to make the process smoother for everyone.

## Getting Started

1. If you're fixing a bug or typo, feel free to submit a Pull Request directly.
2. For new features or bug fix discussions, open an issue or join our [Discord server](https://discord.gg/Jsey3mX98B) to chat with the community.

## Setting Up Your Environment

1. Clone the repository: `git clone https://github.com/guardrails-ai/guardrails.git`
2. Enter the project directory: `cd guardrails`
3. Install the project in developer mode (use a virtual environment if preferred): `make dev`
4. Install [pre-commit](https://pre-commit.com/): `pre-commit install`

## Development Workflow

Follow these steps before committing your changes:

1. Ensure tests pass: `make test`
2. Format your code: `make autoformat`
3. Run static analysis: `make type`
4. Update documentation if needed. Docs are located in the `docs` directory. You can serve docs using `mkdocs serve`.

### Optional: Pre-Commit Hooks

For convenience, consider [installing the pre-commit hooks](https://pre-commit.com/#installation) provided in the repository. These hooks automatically run tests and formatting checks each time you commit, reducing development overhead.

## Submitting a Pull Request

1. Ensure all tests pass and code is formatted.
2. Create a pull request with a clear description of your changes. Link to relevant issues or discussions. Follow [this guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) if needed.
3. Address any failing checks before requesting a review.
4. Engage in the code review process and make any necessary changes.
5. Celebrate when your pull request is merged! Your changes will be available in the next Guardrails release.

Thank you for your contribution and happy coding!

## Documentation

Docs are served via docusaurus. To serve docs locally, run the following

```bash
# install dependencies
pip install -e ".[dev]"

# install npm dependencies
npm i

# serve the docs
npm run start
```
then navigate to `localhost:3000` in your browser.