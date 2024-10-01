/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// populate examples from examples folder. Only include .mdx and .md files
const { triggerAsyncId } = require("async_hooks");
const fs = require("fs");

// @ts-check
/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */

// get examples from the file called examples-toc.json

const examples = JSON.parse(
  fs.readFileSync("./docusaurus/examples-toc.json", "utf8")
).find((x) => x.label === "Examples");

const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  // tutorialSidebar: [{ type: "autogenerated", dirName: "." }],
  docs: [
    "index",
    "getting_started/guardrails_server",
    "getting_started/quickstart",
    // "getting_started/ai_validation",
    // "getting_started/ml_based",
    // "getting_started/structured_data",
    "getting_started/why_use_guardrails",
    "getting_started/contributing",
    "getting_started/help",
    "faq",
    {
      type: "category",
      label: "Migration Guides",
      collapsed: false,
      items: [
        "migration_guides/0-5-migration",
        "migration_guides/0-4-migration",
        "migration_guides/0-3-migration",
        "migration_guides/0-2-migration",
      ],
    },
  ],
  concepts: [
    "concepts/guard",
    "concepts/validators",
    // "concepts/guardrails",
    "concepts/hub",
    "concepts/deploying",
    "concepts/remote_validation_inference",
    {
      type: "category",
      label: "Streaming",
      collapsed: false,
      items: [
        "concepts/streaming",
        "concepts/async_streaming",
        "concepts/streaming_structured_data",
        "concepts/streaming_fixes",
      ],
    },
    "concepts/concurrency",
    "concepts/logs",
    "concepts/telemetry",
    "concepts/error_remediation",
  ],
  tutorials: [
    {
      type: "category",
      label: "How-to Guides",
      collapsed: false,
      items: [
        "how_to_guides/using_llms",
        "how_to_guides/enable_streaming",
        "how_to_guides/generate_structured_data",
        "how_to_guides/custom_validators",
        "how_to_guides/use_on_fail_actions",
        "how_to_guides/hosting_validator_models",
        "how_to_guides/hosting_with_docker",
        "how_to_guides/continuous_integration_continuous_deployment",
      ],
    },
    {
      type: "category",
      label: "Sample Apps",
      collapsed: false,
      items: ["examples/chatbot", "examples/summarizer"],
    },
    {
      type: "link",
      label: "More Examples",
      href: "https://github.com/guardrails-ai/guardrails/tree/main/docs/examples",
    },
  ],
  integrations: [
    // "integrations/azure_openai",
    "integrations/langchain",
    {
      type: "category",
      label: "Telemetry",
      collapsed: false,
      items: [
        {
          type: "link",
          label: "Arize AI",
          href: "https://docs.arize.com/arize/large-language-models/guardrails",
        },
        "integrations/telemetry/grafana",
        {
          type: "link",
          label: "Iudex",
          href: "https://docs.iudex.ai/integrations/guardrails",
        },
        {
          type: "link",
          label: "OpenLIT",
          href: "https://docs.openlit.io/latest/integrations/guardrails",
        },
        "integrations/telemetry/mlflow-tracing",
      ],
    },
    // "integrations/openai_functions",
  ],
  apiReference: [
    {
      type: "category",
      label: "Python",
      collapsed: true,
      items: [
        {
          type: "autogenerated",
          dirName: "api_reference_markdown",
        },
      ],
    },
    "guardrails_server_api",
    "cli",
  ],
};

module.exports = sidebars;
